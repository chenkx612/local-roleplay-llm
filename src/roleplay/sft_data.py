"""Build Student-aware SFT data from frozen training prompts.

For every frozen prompt, a small Student model first produces a baseline answer.
A stronger Teacher then audits that answer and either keeps it or makes the
smallest sufficient correction. The outputs and their provenance are committed together
after every completed item so an interrupted run can safely resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

from .datagen import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    _render_examples,
    load_examples as load_style_examples,
    write_jsonl_bundle,
)
from .inference import (
    MAX_TOKENS,
    PRESENCE_CONTEXT_SIZE,
    PRESENCE_PENALTY,
    REPETITION_CONTEXT_SIZE,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    BaselineGenerationError,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    generate_with_retry,
)
from .persona import PersonaValidationError, load_persona, render_persona_prompt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_REVISION = "674aaa7240b91e8012fcad5d791b7dfe5ba90207"
MAX_TEACHER_ATTEMPTS = 3
TEACHER_MAX_TOKENS = 1536
TEACHER_TEMPERATURE = 0.2
TEACHER_PROMPT_VERSION = 3
METADATA_SCHEMA_VERSION = 1
DECISIONS = {"keep", "light_rewrite", "rewrite"}
SCORE_NAMES = {"persona", "grounding", "style", "format", "quality"}
AUDIT_FIELDS = {
    "user",
    "baseline_assistant",
    "scores",
    "issues",
    "decision",
    "improved_assistant",
}


class StudentAwareSFTError(RuntimeError):
    """Raised when Student-aware SFT generation or resume validation fails."""


def load_prompts(path: Path) -> list[str]:
    """Load a frozen prompt JSONL file and require the exact ``{"user": ...}`` shape."""
    prompts: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Prompt 文件第 {line_no} 行不是合法 JSON: {exc.msg}"
                ) from exc
            if (
                not isinstance(record, dict)
                or set(record) != {"user"}
                or not isinstance(record["user"], str)
                or not record["user"].strip()
            ):
                raise ValueError(
                    f"Prompt 文件第 {line_no} 行必须是仅含非空 user 的对象"
                )
            prompts.append(record["user"])
    if not prompts:
        raise ValueError("Prompt 文件没有有效记录")
    if len(prompts) != len(set(prompts)):
        raise ValueError("Prompt 文件包含精确重复的 user")
    return prompts


def build_teacher_system(persona_text: str, examples_text: str) -> str:
    """Build the fixed Teacher instructions for auditing one Student answer."""
    return (
        "你是一位严格的角色扮演 SFT 数据教师（Teacher）。你要审计 Student 的回答，"
        "并返回一个 JSON 对象。\n\n"
        "【角色设定】\n"
        f"{persona_text}\n\n"
        "【表达风格样例】\n"
        f"{examples_text}\n\n"
        "【事实边界】\n"
        "1. 角色设定是身份、关系、经历、事实和边界的唯一来源。\n"
        "2. 风格样例只用于学习表达方式，其中的事实和共同经历不得作为依据。\n"
        "3. 用户提问中的预设、猜测和二选一不是事实，不能据此确认其中任何一个选项。\n"
        "4. 不得自行补全菜名、家庭成员、出生地、教育、工作履历、旧识或共同经历。\n"
        "5. 对 persona 未说明的信息必须承认不知道、记不清，或自然地向用户确认。\n\n"
        "【审计规则】\n"
        "分别按 0～10 的整数评价角色一致性 persona、事实依据 grounding、"
        "表达风格 style、格式契约 format 和对话质量 quality，并具体列出 issues。\n"
        "format 必须检查回答是否以一组闭合的全角括号描述简短动作或神态，"
        "随后进入口语对白，且没有长篇旁白、额外标签或多层括号。\n"
        "回答完全合格时 decision=keep，improved_assistant 必须逐字保留 baseline。\n"
        "只需少量修正时 decision=light_rewrite；存在明显问题时 decision=rewrite。\n"
        "改写必须是最小充分修改，保持自然、相关、可继续对话，不要解释审计过程。\n\n"
        "【输出格式】\n"
        "只输出 JSON，不要 markdown。字段必须严格为：\n"
        '{"scores":{"persona":0,"grounding":0,"style":0,"format":0,"quality":0},'
        '"issues":["具体问题"],"decision":"keep | light_rewrite | rewrite",'
        '"improved_assistant":"最终回答"}'
    )


def build_teacher_user_prompt(user: str, baseline: str) -> str:
    return (
        "请审计下面这一条对话。\n\n"
        f"【用户消息】\n{user}\n\n"
        f"【Student baseline】\n{baseline}"
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.+?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def parse_teacher_audit(raw: str, user: str, baseline: str) -> dict[str, Any]:
    """Parse and strictly validate a Teacher audit response."""
    try:
        data = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Teacher 返回的不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Teacher 返回值必须是 JSON 对象")

    expected = {"scores", "issues", "decision", "improved_assistant"}
    if set(data) != expected:
        raise ValueError(
            f"Teacher 字段必须严格为 {sorted(expected)}，实际 {sorted(data)}"
        )

    scores = data["scores"]
    if not isinstance(scores, dict) or set(scores) != SCORE_NAMES:
        raise ValueError(f"scores 必须严格包含 {sorted(SCORE_NAMES)}")
    for name, score in scores.items():
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
            raise ValueError(f"scores.{name} 必须是 0～10 的整数")

    issues = data["issues"]
    if not isinstance(issues, list) or any(
        not isinstance(issue, str) or not issue.strip() for issue in issues
    ):
        raise ValueError("issues 必须是由非空字符串组成的数组")

    decision = data["decision"]
    if decision not in DECISIONS:
        raise ValueError(f"decision 必须是 {sorted(DECISIONS)} 之一")

    improved = data["improved_assistant"]
    if not isinstance(improved, str) or not improved.strip():
        raise ValueError("improved_assistant 必须是非空字符串")
    if decision == "keep" and improved != baseline:
        raise ValueError("decision=keep 时 improved_assistant 必须逐字等于 baseline")
    if decision != "keep" and improved == baseline:
        raise ValueError("rewrite decision 必须实际修改 baseline")

    return {
        "user": user,
        "baseline_assistant": baseline,
        "scores": scores,
        "issues": issues,
        "decision": decision,
        "improved_assistant": improved,
    }


def call_teacher_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user: str,
    baseline: str,
    *,
    item_label: str,
) -> tuple[dict[str, Any], int]:
    """Call and validate the Teacher, retrying API and schema failures."""
    last_error = "未知错误"
    for attempt in range(1, MAX_TEACHER_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": build_teacher_user_prompt(user, baseline),
                    },
                ],
                temperature=TEACHER_TEMPERATURE,
                max_tokens=TEACHER_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            finish_reason = choice.finish_reason or ""
            if finish_reason != "stop":
                raise ValueError(f"finish_reason={finish_reason or 'missing'}")
            audit = parse_teacher_audit(
                choice.message.content or "", user=user, baseline=baseline
            )
            return audit, attempt
        except Exception as exc:
            last_error = str(exc)
            print(
                f"{item_label} Teacher 第 {attempt}/{MAX_TEACHER_ATTEMPTS} "
                f"次尝试失败: {last_error}",
                file=sys.stderr,
            )
    raise StudentAwareSFTError(
        f"{item_label} Teacher 连续 {MAX_TEACHER_ATTEMPTS} 次失败: {last_error}"
    )


def _load_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise StudentAwareSFTError(
                    f"已有产物 {path} 第 {line_no} 行不是合法 JSON: {exc.msg}"
                ) from exc
    return records


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_generation_metadata(
    persona_path: Path,
    examples_path: Path,
    prompts_path: Path,
    *,
    student_model: str,
    student_revision: str,
    student_base_url: str,
    teacher_model: str,
    teacher_base_url: str,
) -> dict[str, Any]:
    """Describe every input and generation setting that must stay fixed on resume."""
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "inputs": {
            "persona_sha256": _file_sha256(persona_path),
            "style_examples_sha256": _file_sha256(examples_path),
            "prompts_sha256": _file_sha256(prompts_path),
        },
        "student": {
            "model": student_model,
            "revision": student_revision,
            "base_url": student_base_url,
            "generation": {
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "presence_penalty": PRESENCE_PENALTY,
                "presence_context_size": PRESENCE_CONTEXT_SIZE,
                "repetition_penalty": REPETITION_PENALTY,
                "repetition_context_size": REPETITION_CONTEXT_SIZE,
                "enable_thinking": False,
            },
        },
        "teacher": {
            "model": teacher_model,
            "base_url": teacher_base_url,
            "temperature": TEACHER_TEMPERATURE,
            "max_tokens": TEACHER_MAX_TOKENS,
            "prompt_version": TEACHER_PROMPT_VERSION,
        },
    }


def _validate_resume_records(
    prompts: list[str],
    system_prompt: str,
    baseline_records: list[Any],
    audit_records: list[Any],
    train_records: list[Any],
) -> None:
    lengths = {len(baseline_records), len(audit_records), len(train_records)}
    if len(lengths) != 1:
        raise StudentAwareSFTError("已有三份产物行数不一致，不能安全恢复")
    if len(baseline_records) > len(prompts):
        raise StudentAwareSFTError("已有产物多于当前 Prompt，不能安全恢复")

    for index, (baseline, audit, train) in enumerate(
        zip(baseline_records, audit_records, train_records), 1
    ):
        user = prompts[index - 1]
        if (
            not isinstance(baseline, dict)
            or set(baseline) != {"user", "assistant", "finish_reason", "attempts"}
            or baseline.get("user") != user
            or not isinstance(baseline.get("assistant"), str)
            or not baseline["assistant"].strip()
            or baseline.get("finish_reason") != "stop"
            or isinstance(baseline.get("attempts"), bool)
            or not isinstance(baseline.get("attempts"), int)
            or baseline["attempts"] < 1
        ):
            raise StudentAwareSFTError(f"已有 baseline 第 {index} 行结构或内容无效")
        if (
            not isinstance(audit, dict)
            or set(audit) != AUDIT_FIELDS
            or audit.get("user") != user
            or audit.get("baseline_assistant") != baseline["assistant"]
        ):
            raise StudentAwareSFTError(f"已有 audit 第 {index} 行与 baseline 不对齐")
        try:
            validated_audit = parse_teacher_audit(
                json.dumps(
                    {
                        key: audit[key]
                        for key in audit
                        if key not in {"user", "baseline_assistant"}
                    },
                    ensure_ascii=False,
                ),
                user,
                baseline["assistant"],
            )
        except ValueError as exc:
            raise StudentAwareSFTError(
                f"已有 audit 第 {index} 行无效: {exc}"
            ) from exc
        expected_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": validated_audit["improved_assistant"],
            },
        ]
        if not isinstance(train, dict) or train != {"messages": expected_messages}:
            raise StudentAwareSFTError(f"已有训练数据第 {index} 行与审计记录不对齐")


def load_resume_bundle(
    prompts: list[str],
    system_prompt: str,
    paths: tuple[Path, Path, Path, Path],
    expected_metadata: dict[str, Any],
    *,
    restart: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a valid aligned prefix, or start empty when explicitly requested."""
    if restart:
        return [], [], []
    existing = [path.exists() for path in paths]
    if not any(existing):
        return [], [], []
    if not all(existing):
        raise StudentAwareSFTError(
            "只检测到部分已有产物；请修复文件或使用 --restart 显式重建"
        )
    metadata_records = _load_jsonl(paths[3])
    if metadata_records != [expected_metadata]:
        raise StudentAwareSFTError(
            "已有产物的输入或生成配置与本次运行不一致；请使用原配置，"
            "或使用 --restart 显式重建"
        )
    baseline_records = _load_jsonl(paths[0])
    audit_records = _load_jsonl(paths[1])
    train_records = _load_jsonl(paths[2])
    _validate_resume_records(
        prompts, system_prompt, baseline_records, audit_records, train_records
    )
    return baseline_records, audit_records, train_records


def build_teacher_client(api_key: str | None, base_url: str | None) -> OpenAI:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise StudentAwareSFTError(
            "未提供 Teacher API Key；请设置 DEEPSEEK_API_KEY "
            "或使用 --teacher-api-key"
        )
    return OpenAI(base_url=base_url or DEEPSEEK_BASE_URL, api_key=key)


def run_student_aware_sft(
    persona_path: Path,
    examples_path: Path,
    prompts_path: Path,
    output_dir: Path,
    student_model: str,
    student_base_url: str,
    teacher_model: str,
    student_revision: str = DEFAULT_STUDENT_REVISION,
    teacher_base_url: str | None = None,
    teacher_api_key: str | None = None,
    student_api_key: str = "none",
    student_client: OpenAI | None = None,
    teacher_client: OpenAI | None = None,
    restart: bool = False,
) -> dict[str, Path]:
    """Generate, audit, and atomically publish an aligned Student-aware dataset."""
    persona = load_persona(persona_path)
    system_prompt = render_persona_prompt(persona)
    examples = load_style_examples(examples_path)
    prompts = load_prompts(prompts_path)
    teacher_system = build_teacher_system(system_prompt, _render_examples(examples))
    resolved_teacher_base_url = teacher_base_url or DEEPSEEK_BASE_URL
    metadata = build_generation_metadata(
        persona_path,
        examples_path,
        prompts_path,
        student_model=student_model,
        student_revision=student_revision,
        student_base_url=student_base_url,
        teacher_model=teacher_model,
        teacher_base_url=resolved_teacher_base_url,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "sft_baseline_outputs.jsonl"
    audit_path = output_dir / "sft_teacher_edits.jsonl"
    train_path = output_dir / "sft_train.jsonl"
    metadata_path = output_dir / "sft_generation_meta.json"
    paths = (baseline_path, audit_path, train_path, metadata_path)
    baseline_records, audit_records, train_records = load_resume_bundle(
        prompts, system_prompt, paths, metadata, restart=restart
    )

    start = len(baseline_records)
    if start:
        print(f"检测到 {start}/{len(prompts)} 条有效进度，从下一条恢复")
    if start < len(prompts):
        if student_client is None:
            student_client = OpenAI(base_url=student_base_url, api_key=student_api_key)
        if teacher_client is None:
            teacher_client = build_teacher_client(
                teacher_api_key, resolved_teacher_base_url
            )
    for offset in range(start, len(prompts)):
        user = prompts[offset]
        label = f"[{offset + 1}/{len(prompts)}]"
        answer, finish_reason, student_attempts = generate_with_retry(
            student_client,
            student_model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            item_label=f"{label} Student",
        )
        audit, teacher_attempts = call_teacher_with_retry(
            teacher_client,
            teacher_model,
            teacher_system,
            user,
            answer,
            item_label=label,
        )
        baseline_records.append(
            {
                "user": user,
                "assistant": answer,
                "finish_reason": finish_reason,
                "attempts": student_attempts,
            }
        )
        audit_records.append(audit)
        train_records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                    {
                        "role": "assistant",
                        "content": audit["improved_assistant"],
                    },
                ]
            }
        )
        write_jsonl_bundle(
            {
                baseline_path: baseline_records,
                audit_path: audit_records,
                train_path: train_records,
                metadata_path: [metadata],
            }
        )
        print(
            f"{label} {audit['decision']} "
            f"(Student {student_attempts} 次, Teacher {teacher_attempts} 次)"
        )

    print(f"完成，共 {len(prompts)} 条")
    return {
        "baseline": baseline_path,
        "audit": audit_path,
        "train": train_path,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 Student baseline、Teacher 审计和 ms-swift SFT 数据"
    )
    parser.add_argument(
        "--persona", type=Path, default=ROOT / "data/persona.json", help="角色设定文件"
    )
    parser.add_argument(
        "--style-examples",
        type=Path,
        default=None,
        help="风格样例 JSONL（默认与 persona 同目录）",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "data/sft_train_prompts.jsonl",
        help="冻结的 SFT Train Prompt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data", help="输出文件的目录"
    )
    parser.add_argument("--student-model", default=DEFAULT_MODEL, help="Student 模型名")
    parser.add_argument(
        "--student-revision",
        default=DEFAULT_STUDENT_REVISION,
        help="Student checkpoint revision（用于生成记录和恢复校验）",
    )
    parser.add_argument(
        "--student-base-url", default=DEFAULT_BASE_URL, help="Student API 地址"
    )
    parser.add_argument(
        "--student-api-key", default="none", help="Student API Key（本地服务默认 none）"
    )
    parser.add_argument("--teacher-model", default=DEEPSEEK_MODEL, help="Teacher 模型名")
    parser.add_argument(
        "--teacher-base-url", default=None, help="Teacher API 地址（默认 DeepSeek）"
    )
    parser.add_argument(
        "--teacher-api-key",
        default=None,
        help="Teacher API Key（默认读 DEEPSEEK_API_KEY）",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="忽略已有进度并从第一条开始重建三份产物",
    )
    args = parser.parse_args()
    examples_path = args.style_examples or (
        args.persona.parent / "style_examples.jsonl"
    )
    try:
        run_student_aware_sft(
            persona_path=args.persona,
            examples_path=examples_path,
            prompts_path=args.prompts,
            output_dir=args.output_dir,
            student_model=args.student_model,
            student_revision=args.student_revision,
            student_base_url=args.student_base_url,
            student_api_key=args.student_api_key,
            teacher_model=args.teacher_model,
            teacher_base_url=args.teacher_base_url,
            teacher_api_key=args.teacher_api_key,
            restart=args.restart,
        )
    except (
        OSError,
        ValueError,
        PersonaValidationError,
        BaselineGenerationError,
        StudentAwareSFTError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
