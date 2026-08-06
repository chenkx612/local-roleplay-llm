"""Build baseline-guided, Teacher-corrected SFT data from frozen prompts.

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
    SCENARIOS,
    STYLE_RESPONSE_PATTERN,
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
TEACHER_MAX_TOKENS = 4096
TEACHER_TEMPERATURE: float | None = None
TEACHER_THINKING_TYPE = "enabled"
TEACHER_REASONING_EFFORT = "high"
TEACHER_PROMPT_VERSION = 5
METADATA_SCHEMA_VERSION = 2
DECISIONS = {"keep", "light_rewrite", "rewrite"}
SCORE_NAMES = {"persona", "grounding", "style", "format", "quality"}
FINAL_CHECK_NAMES = {
    "persona_and_addressing",
    "grounding_and_creativity",
    "format",
    "directly_answers_user",
    "natural_dialogue",
}
AUDIT_FIELDS = {
    "user",
    "baseline_assistant",
    "scores",
    "issues",
    "decision",
    "improved_assistant",
    "final_checks",
}
PROMPT_RECORD_FIELDS = {"user"}
PROMPT_METADATA_RECORD_FIELDS = {"id", "scenario", "target_goals", "user"}


class StudentAwareSFTError(RuntimeError):
    """Raised when Teacher-corrected SFT generation or resume validation fails."""


PILOT_SCENARIOS = tuple(scenario["id"] for scenario in SCENARIOS)


def load_prompt_records(path: Path) -> list[dict[str, Any]]:
    """Load and validate frozen prompt records without discarding metadata."""
    records: list[dict[str, Any]] = []
    users: set[str] = set()
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
            record_fields = set(record) if isinstance(record, dict) else set()
            is_metadata_record = record_fields == PROMPT_METADATA_RECORD_FIELDS
            allowed_fields = (PROMPT_RECORD_FIELDS, PROMPT_METADATA_RECORD_FIELDS)
            if (
                not isinstance(record, dict)
                or record_fields not in allowed_fields
                or not isinstance(record["user"], str)
                or not record["user"].strip()
                or (
                    is_metadata_record
                    and (
                        not isinstance(record["id"], str)
                        or not record["id"].strip()
                        or not isinstance(record["scenario"], str)
                        or not record["scenario"].strip()
                        or not isinstance(record["target_goals"], list)
                        or not record["target_goals"]
                        or any(
                            not isinstance(goal, str) or not goal.strip()
                            for goal in record["target_goals"]
                        )
                    )
                )
            ):
                raise ValueError(
                    f"Prompt 文件第 {line_no} 行必须是仅含非空 user 的对象，"
                    "或包含 id、scenario、target_goals、user 的 datagen prompt 对象"
                )
            if record["user"] in users:
                raise ValueError("Prompt 文件包含精确重复的 user")
            users.add(record["user"])
            records.append(record)
    if not records:
        raise ValueError("Prompt 文件没有有效记录")
    return records


def load_prompts(path: Path) -> list[str]:
    """Load frozen prompt JSONL records, accepting optional datagen metadata."""
    return [record["user"] for record in load_prompt_records(path)]


def select_pilot_records(path: Path) -> list[dict[str, Any]]:
    """Select the first frozen SFT prompt for each PLAN 1.3 scenario."""
    records = load_prompt_records(path)
    if any(set(record) != PROMPT_METADATA_RECORD_FIELDS for record in records):
        raise ValueError("Pilot 选样需要带 scenario 元数据的 datagen prompt 文件")
    by_scenario: dict[str, dict[str, Any]] = {}
    for record in records:
        scenario = record["scenario"]
        if scenario in PILOT_SCENARIOS and scenario not in by_scenario:
            by_scenario[scenario] = record
    missing = [scenario for scenario in PILOT_SCENARIOS if scenario not in by_scenario]
    if missing:
        raise ValueError(f"Pilot 缺少场景: {', '.join(missing)}")
    return [by_scenario[scenario] for scenario in PILOT_SCENARIOS]


def _write_json(path: Path, data: Any) -> None:
    """Atomically write one human-readable JSON artifact."""
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


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
        "1. persona 是本项目中身份、关系、当前状态和边界的最高优先级依据，"
        "但不是角色所在原作世界的穷尽百科。\n"
        "2. 可以使用与 persona 不冲突、广为确立且与当前问题直接相关的原作设定；"
        "如果不确定是否属于稳定原作事实，不要冒充确定事实。\n"
        "3. 允许为当前回复做非持久性创造，例如当下选择、假设行为、幽默夸张、"
        "普通生活细节和当场表达的喜好；这些内容不应建立后续必须记住的新事实。\n"
        "4. 不得编造用户个人信息、重大关系变化、具体共同经历、"
        "对后续对话有影响的持续状态，或无依据的重大角色经历。\n"
        "5. 用户提问中的预设、猜测和二选一不会自动成为事实；"
        "可以自然接住假设，但不得把它改写成已发生的确定历史。\n"
        "6. 风格样例只用于学习表达方式，其中的具体共同经历不得迁移到新对话。\n\n"
        "【审计规则】\n"
        "分别按 0～10 的整数评价角色一致性 persona、事实依据 grounding、"
        "表达风格 style、格式契约 format 和对话质量 quality，并具体列出 issues。\n"
        "format 必须检查回答是否以一组闭合的全角括号描述简短动作或神态，"
        "随后进入口语对白，且没有长篇旁白、额外标签或多层括号。\n"
        "回答完全合格时 decision=keep，improved_assistant 必须逐字保留 baseline。\n"
        "只需少量修正时 decision=light_rewrite；存在明显问题时 decision=rewrite。\n"
        "改写必须是最小充分修改，保持自然、相关、可继续对话，不要解释审计过程。\n\n"
        "【改写后必做自检】\n"
        "1. 对 improved_assistant 重新做一次独立的事实检查："
        "不能因为某个细节来自 baseline 就保留它。"
        "重点删除无依据的用户事实、重大共同经历和持续状态；"
        "不要误删符合原作的细节或只服务于当前回复的创造性表达。\n"
        "2. 对 improved_assistant 重新做一次独立的格式检查："
        "只能在回答开头有一组全角括号，口语对白中不得再出现任何动作括号。\n"
        "3. 只有改写后回答同时通过事实、persona、style、format 和 quality "
        "检查，才能把 final_checks 的对应字段标为 true。\n"
        "4. 自称、用户称呼和关系用语必须严格遵循 persona，"
        "不得为了文艺感或变化用词而自行替换。\n"
        "5. 先判断用户的核心问题。用户要求具体选择、判断、建议或信息时，"
        "improved_assistant 必须给出直接答案，角色化吐槽不能代替答案。\n\n"
        "【输出格式】\n"
        "只输出 JSON，不要 markdown。字段必须严格为：\n"
        '{"scores":{"persona":0,"grounding":0,"style":0,"format":0,"quality":0},'
        '"issues":["具体问题"],"decision":"keep | light_rewrite | rewrite",'
        '"improved_assistant":"最终回答",'
        '"final_checks":{"persona_and_addressing":true,'
        '"grounding_and_creativity":true,"format":true,'
        '"directly_answers_user":true,"natural_dialogue":true}}'
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

    expected = {
        "scores",
        "issues",
        "decision",
        "improved_assistant",
        "final_checks",
    }
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

    final_checks = data["final_checks"]
    if not isinstance(final_checks, dict) or set(final_checks) != FINAL_CHECK_NAMES:
        raise ValueError(f"final_checks 必须严格包含 {sorted(FINAL_CHECK_NAMES)}")
    if any(type(value) is not bool for value in final_checks.values()):
        raise ValueError("final_checks 的值必须是布尔值")

    return {
        "user": user,
        "baseline_assistant": baseline,
        "scores": scores,
        "issues": issues,
        "decision": decision,
        "improved_assistant": improved,
        "final_checks": final_checks,
    }


def validate_teacher_final_answer(answer: str) -> None:
    """Reject obvious training pollution before accepting a Teacher response."""
    if not STYLE_RESPONSE_PATTERN.fullmatch(answer.strip()):
        raise ValueError(
            "improved_assistant 不符合一组全角括号动作 + 口语对白格式"
        )
    compact = re.sub(r"\s+", "", answer)
    if re.search(r"(.{1,20}?)\1{3,}", compact):
        raise ValueError("improved_assistant 检测到明显连续复读")


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
            request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": build_teacher_user_prompt(user, baseline),
                    },
                ],
                "max_tokens": TEACHER_MAX_TOKENS,
                "response_format": {"type": "json_object"},
                "extra_body": {
                    "thinking": {"type": TEACHER_THINKING_TYPE},
                    "reasoning_effort": TEACHER_REASONING_EFFORT,
                },
            }
            if TEACHER_TEMPERATURE is not None:
                request["temperature"] = TEACHER_TEMPERATURE
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            finish_reason = choice.finish_reason or ""
            if finish_reason != "stop":
                raise ValueError(f"finish_reason={finish_reason or 'missing'}")
            audit = parse_teacher_audit(
                choice.message.content or "", user=user, baseline=baseline
            )
            failed_checks = [
                name for name, passed in audit["final_checks"].items() if not passed
            ]
            if failed_checks:
                raise ValueError(
                    f"Teacher final_checks 未通过: {', '.join(sorted(failed_checks))}"
                )
            validate_teacher_final_answer(audit["improved_assistant"])
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
            "top_p": None,
            "max_tokens": TEACHER_MAX_TOKENS,
            "thinking": {"type": TEACHER_THINKING_TYPE},
            "reasoning_effort": TEACHER_REASONING_EFFORT,
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
            validate_teacher_final_answer(validated_audit["improved_assistant"])
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
    """Generate, audit, and atomically publish aligned Teacher-corrected data."""
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


def _answer_checks(answer: str) -> dict[str, bool]:
    """Run the minimum automatic answer checks required by PLAN 1.3."""
    compact = re.sub(r"\s+", "", answer)
    return {
        "non_empty": bool(answer.strip()),
        "format_contract": bool(STYLE_RESPONSE_PATTERN.fullmatch(answer.strip())),
        "no_obvious_repetition": not bool(
            re.search(r"(.{1,20}?)\1{3,}", compact)
        ),
    }


def build_pilot_report(
    prompt_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact automatic QA report while preserving manual review scope."""
    if not (
        len(prompt_records) == len(baseline_records) == len(audit_records) == 5
    ):
        raise StudentAwareSFTError("Pilot 报告要求 5 条对齐的 prompt/baseline/audit")
    items: list[dict[str, Any]] = []
    for prompt, baseline, audit in zip(
        prompt_records, baseline_records, audit_records
    ):
        improved = audit["improved_assistant"]
        checks = _answer_checks(improved)
        checks["student_finished"] = baseline["finish_reason"] == "stop"
        items.append(
            {
                "id": prompt["id"],
                "scenario": prompt["scenario"],
                "user": prompt["user"],
                "baseline_assistant": baseline["assistant"],
                "decision": audit["decision"],
                "scores": audit["scores"],
                "issues": audit["issues"],
                "improved_assistant": improved,
                "teacher_final_checks": audit["final_checks"],
                "automatic_checks": checks,
                "automatic_pass": all(checks.values()),
            }
        )
    score_names = sorted(SCORE_NAMES)
    return {
        "status": (
            "automatic_checks_passed"
            if all(item["automatic_pass"] for item in items)
            else "manual_review_required"
        ),
        "sample_count": len(items),
        "scenario_count": len({item["scenario"] for item in items}),
        "decision_counts": {
            decision: sum(item["decision"] == decision for item in items)
            for decision in sorted(DECISIONS)
        },
        "mean_scores": {
            name: round(
                sum(item["scores"][name] for item in items) / len(items), 2
            )
            for name in score_names
        },
        "items": items,
        "manual_review": {
            "required": True,
            "criteria": [
                "persona 与事实边界",
                "风格自然度",
                "格式契约",
                "Teacher 是否为最小充分修改",
            ],
        },
    }


def _pilot_report_digest(report: dict[str, Any]) -> str:
    payload = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _render_pilot_review(report: dict[str, Any], report_digest: str) -> str:
    lines = [
        "# Pilot 人工复核",
        "",
        f"- 自动检查状态：`{report['status']}`",
        f"- 样本：{report['sample_count']} 条 / "
        f"{report['scenario_count']} 个场景",
        f"- 报告 SHA256：`{report_digest}`",
        "- 复核结论：[ ] 通过  [ ] 修改后重跑",
        "- 复核人：",
        "- 复核日期：",
        "- 备注：",
        "",
    ]
    for index, item in enumerate(report["items"], 1):
        lines.extend(
            [
                f"## {index}. {item['scenario']} / {item['id']}",
                "",
                f"**User**：{item['user']}",
                "",
                f"**Student baseline**：{item['baseline_assistant']}",
                "",
                f"**Teacher decision**：`{item['decision']}`",
                "",
                f"**Teacher issues**：{json.dumps(item['issues'], ensure_ascii=False)}",
                "",
                f"**Final answer**：{item['improved_assistant']}",
                "",
                f"**Automatic pass**：`{item['automatic_pass']}`",
                "",
                "- [ ] persona / 事实边界合格",
                "- [ ] 风格自然",
                "- [ ] 格式合格",
                "- [ ] Teacher 修改最小且充分",
                "- 备注：",
                "",
            ]
        )
    return "\n".join(lines)


def _publish_pilot_report(
    output_dir: Path,
    pilot_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    report = build_pilot_report(pilot_records, baseline_records, audit_records)
    report_path = output_dir / "pilot_report.json"
    review_path = output_dir / "pilot_review.md"
    _write_json(report_path, report)
    report_digest = _pilot_report_digest(report)
    digest_marker = f"报告 SHA256：`{report_digest}`"
    review_is_current = (
        review_path.exists()
        and digest_marker in review_path.read_text(encoding="utf-8")
    )
    if not review_is_current:
        temp_review_path = review_path.with_name(f"{review_path.name}.tmp")
        try:
            temp_review_path.write_text(
                _render_pilot_review(report, report_digest), encoding="utf-8"
            )
            temp_review_path.replace(review_path)
        finally:
            if temp_review_path.exists():
                temp_review_path.unlink()
    print(f"Pilot 自动检查: {report['status']}；请完成人工复核 {review_path}")
    return report_path, review_path


def run_pilot(
    persona_path: Path,
    examples_path: Path,
    prompts_path: Path,
    output_dir: Path,
    student_model: str,
    student_base_url: str,
    teacher_model: str,
    **kwargs: Any,
) -> dict[str, Path]:
    """Run the isolated five-scenario pilot and publish its review artifacts."""
    pilot_records = select_pilot_records(prompts_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_prompts_path = output_dir / "pilot_prompts.jsonl"
    write_jsonl_bundle({pilot_prompts_path: pilot_records})
    outputs = run_student_aware_sft(
        persona_path=persona_path,
        examples_path=examples_path,
        prompts_path=pilot_prompts_path,
        output_dir=output_dir,
        student_model=student_model,
        student_base_url=student_base_url,
        teacher_model=teacher_model,
        **kwargs,
    )
    baseline_records = _load_jsonl(outputs["baseline"])
    audit_records = _load_jsonl(outputs["audit"])
    report_path, review_path = _publish_pilot_report(
        output_dir, pilot_records, baseline_records, audit_records
    )
    outputs.update(
        {"prompts": pilot_prompts_path, "report": report_path, "review": review_path}
    )
    return outputs


def rerun_pilot_teacher(
    persona_path: Path,
    examples_path: Path,
    prompts_path: Path,
    output_dir: Path,
    student_model: str,
    student_base_url: str,
    teacher_model: str,
    *,
    student_revision: str = DEFAULT_STUDENT_REVISION,
    teacher_base_url: str | None = None,
    teacher_api_key: str | None = None,
    teacher_client: OpenAI | None = None,
) -> dict[str, Path]:
    """Freeze Pilot baselines and rerun only Teacher correction and QA."""
    pilot_records = select_pilot_records(prompts_path)
    pilot_prompts_path = output_dir / "pilot_prompts.jsonl"
    baseline_path = output_dir / "sft_baseline_outputs.jsonl"
    previous_metadata_path = output_dir / "sft_generation_meta.json"
    if not all(
        path.exists()
        for path in (pilot_prompts_path, baseline_path, previous_metadata_path)
    ):
        raise StudentAwareSFTError(
            "Teacher-only 需要已完成的 Pilot prompt、baseline 和 metadata"
        )
    if _load_jsonl(pilot_prompts_path) != pilot_records:
        raise StudentAwareSFTError("现有 Pilot prompt 与当前 SFT 选样不一致")

    baseline_records = _load_jsonl(baseline_path)
    if len(baseline_records) != 5:
        raise StudentAwareSFTError("Teacher-only 要求 5 条完整 baseline")
    for index, (prompt, baseline) in enumerate(
        zip(pilot_records, baseline_records), 1
    ):
        if (
            not isinstance(baseline, dict)
            or baseline.get("user") != prompt["user"]
            or not isinstance(baseline.get("assistant"), str)
            or not baseline["assistant"].strip()
            or baseline.get("finish_reason") != "stop"
        ):
            raise StudentAwareSFTError(f"Teacher-only baseline 第 {index} 条无效")

    persona = load_persona(persona_path)
    system_prompt = render_persona_prompt(persona)
    examples = load_style_examples(examples_path)
    teacher_system = build_teacher_system(system_prompt, _render_examples(examples))
    resolved_teacher_base_url = teacher_base_url or DEEPSEEK_BASE_URL
    metadata = build_generation_metadata(
        persona_path,
        examples_path,
        pilot_prompts_path,
        student_model=student_model,
        student_revision=student_revision,
        student_base_url=student_base_url,
        teacher_model=teacher_model,
        teacher_base_url=resolved_teacher_base_url,
    )
    previous_metadata = _load_jsonl(previous_metadata_path)
    if (
        len(previous_metadata) != 1
        or previous_metadata[0].get("inputs") != metadata["inputs"]
        or previous_metadata[0].get("student") != metadata["student"]
    ):
        raise StudentAwareSFTError(
            "Teacher-only 的输入或 Student 配置与已冻结 baseline 不一致"
        )
    if teacher_client is None:
        teacher_client = build_teacher_client(
            teacher_api_key, resolved_teacher_base_url
        )

    audit_records: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    for index, (prompt, baseline) in enumerate(
        zip(pilot_records, baseline_records), 1
    ):
        audit, attempts = call_teacher_with_retry(
            teacher_client,
            teacher_model,
            teacher_system,
            prompt["user"],
            baseline["assistant"],
            item_label=f"[{index}/5]",
        )
        audit_records.append(audit)
        train_records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt["user"]},
                    {"role": "assistant", "content": audit["improved_assistant"]},
                ]
            }
        )
        print(f"[{index}/5] {audit['decision']} (Teacher {attempts} 次)")

    audit_path = output_dir / "sft_teacher_edits.jsonl"
    train_path = output_dir / "sft_train.jsonl"
    write_jsonl_bundle(
        {
            baseline_path: baseline_records,
            audit_path: audit_records,
            train_path: train_records,
            previous_metadata_path: [metadata],
        }
    )
    report_path, review_path = _publish_pilot_report(
        output_dir, pilot_records, baseline_records, audit_records
    )
    return {
        "prompts": pilot_prompts_path,
        "baseline": baseline_path,
        "audit": audit_path,
        "train": train_path,
        "metadata": previous_metadata_path,
        "report": report_path,
        "review": review_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 Student baseline、Teacher 纠错和 ms-swift SFT 数据"
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
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="执行 PLAN 1.3 Pilot：五类场景各 1 条，产物写入 output-dir/pilot",
    )
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help="冻结已有 Pilot baseline，只重跑 Teacher 纠错（需与 --pilot 同用）",
    )
    args = parser.parse_args()
    if args.teacher_only and not args.pilot:
        parser.error("--teacher-only 必须与 --pilot 同时使用")
    examples_path = args.style_examples or (
        args.persona.parent / "style_examples.jsonl"
    )
    try:
        output_dir = args.output_dir / "pilot" if args.pilot else args.output_dir
        common_args = {
            "persona_path": args.persona,
            "examples_path": examples_path,
            "prompts_path": args.prompts,
            "output_dir": output_dir,
            "student_model": args.student_model,
            "student_revision": args.student_revision,
            "student_base_url": args.student_base_url,
            "teacher_model": args.teacher_model,
            "teacher_base_url": args.teacher_base_url,
            "teacher_api_key": args.teacher_api_key,
        }
        if args.teacher_only:
            rerun_pilot_teacher(**common_args)
        else:
            runner = run_pilot if args.pilot else run_student_aware_sft
            runner(
                **common_args,
                student_api_key=args.student_api_key,
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
