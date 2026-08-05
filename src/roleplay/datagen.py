"""Generate frozen SFT, GRPO, dev and evaluation prompts with DeepSeek.

The teacher is prompted with the validated persona and a few in-character examples,
then asked to emit user prompts covering the five scenario types required by
PLAN.md §1.2:

    1. 日常对话
    2. 角色背景与人物关系
    3. 情绪与选择
    4. 语言风格
    5. 出戏、冲突与未知事实

The MVP target size is intentionally small: 100 SFT / 30 GRPO / 20 dev /
50 eval prompts.

All records contain only the raw user prompt. Student and Teacher answers are
created in later stages, after these mutually isolated splits have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from .persona import PersonaValidationError, load_persona, render_persona_prompt


def _load_dotenv() -> None:
    """Read KEY=VALUE pairs from the project-root .env into os.environ (no override)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_RUN_ID = "morgana-v1"

MVP_TARGETS: dict[str, int] = {"sft": 100, "grpo": 30, "dev": 20, "eval": 50}
MIN_STYLE_EXAMPLES = 10
MAX_STYLE_EXAMPLES = 20
STYLE_RESPONSE_PATTERN = re.compile(r"^（[^（）\r\n]+）[^（）\r\n]+$")

SCENARIOS: tuple[dict[str, str], ...] = (
    {
        "id": "daily",
        "name": "日常对话",
        "description": "生活化闲聊、日常安排、吃饭出行、兴趣爱好等普通话题。",
        "hints": "用户话题应多样（起居、工作、天气、兴趣、计划等），避免连续追问同一主题。",
    },
    {
        "id": "background",
        "name": "角色背景与人物关系",
        "description": "涉及角色身份、过往经历、已知事实、人物关系的问题。",
        "hints": "部分问题可超出已知事实范围，角色应当场承认不知道或向用户确认，不要编造。",
    },
    {
        "id": "emotion",
        "name": "情绪与选择",
        "description": "安慰、道歉、分歧、价值判断、情感表达等需要情绪拿捏的场景。",
        "hints": "用户情绪可正可负（沮丧、吃醋、兴奋、迷茫），角色表达要符合人设性格。",
    },
    {
        "id": "style",
        "name": "语言风格",
        "description": "最能体现角色说话风格、口头禅、语气特征的场景。",
        "hints": "设计能自然引出角色特色语气、用词和节奏的问题，避免直接要求复述示例。",
    },
    {
        "id": "adversarial",
        "name": "出戏、冲突与未知事实",
        "description": "试图让角色出戏、引用冲突信息，或追问设定中不存在的事实和共同经历。",
        "hints": "混合激将、反问、矛盾事实和未知信息；问题本身不要预设角色已经承认或做过某事。",
    },
)

BATCH_SIZE = 5
# Consecutive teacher batches with no new valid prompts before aborting a scenario.
MAX_CONSECUTIVE_EMPTY = 5


class GenerationShortfallError(RuntimeError):
    """Raised when generation cannot reach the requested number of prompts."""


@dataclass
class GenerationContext:
    persona_text: str
    examples_text: str
    persona_name: str


def _render_examples(examples: list[dict[str, str]]) -> str:
    if not examples:
        return "（暂无样例，请完全依据角色设定生成。）"
    lines = []
    for item in examples:
        lines.append(f"用户：{item['user']}")
        lines.append(f"角色：{item['assistant']}")
    return "\n".join(lines)


def build_teacher_system(
    persona_text: str, examples_text: str, persona_name: str
) -> str:
    return (
        "你是一位角色扮演数据准备教师（Teacher）。你的任务是根据给定的角色设定，\n"
        "只生成自然、多样的用户 Prompt；不要生成角色回答。\n\n"
        "【角色设定】\n"
        f"{persona_text}\n\n"
        "【少量表达风格样例】\n"
        f"{examples_text}\n\n"
        "【生成规则】\n"
        "1. 角色设定是身份、关系、经历和事实的唯一来源。\n"
        "2. 表达风格样例只用于理解适合引出何种表达，不得把样例中的事实或共同经历当成设定。\n"
        f"3. 用户的称呼应自然贴合与{persona_name}的关系，问题要像真实即时聊天。\n"
        "4. Prompt 的长度、语气、话题和句式应变化，避免模板化。\n"
        "5. 每条 Prompt 都应独立成立，不要互相引用或依赖。\n"
        "6. 不要在 Prompt 中代替角色回答，也不要要求复述角色设定。\n\n"
        "【输出要求】\n"
        "严格按 JSON 格式输出，不要包含说明文字或 markdown 标记。\n"
        '格式为：{"prompts": ["用户 Prompt", "..."]}'
    )


def build_scenario_user_prompt(
    scenario: dict[str, str], count: int, offset: int, split_label: str
) -> str:
    return (
        f"请为 {split_label} split 生成 {count} 条属于【{scenario['name']}】场景的用户 Prompt。\n\n"
        f"场景说明：{scenario['description']}\n"
        f"生成提示：{scenario['hints']}\n\n"
        "多样性要求：\n"
        f"- 这是本场景的第 {offset + 1} 至 {offset + count} 条，请与之前的条目明显不同。\n"
        "- 用户提问的句式、话题、情绪、长度都要有变化。\n"
        "- 不要使用相同的开头词或相同的追问套路。\n\n"
        "请严格按 JSON 输出，不要 markdown 代码块包裹：\n"
        '{"prompts": ["用户 Prompt", "..."]}'
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.+?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def parse_prompts(raw: str) -> list[str]:
    """Extract non-empty user prompts from a teacher response."""
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        prompts = data.get("prompts") or data.get("data") or []
    elif isinstance(data, list):
        prompts = data
    else:
        return []
    if not isinstance(prompts, list):
        return []

    valid: list[str] = []
    for item in prompts:
        user = item.get("user") if isinstance(item, dict) else item
        if isinstance(user, str) and user.strip():
            valid.append(user.strip())
    return valid


def normalize_prompt(prompt: str) -> str:
    """Build the conservative comparison key used for exact split deduplication."""
    normalized = unicodedata.normalize("NFKC", prompt)
    return re.sub(r"\s+", " ", normalized).strip()


def _scenario_distribution(total: int) -> dict[str, int]:
    """Evenly split a total across the five scenarios (remainder to the first ones)."""
    base, remainder = divmod(total, len(SCENARIOS))
    return {s["id"]: base + (1 if i < remainder else 0) for i, s in enumerate(SCENARIOS)}


def _batch_sizes(total: int) -> list[int]:
    full, tail = divmod(total, BATCH_SIZE)
    sizes = [BATCH_SIZE] * full
    if tail:
        sizes.append(tail)
    return sizes


def load_examples(path: Path) -> list[dict[str, str]]:
    """Load and strictly validate the style examples required by PLAN.md."""
    examples: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"style_examples.jsonl 第 {line_no} 行不是合法 JSON: {exc.msg}"
                ) from exc
            if (
                not isinstance(item, dict)
                or set(item) != {"user", "assistant"}
                or not isinstance(item["user"], str)
                or not item["user"].strip()
                or not isinstance(item["assistant"], str)
                or not item["assistant"].strip()
            ):
                raise ValueError(
                    f"style_examples.jsonl 第 {line_no} 行必须仅含非空 "
                    "user 和 assistant"
                )
            if not STYLE_RESPONSE_PATTERN.fullmatch(item["assistant"]):
                raise ValueError(
                    f"style_examples.jsonl 第 {line_no} 行 assistant 必须遵循"
                    "“（简短动作或神态）口语对白”格式，且不得使用多层括号或换行"
                )
            examples.append(item)
    if not MIN_STYLE_EXAMPLES <= len(examples) <= MAX_STYLE_EXAMPLES:
        raise ValueError(
            "style_examples.jsonl 必须包含 "
            f"{MIN_STYLE_EXAMPLES}～{MAX_STYLE_EXAMPLES} 条有效对话，"
            f"实际 {len(examples)} 条"
        )
    users = [item["user"] for item in examples]
    if len(users) != len(set(users)):
        raise ValueError("style_examples.jsonl 包含精确重复的 user")
    return examples


def _call_teacher(
    client: OpenAI, model: str, system: str, user: str, max_tokens: int
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.95,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _generate_for_scenario(
    client: OpenAI,
    model: str,
    scenario: dict[str, str],
    total: int,
    context: GenerationContext,
    max_tokens: int,
    label: str,
    seen: set[str],
) -> list[str]:
    """Keep requesting batches until ``total`` unique valid prompts are collected.

    Partial teacher responses are kept and the shortfall is requested in later
    batches. After ``MAX_CONSECUTIVE_EMPTY`` batches that yield no new valid
    prompt (each response is retried once), raise ``GenerationShortfallError``.
    """
    if total <= 0:
        return []

    system = build_teacher_system(
        context.persona_text, context.examples_text, context.persona_name
    )
    collected: list[str] = []
    offset = 0
    consecutive_empty = 0

    while len(collected) < total:
        size = min(BATCH_SIZE, total - len(collected))
        user_prompt = build_scenario_user_prompt(scenario, size, offset, label)
        raw = _call_teacher(client, model, system, user_prompt, max_tokens=max_tokens)
        parsed = parse_prompts(raw)
        accepted = _take_unique_prompts(parsed, seen, total - len(collected))
        if not accepted:
            print(
                f"  [{label}/{scenario['id']}] 无新增有效 Prompt，重试一次",
                file=sys.stderr,
            )
            time.sleep(1.0)
            raw = _call_teacher(client, model, system, user_prompt, max_tokens=max_tokens)
            parsed = parse_prompts(raw)
            accepted = _take_unique_prompts(
                parsed, seen, total - len(collected)
            )

        if accepted:
            collected.extend(accepted)
            offset += len(accepted)
            consecutive_empty = 0
            print(
                f"  [{label}/{scenario['id']}] +{len(accepted)} 条"
                f"（累计 {len(collected)}/{total}）"
            )
        else:
            consecutive_empty += 1
            print(
                f"  [{label}/{scenario['id']}] 重试仍为空"
                f"（连续空批 {consecutive_empty}/{MAX_CONSECUTIVE_EMPTY}）",
                file=sys.stderr,
            )
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                raise GenerationShortfallError(
                    f"[{label}/{scenario['id']}] 目标 {total} 条，"
                    f"连续 {MAX_CONSECUTIVE_EMPTY} 次无有效数据后仅得到 {len(collected)} 条，"
                    f"停止生成以免写出残缺产物"
                )
        time.sleep(0.3)

    return collected[:total]


def _take_unique_prompts(
    prompts: list[str], seen: set[str], limit: int
) -> list[str]:
    accepted: list[str] = []
    for prompt in prompts:
        key = normalize_prompt(prompt)
        if not key or key in seen:
            continue
        seen.add(key)
        accepted.append(prompt)
        if len(accepted) >= limit:
            break
    return accepted


def _generate_all_scenarios(
    client: OpenAI,
    model: str,
    total: int,
    context: GenerationContext,
    max_tokens: int,
    label: str,
    seen: set[str],
) -> list[str]:
    dist = _scenario_distribution(total)
    print(f"[{label}] 共 {total} 条，场景分配：{dist}")
    all_prompts: list[str] = []
    for scenario in SCENARIOS:
        n = dist[scenario["id"]]
        if n == 0:
            continue
        print(f"[{label}] 场景：{scenario['name']}（目标 {n} 条）")
        prompts = _generate_for_scenario(
            client, model, scenario, n, context, max_tokens, label, seen
        )
        if len(prompts) != n:
            raise GenerationShortfallError(
                f"[{label}/{scenario['id']}] 目标 {n} 条，实际 {len(prompts)} 条"
            )
        all_prompts.extend(prompts)
    if len(all_prompts) != total:
        raise GenerationShortfallError(
            f"[{label}] 目标 {total} 条，实际 {len(all_prompts)} 条"
        )
    return all_prompts


def format_prompt_records(prompts: list[str]) -> list[dict[str, str]]:
    return [{"user": prompt} for prompt in prompts]


def write_jsonl(records: list[Any], path: Path) -> None:
    """Write JSONL atomically via a sibling ``.tmp`` file then ``replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _jsonl_bytes(records: list[Any]) -> bytes:
    text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    return text.encode("utf-8")


def write_file_bundle(contents_by_path: dict[Path, bytes]) -> None:
    """Atomically publish a group of byte-for-byte files with rollback."""
    temp_paths: dict[Path, Path] = {}
    backup_paths: dict[Path, Path] = {}
    published: set[Path] = set()
    try:
        for path, content in contents_by_path.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + ".tmp")
            temp_paths[path] = tmp_path
            tmp_path.write_bytes(content)

        for path in contents_by_path:
            backup_path = path.with_name(path.name + ".bak")
            if backup_path.exists():
                raise FileExistsError(f"检测到未清理的备份文件：{backup_path}")
            if path.exists():
                backup_paths[path] = backup_path

        for path, backup_path in backup_paths.items():
            path.replace(backup_path)
        for path, tmp_path in temp_paths.items():
            tmp_path.replace(path)
            published.add(path)
    except BaseException:
        for path in published:
            if path not in backup_paths:
                path.unlink(missing_ok=True)
        for path, backup_path in backup_paths.items():
            if backup_path.exists():
                backup_path.replace(path)
        raise
    else:
        for backup_path in backup_paths.values():
            backup_path.unlink()
    finally:
        for tmp_path in temp_paths.values():
            tmp_path.unlink(missing_ok=True)


def write_jsonl_bundle(records_by_path: dict[Path, list[Any]]) -> None:
    """Stage all JSONL files and roll back the bundle if publication fails."""
    write_file_bundle(
        {path: _jsonl_bytes(records) for path, records in records_by_path.items()}
    )


def _input_snapshot_bundle(
    output_dir: Path,
    persona_snapshot: bytes,
    examples_snapshot: bytes,
    system_prompt_snapshot: bytes,
) -> tuple[dict[Path, bytes], dict[str, Path]]:
    snapshot_dir = output_dir / "inputs"
    paths = {
        "persona": snapshot_dir / "persona.json",
        "style_examples": snapshot_dir / "style_examples.jsonl",
        "system_prompt": output_dir / "system_prompt.txt",
        "manifest": output_dir / "input_manifest.json",
    }
    manifest = {
        "persona": {
            "file": str(paths["persona"].relative_to(output_dir)),
            "sha256": hashlib.sha256(persona_snapshot).hexdigest(),
        },
        "style_examples": {
            "file": str(paths["style_examples"].relative_to(output_dir)),
            "sha256": hashlib.sha256(examples_snapshot).hexdigest(),
        },
        "system_prompt": {
            "file": str(paths["system_prompt"].relative_to(output_dir)),
            "sha256": hashlib.sha256(system_prompt_snapshot).hexdigest(),
        },
    }
    contents = {
        paths["persona"]: persona_snapshot,
        paths["style_examples"]: examples_snapshot,
        paths["system_prompt"]: system_prompt_snapshot,
        paths["manifest"]: (
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8"),
    }
    return contents, paths


def save_input_snapshot(
    persona_path: Path, examples_path: Path, output_dir: Path
) -> dict[str, Path]:
    """Validate and freeze the §1.1 inputs without making an API call."""
    persona = load_persona(persona_path)
    system_prompt = render_persona_prompt(persona)
    load_examples(examples_path)
    contents, paths = _input_snapshot_bundle(
        output_dir,
        persona_path.read_bytes(),
        examples_path.read_bytes(),
        (system_prompt + "\n").encode("utf-8"),
    )
    write_file_bundle(contents)
    return paths


def build_deepseek_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit(
            "未提供 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY，"
            "或使用 --api-key 参数。"
        )
    return OpenAI(base_url=base_url or DEEPSEEK_BASE_URL, api_key=key)


def generate(
    persona_path: Path,
    examples_path: Path,
    output_dir: Path,
    client: OpenAI | None = None,
    model: str = DEEPSEEK_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Path]:
    persona = load_persona(persona_path)
    persona_text = render_persona_prompt(persona)
    examples = load_examples(examples_path)
    persona_snapshot = persona_path.read_bytes()
    examples_snapshot = examples_path.read_bytes()
    system_prompt_snapshot = (persona_text + "\n").encode("utf-8")
    examples_text = _render_examples(examples)
    context = GenerationContext(
        persona_text=persona_text,
        examples_text=examples_text,
        persona_name=persona["name"],
    )

    if client is None:
        client = build_deepseek_client(api_key=api_key, base_url=base_url)

    print(f"=== 开始生成 MVP Prompt，角色：{persona['name']} ===")

    # Generate and validate every split before touching output paths. The shared
    # set enforces both within-split and cross-split exact deduplication.
    seen: set[str] = set()
    sft_prompts = _generate_all_scenarios(
        client,
        model,
        MVP_TARGETS["sft"],
        context,
        max_tokens=1536,
        label="SFT",
        seen=seen,
    )
    grpo_prompts = _generate_all_scenarios(
        client,
        model,
        MVP_TARGETS["grpo"],
        context,
        max_tokens=1536,
        label="GRPO",
        seen=seen,
    )
    dev_prompts = _generate_all_scenarios(
        client,
        model,
        MVP_TARGETS["dev"],
        context,
        max_tokens=1536,
        label="DEV",
        seen=seen,
    )
    eval_prompts = _generate_all_scenarios(
        client,
        model,
        MVP_TARGETS["eval"],
        context,
        max_tokens=1536,
        label="EVAL",
        seen=seen,
    )

    actual = {
        "SFT": len(sft_prompts),
        "GRPO": len(grpo_prompts),
        "DEV": len(dev_prompts),
        "EVAL": len(eval_prompts),
    }
    expected = {
        "SFT": MVP_TARGETS["sft"],
        "GRPO": MVP_TARGETS["grpo"],
        "DEV": MVP_TARGETS["dev"],
        "EVAL": MVP_TARGETS["eval"],
    }
    shortfalls = [
        f"{name}: {actual[name]}/{expected[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if shortfalls:
        raise GenerationShortfallError(
            "生成数量未达目标，已跳过写盘以保留已有产物："
            + "；".join(shortfalls)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = output_dir / "sft_train_prompts.jsonl"
    rl_path = output_dir / "rl_train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    eval_path = output_dir / "eval.jsonl"
    snapshot_contents, snapshot_paths = _input_snapshot_bundle(
        output_dir,
        persona_snapshot,
        examples_snapshot,
        system_prompt_snapshot,
    )

    write_file_bundle(
        {
            sft_path: _jsonl_bytes(format_prompt_records(sft_prompts)),
            rl_path: _jsonl_bytes(format_prompt_records(grpo_prompts)),
            dev_path: _jsonl_bytes(format_prompt_records(dev_prompts)),
            eval_path: _jsonl_bytes(format_prompt_records(eval_prompts)),
            **snapshot_contents,
        }
    )

    print(f"=== 生成完成 ===")
    print(f"SFT Prompt ：{len(sft_prompts)} 条 -> {sft_path}")
    print(f"GRPO Prompt：{len(grpo_prompts)} 条 -> {rl_path}")
    print(f"Dev Prompt ：{len(dev_prompts)} 条 -> {dev_path}")
    print(f"Eval Prompt：{len(eval_prompts)} 条 -> {eval_path}")
    print(f"输入快照   ：{snapshot_paths['persona'].parent}")
    print(f"System Prompt：{snapshot_paths['system_prompt']}")
    return {"sft": sft_path, "rl": rl_path, "dev": dev_path, "eval": eval_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成并冻结 SFT/GRPO/Dev/Eval Prompt split（DeepSeek API）"
    )
    parser.add_argument(
        "--persona",
        type=Path,
        default=Path("data/persona.json"),
        help="persona.json 路径",
    )
    parser.add_argument(
        "--style-examples",
        type=Path,
        default=None,
        help="style_examples.jsonl 路径（默认与 persona 同目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"运行产物目录（默认 data/runs/{DEFAULT_RUN_ID}）",
    )
    parser.add_argument(
        "--api-key", default=None, help="DeepSeek API Key（默认读 DEEPSEEK_API_KEY）"
    )
    parser.add_argument("--base-url", default=None, help="自定义 API base URL")
    parser.add_argument("--model", default=DEEPSEEK_MODEL, help="DeepSeek 模型名")
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="仅校验并保存输入快照与 system prompt，不调用 API",
    )
    args = parser.parse_args()

    examples_path = args.style_examples or (
        args.persona.parent / "style_examples.jsonl"
    )
    output_dir = args.output_dir or Path("data/runs") / DEFAULT_RUN_ID
    try:
        if args.snapshot_only:
            paths = save_input_snapshot(args.persona, examples_path, output_dir)
            print(f"输入校验通过，快照已保存：{paths['persona'].parent}")
            print(f"System Prompt：{paths['system_prompt']}")
            return
        generate(
            persona_path=args.persona,
            examples_path=examples_path,
            output_dir=output_dir,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        )
    except (OSError, ValueError, PersonaValidationError, GenerationShortfallError) as exc:
        raise SystemExit(f"生成失败：{exc}") from exc


if __name__ == "__main__":
    main()
