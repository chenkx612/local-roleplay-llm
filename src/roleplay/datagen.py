"""Generate SFT, GRPO and evaluation data by calling a DeepSeek teacher model.

The teacher is prompted with the validated persona and a few in-character examples,
then asked to emit batches of (user, assistant) dialogue pairs covering five
scenario types required by PLAN.md §1.3:

    1. 普通日常对话
    2. 角色背景与人物关系
    3. 情绪和价值选择
    4. 角色风格表达
    5. 诱导出戏与人设冲突

Two generation profiles are supported:

* ``smoke``  - 100 SFT /  30 GRPO /  50 eval (pipeline sanity check)
* ``mvp``    - 300 SFT / 100 GRPO / 100 eval (first real experiment)

SFT records ship with the persona system prompt baked in so they can be fed to
ms-swift directly. GRPO and eval records deliberately store only the raw user
question: the persona prompt is rendered once, at training/inference time, from
``persona.json`` (the single source of truth, see PLAN.md §1.2 / §1.5) so the
three-stage comparison stays fair.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from .persona import load_persona, render_persona_prompt


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

PROFILES: dict[str, dict[str, int]] = {
    "smoke": {"sft": 100, "grpo": 30, "eval": 50},
    "mvp": {"sft": 300, "grpo": 100, "eval": 100},
}

SCENARIOS: tuple[dict[str, str], ...] = (
    {
        "id": "daily",
        "name": "普通日常对话",
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
        "name": "情绪和价值选择",
        "description": "安慰、道歉、分歧、价值判断、情感表达等需要情绪拿捏的场景。",
        "hints": "用户情绪可正可负（沮丧、吃醋、兴奋、迷茫），角色表达要符合人设性格。",
    },
    {
        "id": "style",
        "name": "角色风格表达",
        "description": "最能体现角色说话风格、口头禅、语气特征的场景。",
        "hints": "重点看角色回答的语气、用词、节奏，而非信息量。",
    },
    {
        "id": "adversarial",
        "name": "诱导出戏与人设冲突",
        "description": "试图让角色承认自己是 AI、跳出角色，或引用与人设矛盾的信息。",
        "hints": "用户可采用激将、反问、引用矛盾事实等手法；角色必须保持人设、不承认是模型。",
    },
)

BATCH_SIZE = 5
# Consecutive teacher batches that yield zero valid pairs before giving up on a scenario.
MAX_CONSECUTIVE_EMPTY = 5


class GenerationShortfallError(RuntimeError):
    """Raised when generation cannot reach the requested number of pairs."""


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


def build_teacher_system(persona_text: str, examples_text: str, persona_name: str) -> str:
    return (
        "你是一位角色扮演对话数据生成教师（Teacher）。你的任务是根据给定的角色设定，\n"
        "生成自然、多样、符合人设的用户-角色对话对，用于训练一个小型对话模型。\n\n"
        "【角色设定】\n"
        f"{persona_text}\n\n"
        "【少量代表性对话样例】\n"
        f"{examples_text}\n\n"
        "【生成规则】\n"
        f"1. 用户的称呼应自然贴合角色与用户的关系（如角色称用户为恋人，用户自称可用“我”）。\n"
        "2. 用户的提问要多样：长度、语气、话题都应变化，避免模板化。\n"
        f"3. 角色（{persona_name}）的回答必须严格遵守角色设定的身份、性格、说话风格、关系和边界。\n"
        "4. 角色回答不要复述角色设定原文，不要堆砌口头禅，不要过度热情或机械化。\n"
        "5. 每条对话都应独立成立，不要互相引用或依赖。\n"
        "6. 严禁在角色回答中出现“我是 AI”“我是语言模型”“作为助手”等出戏表达。\n"
        "7. 回答长度以 1～3 句自然对话为主，不要过长。\n\n"
        "【输出要求】\n"
        "严格按 JSON 格式输出，不要包含任何其他说明文字或 markdown 标记。\n"
        '格式为：{"pairs": [{"user": "...", "assistant": "..."}, ...]}'
    )


def build_scenario_user_prompt(scenario: dict[str, str], count: int, offset: int) -> str:
    return (
        f"请生成 {count} 条属于【{scenario['name']}】场景的用户-角色对话对。\n\n"
        f"场景说明：{scenario['description']}\n"
        f"生成提示：{scenario['hints']}\n\n"
        "多样性要求：\n"
        f"- 这是本场景的第 {offset + 1} 至 {offset + count} 条，请与之前的条目明显不同。\n"
        "- 用户提问的句式、话题、情绪、长度都要有变化。\n"
        "- 不要使用相同的开头词或相同的追问套路。\n\n"
        "请严格按 JSON 输出，不要 markdown 代码块包裹：\n"
        '{"pairs": [{"user": "...", "assistant": "..."}, ...]}'
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.+?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def parse_pairs(raw: str) -> list[dict[str, str]]:
    """Extract a list of {user, assistant} dicts from the teacher response."""
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        pairs = data.get("pairs") or data.get("data") or []
    elif isinstance(data, list):
        pairs = data
    else:
        return []

    valid: list[dict[str, str]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        user = item.get("user")
        assistant = item.get("assistant")
        if (
            isinstance(user, str)
            and isinstance(assistant, str)
            and user.strip()
            and assistant.strip()
        ):
            valid.append({"user": user.strip(), "assistant": assistant.strip()})
    return valid


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
    if not path.exists():
        return []
    examples: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item.get("user"), str) and isinstance(item.get("assistant"), str):
                examples.append({"user": item["user"], "assistant": item["assistant"]})
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
) -> list[dict[str, str]]:
    """Keep requesting batches until ``total`` valid pairs are collected.

    Partial teacher responses are kept and the shortfall is requested in later
    batches. After ``MAX_CONSECUTIVE_EMPTY`` batches that yield zero valid pairs
    (each empty response is retried once), raise ``GenerationShortfallError``
    instead of returning a short list.
    """
    if total <= 0:
        return []

    system = build_teacher_system(context.persona_text, context.examples_text, context.persona_name)
    collected: list[dict[str, str]] = []
    offset = 0
    consecutive_empty = 0

    while len(collected) < total:
        size = min(BATCH_SIZE, total - len(collected))
        user_prompt = build_scenario_user_prompt(scenario, size, offset)
        raw = _call_teacher(client, model, system, user_prompt, max_tokens=max_tokens)
        parsed = parse_pairs(raw)
        if not parsed:
            print(f"  [{label}/{scenario['id']}] 空回复，重试一次", file=sys.stderr)
            time.sleep(1.0)
            raw = _call_teacher(client, model, system, user_prompt, max_tokens=max_tokens)
            parsed = parse_pairs(raw)

        if parsed:
            need = total - len(collected)
            taken = parsed[:need]
            collected.extend(taken)
            offset += len(taken)
            consecutive_empty = 0
            print(
                f"  [{label}/{scenario['id']}] +{len(taken)} 条"
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


def _generate_all_scenarios(
    client: OpenAI,
    model: str,
    total: int,
    context: GenerationContext,
    max_tokens: int,
    label: str,
) -> list[dict[str, str]]:
    dist = _scenario_distribution(total)
    print(f"[{label}] 共 {total} 条，场景分配：{dist}")
    all_pairs: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        n = dist[scenario["id"]]
        if n == 0:
            continue
        print(f"[{label}] 场景：{scenario['name']}（目标 {n} 条）")
        pairs = _generate_for_scenario(
            client, model, scenario, n, context, max_tokens, label
        )
        if len(pairs) != n:
            raise GenerationShortfallError(
                f"[{label}/{scenario['id']}] 目标 {n} 条，实际 {len(pairs)} 条"
            )
        all_pairs.extend(pairs)
    if len(all_pairs) != total:
        raise GenerationShortfallError(
            f"[{label}] 目标 {total} 条，实际 {len(all_pairs)} 条"
        )
    return all_pairs


def format_sft_records(pairs: list[dict[str, str]], system_prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair["user"]},
                {"role": "assistant", "content": pair["assistant"]},
            ]
        }
        for pair in pairs
    ]


def format_prompt_records(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"user": pair["user"]} for pair in pairs]


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
    profile: str,
    output_dir: Path,
    client: OpenAI | None = None,
    model: str = DEEPSEEK_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Path]:
    if profile not in PROFILES:
        raise ValueError(f"未知 profile：{profile}（可选 {sorted(PROFILES)}）")
    targets = PROFILES[profile]

    persona = load_persona(persona_path)
    persona_text = render_persona_prompt(persona)
    examples = load_examples(examples_path)
    examples_text = _render_examples(examples)
    context = GenerationContext(
        persona_text=persona_text,
        examples_text=examples_text,
        persona_name=persona["name"],
    )

    if client is None:
        client = build_deepseek_client(api_key=api_key, base_url=base_url)

    print(f"=== 开始生成（profile={profile}）角色：{persona['name']} ===")

    # Generate fully before touching any output paths so a shortfall never
    # overwrites previous complete artifacts.
    sft_pairs = _generate_all_scenarios(
        client, model, targets["sft"], context, max_tokens=2048, label="SFT"
    )
    grpo_pairs = _generate_all_scenarios(
        client, model, targets["grpo"], context, max_tokens=1536, label="GRPO"
    )
    eval_pairs = _generate_all_scenarios(
        client, model, targets["eval"], context, max_tokens=1536, label="EVAL"
    )

    actual = {
        "SFT": len(sft_pairs),
        "GRPO": len(grpo_pairs),
        "EVAL": len(eval_pairs),
    }
    expected = {
        "SFT": targets["sft"],
        "GRPO": targets["grpo"],
        "EVAL": targets["eval"],
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
    sft_path = output_dir / "sft_train.jsonl"
    rl_path = output_dir / "rl_train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    write_jsonl(format_sft_records(sft_pairs, persona_text), sft_path)
    write_jsonl(format_prompt_records(grpo_pairs), rl_path)
    write_jsonl(format_prompt_records(eval_pairs), eval_path)

    print(f"=== 生成完成 ===")
    print(f"SFT ：{len(sft_pairs)} 条 -> {sft_path}")
    print(f"GRPO：{len(grpo_pairs)} 条 -> {rl_path}")
    print(f"EVAL：{len(eval_pairs)} 条 -> {eval_path}")
    return {"sft": sft_path, "rl": rl_path, "eval": eval_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="自动生成 SFT/GRPO/评测数据（DeepSeek API）")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="smoke",
        help="生成档位：smoke（烟测）或 mvp（正式）",
    )
    parser.add_argument(
        "--persona",
        type=Path,
        default=Path("data/persona.json"),
        help="persona.json 路径",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=None,
        help="examples.jsonl 路径（默认与 persona 同目录下的 examples.jsonl）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="输出 JSONL 目录",
    )
    parser.add_argument("--api-key", default=None, help="DeepSeek API Key（默认读 DEEPSEEK_API_KEY）")
    parser.add_argument("--base-url", default=None, help="自定义 API base URL")
    parser.add_argument("--model", default=DEEPSEEK_MODEL, help="DeepSeek 模型名")
    args = parser.parse_args()

    examples_path = args.examples or (args.persona.parent / "examples.jsonl")
    try:
        generate(
            persona_path=args.persona,
            examples_path=examples_path,
            profile=args.profile,
            output_dir=args.output_dir,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        )
    except GenerationShortfallError as exc:
        raise SystemExit(f"生成失败：{exc}") from exc


if __name__ == "__main__":
    main()
