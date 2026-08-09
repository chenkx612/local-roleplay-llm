"""Generate frozen SFT, GRPO, dev and evaluation prompts with DeepSeek.

The teacher is prompted with the validated persona and a few in-character examples,
then asked to emit user prompts covering the three layered role-play goals and five
scenario types required by PLAN.md §3.2:

    1. 日常对话
    2. 角色背景与人物关系
    3. 情绪与选择
    4. 语言风格
    5. 出戏与冲突

The MVP target size is intentionally small: 50 SFT / 20 GRPO / 10 dev /
20 eval prompts. Prompt generation uses one reasoning-enabled oversampled batch
per scenario, then filters and backfills locally before the seeded split.

Each record contains the raw user prompt plus local metadata for scenario and
target-goal coverage. Student and Teacher answers are created in later stages,
after these mutually isolated splits have been frozen.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import os
import random
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
DEEPSEEK_MODEL = "deepseek-v4-flash"

MVP_TARGETS: dict[str, int] = {"sft": 50, "grpo": 20, "dev": 10, "eval": 20}
SPLIT_ORDER: tuple[str, ...] = ("sft", "grpo", "dev", "eval")
DEFAULT_SPLIT_SEED = 20260806
PROMPT_GENERATION_TEMPERATURE: float | None = None
PROMPT_GENERATION_THINKING_TYPE = "enabled"
PROMPT_GENERATION_REASONING_EFFORT = "high"
PROMPT_GENERATION_MAX_TOKENS = 8192
SCENARIO_OVERSAMPLE_COUNT = 5
BACKFILL_BATCH_SIZE = 5
NEAR_DUPLICATE_MIN_CHARS = 20
NEAR_DUPLICATE_RATIO = 0.88
MIN_STYLE_EXAMPLES = 10
MAX_STYLE_EXAMPLES = 20
STYLE_RESPONSE_PATTERN = re.compile(r"^（[^（）\r\n]+）[^（）\r\n]+$")

GOALS: tuple[dict[str, str], ...] = (
    {
        "id": "generation_stability",
        "name": "生成稳定性",
        "description": "观察回复是否完整、连贯、可读，无乱码、严重复读或破坏性截断。",
    },
    {
        "id": "character_consistency",
        "name": "角色一致性",
        "description": "观察身份、性格、关系、边界和语言风格是否符合 Persona。",
    },
    {
        "id": "dialogue_quality",
        "name": "对话质量",
        "description": "观察回复是否相关、自然、连贯、有信息量，并能承接后续对话。",
    },
)
GOAL_BY_ID = {goal["id"]: goal for goal in GOALS}
GOAL_LEAK_PATTERNS = (
    "generation_stability",
    "character_consistency",
    "dialogue_quality",
    "target_goals",
    "生成稳定性",
    "角色一致性",
    "对话质量",
    "评测目标",
    "评分标准",
)

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "daily",
        "name": "日常对话",
        "description": "生活化闲聊、日常安排、吃饭出行、兴趣爱好等普通话题。",
        "hints": "用户话题应多样（起居、工作、天气、兴趣、计划等），避免连续追问同一主题。",
        "target_goals": (
            "generation_stability",
            "character_consistency",
            "dialogue_quality",
        ),
    },
    {
        "id": "background",
        "name": "角色背景与人物关系",
        "description": "涉及角色身份、过往经历、想象空间和人物关系的问题。",
        "hints": "问题可以给角色留下合理创作空间，但不要强迫角色接受与核心设定冲突的前提。",
        "target_goals": (
            "generation_stability",
            "character_consistency",
            "dialogue_quality",
        ),
    },
    {
        "id": "emotion",
        "name": "情绪与选择",
        "description": "安慰、道歉、分歧、价值判断、情感表达等需要情绪拿捏的场景。",
        "hints": "用户情绪可正可负（沮丧、吃醋、兴奋、迷茫），角色表达要符合人设性格。",
        "target_goals": (
            "generation_stability",
            "character_consistency",
            "dialogue_quality",
        ),
    },
    {
        "id": "style",
        "name": "语言风格",
        "description": "最能体现角色说话风格、口头禅、语气特征的场景。",
        "hints": "设计能自然引出角色特色语气、用词和节奏的问题，避免直接要求复述示例。",
        "target_goals": (
            "generation_stability",
            "character_consistency",
            "dialogue_quality",
        ),
    },
    {
        "id": "adversarial",
        "name": "出戏与冲突",
        "description": "试图让角色出戏、接受冲突身份，或在压力下偏离性格和关系。",
        "hints": "混合激将、反问和矛盾前提，观察角色能否自然守住核心身份而不是机械拒绝。",
        "target_goals": (
            "generation_stability",
            "character_consistency",
            "dialogue_quality",
        ),
    },
)

# Consecutive teacher batches with no new valid prompts before aborting a scenario.
MAX_CONSECUTIVE_EMPTY = 5


class GenerationShortfallError(RuntimeError):
    """Raised when generation cannot reach the requested number of prompts."""


@dataclass
class GenerationContext:
    persona_text: str
    examples_text: str
    persona_name: str
    user_name: str | None
    role_self_references: tuple[str, ...]


def _extract_user_name(persona: dict[str, Any]) -> str | None:
    """Extract a named user relationship such as ``莲：...`` when available."""
    for relationship in persona.get("relationships", []):
        match = re.match(r"^([^：:,，\s]{1,12})[：:]", relationship)
        if match:
            return match.group(1)
    return None


def _extract_role_self_references(persona: dict[str, Any]) -> tuple[str, ...]:
    """Extract explicit role-only self references from speech-style rules."""
    references: list[str] = []
    for rule in persona.get("speech_style", []):
        for match in re.finditer(r"自称[“\"']([^”\"']+)[”\"']", rule):
            reference = match.group(1).strip()
            if reference and reference not in references:
                references.append(reference)
    return tuple(references)


def _render_examples(examples: list[dict[str, str]]) -> str:
    if not examples:
        return "（暂无样例，请完全依据角色设定生成。）"
    lines = []
    for item in examples:
        lines.append(f"用户：{item['user']}")
        lines.append(f"角色：{item['assistant']}")
    return "\n".join(lines)


def _render_goals() -> str:
    return "\n".join(
        f"- {goal['id']}（{goal['name']}）：{goal['description']}"
        for goal in GOALS
    )


def _scenario_goal_summary(scenario: dict[str, Any]) -> str:
    return "、".join(
        f"{goal_id}（{GOAL_BY_ID[goal_id]['name']}）"
        for goal_id in scenario["target_goals"]
    )


def _contains_goal_leak(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern.lower() in lowered for pattern in GOAL_LEAK_PATTERNS)


def build_teacher_system(
    persona_text: str,
    examples_text: str,
    persona_name: str,
    user_name: str | None = None,
    role_self_references: tuple[str, ...] = (),
) -> str:
    user_identity = user_name or "角色设定中与角色对话的用户"
    self_reference_rule = (
        "、".join(f"“{reference}”" for reference in role_self_references)
        if role_self_references
        else "角色专属的自称或口头禅"
    )
    return (
        "你是一位角色扮演数据准备教师（Teacher）。你的任务是根据给定的角色设定，\n"
        "先在思考中规划整批的覆盖面，再只生成自然、多样的用户 Prompt；"
        "不要生成角色回答。\n\n"
        "【角色设定】\n"
        f"{persona_text}\n\n"
        "【少量表达风格样例】\n"
        f"{examples_text}\n\n"
        "【角色扮演目标（仅用于设计覆盖面，不得写入用户 Prompt）】\n"
        f"{_render_goals()}\n\n"
        "【生成规则】\n"
        "1. 角色设定定义核心身份、性格、关系和边界；允许角色在不冲突的前提下自然发挥。\n"
        "2. 表达风格样例只用于理解表达方式，不得把样例中的用户经历或重大共同关系迁移到新对话。\n"
        f"3. 每条消息的说话人始终是{user_identity}，回复者始终是{persona_name}。"
        "不得交换二者视角。\n"
        f"4. 用户不得用{self_reference_rule}自称，不得从{persona_name}的第一人称"
        f"描述{user_identity}做了什么，也不得询问该怎样回复{user_identity}。\n"
        f"5. 用户的称呼应自然贴合与{persona_name}的关系，消息要像真实即时聊天。\n"
        "6. Prompt 的长度、语气、话题、句式和互动目的应变化，避免同义改写和模板化。\n"
        "7. 每条 Prompt 都必须独立成立，不得引用未提供的上一轮对话。\n"
        "8. 不要在 Prompt 中代替角色回答，也不要要求复述角色设定。\n"
        "9. 不要出现生成稳定性、角色一致性、对话质量、评测目标、评分标准等元数据词。\n\n"
        "【输出要求】\n"
        "严格按 JSON 格式输出，不要包含说明文字或 markdown 标记。\n"
        '格式为：{"prompts": ["用户 Prompt", "..."]}'
    )


def build_scenario_user_prompt(
    scenario: dict[str, Any],
    count: int,
    offset: int,
    split_label: str,
    accepted_prompts: list[str] | None = None,
) -> str:
    scope = "共享候选池" if split_label.upper() == "POOL" else f"{split_label} split"
    accepted_prompts = accepted_prompts or []
    avoidance = ""
    if accepted_prompts:
        rendered = "\n".join(
            f"{index}. {prompt}" for index, prompt in enumerate(accepted_prompts, 1)
        )
        avoidance = (
            "\n【已经接受的 Prompt】\n"
            "下面这些条目已经存在。不要复述、同义改写或延续它们的话题模板：\n"
            f"{rendered}\n"
        )
    batch_kind = "首轮过采样候选" if not accepted_prompts else "定向补充候选"
    return (
        f"请为 {scope} 生成 {count} 条属于【{scenario['name']}】场景的用户 Prompt。\n\n"
        f"本次是{batch_kind}；程序会从候选中筛选，因此必须一次返回 {count} 条。\n"
        f"场景说明：{scenario['description']}\n"
        f"生成提示：{scenario['hints']}\n\n"
        f"本场景主要覆盖目标：{_scenario_goal_summary(scenario)}。\n"
        "这些目标只用于你设计覆盖面，不得写入用户 Prompt；用户消息必须像真实聊天，不能像测试说明。\n\n"
        "多样性要求：\n"
        f"- 这是本场景的第 {offset + 1} 至 {offset + count} 条，请与之前的条目明显不同。\n"
        "- 用户提问的句式、话题、情绪、长度都要有变化。\n"
        "- 不要使用相同的开头词或相同的追问套路。\n\n"
        f"{avoidance}\n"
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
        if isinstance(item, dict):
            user = item.get("user") or item.get("user_input") or item.get("prompt")
        else:
            user = item
        if isinstance(user, str) and user.strip():
            valid.append(user.strip())
    return valid


def normalize_prompt(prompt: str) -> str:
    """Build the conservative comparison key used for exact split deduplication."""
    normalized = unicodedata.normalize("NFKC", prompt)
    return re.sub(r"\s+", " ", normalized).strip()


def _comparison_text(prompt: str) -> str:
    normalized = normalize_prompt(prompt).lower()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def _is_near_duplicate(prompt: str, existing_prompts: list[str]) -> bool:
    candidate = _comparison_text(prompt)
    if len(candidate) < NEAR_DUPLICATE_MIN_CHARS:
        return False
    for existing in existing_prompts:
        comparison = _comparison_text(existing)
        if len(comparison) < NEAR_DUPLICATE_MIN_CHARS:
            continue
        if SequenceMatcher(None, candidate, comparison).ratio() >= NEAR_DUPLICATE_RATIO:
            return True
    return False


def _uses_role_self_reference_as_user(
    prompt: str, role_self_references: tuple[str, ...]
) -> bool:
    for reference in role_self_references:
        escaped = re.escape(reference)
        user_usage = (
            rf"(?:把|跟|对|给|让){escaped}"
            rf"|{escaped}(?:觉得|想|要|是|叫|跟|也|就|刚|今天|现在|心里|总|会|能|该|的伙伴)"
            rf"|是{escaped}的"
        )
        if re.search(user_usage, prompt):
            return True
    return False


def _prompt_quality_issue(
    prompt: str, context: GenerationContext | None
) -> str | None:
    if _contains_goal_leak(prompt):
        return "目标元数据泄漏"
    if context is None:
        return None
    if _uses_role_self_reference_as_user(prompt, context.role_self_references):
        return "用户误用角色自称"
    if context.user_name:
        user_name = re.escape(context.user_name)
        if re.match(rf"^\s*{user_name}[：:,，]", prompt):
            return "用户称呼回复者为自己"
        third_person_actions = (
            "把|问|蹲|跟我|跑|拿|翻|说|忽然|居然|带我|看着我|走过来|坐过来"
        )
        if re.search(rf"{user_name}(?:{third_person_actions})", prompt):
            return "从角色视角描述用户"
    if re.match(r"^\s*(?:那是什么感觉|那件事|刚才说的(?:那个|那件事))", prompt):
        return "依赖未提供的上文"
    return None


def _scenario_distribution(total: int) -> dict[str, int]:
    """Evenly split a total across the five scenarios (remainder to the first ones)."""
    base, remainder = divmod(total, len(SCENARIOS))
    return {s["id"]: base + (1 if i < remainder else 0) for i, s in enumerate(SCENARIOS)}


def _split_scenario_distributions(
    targets: dict[str, int] = MVP_TARGETS,
) -> dict[str, dict[str, int]]:
    """Return the per-split scenario quotas used after pool generation."""
    return {
        split: _scenario_distribution(targets[split])
        for split in SPLIT_ORDER
    }


def _candidate_pool_distribution(
    targets: dict[str, int] = MVP_TARGETS,
) -> dict[str, int]:
    """Build the shared-pool scenario quota needed to satisfy every split."""
    distribution = {scenario["id"]: 0 for scenario in SCENARIOS}
    for split_distribution in _split_scenario_distributions(targets).values():
        for scenario_id, count in split_distribution.items():
            distribution[scenario_id] += count
    return distribution


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
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "thinking": {"type": PROMPT_GENERATION_THINKING_TYPE},
            "reasoning_effort": PROMPT_GENERATION_REASONING_EFFORT,
        },
    }
    if PROMPT_GENERATION_TEMPERATURE is not None:
        request["temperature"] = PROMPT_GENERATION_TEMPERATURE
    response = client.chat.completions.create(
        **request
    )
    return response.choices[0].message.content or ""


def _generate_for_scenario(
    client: OpenAI,
    model: str,
    scenario: dict[str, Any],
    total: int,
    context: GenerationContext,
    max_tokens: int,
    label: str,
    seen: set[str],
    accepted_pool: list[str],
) -> list[dict[str, Any]]:
    """Generate one oversampled batch, then backfill only rejected shortfalls."""
    if total <= 0:
        return []

    system = build_teacher_system(
        context.persona_text,
        context.examples_text,
        context.persona_name,
        context.user_name,
        context.role_self_references,
    )
    collected: list[dict[str, Any]] = []
    consecutive_empty = 0
    request_index = 0

    while len(collected) < total:
        missing = total - len(collected)
        size = (
            total + SCENARIO_OVERSAMPLE_COUNT
            if request_index == 0
            else max(BACKFILL_BATCH_SIZE, missing)
        )
        accepted_users = [record["user"] for record in collected]
        user_prompt = build_scenario_user_prompt(
            scenario,
            size,
            len(collected),
            label,
            accepted_prompts=accepted_users,
        )
        raw = _call_teacher(client, model, system, user_prompt, max_tokens=max_tokens)
        parsed = parse_prompts(raw)
        rejection_counts: dict[str, int] = {}
        accepted = _take_unique_prompts(
            parsed,
            seen,
            missing,
            context=context,
            existing_prompts=accepted_pool,
            rejection_counts=rejection_counts,
        )
        if not accepted:
            print(
                f"  [{label}/{scenario['id']}] 无新增有效 Prompt，重试一次",
                file=sys.stderr,
            )
            time.sleep(1.0)
            raw = _call_teacher(client, model, system, user_prompt, max_tokens=max_tokens)
            parsed = parse_prompts(raw)
            accepted = _take_unique_prompts(
                parsed,
                seen,
                missing,
                context=context,
                existing_prompts=accepted_pool,
                rejection_counts=rejection_counts,
            )

        if accepted:
            collected.extend(
                _build_prompt_candidate(prompt, scenario) for prompt in accepted
            )
            consecutive_empty = 0
            rejection_summary = ""
            if rejection_counts:
                rejection_summary = "，过滤 " + "、".join(
                    f"{reason} {count} 条"
                    for reason, count in sorted(rejection_counts.items())
                )
            print(
                f"  [{label}/{scenario['id']}] +{len(accepted)} 条"
                f"（累计 {len(collected)}/{total}{rejection_summary}）"
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
        request_index += 1
        time.sleep(0.3)

    return collected[:total]


def _take_unique_prompts(
    prompts: list[str],
    seen: set[str],
    limit: int,
    *,
    context: GenerationContext | None = None,
    existing_prompts: list[str] | None = None,
    rejection_counts: dict[str, int] | None = None,
) -> list[str]:
    existing_prompts = existing_prompts if existing_prompts is not None else []
    accepted: list[str] = []
    for prompt in prompts:
        issue = _prompt_quality_issue(prompt, context)
        if issue:
            if rejection_counts is not None:
                rejection_counts[issue] = rejection_counts.get(issue, 0) + 1
            continue
        key = normalize_prompt(prompt)
        if not key or key in seen:
            if rejection_counts is not None:
                rejection_counts["精确重复"] = rejection_counts.get("精确重复", 0) + 1
            continue
        if _is_near_duplicate(prompt, existing_prompts + accepted):
            if rejection_counts is not None:
                rejection_counts["高相似重复"] = rejection_counts.get("高相似重复", 0) + 1
            continue
        seen.add(key)
        existing_prompts.append(prompt)
        accepted.append(prompt)
        if len(accepted) >= limit:
            break
    return accepted


def _build_prompt_candidate(prompt: str, scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "user": prompt,
        "scenario": scenario["id"],
        "target_goals": list(scenario["target_goals"]),
    }


def _generate_all_scenarios(
    client: OpenAI,
    model: str,
    total: int,
    context: GenerationContext,
    max_tokens: int,
    label: str,
    seen: set[str],
    scenario_distribution: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    dist = scenario_distribution or _scenario_distribution(total)
    print(f"[{label}] 共 {total} 条，场景分配：{dist}")
    all_prompts: list[dict[str, Any]] = []
    accepted_pool: list[str] = []
    for scenario in SCENARIOS:
        n = dist[scenario["id"]]
        if n == 0:
            continue
        print(f"[{label}] 场景：{scenario['name']}（目标 {n} 条）")
        prompts = _generate_for_scenario(
            client,
            model,
            scenario,
            n,
            context,
            max_tokens,
            label,
            seen,
            accepted_pool,
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


def _generate_candidate_pool(
    client: OpenAI,
    model: str,
    context: GenerationContext,
    max_tokens: int,
    seen: set[str],
) -> list[dict[str, Any]]:
    distribution = _candidate_pool_distribution()
    total = sum(distribution.values())
    return _generate_all_scenarios(
        client,
        model,
        total,
        context,
        max_tokens=max_tokens,
        label="POOL",
        seen=seen,
        scenario_distribution=distribution,
    )


def split_candidate_pool(
    candidate_pool: list[dict[str, Any]], split_seed: int = DEFAULT_SPLIT_SEED
) -> dict[str, list[dict[str, Any]]]:
    """Split the shared candidate pool into isolated splits with a fixed seed."""
    scenario_targets = _candidate_pool_distribution()
    split_targets = _split_scenario_distributions()
    by_scenario = {scenario["id"]: [] for scenario in SCENARIOS}
    for candidate in candidate_pool:
        scenario_id = candidate.get("scenario")
        if scenario_id not in by_scenario:
            raise GenerationShortfallError(f"未知场景：{scenario_id}")
        by_scenario[scenario_id].append(candidate)

    mismatches = []
    for scenario_id, target in scenario_targets.items():
        actual = len(by_scenario[scenario_id])
        if actual != target:
            mismatches.append(f"{scenario_id}: {actual}/{target}")
    if mismatches:
        raise GenerationShortfallError(
            "候选池场景数量不匹配，无法切分：" + "；".join(mismatches)
        )

    rng = random.Random(split_seed)
    for scenario in SCENARIOS:
        rng.shuffle(by_scenario[scenario["id"]])

    cursors = {scenario["id"]: 0 for scenario in SCENARIOS}
    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    for split in SPLIT_ORDER:
        for scenario in SCENARIOS:
            scenario_id = scenario["id"]
            count = split_targets[split][scenario_id]
            start = cursors[scenario_id]
            stop = start + count
            splits[split].extend(by_scenario[scenario_id][start:stop])
            cursors[scenario_id] = stop
        rng.shuffle(splits[split])

    _validate_prompt_splits(splits)
    return splits


def _validate_prompt_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    expected = {split: MVP_TARGETS[split] for split in SPLIT_ORDER}
    all_keys: set[str] = set()
    shortfalls = []
    for split in SPLIT_ORDER:
        actual = len(splits.get(split, []))
        if actual != expected[split]:
            shortfalls.append(f"{split}: {actual}/{expected[split]}")
            continue
        for record in splits[split]:
            key = normalize_prompt(record["user"])
            if key in all_keys:
                raise GenerationShortfallError(
                    f"切分后发现精确重复 Prompt：{record['user']}"
                )
            all_keys.add(key)
    if shortfalls:
        raise GenerationShortfallError(
            "切分数量未达目标，已跳过写盘以保留已有产物："
            + "；".join(shortfalls)
        )


def format_prompt_records(
    prompts: list[dict[str, Any]], split_label: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    prefix = split_label.lower()
    for index, prompt in enumerate(prompts, 1):
        records.append(
            {
                "id": f"{prefix}_{index:04d}",
                "scenario": prompt["scenario"],
                "target_goals": list(prompt["target_goals"]),
                "user": prompt["user"],
            }
        )
    return records


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


def _build_run_manifest_extra(
    *, model: str, base_url: str | None, split_seed: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "1.2_data_specs_and_splits",
        "data": {
            "targets": {split: MVP_TARGETS[split] for split in SPLIT_ORDER},
            "candidate_pool_size": sum(MVP_TARGETS.values()),
            "initial_candidate_target": (
                sum(MVP_TARGETS.values())
                + len(SCENARIOS) * SCENARIO_OVERSAMPLE_COUNT
            ),
            "scenario_targets": {
                "candidate_pool": _candidate_pool_distribution(),
                "splits": _split_scenario_distributions(),
            },
        },
        "split": {
            "method": "shared_candidate_pool_then_seeded_per_scenario_split",
            "seed": split_seed,
            "deduplication": {
                "scope": "global_candidate_pool",
                "match_type": "exact_after_normalization_and_near_duplicate_ratio",
                "normalization": "Unicode NFKC, strip ends, collapse whitespace",
                "near_duplicate_ratio": NEAR_DUPLICATE_RATIO,
                "near_duplicate_min_chars": NEAR_DUPLICATE_MIN_CHARS,
            },
        },
        "prompt_generation": {
            "provider": "deepseek",
            "base_url": base_url or DEEPSEEK_BASE_URL,
            "model": model,
            "temperature": PROMPT_GENERATION_TEMPERATURE,
            "max_tokens": PROMPT_GENERATION_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "thinking": {"type": PROMPT_GENERATION_THINKING_TYPE},
            "reasoning_effort": PROMPT_GENERATION_REASONING_EFFORT,
            "strategy": {
                "mode": "scenario_batch_oversample_filter_backfill",
                "oversample_per_scenario": SCENARIO_OVERSAMPLE_COUNT,
                "backfill_batch_size": BACKFILL_BATCH_SIZE,
                "quality_filters": [
                    "goal_metadata_leak",
                    "speaker_perspective",
                    "missing_context",
                    "exact_duplicate",
                    "near_duplicate",
                ],
            },
        },
    }


def _input_snapshot_bundle(
    output_dir: Path,
    persona_snapshot: bytes,
    examples_snapshot: bytes,
    system_prompt_snapshot: bytes,
    manifest_extra: dict[str, Any] | None = None,
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
    if manifest_extra:
        manifest.update(manifest_extra)
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
    persona_path: Path,
    examples_path: Path,
    output_dir: Path,
    *,
    model: str = DEEPSEEK_MODEL,
    base_url: str | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
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
        _build_run_manifest_extra(
            model=model, base_url=base_url, split_seed=split_seed
        ),
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
    split_seed: int = DEFAULT_SPLIT_SEED,
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
        user_name=_extract_user_name(persona),
        role_self_references=_extract_role_self_references(persona),
    )

    if client is None:
        client = build_deepseek_client(api_key=api_key, base_url=base_url)

    print(f"=== 开始生成 MVP Prompt，角色：{persona['name']} ===")

    # Generate a shared candidate pool before splitting. The shared set enforces
    # pool-wide exact deduplication, and the fixed seed makes split assignment
    # reproducible while preserving the planned scenario quotas per split.
    seen: set[str] = set()
    candidate_pool = _generate_candidate_pool(
        client,
        model,
        context,
        max_tokens=PROMPT_GENERATION_MAX_TOKENS,
        seen=seen,
    )
    splits = split_candidate_pool(candidate_pool, split_seed=split_seed)

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
        _build_run_manifest_extra(
            model=model, base_url=base_url, split_seed=split_seed
        ),
    )

    write_file_bundle(
        {
            sft_path: _jsonl_bytes(format_prompt_records(splits["sft"], "sft")),
            rl_path: _jsonl_bytes(format_prompt_records(splits["grpo"], "grpo")),
            dev_path: _jsonl_bytes(format_prompt_records(splits["dev"], "dev")),
            eval_path: _jsonl_bytes(format_prompt_records(splits["eval"], "eval")),
            **snapshot_contents,
        }
    )

    print(f"=== 生成完成 ===")
    print(f"候选池     ：{len(candidate_pool)} 条，split seed={split_seed}")
    print(f"SFT Prompt ：{len(splits['sft'])} 条 -> {sft_path}")
    print(f"GRPO Prompt：{len(splits['grpo'])} 条 -> {rl_path}")
    print(f"Dev Prompt ：{len(splits['dev'])} 条 -> {dev_path}")
    print(f"Eval Prompt：{len(splits['eval'])} 条 -> {eval_path}")
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
        required=True,
        help="新运行产物目录（必填，避免覆盖已冻结 run）",
    )
    parser.add_argument(
        "--api-key", default=None, help="DeepSeek API Key（默认读 DEEPSEEK_API_KEY）"
    )
    parser.add_argument("--base-url", default=None, help="自定义 API base URL")
    parser.add_argument("--model", default=DEEPSEEK_MODEL, help="DeepSeek 模型名")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="共享候选池切分随机种子",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="仅校验并保存输入快照与 system prompt，不调用 API",
    )
    args = parser.parse_args()

    examples_path = args.style_examples or (
        args.persona.parent / "style_examples.jsonl"
    )
    output_dir = args.output_dir
    try:
        if args.snapshot_only:
            paths = save_input_snapshot(
                args.persona,
                examples_path,
                output_dir,
                model=args.model,
                base_url=args.base_url,
                split_seed=args.split_seed,
            )
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
            split_seed=args.split_seed,
        )
    except (OSError, ValueError, PersonaValidationError, GenerationShortfallError) as exc:
        raise SystemExit(f"生成失败：{exc}") from exc


if __name__ == "__main__":
    main()
