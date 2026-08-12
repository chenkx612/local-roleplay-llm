"""Deterministic continuous rule reward for morgana-v2 Stage 4 GRPO."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from roleplay.sft_eval import (
    has_gibberish,
    has_repeated_span,
    normalize_empty_think_wrapper,
)


PROMPTS_RELATIVE_PATH = Path("data/runs/morgana-v2/rule_grpo_train.jsonl")
PROMPTS_SHA256 = "9a58c6b7011a822900ade1ca31e7dd655559f099d4f41dc58ede977faa4029b2"

INSTRUCTION_WEIGHT = 5.0
PERSONA_WEIGHT = 2.0
STYLE_WEIGHT = 1.0
HARD_INVALID_BASE = -12.0
HARD_INVALID_RECOVERY_RANGE = 2.0
MIN_VALID_REWARD = -8.0
MAX_VALID_REWARD = 8.0

MAX_ACTION_LENGTH = 30
MIN_ACTION_GAP = 12
MAX_ACTION_RATIO = 0.50
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
SIGNATURE_ALIASES = ("吾辈", "吾輩")
WRONG_SELF_ALIASES = ("本大爷", "本喵", "本猫", "俺")
CONSTRAINT_FIELDS = frozenset(
    {
        "min_actions",
        "max_actions",
        "min_sentences",
        "max_sentences",
        "min_chars",
        "max_chars",
        "min_signatures",
        "max_signatures",
    }
)

_QUOTE_PATTERNS = (
    re.compile(r"“[^”]*”"),
    re.compile(r"‘[^’]*’"),
    re.compile(r'"[^"\r\n]*"'),
    re.compile(r"'[^'\r\n]*'"),
)
_WRONG_WO = re.compile(r"(?<![自忘无])我(?!们)")
_TAG = re.compile(r"<[^>\r\n]+>")
_SPEAKER_LABEL = re.compile(
    r"(?:\A|\n)\s*(?:用户|助手|user|assistant|system)\s*[:：]",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[。！？!?]+")


class RuleRewardError(RuntimeError):
    """Raised when the frozen deterministic reward contract is invalid."""


@dataclass(frozen=True)
class RewardConstraints:
    """Prompt-specific targets consumed by one common reward function."""

    min_actions: int | None
    max_actions: int | None
    min_sentences: int | None
    max_sentences: int | None
    min_chars: int
    max_chars: int
    min_signatures: int
    max_signatures: int

    def as_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class ActionAnalysis:
    """Mechanical action-span facts extracted from one completion."""

    segments: tuple[str, ...]
    count: int
    segment_lengths: tuple[int, ...]
    total_action_length: int
    action_ratio: float
    minimum_dialogue_gap: int | None
    unbalanced_count: int
    nested_count: int
    invalid_content_count: int

    def as_log_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["segments"] = list(self.segments)
        value["segment_lengths"] = list(self.segment_lengths)
        return value


@dataclass(frozen=True)
class RuleRewardComponents:
    """Every continuous component used to score one completion."""

    constraints: RewardConstraints
    normalized_length: int
    sentence_count: int
    signature_count: int
    wrong_self_references: tuple[str, ...]
    wrong_self_count: int
    action: ActionAnalysis
    instruction_violation: float
    instruction_score: float
    persona_violation: float
    persona_score: float
    style_violation: float
    style_score: float
    format_reasons: tuple[str, ...]
    hard_invalid_reasons: tuple[str, ...]
    recoverability: float
    raw_reward: float
    total_reward: float

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "constraints": self.constraints.as_dict(),
            "normalized_length": self.normalized_length,
            "sentence_count": self.sentence_count,
            "signature_count": self.signature_count,
            "wrong_self_references": list(self.wrong_self_references),
            "wrong_self_count": self.wrong_self_count,
            "action": self.action.as_log_dict(),
            "instruction_violation": self.instruction_violation,
            "instruction_score": self.instruction_score,
            "persona_violation": self.persona_violation,
            "persona_score": self.persona_score,
            "style_violation": self.style_violation,
            "style_score": self.style_score,
            "format_reasons": list(self.format_reasons),
            "hard_invalid_reasons": list(self.hard_invalid_reasons),
            "recoverability": self.recoverability,
            "raw_reward": self.raw_reward,
            "total_reward": self.total_reward,
        }


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reward_spec() -> dict[str, Any]:
    """Return the serializable frozen reward definition."""
    return {
        "schema_version": 2,
        "name": "morgana_rule_reward",
        "prompt_sha256": PROMPTS_SHA256,
        "valid_reward_range": [MIN_VALID_REWARD, MAX_VALID_REWARD],
        "hard_invalid_reward_range": [
            HARD_INVALID_BASE,
            HARD_INVALID_BASE + HARD_INVALID_RECOVERY_RANGE,
        ],
        "formula": {
            "valid": {
                "instruction_weight": INSTRUCTION_WEIGHT,
                "persona_weight": PERSONA_WEIGHT,
                "style_weight": STYLE_WEIGHT,
            },
            "component_mapping": "(1 - violation) / (1 + violation)",
            "hard_invalid": "-12 + 2 * recoverability",
        },
        "instruction_violation": {
            "action_distance_weight": 1.0,
            "sentence_distance_weight": 0.7,
            "character_distance_scale": 30.0,
            "character_distance_weight": 0.3,
        },
        "persona_violation": {
            "signature_distance_weight": 0.8,
            "wrong_wo_weight": 0.4,
            "wrong_alias_weight": 0.8,
            "signature_aliases": list(SIGNATURE_ALIASES),
            "wrong_aliases": list(WRONG_SELF_ALIASES),
        },
        "style_violation": {
            "length_distance_weight": 0.4,
            "repetition_weight": 0.5,
            "overlong_action_weight": 0.3,
            "action_ratio_weight": 0.3,
            "action_gap_weight": 0.2,
            "format_issue_weight": 0.3,
            "max_action_length": MAX_ACTION_LENGTH,
            "minimum_dialogue_gap": MIN_ACTION_GAP,
            "max_action_ratio": MAX_ACTION_RATIO,
        },
        "hard_invalid_checks": [
            "empty",
            "truncated",
            "gibberish",
            "repeated_span",
            "unbalanced_parentheses",
        ],
    }


def _validate_optional_nonnegative_int(value: Any, name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise RuleRewardError(f"{name} 必须为非负整数或 null")


def parse_constraints(value: Any) -> RewardConstraints:
    """Strictly parse one prompt's continuous reward targets."""
    if not isinstance(value, dict) or set(value) != CONSTRAINT_FIELDS:
        raise RuleRewardError("reward_constraints 字段不正确")
    for name in CONSTRAINT_FIELDS:
        _validate_optional_nonnegative_int(value[name], name)
    for prefix in ("actions", "sentences", "chars", "signatures"):
        minimum = value[f"min_{prefix}"]
        maximum = value[f"max_{prefix}"]
        if (minimum is None) != (maximum is None):
            raise RuleRewardError(f"{prefix} 最小值和最大值必须同时存在或省略")
        if minimum is not None and minimum > maximum:
            raise RuleRewardError(f"{prefix} 最小值不能超过最大值")
    if value["min_chars"] is None or value["min_signatures"] is None:
        raise RuleRewardError("字符和自称范围不能为空")
    return RewardConstraints(**value)


def load_frozen_reward_constraints(
    repo_dir: Path | None = None,
) -> dict[str, RewardConstraints]:
    """Load and strictly validate the frozen per-prompt targets."""
    root = repository_root() if repo_dir is None else repo_dir
    path = root / PROMPTS_RELATIVE_PATH
    if not path.is_file():
        raise RuleRewardError(f"缺少冻结规则 Prompt: {path}")
    actual = _sha256_file(path)
    if actual != PROMPTS_SHA256:
        raise RuleRewardError(
            f"冻结规则 Prompt 哈希不匹配: {actual} != {PROMPTS_SHA256}"
        )
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuleRewardError(
                f"规则 Prompt 第 {line_number} 行不是有效 JSON"
            ) from exc
    if len(rows) != 20:
        raise RuleRewardError(f"规则 Prompt 必须为 20 条，实际 {len(rows)}")
    expected_fields = {
        "id",
        "scenario",
        "user",
        "target_rules",
        "reward_constraints",
    }
    constraints: dict[str, RewardConstraints] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise RuleRewardError(f"规则 Prompt {index} 字段不正确")
        record_id = row["id"]
        target_rules = row["target_rules"]
        if (
            not isinstance(record_id, str)
            or not re.fullmatch(r"rule_grpo_\d{4}", record_id)
            or record_id in constraints
            or not isinstance(row["scenario"], str)
            or not row["scenario"].strip()
            or not isinstance(row["user"], str)
            or not row["user"].strip()
            or not isinstance(target_rules, list)
            or not target_rules
            or any(not isinstance(item, str) or not item for item in target_rules)
        ):
            raise RuleRewardError(f"规则 Prompt {index} 内容不正确")
        constraints[record_id] = parse_constraints(row["reward_constraints"])
    return constraints


def completion_length(text: str) -> int:
    """Count non-whitespace Unicode characters."""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    return sum(not character.isspace() for character in text)


def count_sentences(text: str) -> int:
    """Count non-empty sentence-like spans, including an unterminated tail."""
    compact = text.strip()
    if not compact:
        return 0
    parts = [part.strip() for part in _SENTENCE_END.split(compact)]
    return sum(bool(part) for part in parts)


def range_distance(
    actual: int, minimum: int | None, maximum: int | None
) -> float:
    """Return the non-negative distance from an allowed inclusive range."""
    if minimum is None and maximum is None:
        return 0.0
    if minimum is None or maximum is None:
        raise ValueError("minimum 和 maximum 必须同时存在或省略")
    return float(max(minimum - actual, 0) + max(actual - maximum, 0))


def violation_to_score(violation: float) -> float:
    """Map non-negative violation continuously and monotonically to (-1, 1]."""
    if violation < 0:
        raise ValueError("violation 不能为负数")
    return (1.0 - violation) / (1.0 + violation)


def _without_quoted_text(text: str) -> str:
    value = text
    for pattern in _QUOTE_PATTERNS:
        value = pattern.sub("", value)
    return value


def count_signature_references(text: str) -> int:
    """Count simplified and traditional signature spellings outside quotes."""
    unquoted = _without_quoted_text(text)
    return sum(unquoted.count(alias) for alias in SIGNATURE_ALIASES)


def find_wrong_self_references(text: str) -> tuple[str, ...]:
    """Return every high-confidence wrong first-person occurrence."""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    unquoted = _without_quoted_text(text)
    found = []
    for alias in WRONG_SELF_ALIASES:
        found.extend([alias] * unquoted.count(alias))
    found.extend(["我"] * len(_WRONG_WO.findall(unquoted)))
    return tuple(found)


def _effective_dialogue_length(text: str) -> int:
    return sum(
        unicodedata.category(character)[0] in {"L", "N"}
        for character in text
    )


def _has_action_content(text: str) -> bool:
    return any(
        unicodedata.category(character).startswith("L") for character in text
    )


def analyze_actions(text: str) -> ActionAnalysis:
    """Parse Chinese/ASCII parenthesized action spans without binary saturation."""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    opening_to_closing = {"（": "）", "(": ")"}
    closings = set(opening_to_closing.values())
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int, str]] = []
    nested_count = 0
    unbalanced_count = 0
    for index, character in enumerate(text):
        if character in opening_to_closing:
            nested_count += int(bool(stack))
            stack.append((character, index))
        elif character in closings:
            if not stack:
                unbalanced_count += 1
                continue
            opening, start = stack.pop()
            if opening_to_closing[opening] != character:
                unbalanced_count += 1
            if not stack:
                spans.append((start, index, text[start + 1 : index]))
    unbalanced_count += len(stack)
    segments = tuple(body for _, _, body in spans)
    segment_lengths = tuple(completion_length(body) for body in segments)
    normalized_length = completion_length(text)
    total_action_length = sum(segment_lengths)
    ratio = total_action_length / normalized_length if normalized_length else 0.0
    gaps = [
        _effective_dialogue_length(text[first[1] + 1 : second[0]])
        for first, second in zip(spans, spans[1:])
    ]
    return ActionAnalysis(
        segments=segments,
        count=len(segments),
        segment_lengths=segment_lengths,
        total_action_length=total_action_length,
        action_ratio=ratio,
        minimum_dialogue_gap=min(gaps) if gaps else None,
        unbalanced_count=unbalanced_count,
        nested_count=nested_count,
        invalid_content_count=sum(
            not _has_action_content(body) for body in segments
        ),
    )


def _instruction_violation(
    length: int,
    sentence_count: int,
    action_count: int,
    constraints: RewardConstraints,
) -> float:
    return (
        range_distance(
            action_count, constraints.min_actions, constraints.max_actions
        )
        + 0.7
        * range_distance(
            sentence_count,
            constraints.min_sentences,
            constraints.max_sentences,
        )
        + 0.3
        * range_distance(
            length, constraints.min_chars, constraints.max_chars
        )
        / 30.0
    )


def _persona_violation(
    signature_count: int,
    wrong_references: Sequence[str],
    constraints: RewardConstraints,
) -> float:
    signature_distance = range_distance(
        signature_count,
        constraints.min_signatures,
        constraints.max_signatures,
    )
    wrong_wo_count = wrong_references.count("我")
    wrong_alias_count = len(wrong_references) - wrong_wo_count
    return (
        0.8 * signature_distance
        + 0.4 * wrong_wo_count
        + 0.8 * wrong_alias_count
    )


def _length_style_distance(
    length: int, minimum: int, maximum: int
) -> float:
    midpoint = (minimum + maximum) / 2.0
    scale = max(midpoint, 1.0)
    return abs(length - midpoint) / scale


def _style_violation(
    text: str,
    action: ActionAnalysis,
    constraints: RewardConstraints,
    format_reasons: Sequence[str],
) -> float:
    excess_action_length = sum(
        max(length - MAX_ACTION_LENGTH, 0)
        for length in action.segment_lengths
    )
    ratio_excess = max(action.action_ratio - MAX_ACTION_RATIO, 0.0)
    gap_shortfall = (
        max(MIN_ACTION_GAP - action.minimum_dialogue_gap, 0)
        / MIN_ACTION_GAP
        if action.minimum_dialogue_gap is not None
        else 0.0
    )
    repetition_ratio = _repetition_ratio(text)
    return (
        0.4
        * _length_style_distance(
            completion_length(text),
            constraints.min_chars,
            constraints.max_chars,
        )
        + 0.5 * repetition_ratio
        + 0.3 * excess_action_length / MAX_ACTION_LENGTH
        + 0.3 * ratio_excess / MAX_ACTION_RATIO
        + 0.2 * gap_shortfall
        + 0.3 * len(format_reasons)
        + 0.3 * action.invalid_content_count
        + 0.2 * action.nested_count
    )


def _repetition_ratio(text: str) -> float:
    """Return a smooth proxy for repeated local 3-12 character spans."""
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 6:
        return 0.0
    repeated_characters = 0
    for span in range(3, min(12, len(compact) // 2) + 1):
        counts: dict[str, int] = {}
        for index in range(len(compact) - span + 1):
            part = compact[index : index + span]
            counts[part] = counts.get(part, 0) + 1
        repeated_characters = max(
            repeated_characters,
            max((count - 1) * span for count in counts.values()),
        )
    return min(repeated_characters / len(compact), 1.0)


def _recoverability(
    text: str,
    *,
    truncated: bool,
    gibberish: bool,
    repeated: bool,
    unbalanced_count: int,
) -> float:
    nonempty = float(bool(text))
    complete = float(not truncated)
    readable = float(not gibberish)
    nonrepeated = 1.0 - _repetition_ratio(text) if repeated else 1.0
    balanced = 1.0 / (1.0 + unbalanced_count)
    return (
        nonempty + complete + readable + nonrepeated + balanced
    ) / 5.0


def score_completion(
    completion: str,
    constraints: RewardConstraints | dict[str, Any],
    *,
    finish_reason: str | None = None,
    is_truncated: bool = False,
) -> RuleRewardComponents:
    """Calculate the complete continuous reward for one completion."""
    if not isinstance(completion, str):
        raise TypeError("completion 必须是字符串")
    if not isinstance(is_truncated, bool):
        raise TypeError("is_truncated 必须是布尔值")
    if isinstance(constraints, dict):
        constraints = parse_constraints(constraints)
    if not isinstance(constraints, RewardConstraints):
        raise TypeError("constraints 必须为 RewardConstraints 或字典")

    normalized = normalize_empty_think_wrapper(completion)
    action = analyze_actions(normalized)
    truncated = is_truncated or finish_reason in TRUNCATED_FINISH_REASONS
    gibberish = has_gibberish(normalized)
    repeated = has_repeated_span(normalized)
    hard_reasons = []
    if not normalized:
        hard_reasons.append("empty")
    if truncated:
        hard_reasons.append("truncated")
    if gibberish:
        hard_reasons.append("gibberish")
    if repeated:
        hard_reasons.append("repeated_span")
    if action.unbalanced_count:
        hard_reasons.append("unbalanced_parentheses")

    format_reasons = []
    if _TAG.search(normalized):
        format_reasons.append("extra_tag")
    if _SPEAKER_LABEL.search(normalized):
        format_reasons.append("speaker_label")
    format_reasons.extend(["nested_parentheses"] * action.nested_count)

    length = completion_length(normalized)
    sentences = count_sentences(normalized)
    signatures = count_signature_references(normalized)
    wrong_refs = find_wrong_self_references(normalized)
    instruction_violation = _instruction_violation(
        length, sentences, action.count, constraints
    )
    persona_violation = _persona_violation(
        signatures, wrong_refs, constraints
    )
    style_violation = _style_violation(
        normalized, action, constraints, format_reasons
    )
    instruction_score = violation_to_score(instruction_violation)
    persona_score = violation_to_score(persona_violation)
    style_score = violation_to_score(style_violation)
    raw = (
        INSTRUCTION_WEIGHT * instruction_score
        + PERSONA_WEIGHT * persona_score
        + STYLE_WEIGHT * style_score
    )
    recoverability = _recoverability(
        normalized,
        truncated=truncated,
        gibberish=gibberish,
        repeated=repeated,
        unbalanced_count=action.unbalanced_count,
    )
    total = (
        HARD_INVALID_BASE + HARD_INVALID_RECOVERY_RANGE * recoverability
        if hard_reasons
        else raw
    )
    return RuleRewardComponents(
        constraints=constraints,
        normalized_length=length,
        sentence_count=sentences,
        signature_count=signatures,
        wrong_self_references=wrong_refs,
        wrong_self_count=len(wrong_refs),
        action=action,
        instruction_violation=instruction_violation,
        instruction_score=instruction_score,
        persona_violation=persona_violation,
        persona_score=persona_score,
        style_violation=style_violation,
        style_score=style_score,
        format_reasons=tuple(format_reasons),
        hard_invalid_reasons=tuple(hard_reasons),
        recoverability=recoverability,
        raw_reward=raw,
        total_reward=total,
    )


def _batch_values(
    values: Sequence[Any] | None, count: int, default: Any
) -> list[Any]:
    if values is None:
        return [default] * count
    if len(values) != count:
        raise RuleRewardError("奖励批次元数据长度不一致")
    return list(values)


class RuleRewardEngine:
    """Score and log one ms-swift reward batch without network access."""

    def __init__(
        self,
        *,
        constraints: dict[str, RewardConstraints] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.constraints = (
            load_frozen_reward_constraints()
            if constraints is None
            else constraints
        )
        self.log_path = log_path
        self._log_lock = asyncio.Lock()

    async def score_batch(
        self,
        completions: Sequence[str],
        messages: Sequence[Sequence[dict[str, Any]]],
        *,
        finish_reasons: Sequence[str | None] | None = None,
        is_truncated: Sequence[bool] | None = None,
        prompt_ids: Sequence[Any] | None = None,
        request_ids: Sequence[Any] | None = None,
        record_ids: Sequence[Any] | None = None,
        global_step: int | None = None,
    ) -> list[float]:
        count = len(completions)
        if len(messages) != count:
            raise RuleRewardError("completions/messages 长度不一致")
        finishes = _batch_values(finish_reasons, count, None)
        truncations = _batch_values(is_truncated, count, False)
        prompts = _batch_values(prompt_ids, count, None)
        requests = _batch_values(request_ids, count, None)
        records = _batch_values(record_ids, count, None)
        log_rows = []
        rewards = []
        for index, completion in enumerate(completions):
            record_id = records[index]
            if (
                not isinstance(record_id, str)
                or record_id not in self.constraints
            ):
                raise RuleRewardError(f"奖励缺少有效 record_id: {record_id!r}")
            components = score_completion(
                completion,
                self.constraints[record_id],
                finish_reason=finishes[index],
                is_truncated=truncations[index],
            )
            rewards.append(components.total_reward)
            log_rows.append(
                {
                    "schema_version": 2,
                    "status": "ok",
                    "global_step": global_step,
                    "prompt_id": prompts[index],
                    "request_id": requests[index],
                    "record_id": record_id,
                    "completion": completion,
                    "finish_reason": finishes[index],
                    "is_truncated": truncations[index],
                    "components": components.as_log_dict(),
                    "total_reward": components.total_reward,
                }
            )
        if self.log_path is not None:
            await self._append_logs(log_rows)
        return rewards

    async def _append_logs(self, rows: Sequence[dict[str, Any]]) -> None:
        async with self._log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
