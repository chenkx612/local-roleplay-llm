"""Deterministic rule reward for morgana-v2 Stage 4 GRPO."""

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
PROMPTS_SHA256 = "5927ce66062de01a5b343d2a6fb73fc31f5983016d87905f796fc2e87feaef37"
ACTION_POLICIES = frozenset({"encouraged", "optional", "forbidden"})
ACTION_POLICY_COUNTS = {"encouraged": 10, "optional": 8, "forbidden": 2}

HARD_INVALID_REWARD = -4.0
MIN_VALID_REWARD = -3.0
MAX_VALID_REWARD = 2.6
LENGTH_WEIGHT = 1.2
SIGNATURE_WEIGHT = 0.8
ACTION_WEIGHT = 0.6
WRONG_SELF_WEIGHT = 1.2
FORMAT_WEIGHT = 0.5

MIN_PREFERRED_LENGTH = 30
MAX_PREFERRED_LENGTH = 90
MIN_LENGTH_FLOOR = 15
MAX_LENGTH_CEILING = 180
MAX_ACTION_LENGTH = 30
MIN_ACTION_GAP = 12
MAX_ACTION_RATIO = 0.50

TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
WRONG_SELF_ALIASES = ("本大爷", "本喵", "本猫", "俺")

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


class RuleRewardError(RuntimeError):
    """Raised when the frozen deterministic reward contract is invalid."""


@dataclass(frozen=True)
class ActionAnalysis:
    """Mechanical action-span facts extracted from one completion."""

    segments: tuple[str, ...]
    count: int
    total_action_length: int
    action_ratio: float
    minimum_dialogue_gap: int | None
    unbalanced: bool
    nested: bool
    overlong: bool
    over_ratio: bool
    dense_pair: bool
    invalid_content: bool

    def as_log_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["segments"] = list(self.segments)
        return value


@dataclass(frozen=True)
class RuleRewardComponents:
    """Every deterministic component used to score one completion."""

    action_policy: str
    normalized_length: int
    length_score: float
    signature_count: int
    signature_score: float
    wrong_self_references: tuple[str, ...]
    wrong_self_penalty: float
    action: ActionAnalysis
    action_score: float
    format_reasons: tuple[str, ...]
    format_penalty: float
    hard_invalid_reasons: tuple[str, ...]
    raw_reward: float
    total_reward: float

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "action_policy": self.action_policy,
            "normalized_length": self.normalized_length,
            "length_score": self.length_score,
            "signature_count": self.signature_count,
            "signature_score": self.signature_score,
            "wrong_self_references": list(self.wrong_self_references),
            "wrong_self_penalty": self.wrong_self_penalty,
            "action": self.action.as_log_dict(),
            "action_score": self.action_score,
            "format_reasons": list(self.format_reasons),
            "format_penalty": self.format_penalty,
            "hard_invalid_reasons": list(self.hard_invalid_reasons),
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
        "schema_version": 1,
        "name": "morgana_rule_reward",
        "prompt_sha256": PROMPTS_SHA256,
        "hard_invalid_reward": HARD_INVALID_REWARD,
        "valid_reward_range": [MIN_VALID_REWARD, MAX_VALID_REWARD],
        "formula": {
            "length_weight": LENGTH_WEIGHT,
            "signature_weight": SIGNATURE_WEIGHT,
            "action_weight": ACTION_WEIGHT,
            "wrong_self_weight": WRONG_SELF_WEIGHT,
            "format_weight": FORMAT_WEIGHT,
        },
        "length": {
            "preferred": [MIN_PREFERRED_LENGTH, MAX_PREFERRED_LENGTH],
            "floor": MIN_LENGTH_FLOOR,
            "ceiling": MAX_LENGTH_CEILING,
        },
        "signature": {"token": "吾辈", "wrong_aliases": list(WRONG_SELF_ALIASES)},
        "action": {
            "policies": sorted(ACTION_POLICIES),
            "max_segment_length": MAX_ACTION_LENGTH,
            "minimum_dialogue_gap": MIN_ACTION_GAP,
            "max_ratio": MAX_ACTION_RATIO,
            "requires_letter_content": True,
        },
        "hard_invalid_checks": [
            "empty",
            "truncated",
            "gibberish",
            "repeated_span",
            "unbalanced_parentheses",
        ],
    }


def load_frozen_reward_policies(
    repo_dir: Path | None = None,
) -> dict[str, str]:
    """Load and strictly validate the frozen per-prompt action policies."""
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
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuleRewardError(
                f"规则 Prompt 第 {line_number} 行不是有效 JSON"
            ) from exc
        rows.append(row)
    if len(rows) != 20:
        raise RuleRewardError(f"规则 Prompt 必须为 20 条，实际 {len(rows)}")
    expected_fields = {
        "id",
        "scenario",
        "user",
        "target_rules",
        "reward_policy",
    }
    policies: dict[str, str] = {}
    counts = {name: 0 for name in ACTION_POLICIES}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise RuleRewardError(f"规则 Prompt {index} 字段不正确")
        record_id = row["id"]
        target_rules = row["target_rules"]
        policy = row["reward_policy"]
        if (
            not isinstance(record_id, str)
            or not re.fullmatch(r"rule_grpo_\d{4}", record_id)
            or record_id in policies
            or not isinstance(row["scenario"], str)
            or not row["scenario"].strip()
            or not isinstance(row["user"], str)
            or not row["user"].strip()
            or not isinstance(target_rules, list)
            or not target_rules
            or any(not isinstance(item, str) or not item for item in target_rules)
            or not isinstance(policy, dict)
            or set(policy) != {"action"}
            or policy["action"] not in ACTION_POLICIES
        ):
            raise RuleRewardError(f"规则 Prompt {index} 内容不正确")
        action_policy = policy["action"]
        policies[record_id] = action_policy
        counts[action_policy] += 1
    if counts != ACTION_POLICY_COUNTS:
        raise RuleRewardError(
            f"动作策略分布不正确: {counts} != {ACTION_POLICY_COUNTS}"
        )
    return policies


def completion_length(text: str) -> int:
    """Count non-whitespace Unicode characters."""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    return sum(not character.isspace() for character in text)


def calculate_length_score(length: int) -> float:
    """Return the frozen plateau score with smooth short/long decay."""
    if length < 0:
        raise ValueError("length 不能为负数")
    if length <= MIN_LENGTH_FLOOR:
        return -1.0
    if length < MIN_PREFERRED_LENGTH:
        return -1.0 + 2.0 * (
            (length - MIN_LENGTH_FLOOR)
            / (MIN_PREFERRED_LENGTH - MIN_LENGTH_FLOOR)
        )
    if length <= MAX_PREFERRED_LENGTH:
        return 1.0
    if length < MAX_LENGTH_CEILING:
        return 1.0 - 2.0 * (
            (length - MAX_PREFERRED_LENGTH)
            / (MAX_LENGTH_CEILING - MAX_PREFERRED_LENGTH)
        )
    return -1.0


def calculate_signature_score(count: int) -> float:
    """Reward exactly one signature self-reference without encouraging spam."""
    if count < 0:
        raise ValueError("count 不能为负数")
    if count == 0:
        return 0.0
    if count == 1:
        return 1.0
    if count == 2:
        return 0.25
    return -1.0


def _without_quoted_text(text: str) -> str:
    value = text
    for pattern in _QUOTE_PATTERNS:
        value = pattern.sub("", value)
    return value


def find_wrong_self_references(text: str) -> tuple[str, ...]:
    """Return high-confidence wrong first-person references outside quotes."""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    unquoted = _without_quoted_text(text)
    found = [alias for alias in WRONG_SELF_ALIASES if alias in unquoted]
    if _WRONG_WO.search(unquoted):
        found.append("我")
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
    """Parse sequential Chinese/ASCII parenthesized action spans."""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    opening_to_closing = {"（": "）", "(": ")"}
    closings = set(opening_to_closing.values())
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int, str]] = []
    nested = False
    unbalanced = False
    for index, character in enumerate(text):
        if character in opening_to_closing:
            nested = nested or bool(stack)
            stack.append((character, index))
        elif character in closings:
            if not stack:
                unbalanced = True
                continue
            opening, start = stack.pop()
            if opening_to_closing[opening] != character:
                unbalanced = True
            if not stack:
                spans.append((start, index, text[start + 1 : index]))
    if stack:
        unbalanced = True
    segments = tuple(body for _, _, body in spans)
    segment_lengths = [completion_length(body) for body in segments]
    normalized_length = completion_length(text)
    total_action_length = sum(segment_lengths)
    ratio = total_action_length / normalized_length if normalized_length else 0.0
    gaps = [
        _effective_dialogue_length(text[first[1] + 1 : second[0]])
        for first, second in zip(spans, spans[1:])
    ]
    minimum_gap = min(gaps) if gaps else None
    return ActionAnalysis(
        segments=segments,
        count=len(segments),
        total_action_length=total_action_length,
        action_ratio=ratio,
        minimum_dialogue_gap=minimum_gap,
        unbalanced=unbalanced,
        nested=nested,
        overlong=any(length > MAX_ACTION_LENGTH for length in segment_lengths),
        over_ratio=ratio > MAX_ACTION_RATIO,
        dense_pair=(
            len(segments) == 2
            and minimum_gap is not None
            and minimum_gap < MIN_ACTION_GAP
        ),
        invalid_content=any(not _has_action_content(body) for body in segments),
    )


def calculate_action_score(action: ActionAnalysis, policy: str) -> float:
    """Apply the frozen prompt-specific action policy."""
    if policy not in ACTION_POLICIES:
        raise ValueError(f"未知动作策略: {policy}")
    if (
        action.nested
        or action.overlong
        or action.over_ratio
        or action.invalid_content
        or action.count >= 3
    ):
        return -1.0
    if policy == "forbidden":
        return 1.0 if action.count == 0 else -1.0
    if action.count == 2 and action.dense_pair:
        return 0.2
    if policy == "encouraged":
        return 1.0 if action.count in {1, 2} else 0.0
    return 1.0


def score_completion(
    completion: str,
    action_policy: str,
    *,
    finish_reason: str | None = None,
    is_truncated: bool = False,
) -> RuleRewardComponents:
    """Calculate the complete deterministic reward for one completion."""
    if not isinstance(completion, str):
        raise TypeError("completion 必须是字符串")
    if not isinstance(is_truncated, bool):
        raise TypeError("is_truncated 必须是布尔值")
    normalized = normalize_empty_think_wrapper(completion)
    action = analyze_actions(normalized)
    hard_reasons = []
    if not normalized:
        hard_reasons.append("empty")
    if is_truncated or finish_reason in TRUNCATED_FINISH_REASONS:
        hard_reasons.append("truncated")
    if has_gibberish(normalized):
        hard_reasons.append("gibberish")
    if has_repeated_span(normalized):
        hard_reasons.append("repeated_span")
    if action.unbalanced:
        hard_reasons.append("unbalanced_parentheses")

    format_reasons = []
    if _TAG.search(normalized):
        format_reasons.append("extra_tag")
    if _SPEAKER_LABEL.search(normalized):
        format_reasons.append("speaker_label")
    if action.nested:
        format_reasons.append("nested_parentheses")

    length = completion_length(normalized)
    length_score = calculate_length_score(length)
    signature_count = normalized.count("吾辈")
    signature_score = calculate_signature_score(signature_count)
    wrong_refs = find_wrong_self_references(normalized)
    wrong_penalty = float(bool(wrong_refs))
    action_score = calculate_action_score(action, action_policy)
    format_penalty = float(bool(format_reasons))
    raw = (
        LENGTH_WEIGHT * length_score
        + SIGNATURE_WEIGHT * signature_score
        + ACTION_WEIGHT * action_score
        - WRONG_SELF_WEIGHT * wrong_penalty
        - FORMAT_WEIGHT * format_penalty
    )
    total = (
        HARD_INVALID_REWARD
        if hard_reasons
        else min(MAX_VALID_REWARD, max(MIN_VALID_REWARD, raw))
    )
    return RuleRewardComponents(
        action_policy=action_policy,
        normalized_length=length,
        length_score=length_score,
        signature_count=signature_count,
        signature_score=signature_score,
        wrong_self_references=wrong_refs,
        wrong_self_penalty=wrong_penalty,
        action=action,
        action_score=action_score,
        format_reasons=tuple(format_reasons),
        format_penalty=format_penalty,
        hard_invalid_reasons=tuple(hard_reasons),
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
        policies: dict[str, str] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.policies = policies or load_frozen_reward_policies()
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
            if not isinstance(record_id, str) or record_id not in self.policies:
                raise RuleRewardError(f"奖励缺少有效 record_id: {record_id!r}")
            components = score_completion(
                completion,
                self.policies[record_id],
                finish_reason=finishes[index],
                is_truncated=truncations[index],
            )
            rewards.append(components.total_reward)
            log_rows.append(
                {
                    "schema_version": 1,
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
