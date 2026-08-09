"""Deterministic SFT Dev checks and anonymous manual-review helpers."""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any


EVALUATION_SEEDS = (20260807,)
PRIMARY_EVALUATION_SEED = EVALUATION_SEEDS[0]
MANUAL_REVIEW_ORDER_SEED = 20260807
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
WRONG_SELF_REFERENCES = ("本大爷", "本喵")
MANUAL_SCORE_DIMENSIONS = (
    "generation_stability",
    "role_consistency",
    "dialogue_quality",
)

_EMPTY_THINK_WRAPPER = re.compile(r"\A\s*<think>\s*</think>\s*", re.IGNORECASE)
_TAG = re.compile(r"<[^>\r\n]+>")
_STRICT_OPENING = re.compile(r"\A（([^（）\r\n]+)）([\s\S]+)\Z")
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
)


def normalize_empty_think_wrapper(raw_assistant: str) -> str:
    """Remove one leading empty thinking wrapper and surrounding whitespace."""
    if not isinstance(raw_assistant, str):
        raise TypeError("raw_assistant 必须是字符串")
    return _EMPTY_THINK_WRAPPER.sub("", raw_assistant, count=1).strip()


def has_repeated_span(text: str, span: int = 12, repeats: int = 3) -> bool:
    """Return whether compact text contains a repeated span or short loop."""
    if span <= 0 or repeats <= 1:
        raise ValueError("span 必须为正数且 repeats 必须大于 1")
    compact = re.sub(r"\s+", "", text)
    fixed_span_repeated = any(
        compact.count(compact[index : index + span]) >= repeats
        for index in range(max(0, len(compact) - span + 1))
    )
    if fixed_span_repeated:
        return True

    # Catch degenerate loops such as "哈哈哈……" without rejecting brief,
    # natural emphasis. A loop must occupy at least 24 characters and repeat
    # its short unit at least six times.
    for start in range(max(0, len(compact) - 23)):
        for period in range(1, min(6, len(compact) - start) + 1):
            unit = compact[start : start + period]
            end = start
            while compact.startswith(unit, end):
                end += period
            run_length = end - start
            if run_length >= 24 and run_length // period >= 6:
                return True
    return False


def has_abnormal_symbols(text: str) -> bool:
    """Detect replacement/control characters, emoji, or long symbol runs."""
    symbol_run = 0
    for character in text:
        codepoint = ord(character)
        category = unicodedata.category(character)
        is_control = (
            category.startswith("C")
            and not character.isspace()
            and character != "\u200d"
        )
        if character == "\ufffd" or is_control:
            return True
        if any(start <= codepoint <= end for start, end in _EMOJI_RANGES):
            return True
        if category[0] in {"P", "S"}:
            symbol_run += 1
            if symbol_run >= 4:
                return True
        else:
            symbol_run = 0
    return False


def has_gibberish(text: str) -> bool:
    """Detect unreadable corruption without treating ordinary emoji as failure."""
    foreign_letter_run = 0
    symbol_run = 0
    for character in text:
        category = unicodedata.category(character)
        name = unicodedata.name(character, "")
        is_control = (
            category.startswith("C")
            and not character.isspace()
            and character != "\u200d"
        )
        if character == "\ufffd" or is_control:
            return True

        if category.startswith("L") and (
            "GREEK" in name or "CYRILLIC" in name
        ):
            foreign_letter_run += 1
            if foreign_letter_run >= 12:
                return True
        else:
            foreign_letter_run = 0

        is_emoji = any(
            start <= ord(character) <= end for start, end in _EMOJI_RANGES
        )
        if category[0] in {"P", "S"} and not is_emoji:
            symbol_run += 1
            if symbol_run >= 12:
                return True
        else:
            symbol_run = 0
    return False


def has_unclosed_brackets(text: str) -> bool:
    """Return whether Chinese or ASCII parentheses are unbalanced."""
    for opening, closing in (("（", "）"), ("(", ")")):
        depth = 0
        for character in text:
            if character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if depth < 0:
                    return True
        if depth:
            return True
    return False


def inspect_output(record: dict[str, Any]) -> dict[str, Any]:
    """Inspect one normalized output and return mechanical issue flags."""
    assistant = record.get("assistant")
    if not isinstance(assistant, str):
        raise ValueError("输出记录的 assistant 必须是字符串")
    answer = assistant.strip()
    opening = _STRICT_OPENING.fullmatch(answer)
    action = opening.group(1).strip() if opening else ""
    dialogue = opening.group(2).strip() if opening else ""
    has_nested_brackets = bool(action and any(char in action for char in "（）()"))
    has_later_brackets = bool(dialogue and any(char in dialogue for char in "（）()"))
    has_tags = bool(_TAG.search(answer))
    strict_format = bool(
        opening
        and action
        and dialogue
        and not has_nested_brackets
        and not has_later_brackets
        and not has_tags
    )
    repeated = has_repeated_span(answer)
    abnormal_symbols = has_abnormal_symbols(answer)
    gibberish = has_gibberish(answer)
    unclosed_brackets = has_unclosed_brackets(answer)
    truncated = record.get("finish_reason") in TRUNCATED_FINISH_REASONS
    wrong_self_reference = any(alias in answer for alias in WRONG_SELF_REFERENCES)
    return {
        "nonempty": bool(answer),
        "stop": record.get("finish_reason") == "stop",
        "strict_format": strict_format,
        "truncated": truncated,
        "repeated_span": repeated,
        "abnormal_symbols": abnormal_symbols,
        "gibberish": gibberish,
        "unclosed_brackets": unclosed_brackets,
        "extra_tags": has_tags,
        "uses_signature_self_reference": "吾辈" in answer,
        "uses_wrong_self_reference": wrong_self_reference,
        "degenerate": repeated or abnormal_symbols or unclosed_brackets,
    }


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("输出记录的 seed 必须是整数")
    return seed


def validate_output_grid(
    rows: Sequence[dict[str, Any]],
    expected_seeds: Sequence[int],
    expected_ids: Sequence[str],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Validate a complete, unique ``(seed, id)`` output grid."""
    seeds = tuple(expected_seeds)
    ids = tuple(expected_ids)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds 必须非空且不能重复")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("expected_ids 必须非空且不能重复")
    expected = {(seed, record_id) for seed in seeds for record_id in ids}
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("输出必须是对象列表")
        seed = _validate_seed(row.get("seed"))
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("输出记录的 id 必须是非空字符串")
        key = (seed, record_id)
        if key in indexed:
            raise ValueError(f"输出包含重复键: {seed}:{record_id}")
        indexed[key] = row
    actual = set(indexed)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"输出网格不完整: missing={missing}, unexpected={unexpected}"
        )
    return indexed


def _issue_key(seed: int, record_id: str, include_seed: bool) -> str:
    return f"{seed}:{record_id}" if include_seed else record_id


def _summarize_validated_rows(
    rows: Iterable[dict[str, Any]], *, include_seed_in_ids: bool
) -> dict[str, Any]:
    records = list(rows)
    checks = [(row, inspect_output(row)) for row in records]
    issue_fields = {
        "empty_ids": "nonempty",
        "non_stop_ids": "stop",
        "strict_format_issue_ids": "strict_format",
        "truncated_ids": "truncated",
        "repetition_issue_ids": "repeated_span",
        "abnormal_symbol_ids": "abnormal_symbols",
        "gibberish_ids": "gibberish",
        "unclosed_bracket_ids": "unclosed_brackets",
        "extra_tag_ids": "extra_tags",
        "wrong_self_reference_ids": "uses_wrong_self_reference",
        "degeneration_ids": "degenerate",
    }
    inverse_fields = {"empty_ids", "non_stop_ids", "strict_format_issue_ids"}
    issues: dict[str, list[str]] = {}
    for output_name, check_name in issue_fields.items():
        matching = [
            _issue_key(row["seed"], row["id"], include_seed_in_ids)
            for row, result in checks
            if result[check_name]
        ]
        if output_name in inverse_fields:
            matching = [
                _issue_key(row["seed"], row["id"], include_seed_in_ids)
                for row, result in checks
                if not result[check_name]
            ]
        issues[output_name] = matching
    count = len(records)
    stop_count = sum(result["stop"] for _, result in checks)
    strict_count = sum(result["strict_format"] for _, result in checks)
    signature_count = sum(
        result["uses_signature_self_reference"] for _, result in checks
    )
    wrong_count = sum(result["uses_wrong_self_reference"] for _, result in checks)
    return {
        "records": count,
        "nonempty_count": count - len(issues["empty_ids"]),
        "stop_count": stop_count,
        "strict_format_count": strict_count,
        "strict_format_rate": strict_count / count if count else 0.0,
        "truncation_count": len(issues["truncated_ids"]),
        "degeneration_count": len(issues["degeneration_ids"]),
        "gibberish_count": len(issues["gibberish_ids"]),
        "signature_self_reference_count": signature_count,
        "signature_self_reference_rate": signature_count / count if count else 0.0,
        "wrong_self_reference_count": wrong_count,
        "wrong_self_reference_rate": wrong_count / count if count else 0.0,
        **issues,
    }


def summarize_seed(
    rows: Sequence[dict[str, Any]], seed: int, expected_ids: Sequence[str]
) -> dict[str, Any]:
    """Summarize one complete seed of Dev outputs."""
    indexed = validate_output_grid(rows, (seed,), expected_ids)
    ordered = [indexed[(seed, record_id)] for record_id in expected_ids]
    return {
        "seed": seed,
        **_summarize_validated_rows(ordered, include_seed_in_ids=False),
    }


def summarize_outputs(
    rows: Sequence[dict[str, Any]],
    expected_seeds: Sequence[int],
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    """Summarize each seed and the complete cross-seed grid."""
    indexed = validate_output_grid(rows, expected_seeds, expected_ids)
    ordered = [
        indexed[(seed, record_id)]
        for seed in expected_seeds
        for record_id in expected_ids
    ]
    return {
        "seeds": [
            summarize_seed(
                [indexed[(seed, record_id)] for record_id in expected_ids],
                seed,
                expected_ids,
            )
            for seed in expected_seeds
        ],
        "overall": _summarize_validated_rows(ordered, include_seed_in_ids=True),
    }


def validate_aligned_outputs(
    base_rows: Sequence[dict[str, Any]],
    sft_rows: Sequence[dict[str, Any]],
    expected_seeds: Sequence[int],
    expected_ids: Sequence[str],
) -> bool:
    """Validate both grids and comparable source fields for every pair."""
    base = validate_output_grid(base_rows, expected_seeds, expected_ids)
    sft = validate_output_grid(sft_rows, expected_seeds, expected_ids)
    for key in base:
        for field in ("id", "seed", "scenario", "target_goals", "user"):
            if base[key].get(field) != sft[key].get(field):
                raise ValueError(f"Base/SFT 字段未对齐: {key} field={field}")
    return True


def evaluate_core_behavior_gate(
    base_rows: Sequence[dict[str, Any]],
    sft_rows: Sequence[dict[str, Any]],
    expected_seeds: Sequence[int],
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    """Evaluate the first core goal: readable and stable generation."""
    aligned = validate_aligned_outputs(
        base_rows, sft_rows, expected_seeds, expected_ids
    )
    base = summarize_outputs(base_rows, expected_seeds, expected_ids)
    sft = summarize_outputs(sft_rows, expected_seeds, expected_ids)
    base_all = base["overall"]
    sft_all = sft["overall"]
    expected_count = len(expected_seeds) * len(expected_ids)
    checks = {
        "complete_and_aligned": (
            aligned
            and base_all["records"] == sft_all["records"] == expected_count
            and base_all["nonempty_count"]
            == sft_all["nonempty_count"]
            == expected_count
        ),
        "stop_count_not_lower": sft_all["stop_count"] >= base_all["stop_count"],
        "truncation_count_not_higher": (
            sft_all["truncation_count"] <= base_all["truncation_count"]
        ),
        "no_repeated_spans": not sft_all["repetition_issue_ids"],
        "no_gibberish": sft_all["gibberish_count"] == 0,
    }
    return {
        "goal": "generation_stability",
        "passed": all(checks.values()),
        "checks": checks,
        "base": base,
        "sft": sft,
    }


def evaluate_relative_behavior_gate(
    base_rows: Sequence[dict[str, Any]],
    sft_rows: Sequence[dict[str, Any]],
    expected_seeds: Sequence[int],
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    """Compatibility alias for the core generation-stability gate."""
    return evaluate_core_behavior_gate(
        base_rows, sft_rows, expected_seeds, expected_ids
    )


def build_manual_review(
    base_rows: Sequence[dict[str, Any]],
    sft_rows: Sequence[dict[str, Any]],
    expected_ids: Sequence[str],
    *,
    primary_seed: int = PRIMARY_EVALUATION_SEED,
    order_seed: int = MANUAL_REVIEW_ORDER_SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a reproducible anonymous A/B packet and separate answer key."""
    base_primary = [row for row in base_rows if row.get("seed") == primary_seed]
    sft_primary = [row for row in sft_rows if row.get("seed") == primary_seed]
    validate_aligned_outputs(base_primary, sft_primary, (primary_seed,), expected_ids)
    base = validate_output_grid(base_primary, (primary_seed,), expected_ids)
    sft = validate_output_grid(sft_primary, (primary_seed,), expected_ids)
    rng = random.Random(order_seed)
    items = []
    answers = []
    for index, record_id in enumerate(expected_ids, 1):
        key = (primary_seed, record_id)
        base_row = base[key]
        sft_row = sft[key]
        sft_label = "A" if rng.randrange(2) == 0 else "B"
        base_label = "B" if sft_label == "A" else "A"
        review_id = f"dev-review-{index:02d}"
        candidates = {
            base_label: base_row["assistant"],
            sft_label: sft_row["assistant"],
        }
        items.append(
            {
                "review_id": review_id,
                "id": record_id,
                "scenario": base_row["scenario"],
                "target_goals": base_row["target_goals"],
                "user": base_row["user"],
                "answer_a": candidates["A"],
                "answer_b": candidates["B"],
            }
        )
        answers.append(
            {
                "review_id": review_id,
                "id": record_id,
                "base_label": base_label,
                "sft_label": sft_label,
            }
        )
    packet = {
        "schema_version": 2,
        "primary_seed": primary_seed,
        "order_seed": order_seed,
        "score_dimensions": list(MANUAL_SCORE_DIMENSIONS),
        "severe_issue_examples": [
            "unreadable",
            "role_break",
            "perspective_shift",
        ],
        "items": items,
    }
    answer_key = {
        "schema_version": 2,
        "primary_seed": primary_seed,
        "order_seed": order_seed,
        "answers": answers,
    }
    return packet, answer_key


def empty_manual_review_results(packet: dict[str, Any]) -> dict[str, Any]:
    """Return a standalone blank result artifact with the expected schema."""
    return {
        "schema_version": 2,
        "instructions": {
            "winner": "A、B 或 tie",
            "clearly_worse": "A、B 或 null",
            "scores": "A/B 各按三个核心目标给 0～10 整数分",
            "severe_issues": (
                "A/B 各填写严重问题代码列表，无则为空列表"
            ),
        },
        "expected_review_ids": [item["review_id"] for item in packet["items"]],
        "results": [],
    }


def _validate_scores(scores: Any, review_id: str) -> None:
    if not isinstance(scores, dict) or set(scores) != {"A", "B"}:
        raise ValueError(f"{review_id} scores 必须仅含 A/B")
    for label in ("A", "B"):
        values = scores[label]
        if not isinstance(values, dict) or set(values) != set(MANUAL_SCORE_DIMENSIONS):
            raise ValueError(f"{review_id} {label} scores 维度不完整")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 10
            for value in values.values()
        ):
            raise ValueError(f"{review_id} 分数必须是 0～10 整数")


def evaluate_manual_review(
    packet: dict[str, Any], answer_key: dict[str, Any], results_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Validate submitted reviews and evaluate the fixed manual gate."""
    items = packet.get("items")
    answers = answer_key.get("answers")
    results = results_artifact.get("results")
    if (
        not isinstance(items, list)
        or not isinstance(answers, list)
        or not isinstance(results, list)
    ):
        raise ValueError("人工复核文件缺少 items、answers 或 results 列表")
    item_by_review = {item.get("review_id"): item for item in items}
    answer_by_review = {answer.get("review_id"): answer for answer in answers}
    if None in item_by_review or len(item_by_review) != len(items):
        raise ValueError("匿名复核包包含缺失或重复 review_id")
    if (
        set(answer_by_review) != set(item_by_review)
        or len(answer_by_review) != len(answers)
    ):
        raise ValueError("答案映射与匿名复核包不完整对齐")
    result_by_review = {}
    for result in results:
        review_id = result.get("review_id") if isinstance(result, dict) else None
        if review_id in result_by_review:
            raise ValueError(f"人工结果包含重复 review_id: {review_id}")
        result_by_review[review_id] = result
    if set(result_by_review) != set(item_by_review):
        missing = sorted(set(item_by_review) - set(result_by_review))
        unexpected = sorted(set(result_by_review) - set(item_by_review), key=str)
        raise ValueError(
            f"人工结果不完整: missing={missing}, unexpected={unexpected}"
        )

    sft_wins = 0
    sft_clear_losses = 0
    sft_severe_issue_ids = []
    score_totals = {
        model: {dimension: 0 for dimension in MANUAL_SCORE_DIMENSIONS}
        for model in ("base", "sft")
    }
    for review_id, result in result_by_review.items():
        if set(result) != {
            "review_id",
            "winner",
            "clearly_worse",
            "scores",
            "severe_issues",
        }:
            raise ValueError(f"{review_id} 人工结果字段不正确")
        if result["winner"] not in {"A", "B", "tie"}:
            raise ValueError(f"{review_id} winner 必须为 A、B 或 tie")
        if result["clearly_worse"] not in {"A", "B", None}:
            raise ValueError(f"{review_id} clearly_worse 必须为 A、B 或 null")
        _validate_scores(result["scores"], review_id)
        severe = result["severe_issues"]
        if not isinstance(severe, dict) or set(severe) != {"A", "B"}:
            raise ValueError(f"{review_id} severe_issues 必须仅含 A/B")
        if any(
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            for values in severe.values()
        ):
            raise ValueError(
                f"{review_id} severe_issues 必须是非空字符串列表"
            )

        mapping = answer_by_review[review_id]
        sft_label = mapping["sft_label"]
        base_label = mapping["base_label"]
        if result["winner"] == sft_label:
            sft_wins += 1
        if result["clearly_worse"] == sft_label:
            sft_clear_losses += 1
        sft_severe = severe[sft_label]
        if sft_severe:
            sft_severe_issue_ids.append(review_id)
        for dimension in MANUAL_SCORE_DIMENSIONS:
            score_totals["base"][dimension] += result["scores"][base_label][
                dimension
            ]
            score_totals["sft"][dimension] += result["scores"][sft_label][
                dimension
            ]

    mean_scores = {
        model: {
            dimension: total / len(results)
            for dimension, total in dimensions.items()
        }
        for model, dimensions in score_totals.items()
    }

    checks = {
        "sft_wins_at_least_6": sft_wins >= 6,
        "sft_clear_losses_at_most_2": sft_clear_losses <= 2,
        "sft_has_no_severe_issues": not sft_severe_issue_ids,
        **{
            f"sft_{dimension}_score_not_lower": (
                mean_scores["sft"][dimension] >= mean_scores["base"][dimension]
            )
            for dimension in MANUAL_SCORE_DIMENSIONS
        },
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reviewed_pairs": len(results),
        "sft_wins": sft_wins,
        "sft_clear_losses": sft_clear_losses,
        "sft_severe_issue_ids": sft_severe_issue_ids,
        "mean_scores": mean_scores,
    }
