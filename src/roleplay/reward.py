"""Inspectable deterministic and Judge-backed GRPO rewards."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Sequence

from openai import OpenAI

from .artifacts import read_jsonl
from .sft_eval import has_abnormal_symbols, has_repeated_span, inspect_output


LOOSE_OPENING = re.compile(r"\A\s*（[^（）\r\n]+）\s*\S", re.DOTALL)
EXTRA_LABEL = re.compile(
    r"(?:<[^>]+>|\[(?:动作|旁白|回答|角色|assistant)\]|^(?:动作|旁白|回答|角色)[:：])",
    re.IGNORECASE | re.MULTILINE,
)


class JudgeError(RuntimeError):
    """Raised when a full group cannot receive validated Judge scores."""


def format_consistency(text: str) -> float:
    """Score the frozen strict/loose response format as 10, 5 or 0."""
    if inspect_output({"assistant": text, "finish_reason": "stop"})["strict_format"]:
        return 10.0
    return 5.0 if LOOSE_OPENING.search(text) else 0.0


def _copied_source(text: str, sources: Sequence[str], *, minimum: int) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < minimum:
        return False
    for source in sources:
        candidate = re.sub(r"\s+", "", source)
        match = SequenceMatcher(None, compact, candidate, autojunk=False).find_longest_match()
        if match.size >= minimum and match.size / max(1, len(compact)) >= 0.55:
            return True
    return False


def penalty_breakdown(
    text: str,
    *,
    persona_sources: Sequence[str] = (),
    style_sources: Sequence[str] = (),
) -> dict[str, float]:
    """Return named penalties, capped at five points by ``total``."""
    breakdown = {
        "repetition": 2.0 if has_repeated_span(text) else 0.0,
        "abnormal_symbols": 2.0 if has_abnormal_symbols(text) else 0.0,
        "persona_copy": 2.0 if _copied_source(text, persona_sources, minimum=80) else 0.0,
        "style_copy": 2.0 if _copied_source(text, style_sources, minimum=40) else 0.0,
        "extra_labels": 1.0 if EXTRA_LABEL.search(text) else 0.0,
    }
    breakdown["total"] = min(5.0, sum(breakdown.values()))
    return breakdown


def combine_reward(role: float, format_score: float, dialogue: float, penalty: float) -> float:
    for name, value in (("role", role), ("format", format_score), ("dialogue", dialogue)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 10:
            raise ValueError(f"{name} 必须在 0～10 之间")
    if not isinstance(penalty, (int, float)) or not 0 <= penalty <= 5:
        raise ValueError("penalty 必须在 0～5 之间")
    return max(-5.0, min(10.0, (role + format_score + dialogue) / 3 - penalty))


def build_judge_messages(
    persona_text: str, prompt: str, candidates: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    compact = [{"candidate_id": item["candidate_id"], "assistant": item["assistant"]} for item in candidates]
    return [
        {
            "role": "system",
            "content": (
                "你是角色扮演 GRPO 奖励 Judge。根据角色设定和用户消息，独立评价每个候选的"
                "角色一致性 role_consistency 与对话质量 dialogue_quality，均为 0～10 数字。"
                "只返回 JSON 对象，不评价格式，因为格式由确定性规则评分。不得遗漏候选。\n\n"
                f"角色设定：\n{persona_text}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户消息：\n{prompt}\n\n候选：\n"
                f"{json.dumps(compact, ensure_ascii=False)}\n\n"
                "严格返回：{\"scores\":[{\"candidate_id\":\"...\","
                "\"role_consistency\":0,\"dialogue_quality\":0}]}"
            ),
        },
    ]


def _strip_fence(raw: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", raw, re.DOTALL)
    return match.group(1) if match else raw.strip()


def parse_judge_scores(raw: str, expected_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    try:
        data = json.loads(_strip_fence(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Judge 返回的不是合法 JSON") from exc
    if not isinstance(data, dict) or set(data) != {"scores"} or not isinstance(data["scores"], list):
        raise ValueError("Judge 必须仅返回 scores 数组")
    parsed: dict[str, dict[str, float]] = {}
    for item in data["scores"]:
        expected_fields = {"candidate_id", "role_consistency", "dialogue_quality"}
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError("Judge score 字段不正确")
        candidate_id = item["candidate_id"]
        if candidate_id in parsed:
            raise ValueError("Judge 返回重复 candidate_id")
        scores = {}
        for name in ("role_consistency", "dialogue_quality"):
            value = item[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 10:
                raise ValueError(f"Judge {name} 必须在 0～10 之间")
            scores[name] = float(value)
        parsed[candidate_id] = scores
    if set(parsed) != set(expected_ids) or len(parsed) != len(expected_ids):
        raise ValueError("Judge 未完整返回本组全部候选")
    return parsed


def judge_group_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    expected_ids: Sequence[str],
    *,
    max_attempts: int = 3,
    request: Callable[..., Any] | None = None,
) -> tuple[dict[str, dict[str, float]], int]:
    """Request one group per attempt and reject every incomplete response."""
    requester = request or client.chat.completions.create
    last_error = "未知错误"
    for attempt in range(1, max_attempts + 1):
        try:
            response = requester(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return parse_judge_scores(raw, expected_ids), attempt
        except Exception as exc:
            last_error = str(exc)
    raise JudgeError(f"Judge 连续 {max_attempts} 次未返回完整结构: {last_error}")


def load_reward_sources(persona_path: Path, style_path: Path) -> tuple[list[str], list[str]]:
    persona = json.loads(persona_path.read_text(encoding="utf-8"))
    persona_sources = []
    for value in persona.values():
        if isinstance(value, str):
            persona_sources.append(value)
        elif isinstance(value, list):
            persona_sources.extend(item for item in value if isinstance(item, str))
    persona_sources.append("\n".join(persona_sources))
    style_sources = []
    for row in read_jsonl(style_path):
        if isinstance(row.get("assistant"), str):
            style_sources.append(row["assistant"])
        else:
            messages = row.get("messages", [])
            style_sources.extend(
                item.get("content", "") for item in messages
                if isinstance(item, dict) and item.get("role") == "assistant"
            )
    return persona_sources, style_sources


def score_group(
    prompt: str,
    candidates: Sequence[dict[str, str]],
    judge_scores: dict[str, dict[str, float]],
    *,
    persona_sources: Sequence[str] = (),
    style_sources: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Combine complete Judge scores and deterministic rules for one group."""
    if any(not item["assistant"].strip() for item in candidates):
        raise JudgeError("候选含空回答，整组失败")
    result = []
    for item in candidates:
        candidate_id = item["candidate_id"]
        judge = judge_scores[candidate_id]
        format_score = format_consistency(item["assistant"])
        penalties = penalty_breakdown(
            item["assistant"], persona_sources=persona_sources, style_sources=style_sources
        )
        reward = combine_reward(
            judge["role_consistency"], format_score,
            judge["dialogue_quality"], penalties["total"],
        )
        result.append({
            "candidate_id": candidate_id,
            "prompt": prompt,
            "assistant": item["assistant"],
            "role_consistency": judge["role_consistency"],
            "format_consistency": format_score,
            "dialogue_quality": judge["dialogue_quality"],
            "penalties": penalties,
            "reward": reward,
        })
    return result
