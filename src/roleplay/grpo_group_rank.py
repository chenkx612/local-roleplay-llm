"""Historical four-candidate group ranking for the failed Stage 3 GRPO run."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Sequence

from openai import AsyncOpenAI

from roleplay.grpo_reward import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    JUDGE_MAX_ATTEMPTS,
    JUDGE_MAX_TOKENS,
    JUDGE_REASONING_EFFORT,
    JUDGE_TIMEOUT_SECONDS,
    GRPORewardError,
    JudgeCallError,
    JudgeResponseError,
    VIOLATION_CODES,
    _last_user_message,
    _redact_api_key,
    build_judge_system,
    load_frozen_persona,
)


LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class GroupRanking:
    tiers: tuple[tuple[str, ...], ...]
    violations: dict[str, tuple[str, ...]]
    reasons: dict[str, str]

    def rank_rewards(self) -> dict[str, float]:
        """Map tied average positions onto the historical 0–5 reward range."""
        rewards: dict[str, float] = {}
        position = 0
        for tier in self.tiers:
            positions = range(position, position + len(tier))
            average = sum(positions) / len(tier)
            reward = 5.0 * (len(LABELS) - 1 - average) / (len(LABELS) - 1)
            for label in tier:
                rewards[label] = reward
            position += len(tier)
        return rewards


def parse_group_ranking(raw: str) -> GroupRanking:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise JudgeResponseError("组内 Judge 返回不是有效 JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "tiers",
        "violations",
        "reasons",
    }:
        raise JudgeResponseError("组内 Judge 返回字段不正确")
    tiers = value["tiers"]
    if (
        not isinstance(tiers, list)
        or not tiers
        or any(not isinstance(tier, list) or not tier for tier in tiers)
    ):
        raise JudgeResponseError("tiers 必须是非空二维列表")
    flattened = [label for tier in tiers for label in tier]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(LABELS):
        raise JudgeResponseError("tiers 必须完整且无重复地包含 A/B/C/D")
    violations = value["violations"]
    reasons = value["reasons"]
    if not isinstance(violations, dict) or set(violations) != set(LABELS):
        raise JudgeResponseError("violations 必须按 A/B/C/D 给出")
    if not isinstance(reasons, dict) or set(reasons) != set(LABELS):
        raise JudgeResponseError("reasons 必须按 A/B/C/D 给出")
    parsed_violations: dict[str, tuple[str, ...]] = {}
    for label in LABELS:
        codes = violations[label]
        reason = reasons[label]
        if (
            not isinstance(codes, list)
            or any(not isinstance(code, str) for code in codes)
            or len(codes) != len(set(codes))
            or set(codes) - VIOLATION_CODES
        ):
            raise JudgeResponseError(f"{label} violations 无效")
        if not isinstance(reason, str) or not reason.strip():
            raise JudgeResponseError(f"{label} reason 不能为空")
        parsed_violations[label] = tuple(codes)
    return GroupRanking(
        tiers=tuple(tuple(tier) for tier in tiers),
        violations=parsed_violations,
        reasons={label: reasons[label].strip() for label in LABELS},
    )


def build_group_judge_user(user_message: str, completions: Sequence[str]) -> str:
    if len(completions) != len(LABELS) or any(
        not isinstance(item, str) for item in completions
    ):
        raise GRPORewardError("组内 Judge 必须接收四个字符串候选")
    return "请对以下 JSON 中的四个候选分层排序：\n" + json.dumps(
        {
            "user_message": user_message,
            "candidates": dict(zip(LABELS, completions)),
        },
        ensure_ascii=False,
    )


def build_group_judge_system(persona_text: str) -> str:
    return (
        build_judge_system(persona_text)
        + "\n\n现在不要独立打分，而要对 A/B/C/D 四个候选做从优到劣的"
        "分层排序。"
        "允许同层并列，但必须完整且只出现一次。只输出 JSON："
        '{"tiers":[["A"],["B","C"],["D"]],'
        '"violations":{"A":[],"B":[],"C":[],"D":[]},'
        '"reasons":{"A":"理由","B":"理由","C":"理由","D":"理由"}}'
    )


class GroupRankRewardEngine:
    """Historical one-call group ranker retained for reproducible tests."""

    def __init__(
        self,
        *,
        persona_text: str | None = None,
        client: Any | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.persona_text = persona_text or load_frozen_persona()
        if client is None:
            import os

            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise GRPORewardError("缺少 DEEPSEEK_API_KEY")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=DEEPSEEK_BASE_URL,
                max_retries=0,
                timeout=JUDGE_TIMEOUT_SECONDS,
            )
        self.client = client
        self._sleep = sleep
        self.system = build_group_judge_system(self.persona_text)

    async def score_group(
        self,
        completions: Sequence[str],
        messages: Sequence[Sequence[dict[str, Any]]],
        *,
        record_ids: Sequence[str] | None = None,
    ) -> list[float]:
        if len(completions) != 4 or len(messages) != 4:
            raise GRPORewardError("组内排序要求四个候选和四组消息")
        user_messages = [_last_user_message(item) for item in messages]
        if len(set(user_messages)) != 1:
            raise GRPORewardError("组内候选必须来自同一用户消息")
        last_error = "未知错误"
        for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": self.system},
                        {
                            "role": "user",
                            "content": build_group_judge_user(
                                user_messages[0], completions
                            ),
                        },
                    ],
                    max_tokens=JUDGE_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    extra_body={
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": JUDGE_REASONING_EFFORT,
                    },
                )
                if not getattr(response, "choices", None):
                    raise JudgeResponseError("组内 Judge 返回缺少 choices")
                choice = response.choices[0]
                if getattr(choice, "finish_reason", None) != "stop":
                    raise JudgeResponseError("组内 Judge 未正常结束")
                raw = getattr(getattr(choice, "message", None), "content", None)
                ranking = parse_group_ranking(raw)
                rewards = ranking.rank_rewards()
                return [rewards[label] for label in LABELS]
            except Exception as exc:
                last_error = _redact_api_key(f"{type(exc).__name__}: {exc}")
                if attempt < JUDGE_MAX_ATTEMPTS:
                    await self._sleep(float(2 ** (attempt - 1)))
        raise JudgeCallError(
            f"组内 Judge 连续 {JUDGE_MAX_ATTEMPTS} 次失败: {last_error}"
        )
