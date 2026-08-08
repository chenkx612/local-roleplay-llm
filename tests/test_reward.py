"""Tests for deterministic rewards and strict Judge transactions."""

import json
import unittest
from types import SimpleNamespace

from roleplay.reward import (
    JudgeError,
    combine_reward,
    format_consistency,
    judge_group_with_retry,
    parse_judge_scores,
    penalty_breakdown,
    score_group,
)


IDS = ["c1", "c2", "c3", "c4"]


def raw_scores(ids=IDS):
    return json.dumps({
        "scores": [
            {"candidate_id": item, "role_consistency": 8, "dialogue_quality": 7}
            for item in reversed(ids)
        ]
    })


def response(raw):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
    )


class FormatRewardTests(unittest.TestCase):
    def test_strict_loose_and_invalid_format_scores(self):
        self.assertEqual(format_consistency("（点头）吾辈知道了。"), 10)
        self.assertEqual(format_consistency("（点头）吾辈（又点头）知道了。"), 5)
        self.assertEqual(format_consistency("(点头)吾辈知道了。"), 0)

    def test_penalties_cover_each_frozen_category_and_cap(self):
        self.assertEqual(penalty_breakdown("abcdefghijkl" * 3)["repetition"], 2)
        self.assertEqual(penalty_breakdown("回答😀")["abnormal_symbols"], 2)
        persona = "这是很长的角色设定文本" * 15
        self.assertEqual(
            penalty_breakdown(persona, persona_sources=[persona])["persona_copy"], 2
        )
        style = "（点头）这是一个足够长并且会被逐字照抄的风格回答。" * 3
        self.assertEqual(
            penalty_breakdown(style, style_sources=[style])["style_copy"], 2
        )
        combined = penalty_breakdown(
            persona + "😀<answer>" + "abcdefghijkl" * 3,
            persona_sources=[persona], style_sources=[persona],
        )
        self.assertEqual(combined["extra_labels"], 1)
        self.assertEqual(combined["total"], 5)

    def test_combined_reward_is_bounded(self):
        self.assertEqual(combine_reward(10, 10, 10, 0), 10)
        self.assertEqual(combine_reward(0, 0, 0, 5), -5)


class JudgeTests(unittest.TestCase):
    def test_accepts_out_of_order_complete_scores(self):
        parsed = parse_judge_scores(raw_scores(), IDS)
        self.assertEqual(list(parsed), list(reversed(IDS)))
        self.assertEqual(set(parsed), set(IDS))

    def test_rejects_missing_duplicate_and_invalid_scores(self):
        with self.assertRaisesRegex(ValueError, "完整"):
            parse_judge_scores(raw_scores(IDS[:-1]), IDS)
        duplicate = json.loads(raw_scores())
        duplicate["scores"][1]["candidate_id"] = duplicate["scores"][0]["candidate_id"]
        with self.assertRaisesRegex(ValueError, "重复"):
            parse_judge_scores(json.dumps(duplicate), IDS)
        invalid = json.loads(raw_scores())
        invalid["scores"][0]["dialogue_quality"] = 11
        with self.assertRaisesRegex(ValueError, "0～10"):
            parse_judge_scores(json.dumps(invalid), IDS)

    def test_retries_structural_failures_up_to_three_times(self):
        calls = []
        replies = [response("bad"), response(raw_scores(IDS[:-1])), response(raw_scores())]

        def request(**_kwargs):
            calls.append(1)
            return replies.pop(0)

        scores, attempts = judge_group_with_retry(
            object(), "judge", [], IDS, request=request
        )
        self.assertEqual(attempts, 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(set(scores), set(IDS))

    def test_failed_judge_never_reaches_update_callback(self):
        updates = []

        def always_bad(**_kwargs):
            return response("{}")

        with self.assertRaises(JudgeError):
            judge_group_with_retry(object(), "judge", [], IDS, request=always_bad)
        self.assertEqual(updates, [])

    def test_empty_candidate_fails_entire_group(self):
        candidates = [
            {"candidate_id": item, "assistant": "" if index == 2 else "（点头）回答"}
            for index, item in enumerate(IDS)
        ]
        judge = parse_judge_scores(raw_scores(), IDS)
        with self.assertRaisesRegex(JudgeError, "整组失败"):
            score_group("prompt", candidates, judge)
