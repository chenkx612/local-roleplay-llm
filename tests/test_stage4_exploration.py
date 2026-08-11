"""Tests for the Stage 4 candidate-sampling exploration."""

import unittest

from roleplay.grpo_rule_reward import score_completion
from roleplay.stage4_exploration import (
    SAMPLING_LADDER,
    SamplingConfig,
    Stage4ExplorationError,
    normalize_candidate_rows,
    select_minimum_config,
    summarize_candidate_support,
)


def prompts():
    return [
        {
            "id": "rule_grpo_0001",
            "reward_policy": {"action": "encouraged"},
        },
        {
            "id": "rule_grpo_0002",
            "reward_policy": {"action": "forbidden"},
        },
    ]


def candidate(record_id: str, text: str, policy: str):
    components = score_completion(text, policy).as_log_dict()
    return {
        "record_id": record_id,
        "completion": text,
        "finish_reason": "stop",
        "components": components,
        "total_reward": components["total_reward"],
    }


class CandidateNormalizationTests(unittest.TestCase):
    def test_rejects_invalid_sampling_config(self):
        with self.assertRaisesRegex(ValueError, "采样配置"):
            SamplingConfig("bad", 0, 0.6, 0.8)

    def test_scores_rows_without_logged_components(self):
        config = SamplingConfig("test", 1, 0.6, 0.8)
        rows = normalize_candidate_rows(
            [{"record_id": "rule_grpo_0001", "completion": "（点头）吾辈知道了。"}],
            {"rule_grpo_0001": "encouraged"},
            config,
        )

        self.assertEqual(rows[0]["components"]["signature_count"], 1)
        self.assertTrue(rows[0]["compliance"]["action"])

    def test_rejects_unknown_record(self):
        with self.assertRaisesRegex(Stage4ExplorationError, "record_id"):
            normalize_candidate_rows(
                [{"record_id": "unknown", "completion": "回答"}],
                {"rule_grpo_0001": "encouraged"},
                SamplingConfig("test", 1, 0.6, 0.8),
            )


class SupportSummaryTests(unittest.TestCase):
    def test_full_support_passes_frozen_thresholds(self):
        config = SamplingConfig("test", 2, 0.6, 0.8)
        rows = [
            candidate(
                "rule_grpo_0001",
                "（认真点头）吾辈已经听清楚了，接下来先把最重要的事情妥善处理好。",
                "encouraged",
            ),
            candidate("rule_grpo_0001", "我不知道。", "encouraged"),
            candidate(
                "rule_grpo_0002",
                "吾辈当然不是普通宠物，更不会任由别人随意摆布、命令或者轻视。",
                "forbidden",
            ),
            candidate("rule_grpo_0002", "（甩尾巴）我不是宠物。", "forbidden"),
        ]

        summary = summarize_candidate_support(rows, prompts(), config)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["support_rates"]["fully_compliant"], 1.0)
        self.assertEqual(summary["support_rates"]["forbidden_action"], 1.0)

    def test_missing_signature_support_fails(self):
        config = SamplingConfig("test", 1, 0.6, 0.8)
        rows = [
            candidate(
                "rule_grpo_0001",
                "（认真点头）已经听清楚了，接下来先处理最重要的事情。",
                "encouraged",
            ),
            candidate(
                "rule_grpo_0002",
                "不是普通宠物，也不会任由别人随意摆布。",
                "forbidden",
            ),
        ]

        summary = summarize_candidate_support(rows, prompts(), config)

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["signature"])

    def test_requires_exact_group_size(self):
        with self.assertRaisesRegex(Stage4ExplorationError, "组大小"):
            summarize_candidate_support(
                [candidate("rule_grpo_0001", "吾辈知道了。", "encouraged")],
                prompts(),
                SamplingConfig("test", 1, 0.6, 0.8),
            )

    def test_each_probe_round_must_pass(self):
        config = SamplingConfig("test", 1, 0.6, 0.8, 2)
        rows = [
            {
                **candidate(
                    "rule_grpo_0001",
                    "（认真点头）吾辈已经听清楚了，接下来先把最重要的事情妥善处理好。",
                    "encouraged",
                ),
                "probe_round": 1,
            },
            {
                **candidate(
                    "rule_grpo_0002",
                    "吾辈当然不是普通宠物，更不会任由别人随意摆布、命令或者轻视。",
                    "forbidden",
                ),
                "probe_round": 1,
            },
            {
                **candidate(
                    "rule_grpo_0001", "我不知道。", "encouraged"
                ),
                "probe_round": 2,
            },
            {
                **candidate(
                    "rule_grpo_0002", "我不是宠物。", "forbidden"
                ),
                "probe_round": 2,
            },
        ]

        summary = summarize_candidate_support(rows, prompts(), config)

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["signature"])
        self.assertEqual(
            summary["round_support_rates"]["1"]["signature"], 1.0
        )
        self.assertEqual(
            summary["round_support_rates"]["2"]["signature"], 0.0
        )

    def test_selects_first_passing_ladder_config(self):
        summaries = [
            {"config": {"name": SAMPLING_LADDER[0].name}, "passed": False},
            {"config": {"name": SAMPLING_LADDER[1].name}, "passed": True},
            {"config": {"name": SAMPLING_LADDER[2].name}, "passed": True},
        ]

        self.assertEqual(
            select_minimum_config(summaries), SAMPLING_LADDER[1].name
        )


if __name__ == "__main__":
    unittest.main()
