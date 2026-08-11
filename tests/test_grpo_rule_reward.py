"""Tests for the deterministic Stage 4 GRPO reward."""

import json
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from roleplay.grpo_rule_reward import (
    HARD_INVALID_REWARD,
    RuleRewardEngine,
    analyze_actions,
    calculate_action_score,
    calculate_length_score,
    calculate_signature_score,
    find_wrong_self_references,
    load_frozen_reward_policies,
    score_completion,
)


class LengthAndSignatureTests(unittest.TestCase):
    def test_length_score_boundaries(self):
        expected = {
            0: -1.0,
            15: -1.0,
            30: 1.0,
            90: 1.0,
            135: 0.0,
            180: -1.0,
            240: -1.0,
        }
        for length, score in expected.items():
            with self.subTest(length=length):
                self.assertAlmostEqual(calculate_length_score(length), score)

    def test_signature_score_discourages_spam(self):
        self.assertEqual(calculate_signature_score(0), 0.0)
        self.assertEqual(calculate_signature_score(1), 1.0)
        self.assertEqual(calculate_signature_score(2), 0.25)
        self.assertEqual(calculate_signature_score(3), -1.0)

    def test_wrong_self_reference_exclusions(self):
        text = "吾辈不会说‘我投降’，我们要保持自我，也不能忘我。"
        self.assertEqual(find_wrong_self_references(text), ())
        self.assertEqual(find_wrong_self_references("我才是本大爷！"), ("本大爷", "我"))


class ActionRewardTests(unittest.TestCase):
    def test_rewards_one_or_well_spaced_two_actions(self):
        one = analyze_actions("（竖起耳朵）吾辈已经听清楚了，接下来交给吾辈处理。")
        two = analyze_actions(
            "（竖起耳朵）吾辈先把情况听清楚，再决定最可靠的办法。"
            "（轻轻甩尾巴）这才不会出错。"
        )

        self.assertEqual(calculate_action_score(one, "encouraged"), 1.0)
        self.assertGreaterEqual(two.minimum_dialogue_gap, 12)
        self.assertEqual(calculate_action_score(two, "encouraged"), 1.0)

    def test_dense_pair_receives_only_small_reward(self):
        action = analyze_actions("（竖起耳朵）好。（甩尾巴）吾辈知道了。")

        self.assertTrue(action.dense_pair)
        self.assertEqual(calculate_action_score(action, "encouraged"), 0.2)

    def test_three_actions_or_excessive_action_text_is_penalized(self):
        three = analyze_actions("（一）（二）对白足够长一点。（三）结束。")
        overlong = analyze_actions("（" + "动" * 31 + "）吾辈知道了。")
        over_ratio = analyze_actions("（竖起耳朵甩动尾巴靠近看了又看）好。")

        self.assertEqual(calculate_action_score(three, "encouraged"), -1.0)
        self.assertEqual(calculate_action_score(overlong, "encouraged"), -1.0)
        self.assertEqual(calculate_action_score(over_ratio, "encouraged"), -1.0)

    def test_optional_and_forbidden_policies(self):
        none = analyze_actions("吾辈会直接告诉你答案。")
        one = analyze_actions("（认真点头）吾辈会直接告诉你答案。")

        self.assertEqual(calculate_action_score(none, "optional"), 1.0)
        self.assertEqual(calculate_action_score(one, "optional"), 1.0)
        self.assertEqual(calculate_action_score(none, "forbidden"), 1.0)
        self.assertEqual(calculate_action_score(one, "forbidden"), -1.0)

    def test_empty_numeric_and_symbol_only_parentheses_are_not_actions(self):
        for text in ("（）吾辈知道了。", "（123）吾辈知道了。", "(˘⌒)吾辈知道了。"):
            with self.subTest(text=text):
                action = analyze_actions(text)
                self.assertTrue(action.invalid_content)
                self.assertEqual(calculate_action_score(action, "encouraged"), -1.0)


class CompositeRewardTests(unittest.TestCase):
    def test_hard_invalid_overrides_positive_components(self):
        cases = (
            ("", None, False),
            ("吾辈还没有说完。", "length", False),
            ("αβγδεζηθικλμ", None, False),
            ("abcdefghijkl" * 3, None, False),
            ("（竖起耳朵吾辈知道了。", None, False),
        )
        for text, finish_reason, truncated in cases:
            with self.subTest(text=text):
                result = score_completion(
                    text,
                    "encouraged",
                    finish_reason=finish_reason,
                    is_truncated=truncated,
                )
                self.assertEqual(result.total_reward, HARD_INVALID_REWARD)
                self.assertTrue(result.hard_invalid_reasons)

    def test_format_issues_are_soft_penalties(self):
        result = score_completion(
            "助手：<think>略</think>（认真地（点头））吾辈会把这件事说清楚。",
            "encouraged",
        )

        self.assertEqual(
            set(result.format_reasons),
            {"extra_tag", "speaker_label", "nested_parentheses"},
        )
        self.assertEqual(result.format_penalty, 1.0)

    def test_ignores_framework_empty_think_wrapper(self):
        answer = "（认真点头）吾辈会把这件事说清楚。"

        plain = score_completion(answer, "encouraged")
        wrapped = score_completion(
            f"<think>\n\n</think>\n\n{answer}", "encouraged"
        )

        self.assertEqual(wrapped, plain)

    def test_valid_reward_is_clamped_above_hard_invalid(self):
        result = score_completion(
            "助手：我就是本大爷，吾辈吾辈吾辈。",
            "encouraged",
        )

        self.assertEqual(result.total_reward, -3.0)

    def test_loads_frozen_prompt_policy_distribution(self):
        policies = load_frozen_reward_policies()

        self.assertEqual(len(policies), 20)
        self.assertEqual(list(policies.values()).count("encouraged"), 10)
        self.assertEqual(list(policies.values()).count("optional"), 8)
        self.assertEqual(list(policies.values()).count("forbidden"), 2)


class RuleRewardEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_in_order_and_logs_components_without_api_key(self):
        policies = {"one": "encouraged", "two": "forbidden"}
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "reward_samples.jsonl"
            engine = RuleRewardEngine(policies=policies, log_path=log_path)

            rewards = await engine.score_batch(
                [
                    "（竖起耳朵）吾辈已经听清楚了，接下来交给吾辈处理。",
                    "吾辈会用一句纯对白提醒你，现在就去休息。",
                ],
                [[], []],
                record_ids=["one", "two"],
                prompt_ids=["p1", "p2"],
                request_ids=["r1", "r2"],
                global_step=3,
            )

            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(rewards), 2)
            self.assertEqual([row["record_id"] for row in rows], ["one", "two"])
            self.assertEqual(rows[0]["global_step"], 3)
            self.assertIn("length_score", rows[0]["components"])
            self.assertNotIn("judge", json.dumps(rows))


class PluginTests(unittest.TestCase):
    def test_registers_local_reward_without_openai_dependency(self):
        registry = {}
        rewards = types.ModuleType("swift.rewards")

        class FakeAsyncORM:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        rewards.AsyncORM = FakeAsyncORM
        rewards.orms = registry
        swift = types.ModuleType("swift")
        swift.rewards = rewards
        module_name = "roleplay.grpo_rule_reward_plugin"
        sys.modules.pop(module_name, None)
        with patch.dict(sys.modules, {"swift": swift, "swift.rewards": rewards}):
            module = importlib.import_module(module_name)

        self.assertIs(registry["morgana_rule_reward"], module.MorganaRuleRewardORM)
