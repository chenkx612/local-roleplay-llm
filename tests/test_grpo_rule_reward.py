"""Tests for the continuous deterministic Stage 4 GRPO reward."""

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from roleplay.grpo_rule_reward import (
    HARD_INVALID_BASE,
    MIN_VALID_REWARD,
    RewardConstraints,
    RuleRewardEngine,
    analyze_actions,
    count_sentences,
    count_signature_references,
    find_wrong_self_references,
    load_frozen_reward_constraints,
    range_distance,
    score_completion,
    violation_to_score,
)


def constraints(
    *,
    min_actions=1,
    max_actions=1,
    min_sentences=1,
    max_sentences=2,
    min_chars=20,
    max_chars=70,
    min_signatures=1,
    max_signatures=1,
):
    return RewardConstraints(
        min_actions=min_actions,
        max_actions=max_actions,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        min_chars=min_chars,
        max_chars=max_chars,
        min_signatures=min_signatures,
        max_signatures=max_signatures,
    )


class ContinuousMappingTests(unittest.TestCase):
    def test_range_distance_is_zero_inside_and_linear_outside(self):
        self.assertEqual(range_distance(1, 1, 2), 0.0)
        self.assertEqual(range_distance(2, 1, 2), 0.0)
        self.assertEqual(range_distance(0, 1, 2), 1.0)
        self.assertEqual(range_distance(5, 1, 2), 3.0)
        self.assertEqual(range_distance(99, None, None), 0.0)

    def test_violation_mapping_is_continuous_and_ordered(self):
        scores = [violation_to_score(value) for value in (0, 1, 3, 10)]

        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[1], 0.0)
        self.assertEqual(scores[2], -0.5)
        self.assertGreater(scores[2], scores[3])
        self.assertGreater(scores[3], -1.0)

    def test_sentence_count_includes_unterminated_tail(self):
        self.assertEqual(count_sentences("现在就走。别磨蹭！"), 2)
        self.assertEqual(count_sentences("现在就走"), 1)
        self.assertEqual(count_sentences(""), 0)


class PersonaTests(unittest.TestCase):
    def test_simplified_and_traditional_signature_are_equivalent(self):
        self.assertEqual(count_signature_references("吾辈准备好了。"), 1)
        self.assertEqual(count_signature_references("吾輩准备好了。"), 1)
        self.assertEqual(count_signature_references("他说‘吾辈’。"), 0)

    def test_wrong_self_references_preserve_occurrence_count(self):
        references = find_wrong_self_references("我说我不是本大爷，也不是本喵。")

        self.assertEqual(references.count("我"), 2)
        self.assertEqual(references.count("本大爷"), 1)
        self.assertEqual(references.count("本喵"), 1)

    def test_wrong_self_penalty_remains_ordered_after_first_error(self):
        target = constraints(min_actions=None, max_actions=None)
        clean = score_completion("吾辈已经准备好了。", target)
        one = score_completion("吾辈说我已经准备好了。", target)
        three = score_completion("吾辈说我我我已经准备好了。", target)

        self.assertGreater(clean.persona_score, one.persona_score)
        self.assertGreater(one.persona_score, three.persona_score)


class ActionAndInstructionTests(unittest.TestCase):
    def test_action_analysis_preserves_severity_counts(self):
        action = analyze_actions("（点头）（）（很长的一段动作）好。")

        self.assertEqual(action.count, 3)
        self.assertEqual(action.invalid_content_count, 1)
        self.assertEqual(len(action.segment_lengths), 3)

    def test_forbidden_action_violation_orders_zero_one_and_three(self):
        target = constraints(
            min_actions=0,
            max_actions=0,
            min_sentences=None,
            max_sentences=None,
            min_signatures=0,
            max_signatures=1,
        )
        zero = score_completion("吾辈现在就提醒你去睡觉。", target)
        one = score_completion("（点头）吾辈现在就提醒你去睡觉。", target)
        three = score_completion(
            "（点头）（甩尾巴）（转身）吾辈现在就提醒你去睡觉。",
            target,
        )

        self.assertGreater(zero.instruction_score, one.instruction_score)
        self.assertGreater(one.instruction_score, three.instruction_score)
        self.assertGreater(zero.total_reward, one.total_reward)
        self.assertGreater(one.total_reward, three.total_reward)

    def test_exact_one_action_does_not_reward_two_actions(self):
        target = constraints()
        one = score_completion(
            "（认真点头）吾辈已经准备好出发执行重要任务了。", target
        )
        two = score_completion(
            "（认真点头）（轻甩尾巴）吾辈已经准备好出发了。", target
        )

        self.assertEqual(one.instruction_score, 1.0)
        self.assertLess(two.instruction_score, one.instruction_score)

    def test_length_inside_instruction_range_is_still_style_ordered(self):
        target = constraints(
            min_actions=None,
            max_actions=None,
            min_sentences=None,
            max_sentences=None,
            min_chars=10,
            max_chars=70,
        )
        midpoint = score_completion("吾辈" + "会认真处理这件重要事情" * 4, target)
        edge = score_completion("吾辈会认真处理这件事直到完成。", target)

        self.assertEqual(midpoint.instruction_score, edge.instruction_score)
        self.assertNotEqual(midpoint.style_score, edge.style_score)


class CompositeRewardTests(unittest.TestCase):
    def test_hard_invalid_is_below_every_valid_reward(self):
        target = constraints()
        invalid = score_completion(
            "（认真点头）吾辈还没有说完。",
            target,
            finish_reason="length",
        )
        poor_but_valid = score_completion(
            "（点头）（甩尾巴）（转身）我我我说了很多不合要求的话。",
            target,
        )

        self.assertLess(invalid.total_reward, MIN_VALID_REWARD)
        self.assertGreaterEqual(invalid.total_reward, HARD_INVALID_BASE)
        self.assertGreater(poor_but_valid.total_reward, invalid.total_reward)

    def test_hard_invalid_outputs_are_internally_ordered(self):
        target = constraints()
        truncated = score_completion(
            "（认真点头）吾辈还没有说完。",
            target,
            finish_reason="length",
        )
        truncated_and_unbalanced = score_completion(
            "（认真点头吾辈还没有说完。",
            target,
            finish_reason="length",
        )

        self.assertGreater(
            truncated.recoverability, truncated_and_unbalanced.recoverability
        )
        self.assertGreater(
            truncated.total_reward, truncated_and_unbalanced.total_reward
        )

    def test_format_issues_accumulate(self):
        result = score_completion(
            "助手：<think>略</think>（认真地（点头））吾辈会说清楚。",
            constraints(),
        )

        self.assertEqual(
            set(result.format_reasons),
            {"extra_tag", "speaker_label", "nested_parentheses"},
        )
        self.assertGreater(result.style_violation, 0.9)

    def test_ignores_framework_empty_think_wrapper(self):
        answer = "（认真点头）吾輩会把这件事说清楚。"
        plain = score_completion(answer, constraints())
        wrapped = score_completion(
            f"<think>\n\n</think>\n\n{answer}", constraints()
        )

        self.assertEqual(wrapped, plain)

    def test_loads_twenty_frozen_prompt_constraints(self):
        loaded = load_frozen_reward_constraints()

        self.assertEqual(len(loaded), 20)
        self.assertEqual(loaded["rule_grpo_0001"].min_actions, 1)
        self.assertEqual(loaded["rule_grpo_0019"].max_actions, 0)
        self.assertIsNone(loaded["rule_grpo_0011"].min_actions)


class RuleRewardEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_in_order_and_logs_continuous_components(self):
        targets = {
            "one": constraints(),
            "two": constraints(min_actions=0, max_actions=0),
        }
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "reward_samples.jsonl"
            engine = RuleRewardEngine(constraints=targets, log_path=log_path)

            rewards = await engine.score_batch(
                [
                    "（竖起耳朵）吾辈已经听清楚了，接下来会认真处理。",
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
            self.assertEqual(rows[0]["schema_version"], 2)
            self.assertEqual(rows[0]["global_step"], 3)
            self.assertIn("instruction_violation", rows[0]["components"])
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
