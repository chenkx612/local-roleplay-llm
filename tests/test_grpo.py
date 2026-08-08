"""Tests for completion-only and clipped GRPO math."""

import math
import unittest

from roleplay.grpo import (
    clipped_policy_loss,
    clipped_surrogate_terms,
    completion_target_slice,
    group_should_skip,
    standardized_advantages,
)


class AdvantageTests(unittest.TestCase):
    def test_group_advantages_are_zero_mean_and_unit_variance(self):
        values = standardized_advantages([1, 2, 3, 4])
        self.assertAlmostEqual(sum(values), 0)
        self.assertAlmostEqual(sum(value * value for value in values) / 4, 1)

    def test_equal_rewards_skip_update(self):
        self.assertTrue(group_should_skip([3, 3, 3, 3]))
        self.assertEqual(standardized_advantages([3, 3, 3, 3]), [0, 0, 0, 0])
        self.assertFalse(group_should_skip([3, 3, 3, 4]))

    def test_rejects_nonfinite_rewards(self):
        with self.assertRaisesRegex(ValueError, "有限"):
            standardized_advantages([1, math.nan])


class PolicyLossTests(unittest.TestCase):
    def test_completion_slice_excludes_prompt_targets(self):
        selected = completion_target_slice(prompt_tokens=5, completion_tokens=3)
        self.assertEqual((selected.start, selected.stop), (4, 7))

    def test_old_policy_ratio_is_one(self):
        terms = clipped_surrogate_terms([-1, -2], [-1, -2], 0.5, 0.2)
        self.assertEqual(terms, [0.5, 0.5])
        self.assertEqual(clipped_policy_loss([-1], [-1], 0.5, 0.2), -0.5)

    def test_positive_advantage_clips_excessive_increase(self):
        term = clipped_surrogate_terms([math.log(2)], [0], 1, 0.2)[0]
        self.assertAlmostEqual(term, 1.2)

    def test_negative_advantage_clips_excessive_decrease(self):
        term = clipped_surrogate_terms([math.log(0.2)], [0], -1, 0.2)[0]
        self.assertAlmostEqual(term, -0.8)

    def test_per_candidate_losses_can_be_accumulated_before_one_update(self):
        losses = [
            clipped_policy_loss([0], [0], advantage, 0.2)
            for advantage in standardized_advantages([1, 2, 3, 4])
        ]
        self.assertAlmostEqual(sum(losses), 0)

    def test_rejects_misaligned_old_policy_tokens(self):
        with self.assertRaisesRegex(ValueError, "对齐"):
            clipped_policy_loss([0, 0], [0], 1, 0.2)
