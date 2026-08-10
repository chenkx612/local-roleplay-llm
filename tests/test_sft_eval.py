"""Tests for deterministic SFT evaluation and manual review gates."""

import copy
import unittest

from roleplay.sft_eval import (
    EVALUATION_SEEDS,
    MANUAL_SCORE_DIMENSIONS,
    build_manual_review,
    empty_manual_review_results,
    evaluate_core_behavior_gate,
    evaluate_manual_review,
    has_gibberish,
    inspect_output,
    normalize_empty_think_wrapper,
    summarize_outputs,
    validate_aligned_outputs,
    validate_output_grid,
)


IDS = tuple(f"dev_{index:04d}" for index in range(1, 11))


def make_rows(
    answer_for,
    *,
    seeds=EVALUATION_SEEDS,
    finish_for=lambda _seed, _index: "stop",
):
    rows = []
    for seed in seeds:
        for index, record_id in enumerate(IDS, 1):
            assistant = answer_for(seed, index)
            rows.append(
                {
                    "seed": seed,
                    "id": record_id,
                    "scenario": "emotion" if index in {3, 8} else "daily",
                    "target_goals": ["role_consistency"],
                    "user": f"问题 {index}",
                    "assistant": assistant,
                    "raw_assistant": f"<think></think>{assistant}",
                    "finish_reason": finish_for(seed, index),
                    "attempts": 1,
                }
            )
    return rows


def score_pair(value=8):
    return {
        label: {dimension: value for dimension in MANUAL_SCORE_DIMENSIONS}
        for label in ("A", "B")
    }


def passing_manual_results(packet, answer_key):
    mapping = {item["review_id"]: item for item in answer_key["answers"]}
    results = []
    for index, item in enumerate(packet["items"]):
        labels = mapping[item["review_id"]]
        results.append(
            {
                "review_id": item["review_id"],
                "winner": labels["sft_label"] if index < 6 else "tie",
                "clearly_worse": None,
                "scores": score_pair(),
                "severe_issues": {"A": [], "B": []},
            }
        )
    return {"schema_version": 2, "results": results}


class OutputInspectionTests(unittest.TestCase):
    def inspect(self, assistant, finish_reason="stop"):
        return inspect_output(
            {"assistant": assistant, "finish_reason": finish_reason}
        )

    def test_normalizes_only_a_leading_empty_think_wrapper(self):
        raw = " \n<think>\n</think>\n（点头）吾辈知道了。 "
        self.assertEqual(
            normalize_empty_think_wrapper(raw), "（点头）吾辈知道了。"
        )
        nonempty = "<think>秘密</think>（点头）回答"
        self.assertEqual(normalize_empty_think_wrapper(nonempty), nonempty)

    def test_strict_format_requires_one_full_width_opening_action_and_dialogue(self):
        self.assertTrue(self.inspect("（点头）吾辈知道了。")["strict_format"])
        bad_answers = (
            "(点头)吾辈知道了。",
            "（点头）",
            "（点头）（微笑）吾辈知道了。",
            "（点头）吾辈（微笑）知道了。",
            "（点头）吾辈知道了。(笑)",
            "（点头(轻笑)）吾辈知道了。",
            "<answer>（点头）吾辈知道了。</answer>",
            "旁白：（点头）吾辈知道了。",
        )
        for answer in bad_answers:
            with self.subTest(answer=answer):
                self.assertFalse(self.inspect(answer)["strict_format"])

    def test_detects_truncation_repetition_symbols_and_unclosed_brackets(self):
        truncated = self.inspect("（点头）回答。", "length")
        self.assertTrue(truncated["truncated"])
        repeated = self.inspect("（点头）" + "abcdefghijkl" * 3)
        self.assertTrue(repeated["repeated_span"])
        self.assertTrue(repeated["degenerate"])
        short_loop = self.inspect("哈" * 35)
        self.assertTrue(short_loop["repeated_span"])
        self.assertFalse(self.inspect("吾辈才没有哈哈哈哈！")["repeated_span"])
        for answer in (
            "（点头）回答😀",
            "（点头）回答！！！！",
            "（点头回答",
            "）点头（回答",
        ):
            with self.subTest(answer=answer):
                result = self.inspect(answer)
                self.assertTrue(
                    result["abnormal_symbols"] or result["unclosed_brackets"]
                )
                self.assertTrue(result["degenerate"])

    def test_gibberish_detection_does_not_treat_emoji_as_unreadable(self):
        self.assertFalse(has_gibberish("吾辈知道了 😏🐾✨ 🧑‍💻 👨‍👩‍👧‍👦"))
        self.assertTrue(has_gibberish("ΣΥΧΡΟΣΔΕΖΗΟΙΚΤΣΥΧΡΟΣ"))

    def test_detects_signature_and_wrong_self_references(self):
        good = self.inspect("（叉腰）吾辈当然知道。")
        self.assertTrue(good["uses_signature_self_reference"])
        self.assertFalse(good["uses_wrong_self_reference"])
        for alias in ("本大爷", "本喵"):
            self.assertTrue(
                self.inspect(f"（叉腰）{alias}当然知道。")[
                    "uses_wrong_self_reference"
                ]
            )


class AutomaticGateTests(unittest.TestCase):
    def setUp(self):
        self.base = make_rows(lambda _seed, _index: "(点头)本喵知道了。")
        self.sft = make_rows(lambda _seed, _index: "（点头）吾辈知道了。")

    def test_v2_uses_the_stage1_frozen_seed(self):
        self.assertEqual(EVALUATION_SEEDS, (20260807,))

    def test_aggregates_fixed_seeds_and_reports_concrete_issue_keys(self):
        rows = copy.deepcopy(self.sft)
        rows[0]["assistant"] = "（点头）回答😀"
        rows[1]["finish_reason"] = "max_tokens"
        summary = summarize_outputs(rows, EVALUATION_SEEDS, IDS)
        self.assertEqual(
            summary["overall"]["records"], len(EVALUATION_SEEDS) * len(IDS)
        )
        self.assertEqual(len(summary["seeds"]), len(EVALUATION_SEEDS))
        self.assertIn("20260807:dev_0001", summary["overall"]["abnormal_symbol_ids"])
        self.assertIn("20260807:dev_0002", summary["overall"]["truncated_ids"])
        self.assertEqual(summary["seeds"][0]["seed"], 20260807)
        self.assertIn("dev_0001", summary["seeds"][0]["abnormal_symbol_ids"])

    def test_rejects_duplicate_missing_and_misaligned_pairs(self):
        with self.assertRaisesRegex(ValueError, "重复键"):
            validate_output_grid(self.sft + [self.sft[0]], EVALUATION_SEEDS, IDS)
        with self.assertRaisesRegex(ValueError, "网格不完整"):
            validate_output_grid(self.sft[:-1], EVALUATION_SEEDS, IDS)
        misaligned = copy.deepcopy(self.sft)
        misaligned[0]["user"] = "不同问题"
        with self.assertRaisesRegex(ValueError, "未对齐"):
            validate_aligned_outputs(self.base, misaligned, EVALUATION_SEEDS, IDS)

    def test_core_gate_passes_stable_generation_requirements(self):
        gate = evaluate_core_behavior_gate(
            self.base, self.sft, EVALUATION_SEEDS, IDS
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["goal"], "generation_stability")
        self.assertTrue(all(gate["checks"].values()))

    def test_format_is_diagnostic_but_symbol_regression_fails_gate(self):
        sft = make_rows(lambda _seed, _index: "吾辈知道了 🧑‍💻")
        gate = evaluate_core_behavior_gate(
            self.base, sft, EVALUATION_SEEDS, IDS
        )
        self.assertEqual(gate["sft"]["overall"]["strict_format_rate"], 0.0)
        self.assertEqual(
            len(gate["sft"]["overall"]["abnormal_symbol_ids"]),
            len(EVALUATION_SEEDS) * len(IDS),
        )
        self.assertFalse(gate["checks"]["abnormal_symbol_count_not_higher"])
        self.assertFalse(gate["checks"]["degeneration_count_not_higher"])
        self.assertFalse(gate["passed"])

    def test_core_gate_fails_each_stability_regression(self):
        cases = {}
        fewer_stops = copy.deepcopy(self.sft)
        fewer_stops[0]["finish_reason"] = "cancelled"
        cases["stop_count_not_lower"] = fewer_stops
        truncated = copy.deepcopy(self.sft)
        truncated[0]["finish_reason"] = "length"
        cases["truncation_count_not_higher"] = truncated
        unclosed = copy.deepcopy(self.sft)
        unclosed[0]["assistant"] = "（点头吾辈知道了。"
        cases["unclosed_bracket_count_not_higher"] = unclosed
        abnormal = copy.deepcopy(self.sft)
        abnormal[0]["assistant"] += "🐾"
        cases["abnormal_symbol_count_not_higher"] = abnormal
        repeated = copy.deepcopy(self.sft)
        repeated[0]["assistant"] = "哈" * 35
        cases["no_repeated_spans"] = repeated
        gibberish = copy.deepcopy(self.sft)
        gibberish[0]["assistant"] += "ΣΥΧΡΟΣΔΕΖΗΟΙΚΤΣΥΧΡΟΣ"
        cases["no_gibberish"] = gibberish
        for failed_check, rows in cases.items():
            with self.subTest(failed_check=failed_check):
                gate = evaluate_core_behavior_gate(
                    self.base, rows, EVALUATION_SEEDS, IDS
                )
                self.assertFalse(gate["checks"][failed_check])
                self.assertFalse(gate["passed"])

    def test_core_gate_rejects_new_wrong_self_reference(self):
        base = make_rows(lambda _seed, _index: "（点头）吾辈知道了。")
        sft = copy.deepcopy(base)
        sft[0]["assistant"] = "（点头）本大爷知道了。"

        gate = evaluate_core_behavior_gate(
            base, sft, EVALUATION_SEEDS, IDS
        )

        self.assertFalse(
            gate["checks"]["wrong_self_reference_count_not_higher"]
        )
        self.assertFalse(gate["passed"])


class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        base = make_rows(
            lambda _seed, index: f"(点头)Base {index}",
            seeds=(EVALUATION_SEEDS[0],),
        )
        sft = make_rows(
            lambda _seed, index: f"（点头）SFT {index}",
            seeds=(EVALUATION_SEEDS[0],),
        )
        self.packet, self.answer_key = build_manual_review(base, sft, IDS)

    def test_anonymous_order_is_reproducible_and_mapping_is_separate(self):
        packet_again, answer_key_again = build_manual_review(
            make_rows(lambda _seed, index: f"(点头)Base {index}", seeds=(20260807,)),
            make_rows(
                lambda _seed, index: f"（点头）SFT {index}",
                seeds=(20260807,),
            ),
            IDS,
        )
        self.assertEqual(self.packet, packet_again)
        self.assertEqual(self.answer_key, answer_key_again)
        self.assertNotIn("sft_label", str(self.packet))
        self.assertEqual(len(self.packet["items"]), 10)
        blank = empty_manual_review_results(self.packet)
        self.assertEqual(blank["results"], [])
        self.assertEqual(len(blank["expected_review_ids"]), 10)
        self.assertEqual(
            self.packet["score_dimensions"], list(MANUAL_SCORE_DIMENSIONS)
        )

    def test_manual_gate_passes_thresholds(self):
        result = evaluate_manual_review(
            self.packet,
            self.answer_key,
            passing_manual_results(self.packet, self.answer_key),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["sft_wins"], 6)
        self.assertEqual(result["sft_clear_losses"], 0)

    def test_rejects_missing_and_duplicate_reviews(self):
        artifact = passing_manual_results(self.packet, self.answer_key)
        artifact["results"].pop()
        with self.assertRaisesRegex(ValueError, "不完整"):
            evaluate_manual_review(self.packet, self.answer_key, artifact)
        artifact = passing_manual_results(self.packet, self.answer_key)
        artifact["results"].append(copy.deepcopy(artifact["results"][0]))
        with self.assertRaisesRegex(ValueError, "重复"):
            evaluate_manual_review(self.packet, self.answer_key, artifact)

    def test_manual_gate_enforces_wins_clear_losses_and_severe_issues(self):
        mapping = {
            item["review_id"]: item for item in self.answer_key["answers"]
        }
        cases = {}
        five_wins = passing_manual_results(self.packet, self.answer_key)
        five_wins["results"][5]["winner"] = "tie"
        cases["sft_wins_at_least_6"] = five_wins
        clear_losses = passing_manual_results(self.packet, self.answer_key)
        for result in clear_losses["results"][:3]:
            result["clearly_worse"] = mapping[result["review_id"]]["sft_label"]
        cases["sft_clear_losses_at_most_2"] = clear_losses
        severe = passing_manual_results(self.packet, self.answer_key)
        first = severe["results"][0]
        first["severe_issues"][mapping[first["review_id"]]["sft_label"]] = [
            "role_break"
        ]
        cases["sft_has_no_severe_issues"] = severe
        for failed_check, artifact in cases.items():
            with self.subTest(failed_check=failed_check):
                result = evaluate_manual_review(
                    self.packet, self.answer_key, artifact
                )
                self.assertFalse(result["checks"][failed_check])
                self.assertFalse(result["passed"])

    def test_manual_gate_requires_each_core_score_not_to_regress(self):
        artifact = passing_manual_results(self.packet, self.answer_key)
        mapping = {
            item["review_id"]: item for item in self.answer_key["answers"]
        }
        for result in artifact["results"]:
            labels = mapping[result["review_id"]]
            result["scores"][labels["base_label"]]["role_consistency"] = 9
            result["scores"][labels["sft_label"]]["role_consistency"] = 7
        gate = evaluate_manual_review(self.packet, self.answer_key, artifact)
        self.assertFalse(
            gate["checks"]["sft_role_consistency_score_not_lower"]
        )


if __name__ == "__main__":
    unittest.main()
