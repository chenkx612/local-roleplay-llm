"""Tests for the SFT-to-chosen minimal-edit experiment."""

import json
import tempfile
import unittest
from pathlib import Path

from roleplay.dpo_editability import (
    CONTRACT,
    DPOEditabilityError,
    build_review_artifacts,
    evaluate_result,
    export_training_data,
    finalize_experiment,
)
from roleplay.dpo_data import sha256_file


def labels():
    return {
        "A": {
            "candidate_id": "dpo2_0001-c1",
            "assistant": (
                "（尾巴轻轻一甩）哈？当然可以。"
                "你先自己想想吧，吾辈就在旁边看着。"
            ),
        }
    }


def successful_result(**overrides):
    value = {
        "review_id": "editability-review-01",
        "decision": "successful_minimal_edit",
        "source_label": "A",
        "source_scores": {
            "generation_stability": 8,
            "role_consistency": 8,
            "dialogue_quality": 7,
        },
        "target_preference": "expression_naturalness",
        "improved_assistant": (
            "（尾巴轻轻一甩）哈？当然可以。"
            "你先慢慢想，吾辈会在旁边陪着你。"
        ),
        "improved_scores": {
            "generation_stability": 8,
            "role_consistency": 8,
            "dialogue_quality": 8,
        },
        "changes": "只调整后半句，使陪伴表达更自然。",
        "blocking_reasons": [],
        "notes": "原回答合格，但表达略显疏离。",
    }
    value.update(overrides)
    return value


class ReviewArtifactTests(unittest.TestCase):
    def test_includes_prompt_with_only_one_stable_candidate(self):
        prompts = [{"id": "dpo2_0001", "scenario": "daily", "user": "问题"}]
        candidates = [
            {
                "candidate_id": "dpo2_0001-c1",
                "prompt_id": "dpo2_0001",
                "assistant": "（点头）这是稳定回答。",
                "finish_reason": "stop",
            },
            {
                "candidate_id": "dpo2_0001-c2",
                "prompt_id": "dpo2_0001",
                "assistant": "（括号没有闭合",
                "finish_reason": "stop",
            },
        ]
        packet, key, results = build_review_artifacts(prompts, candidates)

        self.assertEqual(len(packet["items"]), 1)
        self.assertEqual(set(packet["items"][0]["answers"]), {"A"})
        self.assertEqual(
            key["items"][0]["labels"]["A"]["candidate_id"],
            "dpo2_0001-c1",
        )
        self.assertIsNone(results["results"][0]["decision"])


class ResultValidationTests(unittest.TestCase):
    def test_accepts_single_dimension_minimal_edit(self):
        parsed = evaluate_result(successful_result(), labels())

        self.assertGreaterEqual(parsed["similarity"], 0.65)
        self.assertGreaterEqual(parsed["length_ratio"], 0.8)
        self.assertLessEqual(parsed["length_ratio"], 1.2)

    def test_requires_qualified_single_defect_source(self):
        result = successful_result()
        result["source_scores"]["role_consistency"] = 7
        with self.assertRaisesRegex(DPOEditabilityError, "单缺陷"):
            evaluate_result(result, labels())

    def test_rejects_large_rewrite(self):
        with self.assertRaisesRegex(DPOEditabilityError, "相似度|长度比"):
            evaluate_result(
                successful_result(
                    improved_assistant=(
                        "这是一个结构、内容和措辞都完全不同的长篇教师回答，"
                        "它已经远离原模型输出，不能算作局部修改。"
                    )
                ),
                labels(),
            )

    def test_accepts_explicit_not_editable_decision(self):
        result = successful_result(
            decision="not_editable",
            target_preference=None,
            improved_assistant=None,
            improved_scores=None,
            changes="",
            blocking_reasons=["requires_large_rewrite"],
            notes="所有候选都需要重写多个部分。",
        )
        parsed = evaluate_result(result, labels())

        self.assertEqual(parsed["decision"], "not_editable")


class FinalizeExperimentTests(unittest.TestCase):
    def test_reports_insufficient_foundation_below_twenty_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            packet_items = []
            key_items = []
            result_items = []
            for index in range(1, 41):
                review_id = f"editability-review-{index:02d}"
                prompt_id = f"dpo2_{index:04d}"
                packet_items.append(
                    {
                        "review_id": review_id,
                        "prompt_id": prompt_id,
                        "scenario": "daily",
                        "user": "问题",
                        "answers": {"A": labels()["A"]["assistant"]},
                    }
                )
                key_items.append(
                    {
                        "review_id": review_id,
                        "prompt_id": prompt_id,
                        "labels": {
                            "A": {
                                "candidate_id": f"{prompt_id}-c1",
                                "assistant": labels()["A"]["assistant"],
                            }
                        },
                    }
                )
                if index <= 17:
                    result_items.append(successful_result(review_id=review_id))
                else:
                    result_items.append(
                        successful_result(
                            review_id=review_id,
                            decision="not_editable",
                            target_preference=None,
                            improved_assistant=None,
                            improved_scores=None,
                            changes="",
                            blocking_reasons=["no_qualified_source"],
                            notes="没有合格来源。",
                        )
                    )

            packet_path = experiment_dir / "editability_packet.json"
            key_path = experiment_dir / "editability_key.json"
            results_path = experiment_dir / "editability_results.json"
            packet_path.write_text(
                json.dumps({"items": packet_items}, ensure_ascii=False),
                encoding="utf-8",
            )
            key_path.write_text(
                json.dumps({"items": key_items}, ensure_ascii=False),
                encoding="utf-8",
            )
            results_path.write_text(
                json.dumps({"results": result_items}, ensure_ascii=False),
                encoding="utf-8",
            )
            summary_path = experiment_dir / "run_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "contract": CONTRACT,
                        "artifacts": {
                            packet_path.name: {"sha256": sha256_file(packet_path)},
                            key_path.name: {"sha256": sha256_file(key_path)},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report_path, pairs_path = finalize_experiment(
                experiment_dir=experiment_dir
            )
            system_prompt_path = experiment_dir / "system_prompt.txt"
            system_prompt_path.write_text("system", encoding="utf-8")
            training_path = export_training_data(
                experiment_dir=experiment_dir,
                output_path=experiment_dir / "train.jsonl",
                system_prompt_path=system_prompt_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            training_rows = [
                json.loads(line)
                for line in training_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(report["outcome"], "insufficient_sft_foundation")
            self.assertEqual(report["counts"]["successful_minimal_edits"], 17)
            pair_lines = pairs_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(pair_lines), 17)
            self.assertEqual(len(training_rows), 17)
            self.assertEqual(
                [row["role"] for row in training_rows[0]["messages"]],
                ["system", "user", "assistant"],
            )
            self.assertEqual(summary["status"], "completed")


if __name__ == "__main__":
    unittest.main()
