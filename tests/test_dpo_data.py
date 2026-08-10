"""Tests for the high-quality morgana-v2 DPO preference-data workflow."""

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from roleplay.dpo_data import (
    BASE_MODEL,
    BASE_REVISION,
    CANDIDATES_PER_PROMPT,
    CONTRACT,
    DPODataError,
    FROZEN_HASHES,
    GENERATION,
    MIN_FINAL_PAIRS,
    SFT_ADAPTER_PATH,
    _artifact,
    _summary_contract,
    build_codex_review_artifacts,
    candidate_seed,
    finalize_run,
    generate_candidates,
    is_stable_candidate,
    load_prompts,
    parse_codex_result,
    prepare_run,
    validate_cross_split_uniqueness,
    write_json_atomic,
    write_jsonl_atomic,
)


def candidate(prompt_id, index, answer, finish_reason="stop"):
    return {
        "candidate_id": f"{prompt_id}-c{index}",
        "prompt_id": prompt_id,
        "candidate_index": index,
        "seed": candidate_seed(prompt_id, index),
        "assistant": answer,
        "raw_assistant": answer,
        "finish_reason": finish_reason,
        "source": "sft_candidate",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter_sha256": FROZEN_HASHES[
            SFT_ADAPTER_PATH / "adapter_model.safetensors"
        ],
        "generation": GENERATION,
    }


def codex_result(review_id, candidate_labels=("A", "B"), **overrides):
    labels = tuple(candidate_labels)
    value = {
        "review_id": review_id,
        "decision": "clear_preference",
        "source_label": labels[0],
        "rejected_label": labels[1],
        "candidate_scores": {
            label: {
                "generation_stability": 9,
                "role_consistency": 8 if index == 0 else 5,
                "dialogue_quality": 8 if index == 0 else 5,
            }
            for index, label in enumerate(labels)
        },
        "candidate_issues": {label: [] for label in labels},
        "preference_reasons": ["character_consistency"],
        "material_tradeoff": False,
        "hard_rule_only": False,
        "improved_assistant": None,
        "improved_scores": None,
        "improved_issues": None,
        "changes": "",
        "notes": "A 更符合角色。",
    }
    value.update(overrides)
    return value


class FrozenPromptTests(unittest.TestCase):
    def test_v2_prompts_are_balanced_and_cross_split_unique(self):
        prompts = load_prompts()
        self.assertEqual(len(prompts), 40)
        self.assertEqual(prompts[0]["id"], "dpo2_0001")
        scenarios = {name: 0 for name in {row["scenario"] for row in prompts}}
        for row in prompts:
            scenarios[row["scenario"]] += 1
        self.assertEqual(set(scenarios.values()), {8})
        validate_cross_split_uniqueness(prompts)

    def test_candidate_seeds_are_fixed_and_non_overlapping(self):
        self.assertEqual(candidate_seed("dpo2_0001", 1), 20260810)
        self.assertEqual(candidate_seed("dpo2_0001", 2), 20260811)
        self.assertEqual(candidate_seed("dpo2_0001", 4), 20260813)
        self.assertEqual(candidate_seed("dpo2_0002", 1), 20260910)

    def test_stability_filter_rejects_truncation_repetition_and_brackets(self):
        self.assertTrue(
            is_stable_candidate({"assistant": "（点头）可以。", "finish_reason": "stop"})
        )
        self.assertFalse(
            is_stable_candidate({"assistant": "可以。", "finish_reason": "length"})
        )
        self.assertFalse(
            is_stable_candidate({"assistant": "（点头 可以。", "finish_reason": "stop"})
        )
        repeated = "重复内容十二" * 8
        self.assertFalse(
            is_stable_candidate({"assistant": repeated, "finish_reason": "stop"})
        )


class CandidateGenerationTests(unittest.TestCase):
    def test_generates_exactly_four_candidates_once(self):
        fake_mlx = types.ModuleType("mlx_lm")
        fake_mlx.load = lambda *_args, **_kwargs: (object(), object())
        prompt = {
            "id": "dpo2_0001",
            "scenario": "daily",
            "target_goals": [],
            "user": "问题",
        }
        outputs = iter(
            [
                ("回答一", "stop"),
                ("回答二", "stop"),
                ("回答三", "stop"),
                ("回答四", "stop"),
            ]
        )
        with patch.dict(sys.modules, {"mlx_lm": fake_mlx}), patch(
            "roleplay.dpo_data._verify_loaded_adapter"
        ), patch(
            "roleplay.dpo_data._generate_one",
            side_effect=lambda *_args: next(outputs),
        ):
            rows = generate_candidates(
                prompts=[prompt],
                base_model_path=Path("base"),
                mlx_adapter_path=Path("adapter"),
                system_prompt="system",
            )
        self.assertEqual(len(rows), CANDIDATES_PER_PROMPT)
        self.assertEqual(
            [row["seed"] for row in rows],
            [20260810, 20260811, 20260812, 20260813],
        )
        self.assertTrue(all(row["generation"] == CONTRACT["generation"] for row in rows))


class CodexContractTests(unittest.TestCase):
    def setUp(self):
        self.labels = {
            "A": {
                "assistant": "（点头）吾辈当然会认真陪着你。",
                "source": "sft_candidate",
                "source_id": "a",
            },
            "B": {
                "assistant": "（摇头）本大爷才不管你呢。",
                "source": "sft_candidate",
                "source_id": "b",
            },
        }

    def test_accepts_clear_preference(self):
        parsed = parse_codex_result(codex_result("r1"), self.labels)
        self.assertEqual(parsed["decision"], "clear_preference")

    def test_clear_preference_rejects_tradeoff_or_hard_rule_only(self):
        with self.assertRaisesRegex(DPODataError, "语义不一致"):
            parse_codex_result(
                codex_result("r1", material_tradeoff=True), self.labels
            )
        with self.assertRaisesRegex(DPODataError, "语义不一致"):
            parse_codex_result(
                codex_result("r1", hard_rule_only=True), self.labels
            )

    def test_clear_preference_enforces_absolute_chosen_quality(self):
        result = codex_result("r1")
        result["candidate_scores"]["A"]["dialogue_quality"] = 6
        with self.assertRaisesRegex(DPODataError, "绝对质量门槛"):
            parse_codex_result(result, self.labels)

        result = codex_result("r1")
        result["candidate_issues"]["A"] = ["template_overuse"]
        with self.assertRaisesRegex(DPODataError, "绝对质量门槛"):
            parse_codex_result(result, self.labels)

    def test_clear_preference_requires_stable_rejected_and_two_point_gap(self):
        result = codex_result("r1")
        result["candidate_scores"]["B"]["generation_stability"] = 6
        with self.assertRaisesRegex(DPODataError, "稳定性前提"):
            parse_codex_result(result, self.labels)

        result = codex_result("r1")
        result["candidate_scores"]["B"].update(
            {"role_consistency": 7, "dialogue_quality": 7}
        )
        with self.assertRaisesRegex(DPODataError, "偏好分差不足"):
            parse_codex_result(result, self.labels)

    def test_accepts_no_clear_preference_without_winner(self):
        parsed = parse_codex_result(
            codex_result(
                "r1",
                decision="no_clear_preference",
                source_label=None,
                rejected_label=None,
                preference_reasons=[],
                material_tradeoff=True,
                notes="两条各有明显优劣。",
            ),
            self.labels,
        )
        self.assertTrue(parsed["material_tradeoff"])

    def test_teacher_edit_must_be_small_stable_change(self):
        source = self.labels["A"]["assistant"]
        improved = source.replace("认真", "一直认真")
        parsed = parse_codex_result(
            codex_result(
                "r1",
                decision="teacher_edit",
                improved_scores={
                    "generation_stability": 9,
                    "role_consistency": 9,
                    "dialogue_quality": 9,
                },
                improved_issues=[],
                improved_assistant=improved,
                changes="补充持续陪伴的语义。",
                notes="两条原回答都不够自然，最小修改 A。",
            ),
            self.labels,
        )
        self.assertEqual(parsed["improved_assistant"], improved)
        with self.assertRaisesRegex(DPODataError, "语义不一致"):
            parse_codex_result(
                codex_result(
                    "r1",
                    decision="teacher_edit",
                    rejected_label="A",
                    improved_scores={
                        "generation_stability": 9,
                        "role_consistency": 9,
                        "dialogue_quality": 9,
                    },
                    improved_issues=[],
                    improved_assistant=improved,
                    changes="补充持续陪伴的语义。",
                    notes="来源和 rejected 不能相同。",
                ),
                self.labels,
            )
        with self.assertRaisesRegex(DPODataError, "长度变化超过"):
            parse_codex_result(
                codex_result(
                    "r1",
                    decision="teacher_edit",
                    improved_scores={
                        "generation_stability": 9,
                        "role_consistency": 9,
                        "dialogue_quality": 9,
                    },
                    improved_issues=[],
                    improved_assistant=(
                        source
                        + "吾辈还会替你安排路线、准备食物、检查行李，"
                        + "再把每一个可能发生的问题全部提前解决。"
                    ),
                    changes="大幅扩写。",
                    notes="不应通过。",
                ),
                self.labels,
            )


class ReviewArtifactTests(unittest.TestCase):
    def test_packet_is_anonymous_and_excludes_unstable_pairs(self):
        prompts = [
            {"id": "dpo2_0001", "scenario": "daily", "user": "问题一"},
            {"id": "dpo2_0002", "scenario": "daily", "user": "问题二"},
        ]
        candidates = [
            candidate("dpo2_0001", 1, "（点头）回答一。"),
            candidate("dpo2_0001", 2, "（摇头）回答二。"),
            candidate("dpo2_0001", 3, "（认真）回答三。"),
            candidate("dpo2_0001", 4, "（微笑）回答四。"),
            candidate("dpo2_0002", 1, "（点头）回答三。"),
            candidate("dpo2_0002", 2, "（没闭合", finish_reason="stop"),
            candidate("dpo2_0002", 3, "重复内容十二" * 8),
            candidate("dpo2_0002", 4, "（截断）回答", finish_reason="length"),
        ]
        packet, key, results, unresolved = build_codex_review_artifacts(
            prompts=prompts,
            candidates=candidates,
            order_seed=1,
        )
        self.assertEqual(len(packet["items"]), 1)
        self.assertEqual(unresolved, ["dpo2_0002"])
        serialized = json.dumps(packet, ensure_ascii=False)
        self.assertNotIn("sft_candidate", serialized)
        self.assertEqual(set(key["items"][0]["labels"]), {"A", "B", "C", "D"})
        self.assertIsNone(results["results"][0]["decision"])


class PrepareWorkflowTests(unittest.TestCase):
    @staticmethod
    def fake_candidates(prompts, **_kwargs):
        return [
            candidate(
                prompt["id"],
                index,
                f"（点头）{prompt['id']} 的完整回答 {index}。",
            )
            for prompt in prompts
            for index in range(1, CANDIDATES_PER_PROMPT + 1)
        ]

    def test_prepare_builds_codex_packet_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base"
            base.mkdir()
            with patch(
                "roleplay.dpo_data.convert_peft_adapter_to_mlx",
                return_value=root / "adapter",
            ), patch(
                "roleplay.dpo_data.generate_candidates",
                side_effect=self.fake_candidates,
            ):
                run = prepare_run(
                    run_id="test-v2",
                    output_root=root / "output",
                    base_model_path=base,
                )
                second = prepare_run(
                    run_id="test-v2",
                    output_root=root / "output",
                    base_model_path=base,
                )
            self.assertEqual(run, second)
            summary = json.loads((run / "run_summary.json").read_text())
            packet = json.loads((run / "codex_review_packet.json").read_text())
            self.assertEqual(summary["status"], "awaiting_codex_review")
            self.assertEqual(summary["counts"]["candidates"], 160)
            self.assertEqual(len(packet["items"]), 40)
            self.assertFalse((run / "manual_review_packet.json").exists())


class FinalizeTests(unittest.TestCase):
    def make_run(
        self, root: Path, included=30, teacher_count=0, unstable_prompts=0
    ):
        run = root / "run"
        run.mkdir()
        prompts = load_prompts()
        candidates = []
        for prompt_index, prompt in enumerate(prompts):
            for candidate_index in range(1, CANDIDATES_PER_PROMPT + 1):
                finish_reason = (
                    "length"
                    if prompt_index < unstable_prompts and candidate_index > 1
                    else "stop"
                )
                candidates.append(
                    candidate(
                        prompt["id"],
                        candidate_index,
                        (
                            f"（点头）这是 {prompt['id']} 的完整候选回答 "
                            f"{candidate_index}。"
                        ),
                        finish_reason=finish_reason,
                    )
                )
        candidates_path = run / "candidates.jsonl"
        write_jsonl_atomic(candidates_path, candidates)
        packet, key, blank, _ = build_codex_review_artifacts(
            prompts=prompts,
            candidates=candidates,
        )
        results = []
        for index, row in enumerate(blank["results"]):
            review_id = row["review_id"]
            labels = key["items"][index]["labels"]
            if index >= included:
                results.append(
                    codex_result(
                        review_id,
                        candidate_labels=labels,
                        decision="no_clear_preference",
                        source_label=None,
                        rejected_label=None,
                        preference_reasons=[],
                        material_tradeoff=True,
                        notes="两条存在实质权衡。",
                    )
                )
            elif index < teacher_count:
                key_row = key["items"][index]
                source = key_row["labels"]["A"]["assistant"]
                results.append(
                    codex_result(
                        review_id,
                        candidate_labels=labels,
                        decision="teacher_edit",
                        improved_scores={
                            "generation_stability": 9,
                            "role_consistency": 9,
                            "dialogue_quality": 9,
                        },
                        improved_issues=[],
                        improved_assistant=source.replace("完整", "较完整"),
                        changes="修正表达自然度。",
                        notes="两条都不够好，最小修改 A。",
                    )
                )
            else:
                results.append(
                    codex_result(review_id, candidate_labels=labels)
                )
        write_json_atomic(run / "codex_review_packet.json", packet)
        write_json_atomic(run / "codex_review_key.json", key)
        write_json_atomic(
            run / "codex_review_results.json",
            {"schema_version": 3, "reviewer": "codex", "results": results},
        )
        summary = {
            **_summary_contract(),
            "run": {"id": "test"},
            "status": "awaiting_codex_review",
            "artifacts": {
                "candidates.jsonl": _artifact(candidates_path, run),
                "codex_review_packet.json": _artifact(
                    run / "codex_review_packet.json", run
                ),
                "codex_review_key.json": _artifact(
                    run / "codex_review_key.json", run
                ),
            },
        }
        write_json_atomic(run / "run_summary.json", summary)
        return run

    def test_finalizes_ms_swift_format_and_audits_discards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root, included=MIN_FINAL_PAIRS, teacher_count=2)
            train = root / "dpo_train_run2.jsonl"
            audit_path = root / "audit_run2.json"
            finalize_run(run_dir=run, train_output=train, audit_output=audit_path)
            rows = [json.loads(line) for line in train.read_text().splitlines()]
            self.assertEqual(len(rows), MIN_FINAL_PAIRS)
            self.assertEqual(set(rows[0]), {"messages", "rejected_response"})
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit["pairs"], MIN_FINAL_PAIRS)
            self.assertEqual(audit["teacher_involved_pairs"], 2)
            self.assertEqual(audit["filtered_before_codex"], 0)
            self.assertEqual(audit["discarded_pairs"], 10)
            self.assertEqual(len(audit["items"]), 40)
            summary = json.loads((run / "run_summary.json").read_text())
            self.assertEqual(summary["status"], "ready_for_dpo")
            self.assertEqual(
                summary["artifacts"]["codex_review_results.json"]["sha256"],
                hashlib.sha256(
                    (run / "codex_review_results.json").read_bytes()
                ).hexdigest(),
            )

    def test_rejects_too_few_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root, included=MIN_FINAL_PAIRS - 1)
            with self.assertRaisesRegex(DPODataError, "至少需要 30"):
                finalize_run(
                    run_dir=run,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )

    def test_audit_includes_prompts_filtered_before_codex(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root, unstable_prompts=1)
            audit_path = root / "audit.json"
            finalize_run(
                run_dir=run,
                train_output=root / "train.jsonl",
                audit_output=audit_path,
            )
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit["filtered_before_codex"], 1)
            self.assertEqual(audit["discarded_pairs"], 10)
            self.assertEqual(len(audit["items"]), 40)
            self.assertEqual(
                audit["items"][0]["filter_reason"],
                "fewer_than_two_stable_candidates",
            )
            self.assertIsNone(audit["items"][0]["review_id"])

    def test_rejects_teacher_pairs_above_one_fifth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root, included=30, teacher_count=7)
            with self.assertRaisesRegex(DPODataError, "超过五分之一"):
                finalize_run(
                    run_dir=run,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )

    def test_rejects_candidate_changed_after_packet_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root)
            candidates_path = run / "candidates.jsonl"
            rows = [json.loads(line) for line in candidates_path.read_text().splitlines()]
            rows[0]["assistant"] = "（点头）被篡改的回答。"
            rows[0]["raw_assistant"] = "（点头）被篡改的回答。"
            write_jsonl_atomic(candidates_path, rows)
            with self.assertRaisesRegex(DPODataError, "发生变化"):
                finalize_run(
                    run_dir=run,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )

    def test_rejects_packet_and_key_changed_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root)
            packet_path = run / "codex_review_packet.json"
            key_path = run / "codex_review_key.json"
            packet = json.loads(packet_path.read_text())
            key = json.loads(key_path.read_text())
            packet["items"][0]["answers"]["A"] = "（点头）被替换的回答。"
            key["items"][0]["labels"]["A"]["assistant"] = (
                "（点头）被替换的回答。"
            )
            write_json_atomic(packet_path, packet)
            write_json_atomic(key_path, key)
            with self.assertRaisesRegex(DPODataError, "发生变化"):
                finalize_run(
                    run_dir=run,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )

    def test_rejects_incomplete_codex_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root)
            path = run / "codex_review_results.json"
            value = json.loads(path.read_text())
            value["results"].pop()
            write_json_atomic(path, value)
            with self.assertRaisesRegex(DPODataError, "未完整对齐"):
                finalize_run(
                    run_dir=run,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )


if __name__ == "__main__":
    unittest.main()
