"""Tests for the targeted post-GRPO DPO Prompt splits."""

import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from roleplay.post_grpo_dpo_data import (
    BASE_MODEL,
    BASE_REVISION,
    CANDIDATES_PER_PROMPT,
    GENERATION,
    GRPO_ADAPTER_HASHES,
    HOLDOUT_PATH,
    MANIFEST_PATH,
    SYSTEM_PROMPT_PATH,
    TARGET_ISSUES,
    TRAIN_PATH,
    PostGRPODPODataError,
    _artifact,
    _reward_v2,
    build_review_artifacts,
    candidate_id,
    candidate_seed,
    finalize_run,
    generate_candidates,
    load_jsonl,
    prepare_run,
    sha256_file,
    validate_grpo_adapter,
    validate_candidate_rows,
    validate_prompt_data,
    write_json_atomic,
    write_jsonl_atomic,
)


def candidate_record(prompt, index, answer=None, finish_reason="stop"):
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    answer = answer or (
        f"（耳朵轻轻一动）吾辈会认真回应这件事情，候选编号{index}。"
    )
    reward, eligible = _reward_v2(answer, finish_reason)
    return {
        "candidate_id": candidate_id(prompt["id"], index),
        "prompt_id": prompt["id"],
        "target_issue": prompt["target_issue"],
        "scenario": prompt["scenario"],
        "preference_criteria": prompt["preference_criteria"],
        "candidate_index": index,
        "seed": candidate_seed(prompt["id"], index),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt["user"]},
        ],
        "raw_assistant": answer,
        "assistant": answer,
        "finish_reason": finish_reason,
        "is_truncated": finish_reason in {"length", "max_tokens"},
        "source": "grpo_candidate",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter_sha256": GRPO_ADAPTER_HASHES[
            "adapter_model.safetensors"
        ],
        "generation": GENERATION,
        "reward_v2": reward,
        "eligible_for_review": eligible,
    }


class FrozenPromptDataTests(unittest.TestCase):
    def test_repository_data_is_balanced_isolated_and_frozen(self):
        result = validate_prompt_data()

        self.assertEqual(result["training_prompts"], 30)
        self.assertEqual(result["holdout_prompts"], 9)
        self.assertEqual(result["training_per_issue"], 10)
        self.assertEqual(result["holdout_per_issue"], 3)
        self.assertEqual(result["target_issues"], list(TARGET_ISSUES))

    def _copy_frozen_data(self, root: Path) -> tuple[Path, Path]:
        data_dir = root / "data/runs/morgana-v2"
        data_dir.mkdir(parents=True)
        train = data_dir / TRAIN_PATH.name
        holdout = data_dir / HOLDOUT_PATH.name
        manifest = data_dir / MANIFEST_PATH.name
        shutil.copyfile(TRAIN_PATH, train)
        shutil.copyfile(HOLDOUT_PATH, holdout)
        shutil.copyfile(MANIFEST_PATH, manifest)
        return train, holdout

    def test_rejects_prompt_content_drift_against_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, _ = self._copy_frozen_data(root)
            rows = train.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["user"] += "（修改）"
            rows[0] = json.dumps(first, ensure_ascii=False)
            train.write_text("\n".join(rows) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                PostGRPODPODataError, "manifest training_prompts"
            ):
                validate_prompt_data(root)

    def test_rejects_holdout_overlap_with_existing_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, holdout = self._copy_frozen_data(root)
            first = json.loads(
                holdout.read_text(encoding="utf-8").splitlines()[0]
            )
            dev = root / "data/runs/morgana-v2/dev.jsonl"
            dev.write_text(
                json.dumps(
                    {"id": "dev_0001", "user": first["user"]},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                PostGRPODPODataError, "与已有 split 重复"
            ):
                validate_prompt_data(root)


class CandidateGenerationTests(unittest.TestCase):
    def test_generation_and_adapter_contract_are_frozen(self):
        self.assertEqual(
            GENERATION,
            {
                "max_tokens": 512,
                "temperature": 0.6,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.45,
                "repetition_context_size": 128,
                "enable_thinking": False,
            },
        )
        self.assertEqual(
            GRPO_ADAPTER_HASHES["adapter_model.safetensors"],
            "89ca4fa213ea16eeee002b088ea46b3012b6311b02f09b2b27b0937ab6dcd30f",
        )

    def test_adapter_validation_rejects_hash_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            hashes = {}
            for name in GRPO_ADAPTER_HASHES:
                path = adapter / name
                path.write_text(f"frozen {name}", encoding="utf-8")
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch(
                "roleplay.post_grpo_dpo_data.GRPO_ADAPTER_HASHES", hashes
            ):
                self.assertEqual(validate_grpo_adapter(adapter), hashes)
                (adapter / "adapter_config.json").write_text(
                    "drift", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    PostGRPODPODataError, "adapter 哈希不匹配"
                ):
                    validate_grpo_adapter(adapter)

    def test_seeds_are_unique_and_six_per_prompt(self):
        seeds = {
            candidate_seed(f"post_dpo_{prompt:04d}", index)
            for prompt in range(1, 31)
            for index in range(1, CANDIDATES_PER_PROMPT + 1)
        }

        self.assertEqual(len(seeds), 180)
        self.assertEqual(candidate_seed("post_dpo_0001", 1), 20260812)
        self.assertEqual(candidate_seed("post_dpo_0002", 1), 20260912)

    def test_loads_model_once_and_generates_six_candidates(self):
        prompt = load_jsonl(TRAIN_PATH)[0]
        fake_mlx = types.ModuleType("mlx_lm")
        loads = []

        def fake_load(*args, **kwargs):
            loads.append((args, kwargs))
            return object(), object()

        fake_mlx.load = fake_load
        with patch.dict(sys.modules, {"mlx_lm": fake_mlx}), patch(
            "roleplay.post_grpo_dpo_data._verify_loaded_adapter"
        ), patch(
            "roleplay.post_grpo_dpo_data._generate_one",
            side_effect=lambda *_args: (
                "（耳朵一动）吾辈已经听清楚了，会认真回应这件事情。",
                "stop",
            ),
        ):
            rows = generate_candidates(
                prompts=[prompt],
                base_model_path=Path("base"),
                mlx_adapter_path=Path("adapter"),
                system_prompt="system",
            )

        self.assertEqual(len(loads), 1)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["eligible_for_review"] for row in rows))

    def test_validates_180_frozen_records(self):
        prompts = load_jsonl(TRAIN_PATH)
        rows = [
            candidate_record(prompt, index)
            for prompt in prompts
            for index in range(1, CANDIDATES_PER_PROMPT + 1)
        ]

        validate_candidate_rows(
            rows,
            prompts,
            SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        )

    def test_hard_invalid_candidate_is_excluded_from_review(self):
        prompt = load_jsonl(TRAIN_PATH)[0]
        valid = [candidate_record(prompt, index) for index in range(1, 6)]
        invalid = candidate_record(
            prompt,
            6,
            answer="（没有闭合的动作",
            finish_reason="length",
        )

        packet, key, _, unresolved = build_review_artifacts(
            [prompt], [*valid, invalid]
        )

        self.assertFalse(unresolved)
        self.assertEqual(len(packet["items"][0]["answers"]), 5)
        source_ids = {
            value["candidate_id"]
            for value in key["items"][0]["labels"].values()
        }
        self.assertNotIn(invalid["candidate_id"], source_ids)

    def test_failed_prepare_is_inspectable_and_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            with patch(
                "roleplay.post_grpo_dpo_data.validate_prompt_data"
            ), patch(
                "roleplay.post_grpo_dpo_data.validate_grpo_adapter"
            ), patch(
                "roleplay.post_grpo_dpo_data.convert_peft_adapter_to_mlx",
                return_value=root / "mlx-adapter",
            ), patch(
                "roleplay.post_grpo_dpo_data.generate_candidates",
                side_effect=RuntimeError("fake generator failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "fake generator failure"
                ):
                    prepare_run(
                        run_id="failed-run",
                        output_root=root / "runs",
                        base_model_path=base,
                    )
            run_dir = root / "runs/failed-run"
            summary = json.loads(
                (run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "sampling_failed")
            self.assertIn("fake generator failure", summary["error"])
            frozen_hash = sha256_file(run_dir / "run_summary.json")
            with patch(
                "roleplay.post_grpo_dpo_data.validate_prompt_data"
            ), patch(
                "roleplay.post_grpo_dpo_data.validate_grpo_adapter"
            ):
                with self.assertRaisesRegex(
                    PostGRPODPODataError, "运行目录已存在"
                ):
                    prepare_run(
                        run_id="failed-run",
                        output_root=root / "runs",
                        base_model_path=base,
                    )
            self.assertEqual(
                sha256_file(run_dir / "run_summary.json"), frozen_hash
            )


class PairFinalizationTests(unittest.TestCase):
    def _make_run(self, root: Path, *, teacher_pairs: int = 0) -> Path:
        run_dir = root / "run"
        run_dir.mkdir()
        prompts = load_jsonl(TRAIN_PATH)
        candidates = [
            candidate_record(prompt, index)
            for prompt in prompts
            for index in range(1, CANDIDATES_PER_PROMPT + 1)
        ]
        packet, key, _, unresolved = build_review_artifacts(
            prompts, candidates
        )
        self.assertFalse(unresolved)
        results = []
        key_by_id = {row["review_id"]: row for row in key["items"]}
        for item_index, item in enumerate(packet["items"]):
            labels = key_by_id[item["review_id"]]["labels"]
            label_by_candidate = {
                value["candidate_id"]: label
                for label, value in labels.items()
            }
            chosen_label = label_by_candidate[f"{item['prompt_id']}-c1"]
            rejected_label = label_by_candidate[f"{item['prompt_id']}-c2"]
            use_teacher = item_index < teacher_pairs
            candidate_reviews = {}
            for label, value in labels.items():
                source_id = value["candidate_id"]
                if source_id.endswith("-c1") and not use_teacher:
                    status, quality = "pass", 9
                elif source_id.endswith("-c2"):
                    status, quality = "fail", 8
                else:
                    status, quality = "ambiguous", 6
                candidate_reviews[label] = {
                    "target_status": status,
                    "non_target_quality": quality,
                    "off_target_issues": [],
                    "evidence": f"测试裁决 {source_id}",
                }
            if use_teacher:
                rejected = labels[rejected_label]["assistant"]
                teacher = rejected.replace("候选编号2", "修正候选编号2")
                result = {
                    "review_id": item["review_id"],
                    "decision": "teacher_chosen",
                    "chosen_label": None,
                    "rejected_label": rejected_label,
                    "candidate_reviews": candidate_reviews,
                    "teacher_assistant": teacher,
                    "teacher_target_status": "pass",
                    "teacher_non_target_quality": 9,
                    "teacher_off_target_issues": [],
                    "teacher_evidence": "Teacher 已修正唯一目标问题。",
                    "teacher_changes": "只修正目标语义。",
                    "notes": "没有原生 chosen，使用定向 Teacher chosen。",
                }
            else:
                result = {
                    "review_id": item["review_id"],
                    "decision": "native_pair",
                    "chosen_label": chosen_label,
                    "rejected_label": rejected_label,
                    "candidate_reviews": candidate_reviews,
                    "teacher_assistant": None,
                    "teacher_target_status": None,
                    "teacher_non_target_quality": None,
                    "teacher_off_target_issues": None,
                    "teacher_evidence": "",
                    "teacher_changes": "",
                    "notes": "原生 chosen 与最佳 hard negative。",
                }
            results.append(result)
        paths = {
            "candidates.jsonl": run_dir / "candidates.jsonl",
            "review_packet.json": run_dir / "review_packet.json",
            "review_key.json": run_dir / "review_key.json",
        }
        write_jsonl_atomic(paths["candidates.jsonl"], candidates)
        write_json_atomic(paths["review_packet.json"], packet)
        write_json_atomic(paths["review_key.json"], key)
        write_json_atomic(
            run_dir / "review_results.json",
            {"schema_version": 1, "reviewer": "codex", "results": results},
        )
        write_json_atomic(
            run_dir / "run_summary.json",
            {
                "status": "awaiting_codex_review",
                "artifacts": {
                    name: _artifact(path, run_dir)
                    for name, path in paths.items()
                },
            },
        )
        return run_dir

    def test_exports_one_balanced_pair_per_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._make_run(root)
            train = root / "train.jsonl"
            audit = root / "audit.json"

            status, train_path, _ = finalize_run(
                run_dir=run_dir,
                train_output=train,
                audit_output=audit,
            )

            self.assertEqual(status, "ready_for_dpo")
            self.assertEqual(train_path, train)
            self.assertEqual(len(load_jsonl(train)), 30)
            report = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(
                report["pairs_by_issue"],
                {issue: 10 for issue in TARGET_ISSUES},
            )
            self.assertEqual(report["teacher_pairs"], 0)
            holdout_users = {
                row["user"] for row in load_jsonl(HOLDOUT_PATH)
            }
            exported_users = {
                row["messages"][1]["content"] for row in load_jsonl(train)
            }
            self.assertTrue(exported_users.isdisjoint(holdout_users))

    def test_rejects_non_best_or_artificial_rejected_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._make_run(root)
            results_path = run_dir / "review_results.json"
            submitted = json.loads(results_path.read_text(encoding="utf-8"))
            first = submitted["results"][0]
            original_rejected = first["rejected_label"]
            first["rejected_label"] = next(
                label
                for label in first["candidate_reviews"]
                if label not in {first["chosen_label"], original_rejected}
            )
            write_json_atomic(results_path, submitted)

            with self.assertRaisesRegex(
                PostGRPODPODataError, "最佳 hard negative"
            ):
                finalize_run(
                    run_dir=run_dir,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )

    def test_rejects_teacher_edit_outside_similarity_and_length_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._make_run(root, teacher_pairs=1)
            results_path = run_dir / "review_results.json"
            submitted = json.loads(results_path.read_text(encoding="utf-8"))
            submitted["results"][0]["teacher_assistant"] = "完全重写。"
            write_json_atomic(results_path, submitted)

            with self.assertRaisesRegex(
                PostGRPODPODataError, "长度比"
            ):
                finalize_run(
                    run_dir=run_dir,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )

    def test_rejects_teacher_pairs_above_one_quarter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._make_run(root, teacher_pairs=8)

            with self.assertRaisesRegex(
                PostGRPODPODataError, "超过25%"
            ):
                finalize_run(
                    run_dir=run_dir,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )


if __name__ == "__main__":
    unittest.main()
