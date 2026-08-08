"""Tests for the AutoDL Stage 2 SFT orchestration helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from roleplay.sft_eval import MANUAL_SCORE_DIMENSIONS, build_manual_review
from roleplay.stage2_sft import (
    EXPECTED_ARCHIVE_FILES,
    Stage2SFTError,
    _record_failure,
    create_exclusive_directory,
    ensure_clean_tracked_status,
    review_run,
    sha256_file,
    validate_archive_contract,
    validate_environment_snapshot,
    validate_file_manifest,
    write_json_atomic,
)


def make_environment(**overrides):
    environment = {
        "platform": "Linux",
        "python_version": [3, 12],
        "pytorch": "2.10.0+cu128",
        "cuda": "12.8",
        "cuda_available": True,
        "gpu_count": 1,
        "gpu_memory_gib": 23.7,
        "cxx11_abi": True,
    }
    environment.update(overrides)
    return environment


def make_review_files(run_dir: Path, *, passing: bool, empty: bool = False):
    ids = [f"dev_{index:04d}" for index in range(1, 11)]
    base = []
    sft = []
    for record_id in ids:
        common = {
            "seed": 20260807,
            "id": record_id,
            "scenario": "daily",
            "target_goals": ["role_consistency"],
            "user": f"问题 {record_id}",
            "finish_reason": "stop",
            "attempts": 1,
        }
        base.append(
            common
            | {"assistant": "Base", "raw_assistant": "<think></think>Base"}
        )
        sft.append(
            common
            | {"assistant": "SFT", "raw_assistant": "<think></think>SFT"}
        )
    packet, answer_key = build_manual_review(base, sft, ids)
    mapping = {item["review_id"]: item for item in answer_key["answers"]}
    results = []
    if not empty:
        for index, item in enumerate(packet["items"]):
            labels = mapping[item["review_id"]]
            winner = labels["sft_label"] if passing and index < 6 else "tie"
            results.append(
                {
                    "review_id": item["review_id"],
                    "winner": winner,
                    "clearly_worse": None,
                    "scores": {
                        label: {
                            dimension: 8
                            for dimension in MANUAL_SCORE_DIMENSIONS
                        }
                        for label in ("A", "B")
                    },
                    "severe_issues": {"A": [], "B": []},
                }
            )
    write_json_atomic(run_dir / "manual_review_packet.json", packet)
    write_json_atomic(run_dir / "manual_review_answer_key.json", answer_key)
    write_json_atomic(
        run_dir / "manual_review_results.json",
        {
            "schema_version": 2,
            "expected_review_ids": [
                item["review_id"] for item in packet["items"]
            ],
            "results": results,
        },
    )
    summary = {
        "schema_version": 2,
        "status": "awaiting_manual_review",
        "technically_valid": True,
        "core_behavior_gate": {"passed": True},
        "manual_review": {"status": "awaiting_manual_review"},
        "ready_for_grpo": False,
        "artifacts": {"manual_review_results.json": {}},
    }
    write_json_atomic(run_dir / "run_summary.json", summary)


class EnvironmentValidationTests(unittest.TestCase):
    def test_accepts_frozen_autodl_environment(self):
        validate_environment_snapshot(make_environment())

    def test_rejects_each_environment_mismatch(self):
        cases = {
            "platform": {"platform": "Darwin"},
            "python": {"python_version": [3, 11]},
            "torch": {"pytorch": "2.8.0+cu128"},
            "cuda": {"cuda": "13.0"},
            "availability": {"cuda_available": False},
            "gpu_count": {"gpu_count": 2},
            "memory": {"gpu_memory_gib": 16.0},
            "abi": {"cxx11_abi": False},
        }
        for name, override in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(Stage2SFTError):
                    validate_environment_snapshot(make_environment(**override))

    def test_rejects_tracked_changes_but_accepts_clean_status(self):
        ensure_clean_tracked_status("")
        with self.assertRaisesRegex(Stage2SFTError, "tracked"):
            ensure_clean_tracked_status(" M RUNLOG.md\n")


class FileContractTests(unittest.TestCase):
    def test_run_directory_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            self.assertEqual(create_exclusive_directory(path), path)
            with self.assertRaisesRegex(Stage2SFTError, "已存在"):
                create_exclusive_directory(path)

    def test_manifest_checks_hash_and_record_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "input.jsonl"
            path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
            manifest = {
                "input.jsonl": {
                    "records": 2,
                    "sha256": sha256_file(path),
                }
            }
            validated, loaded = validate_file_manifest(root, manifest)
            self.assertEqual(validated["input.jsonl"]["records"], 2)
            self.assertEqual(len(loaded["input.jsonl"]), 2)
            manifest["input.jsonl"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(Stage2SFTError, "哈希"):
                validate_file_manifest(root, manifest)

    def test_archive_contract_rejects_missing_and_unexpected_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in EXPECTED_ARCHIVE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test", encoding="utf-8")
            validate_archive_contract(root)
            (root / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(Stage2SFTError, "unexpected"):
                validate_archive_contract(root)

    def test_failure_summary_is_preserved_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "run_summary.json"
            work_root = root / ".work" / "run"
            error = ValueError("boom")
            _record_failure(
                {"status": "initialized"},
                summary_path,
                "training_failed",
                error,
                work_root,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "training_failed")
            self.assertEqual(summary["error"]["type"], "ValueError")
            self.assertEqual(summary["retained_work_dir"], str(work_root))
            self.assertFalse(list(root.glob(".*.tmp")))


class ReviewCommandTests(unittest.TestCase):
    def test_empty_review_keeps_summary_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            make_review_files(run_dir, passing=False, empty=True)
            summary = review_run(run_dir)
            self.assertEqual(summary["status"], "awaiting_manual_review")
            self.assertNotIn("manual_reviewed_at_utc", summary)

    def test_passing_review_updates_summary_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            make_review_files(run_dir, passing=True)
            summary = review_run(run_dir)
            self.assertEqual(summary["status"], "ready_for_grpo")
            self.assertTrue(summary["ready_for_grpo"])
            self.assertEqual(summary["manual_review"]["status"], "passed")
            with self.assertRaisesRegex(Stage2SFTError, "重复"):
                review_run(run_dir)

    def test_failing_review_records_manual_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            make_review_files(run_dir, passing=False)
            summary = review_run(run_dir)
            self.assertEqual(summary["status"], "manual_failed")
            self.assertFalse(summary["ready_for_grpo"])
            self.assertEqual(summary["manual_review"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
