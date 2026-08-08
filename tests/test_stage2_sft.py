"""Tests for the AutoDL Stage 2 SFT orchestration helpers."""

import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roleplay.sft_eval import MANUAL_SCORE_DIMENSIONS, build_manual_review
from roleplay.stage2_sft import (
    DEFAULT_HF_ENDPOINT,
    DEFAULT_HF_HOME,
    DISABLED_ACCELERATION_PACKAGES,
    EXPECTED_ARCHIVE_FILES,
    PINNED_PACKAGES,
    Stage2SFTError,
    _record_failure,
    configure_huggingface_environment,
    create_exclusive_directory,
    ensure_clean_tracked_status,
    review_run,
    sha256_file,
    validate_archive_contract,
    validate_effective_training_args,
    validate_environment_snapshot,
    validate_file_manifest,
    validate_pinned_packages,
    validate_training_config,
    write_json_atomic,
)


def make_environment(**overrides):
    environment = {
        "platform": "Linux",
        "python_version": [3, 12],
        "pytorch": "2.8.0+cu128",
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
    def test_configures_autodl_huggingface_defaults(self):
        environment = {}

        configured = configure_huggingface_environment(environment)

        self.assertEqual(environment["HF_ENDPOINT"], DEFAULT_HF_ENDPOINT)
        self.assertEqual(environment["HF_HOME"], DEFAULT_HF_HOME)
        self.assertEqual(
            configured,
            {
                "endpoint": DEFAULT_HF_ENDPOINT,
                "cache_home": DEFAULT_HF_HOME,
            },
        )

    def test_preserves_explicit_huggingface_configuration(self):
        environment = {
            "HF_ENDPOINT": "https://huggingface.example.test",
            "HF_HOME": "/tmp/custom-huggingface",
        }

        configured = configure_huggingface_environment(environment)

        self.assertEqual(
            configured,
            {
                "endpoint": "https://huggingface.example.test",
                "cache_home": "/tmp/custom-huggingface",
            },
        )

    def test_accepts_frozen_autodl_environment(self):
        validate_environment_snapshot(make_environment())

    def test_rejects_each_environment_mismatch(self):
        cases = {
            "platform": {"platform": "Darwin"},
            "python": {"python_version": [3, 11]},
            "torch": {"pytorch": "2.10.0+cu128"},
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


class PackageValidationTests(unittest.TestCase):
    @staticmethod
    def package_versions(overrides=None, missing=()):
        versions = dict(PINNED_PACKAGES)
        versions.update(overrides or {})
        for name in missing:
            versions.pop(name, None)

        def version(name):
            if name in versions:
                return versions[name]
            raise importlib.metadata.PackageNotFoundError(name)

        return version

    @patch("roleplay.stage2_sft.importlib.metadata.version")
    def test_accepts_frozen_direct_dependencies(self, version):
        version.side_effect = self.package_versions()
        self.assertEqual(validate_pinned_packages(), PINNED_PACKAGES)

    @patch("roleplay.stage2_sft.importlib.metadata.version")
    def test_rejects_missing_direct_dependency(self, version):
        version.side_effect = self.package_versions(missing=("ms-swift",))
        with self.assertRaisesRegex(Stage2SFTError, "缺少固定依赖"):
            validate_pinned_packages()

    @patch("roleplay.stage2_sft.importlib.metadata.version")
    def test_rejects_mismatched_direct_dependency(self, version):
        version.side_effect = self.package_versions(
            overrides={"transformers": "0.0.0"}
        )
        with self.assertRaisesRegex(Stage2SFTError, "版本不匹配"):
            validate_pinned_packages()

    @patch("roleplay.stage2_sft.importlib.metadata.version")
    def test_rejects_disabled_acceleration_packages(self, version):
        disabled = DISABLED_ACCELERATION_PACKAGES[0]
        version.side_effect = self.package_versions(
            overrides={disabled: "0.5.1"}
        )
        with self.assertRaisesRegex(Stage2SFTError, "pip uninstall"):
            validate_pinned_packages()

    def test_autodl_requirements_reuse_base_torch_and_disable_kernels(self):
        requirements = (
            Path(__file__).resolve().parents[1]
            / "requirements"
            / "stage2_sft_autodl.txt"
        ).read_text(encoding="utf-8")
        forbidden = (
            "download.pytorch.org",
            "torch==",
            "torchvision==",
            "torchaudio==",
            "flash-linear-attention",
            "causal-conv1d",
            "ninja==",
        )
        for dependency in forbidden:
            with self.subTest(dependency=dependency):
                self.assertNotIn(dependency, requirements)


class TrainingPrecisionValidationTests(unittest.TestCase):
    @staticmethod
    def pure_fp32_config():
        return {
            "torch_dtype": "float32",
            "bnb_4bit_compute_dtype": "float32",
            "lora_dtype": "float32",
            "fp16": False,
            "bf16": False,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "num_train_epochs": 3,
        }

    def test_accepts_pure_fp32_config_and_plans_39_steps(self):
        self.assertEqual(
            validate_training_config(self.pure_fp32_config(), 50), 39
        )

    def test_rejects_each_mixed_precision_or_dtype_change(self):
        cases = {
            "fp16": True,
            "bf16": True,
            "torch_dtype": "float16",
            "bnb_4bit_compute_dtype": "float16",
            "lora_dtype": "float16",
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                config = self.pure_fp32_config()
                config[name] = value
                with self.assertRaisesRegex(
                    Stage2SFTError, "冻结训练配置不正确"
                ):
                    validate_training_config(config, 50)

    def test_accepts_effective_pure_fp32_args(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            args = self.pure_fp32_config()
            write_json_atomic(output_dir / "args.json", args)

            self.assertEqual(
                validate_effective_training_args(output_dir),
                {
                    "torch_dtype": "float32",
                    "bnb_4bit_compute_dtype": "float32",
                    "lora_dtype": "float32",
                    "fp16": False,
                    "bf16": False,
                },
            )

    def test_rejects_hidden_effective_fp16(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            args = self.pure_fp32_config()
            args["fp16"] = True
            write_json_atomic(output_dir / "args.json", args)

            with self.assertRaisesRegex(
                Stage2SFTError, "ms-swift 实际训练精度不正确"
            ):
                validate_effective_training_args(output_dir)


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
