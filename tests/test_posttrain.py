"""Offline command/gate and artifact tests for the post-training workflow."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roleplay.artifacts import atomic_write_json, atomic_write_jsonl, read_json
from roleplay.posttrain import (
    FROZEN_HASHES,
    PosttrainError,
    _cloud_manifest,
    _require_reward_approval,
    _require_sft_ready,
    _trajectory_audit,
    build_parser,
    resolve_run_dir,
    run_doctor,
)
from roleplay.mlx_backend import make_temporary_mlx_dataset
from roleplay.posttrain_config import load_grpo_config


ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_json_and_jsonl_publish_without_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(root / "summary.json", {"状态": "完成"})
            atomic_write_jsonl(root / "metrics.jsonl", [{"loss": 1.0}])
            self.assertEqual(read_json(root / "summary.json")["状态"], "完成")
            self.assertEqual(json.loads((root / "metrics.jsonl").read_text())["loss"], 1.0)
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_runtime_mlx_dataset_is_ephemeral_and_preserves_messages(self):
        rows = [{"messages": [{"role": "assistant", "content": "回答"}]}]
        with tempfile.TemporaryDirectory() as directory:
            handle, path = make_temporary_mlx_dataset(rows, Path(directory))
            self.assertEqual(
                json.loads((path / "train.jsonl").read_text()), rows[0]
            )
            handle.cleanup()
            self.assertFalse(path.exists())

    def test_grpo_trajectory_audit_preserves_policy_ratio_inputs(self):
        candidate = {
            "candidate_id": "g1-c1", "assistant": "回答",
            "prompt_tokens": [1, 2], "completion_tokens": [3, 4],
            "old_logprobs": [-0.3, -0.4], "finish_reason": "stop",
            "peak_memory_gb": 1.0,
        }
        self.assertEqual(_trajectory_audit([candidate]), [{
            "candidate_id": "g1-c1", "prompt_tokens": [1, 2],
            "completion_tokens": [3, 4], "old_logprobs": [-0.3, -0.4],
            "finish_reason": "stop",
        }])


class CliTests(unittest.TestCase):
    def test_all_six_commands_are_registered(self):
        parser = build_parser()
        commands = ("doctor", "sft", "gate-sft", "reward-preview", "grpo", "evaluate")
        for command in commands:
            args = [command]
            if command not in {"doctor", "sft"}:
                args += ["--run-id", "run"]
            with self.subTest(command=command):
                self.assertEqual(parser.parse_args(args).command, command)

    def test_doctor_writes_a_complete_passing_report_with_fake_probe(self):
        probe = {
            "metal_available": True,
            "device": {"architecture": "Apple M4"},
            "snapshot_revision": "674aaa7240b91e8012fcad5d791b7dfe5ba90207",
            "records": 50,
            "maximum": 816,
            "limit": 1024,
            "all_within_limit": True,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "roleplay.posttrain.sys.version_info", (3, 13)
        ), patch(
            "roleplay.posttrain.platform.system", return_value="Darwin"
        ), patch(
            "roleplay.posttrain.platform.machine", return_value="arm64"
        ), patch(
            "roleplay.posttrain._version",
            side_effect=lambda name: "0.32.0" if name == "mlx" else "0.31.3",
        ), patch(
            "roleplay.posttrain._default_doctor_probe", return_value=probe
        ):
            report = run_doctor(Path(directory))
            self.assertTrue(report["passed"])
            self.assertTrue((Path(directory) / "doctor.json").is_file())

    def test_run_directory_is_scoped_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "roleplay.posttrain.OUTPUT_ROOT", Path(directory)
        ):
            run = resolve_run_dir("safe-run", create=True)
            self.assertTrue((run / "sft").is_dir())
            with self.assertRaises(PosttrainError):
                resolve_run_dir("../escape", create=True)


class StageGateTests(unittest.TestCase):
    def test_sft_gate_defaults_to_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "sft").mkdir()
            atomic_write_json(run / "sft/summary.json", {"ready_for_grpo": False})
            with self.assertRaisesRegex(PosttrainError, "拒绝"):
                _require_sft_ready(run)

    def test_reward_gate_requires_exact_five_reviewed_groups_and_reviewer(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "grpo").mkdir()
            expected = [f"g{i}" for i in range(5)]
            base = {
                "approved": True, "reviewer": "human",
                "expected_group_ids": expected, "reviewed_group_ids": expected,
            }
            atomic_write_json(run / "grpo/reward_review_results.json", base)
            _require_reward_approval(run)
            base["reviewed_group_ids"] = expected[:-1]
            atomic_write_json(run / "grpo/reward_review_results.json", base)
            with self.assertRaisesRegex(PosttrainError, "拒绝"):
                _require_reward_approval(run)

    def test_cloud_fallback_requires_base_rerun_and_no_spend(self):
        config = load_grpo_config(ROOT / "configs/morgana_v1_grpo_mlx.json")
        manifest = _cloud_manifest(config, Path("/run"), [{"error": "OOM"}])
        self.assertFalse(manifest["automatic_cloud_spend"])
        self.assertTrue(manifest["must_restart_from_base"])
        self.assertTrue(manifest["do_not_convert_mlx_adapter"])
        self.assertEqual(manifest["base"]["revision"], config["model_revision"])
        self.assertEqual(manifest["inputs"], FROZEN_HASHES)
