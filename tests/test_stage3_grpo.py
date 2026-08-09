"""Tests for the minimal AutoDL Stage 3 GRPO workflow."""

import json
import tarfile
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from roleplay.stage2_sft import write_json_atomic
from roleplay.stage3_grpo import (
    ADAPTER_FILES,
    EXPECTED_ARCHIVE_FILES,
    Stage3GRPOError,
    _load_yaml,
    create_release_bundle,
    extract_release_bundle,
    main,
    prepare_training_rows,
    publish_run,
    review_run,
    run_stage3,
    validate_archive_contract,
    validate_reward_samples,
    validate_training_config,
)


def prompt_rows():
    return [
        {
            "id": f"grpo_{index:04d}",
            "scenario": "daily",
            "target_goals": ["dialogue_quality"],
            "user": f"问题 {index}",
        }
        for index in range(1, 21)
    ]


def make_successful_run(run_dir: Path) -> None:
    (run_dir / "adapter").mkdir(parents=True)
    for name in ADAPTER_FILES:
        (run_dir / "adapter" / name).write_bytes(name.encode())
    for name in (
        "training_config.yaml",
        "train.jsonl",
        "train.log",
        "reward_samples.jsonl",
        "sft_dev_outputs.jsonl",
        "grpo_dev_outputs.jsonl",
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
    ):
        (run_dir / name).write_text(name, encoding="utf-8")
    write_json_atomic(
        run_dir / "run_summary.json",
        {
            "schema_version": 1,
            "status": "awaiting_manual_review",
            "run": {"id": "20260809-1800", "commit": "abc123"},
        },
    )


class FrozenInputTests(unittest.TestCase):
    def test_prepares_twenty_message_records(self):
        records = prepare_training_rows(prompt_rows(), "system")

        self.assertEqual(len(records), 20)
        self.assertEqual(
            records[0],
            {
                "id": "grpo_0001",
                "prompt_id": "grpo_0001",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "问题 1"},
                ],
            },
        )

    def test_rejects_duplicate_or_incomplete_prompts(self):
        duplicate = prompt_rows()
        duplicate[1]["id"] = duplicate[0]["id"]
        with self.assertRaises(Stage3GRPOError):
            prepare_training_rows(duplicate, "system")

        with self.assertRaisesRegex(Stage3GRPOError, "20"):
            prepare_training_rows(prompt_rows()[:-1], "system")

    def test_accepts_repository_grpo_config(self):
        config = _load_yaml(Path("configs/morgana_v2_grpo.yaml"))

        validate_training_config(config)


class RewardValidationTests(unittest.TestCase):
    def test_requires_four_successful_rewards_per_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reward_samples.jsonl"
            with path.open("w", encoding="utf-8") as output:
                for prompt_index, prompt in enumerate(prompt_rows()):
                    for index in range(4):
                        output.write(
                            json.dumps(
                                {
                                    "prompt_id": f"prompt_{prompt_index}",
                                    "record_id": prompt["id"],
                                    "status": "ok",
                                    "total_reward": float(index),
                                }
                            )
                            + "\n"
                        )

            result = validate_reward_samples(
                path, {row["id"] for row in prompt_rows()}
            )

            self.assertEqual(result["rows"], 80)
            self.assertEqual(result["minimum"], 0.0)
            self.assertEqual(result["maximum"], 3.0)
            self.assertEqual(result["record_counts"]["grpo_0001"], 4)

    def test_rejects_failed_reward(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reward_samples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt_id": "prompt_0",
                        "record_id": "grpo_0001",
                        "status": "error",
                        "total_reward": None,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Stage3GRPOError, "失败"):
                validate_reward_samples(path, {"grpo_0001"})


class RunWorkflowTests(unittest.TestCase):
    def test_run_builds_and_atomically_publishes_valid_artifacts(self):
        config = _load_yaml(Path("configs/morgana_v2_grpo.yaml"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            prompts_path = repo / "data/runs/morgana-v2/rl_train.jsonl"
            system_path = repo / "data/runs/morgana-v2/system_prompt.txt"
            config_path = repo / "configs/morgana_v2_grpo.yaml"
            adapter_path = (
                repo / "output/morgana-v2/stage2-sft/final/adapter"
            )
            prompts_path.parent.mkdir(parents=True)
            config_path.parent.mkdir(parents=True)
            adapter_path.mkdir(parents=True)
            prompts_path.write_text(
                "\n".join(json.dumps(row) for row in prompt_rows()) + "\n",
                encoding="utf-8",
            )
            system_path.write_text("system", encoding="utf-8")
            config_path.write_text("frozen", encoding="utf-8")
            for name in ADAPTER_FILES:
                (adapter_path / name).write_text(name, encoding="utf-8")

            commands = []

            def fake_run_logged(
                command,
                log_path,
                repo_dir,
                environment,
                expected_steps,
            ):
                del repo_dir, environment
                commands.append((command, expected_steps))
                import yaml

                effective = yaml.safe_load(
                    Path(command[2]).read_text(encoding="utf-8")
                )
                output = Path(effective["output_dir"])
                trained_adapter = output / "adapter"
                trained_adapter.mkdir(parents=True)
                for name in ADAPTER_FILES:
                    (trained_adapter / name).write_text(name, encoding="utf-8")
                with (output / "reward_samples.jsonl").open(
                    "w", encoding="utf-8"
                ) as rewards:
                    for prompt_index, prompt in enumerate(prompt_rows()):
                        for _ in range(4):
                            rewards.write(
                                json.dumps(
                                    {
                                        "prompt_id": f"prompt_{prompt_index}",
                                        "record_id": prompt["id"],
                                        "status": "ok",
                                        "total_reward": 4.0,
                                    }
                                )
                                + "\n"
                            )
                (output / "args.json").write_text("{}", encoding="utf-8")
                log_path.write_text("trained", encoding="utf-8")
                return 1.25

            def fake_generate_review(
                repo_dir, sft_adapter, grpo_adapter, output
            ):
                del repo_dir, sft_adapter, grpo_adapter
                names = (
                    "sft_dev_outputs.jsonl",
                    "grpo_dev_outputs.jsonl",
                    "manual_review_packet.json",
                    "manual_review_answer_key.json",
                    "manual_review_results.json",
                )
                paths = {}
                for name in names:
                    paths[name] = output / name
                    paths[name].write_text("{}", encoding="utf-8")
                return {
                    "paths": paths,
                    "automatic_gate": {
                        "passed": True,
                        "checks": {"complete": True},
                        "base": {"overall": {"records": 10}},
                        "sft": {"overall": {"records": 10}},
                    },
                }

            with (
                patch("roleplay.stage3_grpo.repository_root", return_value=repo),
                patch("roleplay.stage3_grpo.generate_run_id", return_value="run-1"),
                patch(
                    "roleplay.stage3_grpo.configure_huggingface_environment",
                    return_value={},
                ),
                patch(
                    "roleplay.stage3_grpo.capture_environment",
                    return_value=({}, object()),
                ),
                patch(
                    "roleplay.stage3_grpo.validate_pinned_packages",
                    return_value={},
                ),
                patch(
                    "roleplay.stage3_grpo.git_context",
                    return_value={"commit": "abc123", "branch": "main"},
                ),
                patch(
                    "roleplay.stage3_grpo.validate_frozen_file",
                    return_value="digest",
                ),
                patch(
                    "roleplay.stage3_grpo.validate_sft_adapter",
                    return_value={"adapter_model.safetensors": "digest"},
                ),
                patch(
                    "roleplay.stage3_grpo._load_yaml",
                    return_value=deepcopy(config),
                ),
                patch(
                    "roleplay.stage3_grpo.run_logged",
                    side_effect=fake_run_logged,
                ),
                patch(
                    "roleplay.stage3_grpo.find_final_adapter",
                    side_effect=lambda output: output / "adapter",
                ),
                patch(
                    "roleplay.stage3_grpo.inspect_adapter_change",
                    return_value={"changed_tensors": 1},
                ),
                patch(
                    "roleplay.stage3_grpo.generate_dev_review_artifacts",
                    side_effect=fake_generate_review,
                ),
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}),
            ):
                run_dir = run_stage3(root / "runs")

            self.assertEqual(run_dir, (root / "runs/run-1").resolve())
            validate_archive_contract(run_dir)
            self.assertEqual(commands[0][0][:2], ["swift", "rlhf"])
            self.assertEqual(commands[0][1], 20)
            self.assertFalse((root / "runs/.work/run-1").exists())
            summary = json.loads(
                (run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "awaiting_manual_review")
            self.assertEqual(summary["reward"]["rows"], 80)


class PublicationTests(unittest.TestCase):
    def test_creates_exact_release_bundle_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            output_dir = root / "dist"
            make_successful_run(run_dir)

            validate_archive_contract(run_dir)
            first = create_release_bundle(run_dir, output_dir)
            second = create_release_bundle(run_dir, output_dir)

            self.assertEqual(first, second)
            bundle, manifest, tag = first
            self.assertEqual(tag, "morgana-v2-stage3-grpo-20260809-1800")
            self.assertTrue(manifest.is_file())
            with tarfile.open(bundle, "r:gz") as archive:
                names = {member.name for member in archive.getmembers()}
            self.assertEqual(
                names,
                {f"20260809-1800/{name}" for name in EXPECTED_ARCHIVE_FILES},
            )

    def test_publish_uploads_bundle_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_successful_run(run_dir)

            with patch("subprocess.run") as run:
                bundle, manifest, tag = publish_run(
                    run_dir, root / "dist", "owner/repo"
                )

            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["gh", "release", "create"])
            self.assertIn(str(bundle), command)
            self.assertIn(str(manifest), command)
            self.assertIn(tag, command)
            self.assertIn("owner/repo", command)


class DownloadTests(unittest.TestCase):
    def test_extracts_verified_bundle_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_successful_run(run_dir)
            bundle, manifest, _ = create_release_bundle(
                run_dir, root / "dist"
            )

            destination = extract_release_bundle(
                bundle, manifest, root / "downloads"
            )

            self.assertEqual(destination.name, "20260809-1800")
            validate_archive_contract(destination)
            with self.assertRaisesRegex(Stage3GRPOError, "拒绝覆盖"):
                extract_release_bundle(bundle, manifest, root / "downloads")


class ReviewTests(unittest.TestCase):
    def test_records_passing_manual_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            items = [
                {"review_id": f"dev-review-{index:02d}"}
                for index in range(1, 11)
            ]
            packet = {"items": items}
            answer_key = {
                "answers": [
                    {
                        "review_id": item["review_id"],
                        "id": f"dev_{index:04d}",
                        "sft_label": "A",
                        "grpo_label": "B",
                    }
                    for index, item in enumerate(items, 1)
                ]
            }
            scores = {
                "A": {
                    "generation_stability": 8,
                    "role_consistency": 8,
                    "dialogue_quality": 8,
                },
                "B": {
                    "generation_stability": 9,
                    "role_consistency": 9,
                    "dialogue_quality": 9,
                },
            }
            results = {
                "results": [
                    {
                        "review_id": item["review_id"],
                        "winner": "B",
                        "clearly_worse": None,
                        "scores": scores,
                        "severe_issues": {"A": [], "B": []},
                    }
                    for item in items
                ]
            }
            write_json_atomic(run_dir / "manual_review_packet.json", packet)
            write_json_atomic(
                run_dir / "manual_review_answer_key.json", answer_key
            )
            write_json_atomic(
                run_dir / "manual_review_results.json", results
            )
            write_json_atomic(
                run_dir / "run_summary.json",
                {
                    "status": "awaiting_manual_review",
                    "automatic_review": {"passed": True},
                    "manual_review": {"status": "awaiting_manual_review"},
                },
            )

            with redirect_stdout(StringIO()):
                summary = review_run(run_dir)

            self.assertEqual(summary["status"], "ready_for_eval")
            self.assertTrue(summary["ready_for_eval"])
            self.assertEqual(
                summary["manual_review"]["gate"]["grpo_wins"], 10
            )
            self.assertNotIn(
                "sft_wins", summary["manual_review"]["gate"]
            )


class CliTests(unittest.TestCase):
    def test_run_command_reports_publish_readiness(self):
        with patch(
            "roleplay.stage3_grpo.run_stage3", return_value=Path("run")
        ) as run:
            output = StringIO()
            with redirect_stdout(output):
                result = main(["run"])

        self.assertEqual(result, 0)
        run.assert_called_once_with(None)
        self.assertIn("ready_to_publish=True", output.getvalue())

    def test_download_and_review_commands_dispatch(self):
        with patch(
            "roleplay.stage3_grpo.download_release",
            return_value=Path("output/run"),
        ) as download:
            with redirect_stdout(StringIO()):
                result = main(["download", "--tag", "release-1"])
        self.assertEqual(result, 0)
        download.assert_called_once()

        with patch("roleplay.stage3_grpo.review_run") as review:
            result = main(["review", "--run-dir", "output/run"])
        self.assertEqual(result, 0)
        review.assert_called_once_with(Path("output/run"))

    def test_cli_reports_run_failure_without_traceback(self):
        with patch(
            "roleplay.stage3_grpo.run_stage3",
            side_effect=Stage3GRPOError("broken"),
        ):
            error = StringIO()
            with redirect_stderr(error):
                result = main(["run"])

        self.assertEqual(result, 1)
        self.assertIn("broken", error.getvalue())


if __name__ == "__main__":
    unittest.main()
