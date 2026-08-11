"""Tests for the Stage 4 rule GRPO workflow."""

import json
import tarfile
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from roleplay.grpo_rule_reward import score_completion
from roleplay.stage2_sft import sha256_file, write_json_atomic
from roleplay.stage4_grpo import (
    ADAPTER_FILES,
    CONFIG_SHA256,
    EXPECTED_ARCHIVE_FILES,
    Stage4GRPOError,
    _load_yaml,
    build_parser,
    create_release_bundle,
    evaluate_rule_dev,
    evaluate_stage4_manual_review,
    extract_release_bundle,
    prepare_training_rows,
    publish_run,
    review_run,
    run_stage4,
    validate_archive_contract,
    validate_prompt_isolation,
    validate_reward_samples,
    validate_training_config,
)


def prompt_rows():
    policies = ["encouraged"] * 10 + ["optional"] * 8 + ["forbidden"] * 2
    return [
        {
            "id": f"rule_grpo_{index:04d}",
            "scenario": "daily",
            "user": f"规则问题 {index}",
            "target_rules": ["brevity", "signature", "action"],
            "reward_policy": {"action": policy},
        }
        for index, policy in enumerate(policies, 1)
    ]


def dev_rows(answer: str, *, finish_reason: str = "stop"):
    return [
        {
            "seed": 20260807,
            "id": f"dev_{index:04d}",
            "scenario": "daily",
            "target_goals": ["generation_stability"],
            "user": f"Dev 问题 {index}",
            "assistant": answer,
            "raw_assistant": answer,
            "finish_reason": finish_reason,
            "attempts": 1,
        }
        for index in range(1, 11)
    ]


def reward_components(total_reward: float = 2.0):
    components = score_completion(
        "（认真点头）吾辈已经听清楚了，接下来会把这件事情处理妥当。",
        "encouraged",
    ).as_log_dict()
    components["total_reward"] = total_reward
    return components


def review_artifacts(new_severe=None):
    items = [
        {"review_id": f"dev-review-{index:02d}"}
        for index in range(1, 11)
    ]
    answers = [
        {
            "review_id": item["review_id"],
            "id": f"dev_{index:04d}",
            "sft_label": "A",
            "grpo_label": "B",
        }
        for index, item in enumerate(items, 1)
    ]
    results = []
    for index, item in enumerate(items, 1):
        results.append(
            {
                "review_id": item["review_id"],
                "winner": "B",
                "clearly_worse": None,
                "scores": {
                    "A": {
                        "generation_stability": 8,
                        "role_consistency": 6,
                        "dialogue_quality": 6,
                    },
                    "B": {
                        "generation_stability": 8,
                        "role_consistency": 6,
                        "dialogue_quality": 6,
                    },
                },
                "severe_issues": {
                    "A": [],
                    "B": list(new_severe or []) if index == 1 else [],
                },
            }
        )
    return {"items": items}, {"answers": answers}, {"results": results}


def make_successful_run(run_dir: Path) -> None:
    (run_dir / "adapter").mkdir(parents=True)
    for name in ADAPTER_FILES:
        (run_dir / "adapter" / name).write_bytes(name.encode())
    for name in EXPECTED_ARCHIVE_FILES - {
        "run_summary.json",
        *(f"adapter/{item}" for item in ADAPTER_FILES),
    }:
        (run_dir / name).write_text(name, encoding="utf-8")
    write_json_atomic(
        run_dir / "run_summary.json",
        {
            "schema_version": 1,
            "stage": "stage4_rule_grpo",
            "status": "awaiting_manual_review",
            "run": {"id": "20260811-1200", "commit": "abc123"},
            "automatic_review": {"passed": True},
            "manual_review": {},
        },
    )


class FrozenInputTests(unittest.TestCase):
    def test_prepares_twenty_policy_validated_records(self):
        records = prepare_training_rows(prompt_rows(), "system")

        self.assertEqual(len(records), 20)
        self.assertEqual(records[0]["id"], "rule_grpo_0001")
        self.assertEqual(records[0]["messages"][-1]["content"], "规则问题 1")

    def test_rejects_bad_policy_distribution(self):
        rows = prompt_rows()
        rows[0]["reward_policy"]["action"] = "optional"

        with self.assertRaisesRegex(Stage4GRPOError, "分布"):
            prepare_training_rows(rows, "system")

    def test_repository_prompts_are_isolated(self):
        root = Path(__file__).resolve().parents[1]
        rows = [
            json.loads(line)
            for line in (
                root / "data/runs/morgana-v2/rule_grpo_train.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]

        validate_prompt_isolation(root, rows)

    def test_accepts_repository_stage4_config(self):
        path = Path("configs/morgana_v2_stage4_grpo.yaml")
        config = _load_yaml(path)

        validate_training_config(config)
        self.assertEqual(sha256_file(path), CONFIG_SHA256)

    def test_rejects_weaker_kl_constraint(self):
        config = _load_yaml(Path("configs/morgana_v2_stage4_grpo.yaml"))
        config["beta"] = 0.04

        with self.assertRaisesRegex(Stage4GRPOError, "beta"):
            validate_training_config(config)

    def test_parser_exposes_exact_four_commands(self):
        parser = build_parser()

        self.assertEqual(parser.parse_args(["run"]).command, "run")
        self.assertEqual(
            parser.parse_args(["publish", "--run-dir", "run"]).command,
            "publish",
        )
        self.assertEqual(
            parser.parse_args(["download", "--tag", "tag"]).command,
            "download",
        )
        self.assertEqual(
            parser.parse_args(["review", "--run-dir", "run"]).command,
            "review",
        )

    def test_stage4_dependencies_include_grpo_kernel_only(self):
        requirements = Path("requirements/stage4_grpo_autodl.txt").read_text()

        self.assertIn("-r stage2_sft_autodl.txt", requirements)
        self.assertIn("flash-linear-attention==0.4.2", requirements)
        self.assertIn("msgspec==0.21.1", requirements)
        self.assertNotIn("causal-conv1d", requirements)


class RewardLogAndDevGateTests(unittest.TestCase):
    def test_requires_component_complete_rewards(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reward_samples.jsonl"
            with path.open("w", encoding="utf-8") as output:
                for prompt in prompt_rows():
                    for _ in range(4):
                        output.write(
                            json.dumps(
                                {
                                    "status": "ok",
                                    "record_id": prompt["id"],
                                    "components": reward_components(),
                                    "total_reward": 2.0,
                                }
                            )
                            + "\n"
                        )

            result = validate_reward_samples(
                path, {row["id"] for row in prompt_rows()}
            )

            self.assertEqual(result["rows"], 80)
            self.assertEqual(result["mean"], 2.0)

    def test_rejects_incomplete_reward_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reward_samples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "record_id": "rule_grpo_0001",
                        "components": {"total_reward": 2.0},
                        "total_reward": 2.0,
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(Stage4GRPOError, "子分"):
                validate_reward_samples(path, {"rule_grpo_0001"})

    def test_rule_dev_passes_clear_improvement(self):
        sft = dev_rows("我知道了。")
        grpo = dev_rows(
            "（轻轻甩了甩尾巴）吾辈已经听清楚了，莲，先做好最重要的一件事，再回来休息。"
        )

        gate, rows = evaluate_rule_dev(
            sft, grpo, [f"dev_{index:04d}" for index in range(1, 11)]
        )

        self.assertTrue(gate["passed"])
        self.assertGreaterEqual(gate["mean_rule_reward_delta"], 0.3)
        self.assertEqual(gate["grpo_rule_wins"], 10)
        self.assertEqual(len(rows), 10)

    def test_rule_dev_fails_without_six_wins(self):
        answer = "（轻轻甩了甩尾巴）吾辈已经听清楚了，接下来会认真处理这件事情。"

        gate, _ = evaluate_rule_dev(
            dev_rows(answer),
            dev_rows(answer),
            [f"dev_{index:04d}" for index in range(1, 11)],
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["rule_reward_wins_at_least_6"])


class ManualReviewTests(unittest.TestCase):
    def test_only_new_severe_issue_blocks_manual_gate(self):
        packet, key, clean = review_artifacts()
        clean_gate = evaluate_stage4_manual_review(packet, key, clean)
        _, _, regressed = review_artifacts(["role_break"])
        failed_gate = evaluate_stage4_manual_review(packet, key, regressed)

        self.assertTrue(clean_gate["passed"])
        self.assertEqual(clean_gate["grpo_wins"], 10)
        self.assertFalse(failed_gate["passed"])
        self.assertEqual(
            failed_gate["new_grpo_severe_issues"][0]["codes"],
            ["role_break"],
        )

    def test_rejects_unknown_severe_code(self):
        packet, key, submitted = review_artifacts(["unknown"])

        with self.assertRaisesRegex(Stage4GRPOError, "未知"):
            evaluate_stage4_manual_review(packet, key, submitted)

    def test_review_sets_ready_for_eval_without_subjective_win_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            make_successful_run(run_dir)
            packet, key, submitted = review_artifacts()
            for result in submitted["results"]:
                result["winner"] = "A"
                result["clearly_worse"] = "B"
            write_json_atomic(run_dir / "manual_review_packet.json", packet)
            write_json_atomic(run_dir / "manual_review_answer_key.json", key)
            write_json_atomic(run_dir / "manual_review_results.json", submitted)

            summary = review_run(run_dir)

            self.assertTrue(summary["ready_for_eval"])
            self.assertEqual(summary["status"], "ready_for_eval")
            self.assertEqual(summary["manual_review"]["gate"]["grpo_wins"], 0)


class RunWorkflowTests(unittest.TestCase):
    def test_run_is_offline_and_atomically_archives(self):
        config = _load_yaml(Path("configs/morgana_v2_stage4_grpo.yaml"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            prompts_path = repo / "data/runs/morgana-v2/rule_grpo_train.jsonl"
            dev_path = repo / "data/runs/morgana-v2/dev.jsonl"
            system_path = repo / "data/runs/morgana-v2/system_prompt.txt"
            config_path = repo / "configs/morgana_v2_stage4_grpo.yaml"
            adapter_path = repo / "output/morgana-v2/stage2-sft/final/adapter"
            prompts_path.parent.mkdir(parents=True)
            config_path.parent.mkdir(parents=True)
            adapter_path.mkdir(parents=True)
            prompts_path.write_text(
                "\n".join(json.dumps(row) for row in prompt_rows()) + "\n"
            )
            source_dev = [
                {
                    "id": f"dev_{index:04d}",
                    "scenario": "daily",
                    "target_goals": ["generation_stability"],
                    "user": f"Dev 问题 {index}",
                }
                for index in range(1, 11)
            ]
            dev_path.write_text(
                "\n".join(json.dumps(row) for row in source_dev) + "\n"
            )
            system_path.write_text("system")
            config_path.write_text("frozen")
            for name in ADAPTER_FILES:
                (adapter_path / name).write_text(name)
            commands = []

            def fake_run_logged(command, log_path, repo_dir, environment, expected_steps):
                del repo_dir, environment
                commands.append((command, expected_steps))
                import yaml

                effective = yaml.safe_load(Path(command[2]).read_text())
                output = Path(effective["output_dir"])
                trained = output / "adapter"
                trained.mkdir(parents=True)
                for name in ADAPTER_FILES:
                    (trained / name).write_text(name)
                with (output / "reward_samples.jsonl").open("w") as rewards:
                    for prompt in prompt_rows():
                        for _ in range(4):
                            rewards.write(
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "record_id": prompt["id"],
                                        "components": reward_components(),
                                        "total_reward": 2.0,
                                    }
                                )
                                + "\n"
                            )
                (output / "args.json").write_text("{}")
                log_path.write_text("trained")
                return 1.5

            def fake_review(repo_dir, sft_adapter, grpo_adapter, output):
                del repo_dir, sft_adapter, grpo_adapter
                paths = {}
                for name, rows in (
                    ("sft_dev_outputs.jsonl", dev_rows("我知道了。")),
                    (
                        "grpo_dev_outputs.jsonl",
                        dev_rows(
                            "（轻轻甩了甩尾巴）吾辈已经听清楚了，莲，"
                            "先做好最重要的一件事，再回来休息。"
                        ),
                    ),
                ):
                    paths[name] = output / name
                    paths[name].write_text(
                        "\n".join(json.dumps(row) for row in rows) + "\n"
                    )
                packet, key, submitted = review_artifacts()
                submitted["results"] = []
                for name, value in (
                    ("manual_review_packet.json", packet),
                    ("manual_review_answer_key.json", key),
                    ("manual_review_results.json", submitted),
                ):
                    paths[name] = output / name
                    write_json_atomic(paths[name], value)
                return {"paths": paths, "automatic_gate": {"passed": True}}

            with (
                patch("roleplay.stage4_grpo.repository_root", return_value=repo),
                patch("roleplay.stage4_grpo.generate_run_id", return_value="run-1"),
                patch("roleplay.stage4_grpo.configure_huggingface_environment", return_value={}),
                patch("roleplay.stage4_grpo.capture_environment", return_value=({}, object())),
                patch("roleplay.stage4_grpo.validate_pinned_packages", return_value={}),
                patch("roleplay.stage4_grpo.git_context", return_value={"commit": "abc", "branch": "main"}),
                patch("roleplay.stage4_grpo.validate_frozen_file", return_value="digest"),
                patch("roleplay.stage4_grpo.validate_sft_adapter", return_value={"adapter_model.safetensors": "digest"}),
                patch("roleplay.stage4_grpo.validate_prompt_isolation"),
                patch("roleplay.stage4_grpo._load_yaml", return_value=deepcopy(config)),
                patch("roleplay.stage4_grpo.run_logged", side_effect=fake_run_logged),
                patch("roleplay.stage4_grpo.find_final_adapter", side_effect=lambda output: output / "adapter"),
                patch("roleplay.stage4_grpo.inspect_adapter_change", return_value={"changed_tensors": 1}),
                patch("roleplay.stage4_grpo.generate_dev_review_artifacts", side_effect=fake_review),
                patch.dict("os.environ", {}, clear=True),
            ):
                run_dir = run_stage4(root / "runs")

            self.assertEqual(run_dir, (root / "runs/run-1").resolve())
            validate_archive_contract(run_dir)
            self.assertEqual(commands[0][0][:2], ["swift", "rlhf"])
            self.assertEqual(commands[0][1], 20)
            self.assertFalse((root / "runs/.work/run-1").exists())
            summary = json.loads((run_dir / "run_summary.json").read_text())
            self.assertEqual(summary["status"], "awaiting_manual_review")
            self.assertTrue(summary["automatic_review"]["passed"])


class PublicationTests(unittest.TestCase):
    def test_publish_download_bundle_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_successful_run(run_dir)

            bundle, manifest, tag = create_release_bundle(run_dir, root / "dist")
            self.assertEqual(tag, "morgana-v2-stage4-grpo-20260811-1200")
            with tarfile.open(bundle, "r:gz") as archive:
                self.assertEqual(
                    {member.name for member in archive.getmembers()},
                    {f"20260811-1200/{name}" for name in EXPECTED_ARCHIVE_FILES},
                )
            destination = extract_release_bundle(bundle, manifest, root / "downloads")
            validate_archive_contract(destination)
            with self.assertRaisesRegex(Stage4GRPOError, "拒绝覆盖"):
                extract_release_bundle(bundle, manifest, root / "downloads")

    def test_publish_uses_stage4_release_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_successful_run(run_dir)

            with patch("subprocess.run") as run:
                _, _, tag = publish_run(run_dir, root / "dist", "owner/repo")

            self.assertEqual(tag, "morgana-v2-stage4-grpo-20260811-1200")
            self.assertIn(tag, run.call_args.args[0])
