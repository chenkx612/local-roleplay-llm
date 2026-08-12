"""Tests for the frozen post-GRPO DPO training workflow."""

import json
import tarfile
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from roleplay.post_grpo_dpo import (
    ADAPTER_FILES,
    EXPANSION_TRAIN_PAIR_COUNT,
    EXPANSION_TRAIN_RELATIVE_PATH,
    EXPANSION_TRAIN_SHA256,
    EXPECTED_ARCHIVE_FILES,
    ORIGINAL_TRAIN_PAIR_COUNT,
    ORIGINAL_TRAIN_RELATIVE_PATH,
    ORIGINAL_TRAIN_SHA256,
    PostGRPODPOError,
    _load_yaml,
    build_parser,
    create_release_bundle,
    evaluate_post_dpo_review,
    extract_release_bundle,
    main,
    publish_run,
    review_run,
    run_post_grpo_dpo,
    validate_archive_contract,
    validate_grpo_adapter,
    validate_holdout_rows,
    validate_training_config,
    validate_training_rows,
)
from roleplay.stage2_sft import sha256_file, write_json_atomic


def dpo_rows(count=20, start=1):
    return [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"问题 {index}"},
                {"role": "assistant", "content": f"优选 {index}"},
            ],
            "rejected_response": f"拒选 {index}",
        }
        for index in range(start, start + count)
    ]


def holdout_rows():
    issues = [
        "fabricated_background",
        "perspective_shift",
        "emotion_response",
    ]
    return [
        {
            "id": f"post_dpo_dev_{index:04d}",
            "target_issue": issues[(index - 1) // 3],
            "scenario": "test",
            "user": f"holdout {index}",
            "preference_criteria": "只检查一个目标。",
        }
        for index in range(1, 10)
    ]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_archive(run_dir: Path, status="awaiting_manual_review") -> None:
    (run_dir / "adapter").mkdir(parents=True)
    for name in ADAPTER_FILES:
        (run_dir / "adapter" / name).write_text(name, encoding="utf-8")
    for name in EXPECTED_ARCHIVE_FILES - {
        "run_summary.json",
        *(f"adapter/{item}" for item in ADAPTER_FILES),
    }:
        (run_dir / name).write_text(name, encoding="utf-8")
    write_json_atomic(
        run_dir / "run_summary.json",
        {
            "schema_version": 1,
            "stage": "post_grpo_dpo",
            "status": status,
            "run": {"id": "20260813-0100", "commit": "abc123"},
        },
    )


def review_artifacts(run_dir: Path, *, submitted=True, automatic=True) -> None:
    items = [
        {"review_id": f"dev-review-{index:02d}"}
        for index in range(1, 10)
    ]
    packet = {"items": items}
    answer_key = {
        "answers": [
            {
                "review_id": item["review_id"],
                "id": f"post_dpo_dev_{index:04d}",
                "grpo_label": "A",
                "dpo_label": "B",
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
        if submitted
        else []
    }
    write_json_atomic(run_dir / "manual_review_packet.json", packet)
    write_json_atomic(run_dir / "manual_review_answer_key.json", answer_key)
    write_json_atomic(run_dir / "manual_review_results.json", results)
    write_json_atomic(
        run_dir / "run_summary.json",
        {
            "status": "awaiting_manual_review",
            "technically_valid": True,
            "automatic_review": {"passed": automatic},
            "manual_review": {"status": "awaiting_manual_review"},
            "artifacts": {},
        },
    )


class FrozenInputTests(unittest.TestCase):
    def test_repository_contract_is_frozen(self):
        original = validate_training_rows(
            ORIGINAL_TRAIN_RELATIVE_PATH,
            ORIGINAL_TRAIN_SHA256,
            ORIGINAL_TRAIN_PAIR_COUNT,
        )
        expansion = validate_training_rows(
            EXPANSION_TRAIN_RELATIVE_PATH,
            EXPANSION_TRAIN_SHA256,
            EXPANSION_TRAIN_PAIR_COUNT,
        )
        holdout = validate_holdout_rows(
            Path("data/runs/morgana-v2/post_grpo_dpo_holdout.jsonl")
        )
        adapter = validate_grpo_adapter(
            Path("output/morgana-v2/stage4-grpo/20260812-2144/adapter")
        )
        steps = validate_training_config(
            _load_yaml(Path("configs/morgana_v2_post_grpo_dpo.yaml"))
        )

        self.assertEqual(len(original), 20)
        self.assertEqual(len(expansion), 41)
        self.assertEqual(len(holdout), 9)
        self.assertEqual(len(adapter), 3)
        self.assertEqual(steps, 31)

    def test_rejects_data_config_holdout_and_adapter_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            write_jsonl(train, dpo_rows(19))
            with self.assertRaisesRegex(PostGRPODPOError, "必须为 20"):
                validate_training_rows(train, sha256_file(train), 20)

            holdout = root / "holdout.jsonl"
            rows = holdout_rows()
            rows[0]["id"] = "wrong"
            write_jsonl(holdout, rows)
            with patch(
                "roleplay.post_grpo_dpo.HOLDOUT_SHA256", sha256_file(holdout)
            ):
                with self.assertRaisesRegex(PostGRPODPOError, "id 不正确"):
                    validate_holdout_rows(holdout)

            config = _load_yaml(Path("configs/morgana_v2_post_grpo_dpo.yaml"))
            config["learning_rate"] = 1.0e-5
            with self.assertRaisesRegex(PostGRPODPOError, "learning_rate"):
                validate_training_config(config)

            adapter = root / "adapter"
            adapter.mkdir()
            for name in ADAPTER_FILES:
                (adapter / name).write_text("wrong", encoding="utf-8")
            with self.assertRaisesRegex(PostGRPODPOError, "哈希"):
                validate_grpo_adapter(adapter)


class RunWorkflowTests(unittest.TestCase):
    def _run(self, root: Path, *, fail=False):
        repo = root / "repo"
        train = repo / "data/runs/morgana-v2/post_grpo_dpo_train.jsonl"
        expansion_train = (
            repo
            / "data/runs/morgana-v2/post_grpo_dpo_train_expansion.jsonl"
        )
        config_path = repo / "configs/morgana_v2_post_grpo_dpo.yaml"
        write_jsonl(train, dpo_rows())
        write_jsonl(expansion_train, dpo_rows(41, start=21))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            Path("configs/morgana_v2_post_grpo_dpo.yaml").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        adapter = (
            repo / "output/morgana-v2/stage4-grpo/20260812-2144/adapter"
        )
        adapter.mkdir(parents=True)
        commands = []

        def fake_run_logged(command, log, repo_dir, environment, expected_steps):
            del repo_dir, environment
            commands.append((command, expected_steps))
            if fail:
                log.write_text("failed", encoding="utf-8")
                raise PostGRPODPOError("train broke")
            import yaml

            effective = yaml.safe_load(Path(command[2]).read_text(encoding="utf-8"))
            output = Path(effective["output_dir"])
            trained = output / "adapter"
            trained.mkdir(parents=True)
            for name in ADAPTER_FILES:
                (trained / name).write_text(name, encoding="utf-8")
            (output / "args.json").write_text(
                json.dumps(
                    {
                        "torch_dtype": "float32",
                        "bnb_4bit_compute_dtype": "float32",
                        "lora_dtype": "float32",
                        "fp16": False,
                        "bf16": False,
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                output / "logging.jsonl",
                [
                    {"loss": 0.5, "grad_norm": 1.0}
                    for _ in range(expected_steps)
                ],
            )
            log.write_text("trained", encoding="utf-8")
            return 1.0

        def fake_review(repo_dir, grpo, dpo, output):
            del repo_dir, grpo, dpo
            paths = {}
            for name in (
                "grpo_holdout_outputs.jsonl",
                "dpo_holdout_outputs.jsonl",
                "manual_review_packet.json",
                "manual_review_answer_key.json",
                "manual_review_results.json",
            ):
                paths[name] = output / name
                paths[name].write_text("{}", encoding="utf-8")
            return {
                "paths": paths,
                "automatic_gate": {
                    "passed": True,
                    "checks": {"generation_stability": True},
                },
            }

        class FakeHfApi:
            def model_info(self, model, revision):
                del model, revision
                return types.SimpleNamespace(
                    sha="965dcc54bc9c0591873df0e9869c056a54d323d1"
                )

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, model, revision):
                del model, revision
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                del messages, kwargs
                return [1, 2, 3]

        with (
            patch("roleplay.post_grpo_dpo.repository_root", return_value=repo),
            patch("roleplay.post_grpo_dpo.generate_run_id", return_value="run-1"),
            patch(
                "roleplay.post_grpo_dpo.configure_huggingface_environment",
                return_value={},
            ),
            patch(
                "roleplay.post_grpo_dpo.capture_environment",
                return_value=({}, object()),
            ),
            patch(
                "roleplay.post_grpo_dpo.validate_pinned_packages",
                return_value={},
            ),
            patch(
                "roleplay.post_grpo_dpo.git_context",
                return_value={"commit": "abc123", "branch": "main"},
            ),
            patch(
                "roleplay.post_grpo_dpo.validate_training_rows",
                side_effect=[dpo_rows(), dpo_rows(41, start=21)],
            ),
            patch(
                "roleplay.post_grpo_dpo.validate_holdout_rows",
                return_value=holdout_rows(),
            ),
            patch(
                "roleplay.post_grpo_dpo.validate_frozen_file",
                return_value="digest",
            ),
            patch(
                "roleplay.post_grpo_dpo.validate_grpo_adapter",
                return_value={"adapter_model.safetensors": "digest"},
            ),
            patch(
                "roleplay.post_grpo_dpo.run_logged",
                side_effect=fake_run_logged,
            ),
            patch(
                "roleplay.post_grpo_dpo.find_final_adapter",
                side_effect=lambda output: output / "adapter",
            ),
            patch(
                "roleplay.post_grpo_dpo.inspect_adapter_change",
                return_value={"changed_tensors": 1},
            ),
            patch(
                "roleplay.post_grpo_dpo.generate_holdout_review_artifacts",
                side_effect=fake_review,
            ),
            patch.dict(
                "sys.modules",
                {
                    "huggingface_hub": types.SimpleNamespace(HfApi=FakeHfApi),
                    "transformers": types.SimpleNamespace(
                        AutoProcessor=FakeProcessor
                    ),
                },
            ),
        ):
            if fail:
                with self.assertRaisesRegex(PostGRPODPOError, "train broke"):
                    run_post_grpo_dpo(root / "runs")
                return commands
            return run_post_grpo_dpo(root / "runs"), commands

    def test_run_archives_valid_merged_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, commands = self._run(root)
            validate_archive_contract(run_dir)
            self.assertEqual(commands[0][0][:2], ["swift", "rlhf"])
            self.assertEqual(commands[0][1], 31)
            self.assertEqual(
                len(
                    [
                        line
                        for line in (run_dir / "train.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line
                    ]
                ),
                61,
            )
            summary = json.loads(
                (run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "awaiting_manual_review")
            self.assertEqual(summary["training"]["optimizer_steps"], 31)
            self.assertEqual(summary["training"]["training_pairs"], 61)
            self.assertEqual(
                summary["training"]["source_pair_counts"],
                {"original": 20, "expansion": 41},
            )

    def test_failure_retains_inspectable_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(root, fail=True)
            self.assertTrue((root / "runs/.work/run-1/train.log").is_file())
            summary = json.loads(
                (root / "runs/run-1/run_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "training_failed")
            self.assertFalse(summary["ready_to_publish"])


class PublicationTests(unittest.TestCase):
    def test_bundle_is_reusable_and_extract_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_archive(run_dir)
            first = create_release_bundle(run_dir, root / "dist")
            self.assertEqual(first, create_release_bundle(run_dir, root / "dist"))
            bundle, manifest, tag = first
            self.assertEqual(tag, "morgana-v2-post-grpo-dpo-20260813-0100")
            with tarfile.open(bundle, "r:gz") as archive:
                self.assertEqual(
                    {item.name for item in archive.getmembers()},
                    {
                        f"20260813-0100/{name}"
                        for name in EXPECTED_ARCHIVE_FILES
                    },
                )
            destination = extract_release_bundle(
                bundle, manifest, root / "downloads"
            )
            validate_archive_contract(destination)
            with self.assertRaisesRegex(PostGRPODPOError, "拒绝覆盖"):
                extract_release_bundle(bundle, manifest, root / "downloads")

    def test_publish_uses_post_grpo_release_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_archive(run_dir)
            with patch("subprocess.run") as command:
                publish_run(run_dir, root / "dist", "owner/repo")
            args = command.call_args.args[0]
            self.assertEqual(args[:3], ["gh", "release", "create"])
            self.assertIn("morgana-v2-post-grpo-dpo-20260813-0100", args)
            self.assertIn("owner/repo", args)


class ReviewTests(unittest.TestCase):
    def test_review_gate_maps_grpo_and_dpo(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            review_artifacts(run_dir)
            packet = json.loads(
                (run_dir / "manual_review_packet.json").read_text()
            )
            key = json.loads(
                (run_dir / "manual_review_answer_key.json").read_text()
            )
            results = json.loads(
                (run_dir / "manual_review_results.json").read_text()
            )
            gate = evaluate_post_dpo_review(packet, key, results)
            self.assertEqual(gate["dpo_wins"], 9)
            self.assertIn("grpo", gate["mean_scores"])

            with redirect_stdout(StringIO()):
                summary = review_run(run_dir)
            self.assertEqual(summary["status"], "ready_for_final_eval")
            self.assertTrue(summary["ready_for_final_eval"])
            with self.assertRaisesRegex(PostGRPODPOError, "重复"):
                review_run(run_dir)

    def test_automatic_failure_blocks_final_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            review_artifacts(run_dir, automatic=False)
            with redirect_stdout(StringIO()):
                summary = review_run(run_dir)
            self.assertEqual(summary["status"], "dpo_failed")
            self.assertFalse(summary["ready_for_final_eval"])


class CliTests(unittest.TestCase):
    def test_parser_exposes_exact_four_commands(self):
        parser = build_parser()
        for command, args in (
            ("run", []),
            ("publish", ["--run-dir", "run"]),
            ("download", ["--tag", "tag"]),
            ("review", ["--run-dir", "run"]),
        ):
            self.assertEqual(parser.parse_args([command, *args]).command, command)

    def test_commands_dispatch_and_fail_without_traceback(self):
        with patch(
            "roleplay.post_grpo_dpo.run_post_grpo_dpo", return_value=Path("run")
        ) as run, redirect_stdout(StringIO()):
            self.assertEqual(main(["run"]), 0)
        run.assert_called_once_with(None)

        with patch("roleplay.post_grpo_dpo.publish_run") as publish:
            publish.return_value = (Path("bundle"), Path("manifest"), "tag")
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(["publish", "--run-dir", "run"]), 0
                )

        with patch(
            "roleplay.post_grpo_dpo.download_release", return_value=Path("run")
        ), redirect_stdout(StringIO()):
            self.assertEqual(main(["download", "--tag", "tag"]), 0)

        with patch("roleplay.post_grpo_dpo.review_run") as review:
            self.assertEqual(main(["review", "--run-dir", "run"]), 0)
        review.assert_called_once_with(Path("run"))

        with patch(
            "roleplay.post_grpo_dpo.run_post_grpo_dpo",
            side_effect=PostGRPODPOError("broken"),
        ):
            error = StringIO()
            with redirect_stderr(error):
                self.assertEqual(main(["run"]), 1)
            self.assertIn("broken", error.getvalue())
            self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
