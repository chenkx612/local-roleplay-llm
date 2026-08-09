"""Tests for the frozen AutoDL Stage 3 DPO workflow."""

import json
import tarfile
import tempfile
import types
import unittest
from collections import UserDict
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from roleplay.stage2_sft import write_json_atomic
from roleplay.stage3_dpo import (
    ADAPTER_FILES,
    EXPECTED_ARCHIVE_FILES,
    Stage3DPOError,
    _load_yaml,
    _token_length,
    create_release_bundle,
    extract_release_bundle,
    main,
    publish_run,
    review_run,
    run_stage3_dpo,
    sha256_file,
    validate_archive_contract,
    validate_sft_adapter,
    validate_training_config,
    validate_training_rows,
)


def dpo_rows(count=30):
    return [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"问题 {index}"},
                {"role": "assistant", "content": f"优选 {index}"},
            ],
            "rejected_response": f"拒选 {index}",
        }
        for index in range(1, count + 1)
    ]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_successful_run(run_dir: Path, status="awaiting_manual_review") -> None:
    (run_dir / "adapter").mkdir(parents=True)
    for name in ADAPTER_FILES:
        (run_dir / "adapter" / name).write_bytes(name.encode())
    for name in (
        "training_config.yaml",
        "train.jsonl",
        "train.log",
        "sft_dev_outputs.jsonl",
        "dpo_dev_outputs.jsonl",
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
    ):
        (run_dir / name).write_text(name, encoding="utf-8")
    write_json_atomic(
        run_dir / "run_summary.json",
        {
            "schema_version": 1,
            "status": status,
            "run": {"id": "20260809-2300", "commit": "abc123"},
        },
    )


def review_artifacts(run_dir: Path, *, submitted=True, automatic=True) -> None:
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
    def test_accepts_repository_data_and_config(self):
        rows = validate_training_rows(
            Path("data/runs/morgana-v2/dpo_train_run2.jsonl")
        )
        steps = validate_training_config(
            _load_yaml(Path("configs/morgana_v2_dpo.yaml"))
        )

        self.assertEqual(len(rows), 30)
        self.assertEqual(steps, 24)

    def test_rejects_hash_count_schema_roles_and_equal_answers(self):
        cases = []
        cases.append(dpo_rows(29))
        wrong_fields = dpo_rows()
        wrong_fields[0]["extra"] = True
        cases.append(wrong_fields)
        wrong_roles = dpo_rows()
        wrong_roles[0]["messages"][1]["role"] = "assistant"
        cases.append(wrong_roles)
        equal_answers = dpo_rows()
        equal_answers[0]["rejected_response"] = equal_answers[0]["messages"][-1][
            "content"
        ]
        cases.append(equal_answers)

        with tempfile.TemporaryDirectory() as temporary:
            for index, rows in enumerate(cases):
                path = Path(temporary) / f"case-{index}.jsonl"
                write_jsonl(path, rows)
                with patch(
                    "roleplay.stage3_dpo.TRAIN_SHA256", sha256_file(path)
                ):
                    with self.assertRaises(Stage3DPOError):
                        validate_training_rows(path)

            path = Path(temporary) / "tampered.jsonl"
            write_jsonl(path, dpo_rows())
            with self.assertRaisesRegex(Stage3DPOError, "哈希"):
                validate_training_rows(path)

    def test_rejects_training_config_drift(self):
        config = _load_yaml(Path("configs/morgana_v2_dpo.yaml"))
        for name, value in (
            ("learning_rate", 5.0e-6),
            ("beta", 0.2),
            ("padding_free", True),
            ("ref_adapters", []),
        ):
            changed = deepcopy(config)
            changed[name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(Stage3DPOError, name):
                    validate_training_config(changed)

    def test_rejects_missing_or_unaccepted_sft_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter"
            with self.assertRaisesRegex(Stage3DPOError, "缺少"):
                validate_sft_adapter(adapter)
            adapter.mkdir()
            for name in ADAPTER_FILES:
                (adapter / name).write_text("wrong", encoding="utf-8")
            with self.assertRaisesRegex(Stage3DPOError, "哈希"):
                validate_sft_adapter(adapter)

    def test_dpo_reuses_stage2_dependencies_without_optional_kernels(self):
        requirements = Path("requirements/stage3_dpo_autodl.txt").read_text(
            encoding="utf-8"
        )

        self.assertEqual(requirements.strip(), "-r stage2_sft_autodl.txt")
        self.assertNotIn("flash-linear-attention", requirements)
        self.assertNotIn("causal-conv1d", requirements)

    def test_token_length_reads_input_ids_from_mapping(self):
        class MappingProcessor:
            def apply_chat_template(self, messages, **kwargs):
                del messages, kwargs
                return UserDict(
                    {"input_ids": list(range(807)), "attention_mask": []}
                )

        self.assertEqual(_token_length(MappingProcessor(), []), 807)


class RunWorkflowTests(unittest.TestCase):
    def _run(self, root: Path, *, fail_training=False):
        repo = root / "repo"
        train_path = repo / "data/runs/morgana-v2/dpo_train_run2.jsonl"
        dev_path = repo / "data/runs/morgana-v2/dev.jsonl"
        system_path = repo / "data/runs/morgana-v2/system_prompt.txt"
        config_path = repo / "configs/morgana_v2_dpo.yaml"
        adapter_path = repo / "output/morgana-v2/stage2-sft/final/adapter"
        write_jsonl(train_path, dpo_rows())
        dev_path.write_text("{}\n", encoding="utf-8")
        system_path.write_text("system", encoding="utf-8")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("frozen", encoding="utf-8")
        adapter_path.mkdir(parents=True)
        for name in ADAPTER_FILES:
            (adapter_path / name).write_text(name, encoding="utf-8")

        config = _load_yaml(Path("configs/morgana_v2_dpo.yaml"))
        commands = []

        def fake_run_logged(command, log_path, repo_dir, environment, expected_steps):
            del repo_dir, environment
            commands.append((command, expected_steps))
            if fail_training:
                log_path.write_text("failed", encoding="utf-8")
                raise Stage3DPOError("train broke")
            import yaml

            effective = yaml.safe_load(Path(command[2]).read_text(encoding="utf-8"))
            output = Path(effective["output_dir"])
            trained_adapter = output / "adapter"
            trained_adapter.mkdir(parents=True)
            for name in ADAPTER_FILES:
                (trained_adapter / name).write_text(name, encoding="utf-8")
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
            log_path.write_text("trained", encoding="utf-8")
            return 1.25

        def fake_review(repo_dir, sft_adapter, dpo_adapter, output):
            del repo_dir, sft_adapter, dpo_adapter
            names = (
                "sft_dev_outputs.jsonl",
                "dpo_dev_outputs.jsonl",
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

        fake_hub = types.SimpleNamespace(HfApi=FakeHfApi)
        fake_transformers = types.SimpleNamespace(AutoProcessor=FakeProcessor)
        with (
            patch("roleplay.stage3_dpo.repository_root", return_value=repo),
            patch("roleplay.stage3_dpo.generate_run_id", return_value="run-1"),
            patch(
                "roleplay.stage3_dpo.configure_huggingface_environment",
                return_value={},
            ),
            patch(
                "roleplay.stage3_dpo.capture_environment",
                return_value=({}, object()),
            ),
            patch(
                "roleplay.stage3_dpo.validate_pinned_packages", return_value={}
            ),
            patch(
                "roleplay.stage3_dpo.git_context",
                return_value={"commit": "abc123", "branch": "main"},
            ),
            patch(
                "roleplay.stage3_dpo.validate_training_rows",
                return_value=dpo_rows(),
            ),
            patch(
                "roleplay.stage3_dpo.validate_frozen_file",
                return_value="digest",
            ),
            patch(
                "roleplay.stage3_dpo.validate_sft_adapter",
                return_value={"adapter_model.safetensors": "digest"},
            ),
            patch("roleplay.stage3_dpo._load_yaml", return_value=deepcopy(config)),
            patch("roleplay.stage3_dpo.run_logged", side_effect=fake_run_logged),
            patch(
                "roleplay.stage3_dpo.find_final_adapter",
                side_effect=lambda output: output / "adapter",
            ),
            patch(
                "roleplay.stage3_dpo.inspect_adapter_change",
                return_value={"changed_tensors": 1},
            ),
            patch(
                "roleplay.stage3_dpo.generate_dev_review_artifacts",
                side_effect=fake_review,
            ),
            patch.dict(
                "sys.modules",
                {
                    "huggingface_hub": fake_hub,
                    "transformers": fake_transformers,
                },
            ),
        ):
            if fail_training:
                with self.assertRaisesRegex(Stage3DPOError, "train broke"):
                    run_stage3_dpo(root / "runs")
                return commands
            return run_stage3_dpo(root / "runs"), commands

    def test_run_builds_and_atomically_archives_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, commands = self._run(root)

            self.assertEqual(run_dir, (root / "runs/run-1").resolve())
            validate_archive_contract(run_dir)
            self.assertEqual(commands[0][0][:2], ["swift", "rlhf"])
            self.assertEqual(commands[0][1], 24)
            self.assertFalse((root / "runs/.work/run-1").exists())
            summary = json.loads(
                (run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "awaiting_manual_review")
            self.assertEqual(summary["training"]["optimizer_steps"], 24)
            self.assertTrue(summary["technically_valid"])

    def test_failure_retains_work_and_minimal_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(root, fail_training=True)

            self.assertTrue((root / "runs/.work/run-1/train.log").is_file())
            summary = json.loads(
                (root / "runs/run-1/run_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "training_failed")
            self.assertFalse(summary["ready_to_publish"])


class PublicationTests(unittest.TestCase):
    def test_creates_exact_reusable_release_and_uploads_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_successful_run(run_dir)

            first = create_release_bundle(run_dir, root / "dist")
            second = create_release_bundle(run_dir, root / "dist")

            self.assertEqual(first, second)
            bundle, manifest, tag = first
            self.assertEqual(tag, "morgana-v2-stage3-dpo-20260809-2300")
            with tarfile.open(bundle, "r:gz") as archive:
                names = {member.name for member in archive.getmembers()}
            self.assertEqual(
                names,
                {f"20260809-2300/{name}" for name in EXPECTED_ARCHIVE_FILES},
            )
            self.assertTrue(manifest.is_file())

            with patch("subprocess.run") as run:
                publish_run(run_dir, root / "dist", "owner/repo")
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["gh", "release", "create"])
            self.assertIn("owner/repo", command)

    def test_extract_verifies_files_refuses_overwrite_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_successful_run(run_dir)
            bundle, manifest, _ = create_release_bundle(run_dir, root / "dist")

            destination = extract_release_bundle(
                bundle, manifest, root / "downloads"
            )
            validate_archive_contract(destination)
            with self.assertRaisesRegex(Stage3DPOError, "拒绝覆盖"):
                extract_release_bundle(bundle, manifest, root / "downloads")

            bundle.write_bytes(bundle.read_bytes() + b"tampered")
            with self.assertRaisesRegex(Stage3DPOError, "SHA-256"):
                extract_release_bundle(bundle, manifest, root / "other")

    def test_missing_archive_file_is_not_publishable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_successful_run(run_dir)
            (run_dir / "train.log").unlink()

            with self.assertRaisesRegex(Stage3DPOError, "missing"):
                create_release_bundle(run_dir, root / "dist")

    def test_rejects_archive_with_malicious_extra_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            make_successful_run(run_dir)
            bundle, manifest, _ = create_release_bundle(run_dir, root / "dist")
            malicious = root / "malicious.tar.gz"
            with tarfile.open(bundle, "r:gz") as source, tarfile.open(
                malicious, "w:gz"
            ) as target:
                for member in source.getmembers():
                    extracted = source.extractfile(member)
                    target.addfile(member, extracted)
                info = tarfile.TarInfo("../escape")
                info.size = 0
                target.addfile(info)
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_data["bundle"]["sha256"] = sha256_file(malicious)
            write_json_atomic(root / "malicious.manifest.json", manifest_data)

            with self.assertRaisesRegex(Stage3DPOError, "内容"):
                extract_release_bundle(
                    malicious,
                    root / "malicious.manifest.json",
                    root / "downloads",
                )


class ReviewTests(unittest.TestCase):
    def test_passing_review_records_ready_for_grpo_and_rejects_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            review_artifacts(run_dir)

            with redirect_stdout(StringIO()):
                summary = review_run(run_dir)

            self.assertEqual(summary["status"], "ready_for_grpo")
            self.assertTrue(summary["ready_for_grpo"])
            self.assertEqual(summary["manual_review"]["gate"]["dpo_wins"], 10)
            self.assertNotIn("sft_wins", summary["manual_review"]["gate"])
            with self.assertRaisesRegex(Stage3DPOError, "重复"):
                review_run(run_dir)

    def test_empty_review_keeps_summary_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            review_artifacts(run_dir, submitted=False)
            before = (run_dir / "run_summary.json").read_bytes()

            with redirect_stdout(StringIO()):
                review_run(run_dir)

            self.assertEqual((run_dir / "run_summary.json").read_bytes(), before)

    def test_automatic_failure_blocks_ready_for_grpo(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            review_artifacts(run_dir, automatic=False)

            with redirect_stdout(StringIO()):
                summary = review_run(run_dir)

            self.assertEqual(summary["status"], "dpo_failed")
            self.assertFalse(summary["ready_for_grpo"])

    def test_failing_human_review_records_dpo_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            review_artifacts(run_dir)
            results_path = run_dir / "manual_review_results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            for row in results["results"]:
                row["winner"] = "A"
            write_json_atomic(results_path, results)

            with redirect_stdout(StringIO()):
                summary = review_run(run_dir)

            self.assertEqual(summary["status"], "dpo_failed")
            self.assertFalse(summary["ready_for_grpo"])
            self.assertFalse(summary["manual_review"]["gate"]["passed"])


class CliTests(unittest.TestCase):
    def test_four_commands_dispatch_and_fail_cleanly(self):
        with patch(
            "roleplay.stage3_dpo.run_stage3_dpo", return_value=Path("run")
        ) as run:
            with redirect_stdout(StringIO()):
                result = main(["run"])
        self.assertEqual(result, 0)
        run.assert_called_once_with(None)

        with patch("roleplay.stage3_dpo.publish_run") as publish:
            publish.return_value = (Path("bundle"), Path("manifest"), "tag")
            with redirect_stdout(StringIO()):
                result = main(["publish", "--run-dir", "run"])
        self.assertEqual(result, 0)
        publish.assert_called_once()

        with patch(
            "roleplay.stage3_dpo.download_release", return_value=Path("run")
        ) as download:
            with redirect_stdout(StringIO()):
                result = main(["download", "--tag", "tag"])
        self.assertEqual(result, 0)
        download.assert_called_once()

        with patch("roleplay.stage3_dpo.review_run") as review:
            result = main(["review", "--run-dir", "run"])
        self.assertEqual(result, 0)
        review.assert_called_once_with(Path("run"))

        with patch(
            "roleplay.stage3_dpo.run_stage3_dpo",
            side_effect=Stage3DPOError("broken"),
        ):
            error = StringIO()
            with redirect_stderr(error):
                result = main(["run"])
        self.assertEqual(result, 1)
        self.assertIn("broken", error.getvalue())

    def test_malformed_review_json_fails_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            review_artifacts(run_dir, submitted=False)
            (run_dir / "manual_review_results.json").write_text(
                "{broken", encoding="utf-8"
            )
            error = StringIO()

            with redirect_stderr(error):
                result = main(["review", "--run-dir", str(run_dir)])

            self.assertEqual(result, 1)
            self.assertIn("不是有效 JSON", error.getvalue())
            self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
