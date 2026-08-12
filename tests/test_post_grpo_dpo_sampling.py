"""Tests for AutoDL post-GRPO DPO bulk sampling and exchange."""

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from roleplay.post_grpo_dpo_data import (
    _artifact as review_artifact,
    build_review_artifacts,
    finalize_run,
    load_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from roleplay.post_grpo_dpo_sampling import (
    CANDIDATES_PER_PROMPT,
    EXPECTED_ARCHIVE_FILES,
    GENERATION,
    MIN_FINAL_PAIRS,
    MODEL_REVISION,
    PROMPT_COUNT,
    TOTAL_CANDIDATES,
    PostGRPODPOSamplingError,
    TransformersSamplingRuntime,
    _artifact,
    batch_seed,
    build_candidate_records,
    candidate_id,
    create_release_bundle,
    extract_release_bundle,
    main,
    run_sampling,
    validate_archive_contract,
    validate_expansion_prompt_data,
    validate_sampling_candidate_rows,
)


def valid_answer(prompt_number: int, position: int) -> str:
    return (
        f"（耳朵认真一动）吾辈听清楚第{prompt_number}件事了，"
        f"会先陪莲把感受和事实分清楚，这是候选{position}。"
    )


class FakeRuntime:
    def __init__(self, *, fail_on_call=None):
        self.calls = []
        self.closed = False
        self.fail_on_call = fail_on_call

    def generate(self, messages, count, seed):
        call_number = len(self.calls) + 1
        self.calls.append((messages, count, seed))
        if call_number == self.fail_on_call:
            raise RuntimeError("fake batch failure")
        return [
            (valid_answer(call_number, position), "stop")
            for position in range(1, count + 1)
        ]

    def close(self):
        self.closed = True


class FrozenExpansionPromptTests(unittest.TestCase):
    def test_repository_expansion_is_balanced_frozen_and_isolated(self):
        rows = validate_expansion_prompt_data()

        self.assertEqual(len(rows), 60)
        self.assertEqual(rows[0]["id"], "post_dpo_exp_0001")
        self.assertEqual(rows[-1]["id"], "post_dpo_exp_0060")
        counts = {}
        for row in rows:
            counts[row["target_issue"]] = counts.get(row["target_issue"], 0) + 1
        self.assertEqual(set(counts.values()), {20})

    def test_batch_seed_and_candidate_ids_are_stable(self):
        seeds = {
            batch_seed(f"post_dpo_exp_{index:04d}")
            for index in range(1, PROMPT_COUNT + 1)
        }
        ids = {
            candidate_id(f"post_dpo_exp_{index:04d}", position)
            for index in range(1, PROMPT_COUNT + 1)
            for position in range(1, CANDIDATES_PER_PROMPT + 1)
        }

        self.assertEqual(len(seeds), 60)
        self.assertEqual(len(ids), 480)
        self.assertEqual(batch_seed("post_dpo_exp_0001"), 20260813)
        self.assertEqual(batch_seed("post_dpo_exp_0060"), 20260872)


class CandidateContractTests(unittest.TestCase):
    def setUp(self):
        self.prompt = validate_expansion_prompt_data()[0]
        self.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": self.prompt["user"]},
        ]

    def test_generation_contract_is_frozen(self):
        self.assertEqual(
            GENERATION,
            {
                "max_tokens": 512,
                "temperature": 0.6,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.45,
                "enable_thinking": False,
            },
        )

    def test_duplicate_and_hard_invalid_candidates_are_audited_not_reviewed(self):
        first = valid_answer(1, 1)
        responses = [
            (first, "stop"),
            (first, "stop"),
            ("（没有闭合的动作", "length"),
            *[
                (valid_answer(1, position), "stop")
                for position in range(4, 9)
            ],
        ]

        rows = build_candidate_records(self.prompt, self.messages, responses)

        self.assertTrue(rows[0]["eligible_for_review"])
        self.assertEqual(
            rows[1]["duplicate_of_candidate_id"], rows[0]["candidate_id"]
        )
        self.assertEqual(
            rows[1]["review_exclusion_reasons"], ["duplicate_candidate"]
        )
        self.assertFalse(rows[1]["eligible_for_review"])
        self.assertFalse(rows[2]["hard_valid"])
        self.assertFalse(rows[2]["eligible_for_review"])
        packet, key, _, unresolved = build_review_artifacts(
            [self.prompt], rows
        )
        self.assertFalse(unresolved)
        self.assertEqual(len(packet["items"][0]["answers"]), 6)
        self.assertEqual(len(key["items"][0]["labels"]), 6)


class SamplingRuntimeTests(unittest.TestCase):
    def test_transformers_engine_batches_all_candidates_per_prompt(self):
        engine_calls = []
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)

        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = types.SimpleNamespace(
            from_pretrained=lambda model, adapter_dir: model
        )

        fake_swift = types.ModuleType("swift")
        fake_swift.InferRequest = object
        fake_swift.RequestConfig = lambda **kwargs: kwargs
        fake_swift.get_model_processor = lambda *args, **kwargs: (
            object(),
            object(),
        )
        fake_swift.get_template = lambda *args, **kwargs: object()

        def fake_engine(*args, **kwargs):
            engine_calls.append((args, kwargs))
            return object()

        fake_swift.TransformersEngine = fake_engine

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BitsAndBytesConfig = lambda **kwargs: kwargs

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "peft": fake_peft,
                "swift": fake_swift,
                "transformers": fake_transformers,
            },
        ):
            runtime = TransformersSamplingRuntime(Path("adapter"))

        self.assertEqual(len(engine_calls), 1)
        self.assertEqual(
            engine_calls[0][1]["max_batch_size"], CANDIDATES_PER_PROMPT
        )
        runtime.close()


class SamplingRunTests(unittest.TestCase):
    def _run(self, root: Path, runtime: FakeRuntime):
        output = io.StringIO()
        with (
            patch(
                "roleplay.post_grpo_dpo_sampling.generate_run_id",
                return_value="sample-run",
            ),
            patch(
                "roleplay.post_grpo_dpo_sampling.configure_huggingface_environment",
                return_value={"endpoint": "fake", "cache_home": "fake"},
            ),
            patch(
                "roleplay.post_grpo_dpo_sampling.capture_environment",
                return_value=({"gpu": "fake"}, object()),
            ),
            patch(
                "roleplay.post_grpo_dpo_sampling.validate_pinned_packages",
                return_value={"ms-swift": "4.4.1"},
            ),
            patch(
                "roleplay.post_grpo_dpo_sampling.git_context",
                return_value={"commit": "abc123", "branch": "test"},
            ),
            patch(
                "roleplay.post_grpo_dpo_sampling.validate_grpo_adapter",
                return_value={"adapter_model.safetensors": "fake"},
            ),
            patch(
                "roleplay.post_grpo_dpo_sampling.validate_model_revision",
                return_value=MODEL_REVISION,
            ),
            patch(
                "roleplay.post_grpo_dpo_sampling.load_sampling_runtime",
                return_value=runtime,
            ),
            redirect_stdout(output),
        ):
            run_dir = run_sampling(root / "sampling")
        return run_dir, output.getvalue()

    def test_fake_run_loads_once_batches_60_times_and_prints_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = FakeRuntime()
            run_dir, stdout = self._run(Path(temporary), runtime)

            self.assertTrue(runtime.closed)
            self.assertEqual(len(runtime.calls), 60)
            self.assertTrue(all(call[1] == 8 for call in runtime.calls))
            self.assertEqual(
                [call[2] for call in runtime.calls],
                list(range(20260813, 20260873)),
            )
            self.assertIn("[1/4] 校验环境和冻结输入", stdout)
            self.assertIn("[2/4] 加载 4-bit Base", stdout)
            self.assertIn("Prompt 1/60 | candidates 8/480", stdout)
            self.assertIn("Prompt 60/60 | candidates 480/480", stdout)
            self.assertIn("elapsed", stdout)
            self.assertIn("ETA", stdout)
            self.assertIn("[4/4]", stdout)
            self.assertEqual(
                {
                    str(path.relative_to(run_dir))
                    for path in run_dir.rglob("*")
                    if path.is_file()
                },
                EXPECTED_ARCHIVE_FILES,
            )
            self.assertIn(
                "Prompt 60/60",
                (run_dir / "sampling.log").read_text(encoding="utf-8"),
            )
            summary = json.loads(
                (run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "awaiting_codex_review")
            self.assertEqual(summary["counts"]["candidates"], 480)
            self.assertEqual(summary["counts"]["review_items"], 60)
            candidates = load_jsonl(run_dir / "candidates.jsonl")
            self.assertEqual(len(candidates), TOTAL_CANDIDATES)
            validate_sampling_candidate_rows(
                candidates,
                validate_expansion_prompt_data(),
                Path("data/runs/morgana-v2/system_prompt.txt").read_text(
                    encoding="utf-8"
                ),
            )
            validate_archive_contract(run_dir)

    def test_failed_run_retains_partial_progress_without_final_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = FakeRuntime(fail_on_call=2)
            with self.assertRaisesRegex(RuntimeError, "fake batch failure"):
                self._run(root, runtime)
            work_dir = root / "sampling/.work/sample-run"
            self.assertTrue(work_dir.is_dir())
            self.assertFalse((root / "sampling/sample-run").exists())
            self.assertEqual(len(load_jsonl(work_dir / "candidates.jsonl")), 8)
            summary = json.loads(
                (work_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "sampling_failed")
            self.assertEqual(summary["progress"]["completed_prompts"], 1)


class ReleaseExchangeTests(unittest.TestCase):
    def _make_archive_run(self, root: Path) -> Path:
        run_dir = root / "run-1"
        run_dir.mkdir()
        for name in EXPECTED_ARCHIVE_FILES - {"run_summary.json"}:
            (run_dir / name).write_text(f"{name}\n", encoding="utf-8")
        summary = {
            "schema_version": 2,
            "stage": "post_grpo_dpo_sampling",
            "status": "awaiting_codex_review",
            "run": {"id": "run-1", "commit": "abc123"},
            "counts": {"candidates": 480},
            "artifacts": {
                name: _artifact(run_dir / name, run_dir)
                for name in EXPECTED_ARCHIVE_FILES - {"run_summary.json"}
            },
        }
        write_json_atomic(run_dir / "run_summary.json", summary)
        return run_dir

    def test_bundle_round_trip_verifies_exact_contract_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._make_archive_run(root)
            bundle, manifest, tag = create_release_bundle(
                run_dir, root / "dist"
            )

            self.assertEqual(tag, "morgana-v2-post-grpo-dpo-sampling-run-1")
            destination = extract_release_bundle(
                bundle, manifest, root / "downloads"
            )
            validate_archive_contract(destination)
            with self.assertRaisesRegex(
                PostGRPODPOSamplingError, "拒绝覆盖"
            ):
                extract_release_bundle(bundle, manifest, root / "downloads")

    def test_extract_rejects_archive_with_unexpected_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._make_archive_run(root)
            bundle, manifest, _ = create_release_bundle(run_dir, root / "dist")
            with tarfile.open(bundle, "w:gz") as archive:
                archive.add(
                    run_dir / "sampling.log",
                    arcname="../sampling.log",
                )
            release_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            release_manifest["bundle"]["sha256"] = hashlib.sha256(
                bundle.read_bytes()
            ).hexdigest()
            write_json_atomic(manifest, release_manifest)

            with self.assertRaisesRegex(
                PostGRPODPOSamplingError, "发布包内容"
            ):
                extract_release_bundle(bundle, manifest, root / "downloads")

    def test_download_command_dispatches(self):
        with patch(
            "roleplay.post_grpo_dpo_sampling.download_release",
            return_value=Path("downloaded"),
        ) as download:
            code = main(["download", "--tag", "sample-tag"])

        self.assertEqual(code, 0)
        download.assert_called_once()


class ExpansionFinalizationTests(unittest.TestCase):
    def _make_review_run(self, root: Path) -> Path:
        run_dir = root / "review-run"
        run_dir.mkdir()
        prompts = validate_expansion_prompt_data()
        system_prompt = Path(
            "data/runs/morgana-v2/system_prompt.txt"
        ).read_text(encoding="utf-8")
        candidates = []
        for prompt_number, prompt in enumerate(prompts, 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt["user"]},
            ]
            candidates.extend(
                build_candidate_records(
                    prompt,
                    messages,
                    [
                        (valid_answer(prompt_number, position), "stop")
                        for position in range(1, 9)
                    ],
                )
            )
        packet, key, _, unresolved = build_review_artifacts(
            prompts, candidates
        )
        self.assertFalse(unresolved)
        key_by_id = {item["review_id"]: item for item in key["items"]}
        results = []
        for item in packet["items"]:
            labels = key_by_id[item["review_id"]]["labels"]
            label_by_id = {
                value["candidate_id"]: label
                for label, value in labels.items()
            }
            chosen_label = label_by_id[f"{item['prompt_id']}-c1"]
            rejected_label = label_by_id[f"{item['prompt_id']}-c2"]
            candidate_reviews = {}
            for label, value in labels.items():
                if value["candidate_id"].endswith("-c1"):
                    status, quality = "pass", 9
                elif value["candidate_id"].endswith("-c2"):
                    status, quality = "fail", 8
                else:
                    status, quality = "ambiguous", 6
                candidate_reviews[label] = {
                    "target_status": status,
                    "non_target_quality": quality,
                    "off_target_issues": [],
                    "evidence": f"测试裁决 {value['candidate_id']}",
                }
            results.append(
                {
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
                    "notes": "原生 chosen 与最佳真实 hard negative。",
                }
            )
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
                "schema_version": 2,
                "stage": "post_grpo_dpo_sampling",
                "status": "awaiting_codex_review",
                "artifacts": {
                    name: review_artifact(path, run_dir)
                    for name, path in paths.items()
                },
            },
        )
        return run_dir

    def test_finalize_exports_60_balanced_expansion_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._make_review_run(root)
            train = root / "expansion.jsonl"
            audit = root / "expansion_audit.json"

            status, train_path, _ = finalize_run(
                run_dir=run_dir,
                train_output=train,
                audit_output=audit,
            )

            self.assertEqual(status, "ready_for_dpo")
            self.assertEqual(train_path, train)
            self.assertEqual(len(load_jsonl(train)), 60)
            report = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(report["pairs"], 60)
            self.assertEqual(set(report["pairs_by_issue"].values()), {20})
            self.assertEqual(
                report["readiness_contract"]["minimum_pairs"],
                MIN_FINAL_PAIRS,
            )

    def test_expansion_finalize_requires_new_explicit_output_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._make_review_run(Path(temporary))

            with self.assertRaisesRegex(
                RuntimeError, "必须显式指定新的"
            ):
                finalize_run(run_dir=run_dir)


if __name__ == "__main__":
    unittest.main()
