"""Tests for the minimal morgana-v2 DPO preference-data workflow."""

import asyncio
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from roleplay.dpo_data import (
    CONTRACT,
    DEEPSEEK_MODEL,
    DPODataError,
    PREFERENCE_REASONS,
    PreferenceService,
    _judge_prompts,
    _summary_contract,
    build_judge_system,
    build_review_artifacts,
    candidate_set_sha256,
    candidate_seed,
    finalize_run,
    generate_candidate_round,
    is_stable_candidate,
    load_prompts,
    parse_judge_decision,
    parse_teacher_edit,
    prepare_run,
    validate_cross_split_uniqueness,
    write_json_atomic,
)


def judge_json(**overrides):
    value = {
        "decision": "clear_preference",
        "chosen_id": "p-r1-c1",
        "rejected_id": "p-r1-c2",
        "best_id": "p-r1-c1",
        "preference_reasons": ["character_consistency"],
        "hard_rule_only": False,
        "reason": "第一条更自然地保持角色。",
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


def teacher_json(**overrides):
    value = {
        "improved_assistant": "吾辈会陪着你的。",
        "changes": "补足角色态度和情绪承接。",
        "final_checks": {
            "generation_stability": True,
            "role_consistency": True,
            "dialogue_quality": True,
        },
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


def fake_response(raw, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=raw),
            )
        ],
        model="deepseek-v4-pro",
        system_fingerprint="fp-test",
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
    )


class QueueCompletions:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return fake_response(value)


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def candidate(prompt_id, number, answer, source="sft_candidate"):
    return {
        "candidate_id": f"{prompt_id}-r1-c{number}",
        "prompt_id": prompt_id,
        "assistant": answer,
        "finish_reason": "stop",
        "source": source,
    }


class FrozenPromptTests(unittest.TestCase):
    def test_migrated_prompts_are_balanced_and_cross_split_unique(self):
        prompts = load_prompts()
        self.assertEqual(len(prompts), 20)
        self.assertEqual(prompts[0]["id"], "dpo_0001")
        validate_cross_split_uniqueness(prompts)

    def test_candidate_seeds_are_fixed_and_rounds_do_not_overlap(self):
        self.assertEqual(candidate_seed("dpo_0001", 1, 1), 20260807)
        self.assertEqual(candidate_seed("dpo_0002", 1, 1), 20260907)
        self.assertEqual(candidate_seed("dpo_0001", 2, 1), 20270807)

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
    def test_generates_exactly_three_candidates_with_frozen_seeds(self):
        fake_mlx = types.ModuleType("mlx_lm")
        fake_mlx.load = lambda *_args, **_kwargs: (object(), object())
        prompt = {
            "id": "dpo_0001",
            "scenario": "daily",
            "target_goals": [],
            "user": "问题",
        }
        outputs = iter([("回答一", "stop"), ("回答二", "stop"), ("回答三", "stop")])
        with patch.dict(sys.modules, {"mlx_lm": fake_mlx}), patch(
            "roleplay.dpo_data._verify_loaded_adapter"
        ), patch("roleplay.dpo_data._generate_one", side_effect=lambda *_args: next(outputs)):
            rows = generate_candidate_round(
                prompts=[prompt],
                round_number=1,
                base_model_path=Path("base"),
                mlx_adapter_path=Path("adapter"),
                system_prompt="system",
            )
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["seed"] for row in rows], [20260807, 20260808, 20260809])
        self.assertTrue(all(row["generation"] == CONTRACT["generation"] for row in rows))


class JudgeTeacherContractTests(unittest.TestCase):
    def test_judge_selects_strongest_comparable_rejected_candidate(self):
        prompt = build_judge_system("persona", "examples", "system")
        self.assertIn("质量最高", prompt)
        self.assertIn("不得为了扩大差距故意选择最差候选", prompt)
        self.assertIn("no_clear_preference", prompt)

    def test_parses_clear_subjective_preference(self):
        parsed = parse_judge_decision(
            judge_json(), ["p-r1-c1", "p-r1-c2", "p-r1-c3"]
        )
        self.assertEqual(parsed["decision"], "clear_preference")

    def test_rejects_hard_rule_only_clear_preference(self):
        with self.assertRaisesRegex(DPODataError, "语义不一致"):
            parse_judge_decision(
                judge_json(hard_rule_only=True),
                ["p-r1-c1", "p-r1-c2", "p-r1-c3"],
            )

    def test_all_inadequate_requires_best_candidate(self):
        with self.assertRaisesRegex(DPODataError, "best_id"):
            parse_judge_decision(
                judge_json(
                    decision="all_inadequate",
                    chosen_id=None,
                    rejected_id=None,
                    best_id=None,
                    preference_reasons=[],
                ),
                ["p-r1-c1", "p-r1-c2"],
            )

    def test_teacher_requires_all_final_checks(self):
        parsed = parse_teacher_edit(teacher_json())
        self.assertIn("吾辈", parsed["improved_assistant"])
        checks = {
            "generation_stability": True,
            "role_consistency": False,
            "dialogue_quality": True,
        }
        with self.assertRaisesRegex(DPODataError, "全部通过"):
            parse_teacher_edit(teacher_json(final_checks=checks))

    def test_service_uses_pro_max_json_contract_without_sampling(self):
        completions = QueueCompletions([judge_json()])
        service = PreferenceService(
            persona="persona",
            examples="examples",
            system_prompt="system",
            client=fake_client(completions),
            sleep=lambda _seconds: asyncio.sleep(0),
        )
        prompt = {"id": "p", "user": "问题"}
        candidates = [
            candidate("p", 1, "回答一"),
            candidate("p", 2, "回答二"),
            candidate("p", 3, "回答三"),
        ]
        result = asyncio.run(service.judge(prompt, candidates, 1))
        self.assertEqual(result["decision"], "clear_preference")
        call = completions.calls[0]
        self.assertEqual(call["model"], DEEPSEEK_MODEL)
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertEqual(call["extra_body"]["reasoning_effort"], "max")
        self.assertNotIn("temperature", call)
        self.assertNotIn("top_p", call)

    def test_service_retries_invalid_api_results(self):
        completions = QueueCompletions([RuntimeError("one"), judge_json()])
        service = PreferenceService(
            persona="persona",
            examples="examples",
            system_prompt="system",
            client=fake_client(completions),
            sleep=lambda _seconds: asyncio.sleep(0),
        )
        result = asyncio.run(
            service.judge(
                {"id": "p", "user": "问题"},
                [candidate("p", 1, "一"), candidate("p", 2, "二")],
                1,
            )
        )
        self.assertEqual(result["attempts"], 2)


class ReviewArtifactTests(unittest.TestCase):
    def test_packet_hides_sources_and_human_can_override_judge(self):
        prompt = {"id": "dpo_0001", "scenario": "daily", "user": "问题"}
        candidates = [
            candidate("dpo_0001", 1, "回答一"),
            candidate("dpo_0001", 2, "回答二"),
        ]
        judges = [
            {
                "prompt_id": "dpo_0001",
                "decision": "clear_preference",
                "chosen_id": "dpo_0001-r1-c1",
                "rejected_id": "dpo_0001-r1-c2",
                "preference_reasons": ["character_consistency"],
            }
        ]
        packet, key, results, unresolved = build_review_artifacts(
            prompts=[prompt],
            candidates=candidates,
            judges=judges,
            teacher_edits=[],
            order_seed=1,
        )
        self.assertFalse(unresolved)
        serialized = json.dumps(packet, ensure_ascii=False)
        self.assertNotIn("sft_candidate", serialized)
        self.assertNotIn("judge", serialized.lower())
        self.assertIn("labels", key["items"][0])
        self.assertIsNone(results["results"][0]["winner"])


class PrepareWorkflowTests(unittest.TestCase):
    class ClearService:
        def __init__(self):
            self.calls = 0

        async def judge(self, prompt, candidates, round_number):
            self.calls += 1
            return {
                "prompt_id": prompt["id"],
                "round": round_number,
                "candidate_ids": [row["candidate_id"] for row in candidates],
                "candidate_set_sha256": candidate_set_sha256(candidates),
                "decision": "clear_preference",
                "chosen_id": candidates[0]["candidate_id"],
                "rejected_id": candidates[1]["candidate_id"],
                "best_id": candidates[0]["candidate_id"],
                "preference_reasons": ["character_consistency"],
                "hard_rule_only": False,
                "reason": "第一条更符合角色。",
                "attempts": 1,
                "api": {},
            }

        async def teacher(self, *_args):
            raise AssertionError("clear preference 不应调用 Teacher")

    @staticmethod
    def fake_candidates(prompts, round_number, **_kwargs):
        rows = []
        for prompt in prompts:
            for index in range(1, 4):
                rows.append(
                    {
                        "candidate_id": f"{prompt['id']}-r{round_number}-c{index}",
                        "prompt_id": prompt["id"],
                        "round": round_number,
                        "candidate_index": index,
                        "seed": candidate_seed(prompt["id"], round_number, index),
                        "assistant": f"（点头）{prompt['id']} 回答 {index}",
                        "raw_assistant": f"（点头）{prompt['id']} 回答 {index}",
                        "finish_reason": "stop",
                        "source": "sft_candidate",
                        "base_model": "mlx-community/Qwen3.5-2B-4bit",
                        "base_revision": "674aaa7240b91e8012fcad5d791b7dfe5ba90207",
                        "adapter_sha256": (
                            "617e6e00535fa356272d32fb16d8fe8d451a9c3cfd2f766f56af02cdf2f9b78d"
                        ),
                        "generation": CONTRACT["generation"],
                    }
                )
        return rows

    def test_prepare_builds_packet_and_completed_run_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base"
            base.mkdir()
            service = self.ClearService()
            with patch(
                "roleplay.dpo_data.convert_peft_adapter_to_mlx",
                return_value=root / "adapter",
            ), patch(
                "roleplay.dpo_data.generate_candidate_round",
                side_effect=self.fake_candidates,
            ):
                run = prepare_run(
                    run_id="test-run",
                    output_root=root / "output",
                    base_model_path=base,
                    service=service,
                )
                second = prepare_run(
                    run_id="test-run",
                    output_root=root / "output",
                    base_model_path=base,
                    service=service,
                )
                summary_path = run / "run_summary.json"
                frozen_summary = json.loads(summary_path.read_text())
                frozen_summary["status"] = "ready_for_dpo"
                write_json_atomic(summary_path, frozen_summary)
                third = prepare_run(
                    run_id="test-run",
                    output_root=root / "output",
                    base_model_path=base,
                    service=service,
                )
            self.assertEqual(run, second)
            self.assertEqual(run, third)
            self.assertEqual(service.calls, 20)
            summary = json.loads((run / "run_summary.json").read_text())
            packet = json.loads((run / "manual_review_packet.json").read_text())
            self.assertEqual(summary["status"], "ready_for_dpo")
            self.assertEqual(len(packet["items"]), 20)

    def test_resume_rejects_candidate_content_changed_after_judging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base"
            base.mkdir()
            service = self.ClearService()
            with patch(
                "roleplay.dpo_data.convert_peft_adapter_to_mlx",
                return_value=root / "adapter",
            ), patch(
                "roleplay.dpo_data.generate_candidate_round",
                side_effect=self.fake_candidates,
            ):
                run = prepare_run(
                    run_id="changed-candidate",
                    output_root=root / "output",
                    base_model_path=base,
                    service=service,
                )
                summary = json.loads((run / "run_summary.json").read_text())
                summary["status"] = "preparing"
                write_json_atomic(run / "run_summary.json", summary)
                candidates_path = run / "candidates.jsonl"
                candidates = [
                    json.loads(line) for line in candidates_path.read_text().splitlines()
                ]
                candidates[0]["assistant"] = "（点头）被修改的回答"
                candidates[0]["raw_assistant"] = "（点头）被修改的回答"
                candidates_path.write_text(
                    "".join(
                        json.dumps(row, ensure_ascii=False) + "\n"
                        for row in candidates
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(DPODataError, "候选内容不一致"):
                    prepare_run(
                        run_id="changed-candidate",
                        output_root=root / "output",
                        base_model_path=base,
                        service=service,
                    )

    def test_successful_prompt_is_persisted_when_another_api_call_fails(self):
        class PartialService:
            async def judge(self, prompt, candidates, round_number):
                if prompt["id"] == "p2":
                    raise DPODataError("broken")
                return {
                    "prompt_id": prompt["id"],
                    "round": round_number,
                    "candidate_ids": [row["candidate_id"] for row in candidates],
                    "candidate_set_sha256": candidate_set_sha256(candidates),
                    "decision": "clear_preference",
                    "chosen_id": candidates[0]["candidate_id"],
                    "rejected_id": candidates[1]["candidate_id"],
                    "best_id": candidates[0]["candidate_id"],
                    "preference_reasons": ["character_consistency"],
                    "hard_rule_only": False,
                    "reason": "清晰偏好。",
                    "attempts": 1,
                    "api": {},
                }

        prompts = [{"id": "p1", "user": "一"}, {"id": "p2", "user": "二"}]
        candidates = [
            candidate(prompt_id, index, f"{prompt_id}-{index}")
            for prompt_id in ("p1", "p2")
            for index in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "judge.jsonl"
            with self.assertRaisesRegex(DPODataError, "broken"):
                asyncio.run(
                    _judge_prompts(
                        PartialService(),
                        prompts,
                        candidates,
                        1,
                        persist_path=path,
                    )
                )
            persisted = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["prompt_id"] for row in persisted], ["p1"])


class FinalizeTests(unittest.TestCase):
    def make_run(self, root: Path, count: int = 16, teacher_count: int = 0):
        run = root / "run"
        run.mkdir()
        summary = {
            **_summary_contract(),
            "run": {"id": "test"},
            "status": "awaiting_human_review",
        }
        write_json_atomic(run / "run_summary.json", summary)
        packet_items = []
        key_items = []
        results = []
        for index in range(1, count + 1):
            review_id = f"dpo-review-{index:02d}"
            prompt_id = f"dpo_{index:04d}"
            packet_items.append(
                {
                    "review_id": review_id,
                    "prompt_id": prompt_id,
                    "scenario": "daily",
                    "user": f"问题{index}",
                    "answer_a": f"甲{index}",
                    "answer_b": f"乙{index}",
                }
            )
            key_items.append(
                {
                    "review_id": review_id,
                    "prompt_id": prompt_id,
                    "labels": {
                        "A": {
                            "assistant": f"甲{index}",
                            "source": "teacher_edit" if index <= teacher_count else "sft_candidate",
                            "source_id": f"a{index}",
                        },
                        "B": {
                            "assistant": f"乙{index}",
                            "source": "sft_candidate",
                            "source_id": f"b{index}",
                        },
                    },
                }
            )
            results.append(
                {
                    "review_id": review_id,
                    "winner": "B",
                    "preference_reasons": ["expression_naturalness"],
                    "material_tradeoff": False,
                    "notes": "人工选择 B。",
                }
            )
        write_json_atomic(run / "manual_review_packet.json", {"items": packet_items})
        write_json_atomic(run / "manual_review_key.json", {"items": key_items})
        write_json_atomic(run / "manual_review_results.json", {"results": results})
        return run

    def test_finalizes_ms_swift_format_and_accepts_human_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root)
            train = root / "dpo_train.jsonl"
            audit = root / "audit.json"
            finalize_run(run_dir=run, train_output=train, audit_output=audit)
            rows = [json.loads(line) for line in train.read_text().splitlines()]
            self.assertEqual(len(rows), 16)
            self.assertEqual(set(rows[0]), {"messages", "rejected_response"})
            self.assertEqual(rows[0]["messages"][-1]["content"], "乙1")
            self.assertEqual(rows[0]["rejected_response"], "甲1")
            written_summary = json.loads((run / "run_summary.json").read_text())
            result_artifact = written_summary["artifacts"][
                "manual_review_results.json"
            ]
            self.assertEqual(
                result_artifact["sha256"],
                hashlib.sha256(
                    (run / "manual_review_results.json").read_bytes()
                ).hexdigest(),
            )
            written_audit = json.loads(audit.read_text())
            self.assertEqual(
                written_audit["manual_review_results_sha256"],
                result_artifact["sha256"],
            )

    def test_rejects_too_few_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root, count=15)
            with self.assertRaisesRegex(DPODataError, "至少需要 16"):
                finalize_run(
                    run_dir=run,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )

    def test_rejects_teacher_pairs_above_one_third(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root, teacher_count=6)
            with self.assertRaisesRegex(DPODataError, "超过三分之一"):
                finalize_run(
                    run_dir=run,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )

    def test_rejects_packet_key_answer_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root)
            key_path = run / "manual_review_key.json"
            key = json.loads(key_path.read_text())
            key["items"][0]["labels"]["A"]["assistant"] = "未展示给复核者"
            write_json_atomic(key_path, key)
            with self.assertRaisesRegex(DPODataError, "A/B 映射不一致"):
                finalize_run(
                    run_dir=run,
                    train_output=root / "train.jsonl",
                    audit_output=root / "audit.json",
                )


if __name__ == "__main__":
    unittest.main()
