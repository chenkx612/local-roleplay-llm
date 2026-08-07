"""Unit tests for Student-aware SFT dataset construction."""

import json
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from roleplay.inference import BaselineGenerationError
from roleplay.sft_data import (
    DEFAULT_STUDENT_REVISION,
    StudentAwareSFTError,
    TEACHER_MAX_TOKENS,
    TEACHER_REASONING_EFFORT,
    TEACHER_TEMPERATURE,
    TEACHER_THINKING_TYPE,
    build_teacher_system,
    build_pilot_report,
    call_teacher_with_retry,
    load_prompts,
    load_style_examples,
    main,
    parse_teacher_audit,
    rerun_pilot_teacher,
    run_pilot,
    run_student_aware_sft,
    select_pilot_records,
)


PERSONA = {
    "name": "小衣",
    "identity": ["女仆恋人"],
    "personality": ["可爱"],
    "speech_style": ["简洁"],
    "relationships": ["与用户是恋人"],
    "facts": [],
    "boundaries": ["不承认是模型"],
}


def audit_json(
    baseline: str,
    *,
    decision: str = "keep",
    improved: str | None = None,
    final_checks: dict[str, bool] | None = None,
) -> str:
    return json.dumps(
        {
            "scores": {
                "persona": 9,
                "grounding": 8,
                "style": 9,
                "format": 9,
                "quality": 8,
            },
            "issues": [],
            "decision": decision,
            "improved_assistant": baseline if improved is None else improved,
            "final_checks": final_checks
            or {
                "persona_and_addressing": True,
                "grounding_and_creativity": True,
                "format": True,
                "directly_answers_user": True,
                "natural_dialogue": True,
            },
        },
        ensure_ascii=False,
    )


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        content, finish_reason = response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class TeacherAuditTests(unittest.TestCase):
    def test_teacher_prompt_contains_fact_boundary_and_examples(self):
        prompt = build_teacher_system("PERSONA", "EXAMPLES")
        self.assertIn("PERSONA", prompt)
        self.assertIn("EXAMPLES", prompt)
        self.assertIn("最高优先级依据", prompt)
        self.assertIn("原作设定", prompt)
        self.assertIn("非持久性创造", prompt)
        self.assertIn("最小充分修改", prompt)
        self.assertIn("预设", prompt)
        self.assertIn("用户个人信息", prompt)
        self.assertIn("持续状态", prompt)
        self.assertIn("格式契约 format", prompt)
        self.assertIn("全角括号", prompt)
        self.assertIn("改写后必做自检", prompt)
        self.assertIn("不能因为某个细节来自 baseline 就保留它", prompt)
        self.assertIn("不要误删符合原作的细节", prompt)
        self.assertIn("口语对白中不得再出现任何动作括号", prompt)
        self.assertIn('"format":0', prompt)

    def test_parses_keep_audit_and_adds_context_fields(self):
        audit = parse_teacher_audit(audit_json("原回答"), "问题", "原回答")
        self.assertEqual(audit["user"], "问题")
        self.assertEqual(audit["baseline_assistant"], "原回答")
        self.assertEqual(audit["decision"], "keep")

    def test_rejects_invalid_scores_and_inconsistent_decisions(self):
        data = json.loads(audit_json("原回答"))
        data["scores"]["persona"] = True
        with self.assertRaisesRegex(ValueError, "0～10"):
            parse_teacher_audit(json.dumps(data), "问题", "原回答")

        with self.assertRaisesRegex(ValueError, "逐字等于"):
            parse_teacher_audit(
                audit_json("原回答", improved="修改回答"), "问题", "原回答"
            )
        with self.assertRaisesRegex(ValueError, "实际修改"):
            parse_teacher_audit(
                audit_json("原回答", decision="rewrite"), "问题", "原回答"
            )

    def test_rejects_unknown_or_missing_fields(self):
        data = json.loads(audit_json("原回答"))
        data["extra"] = "no"
        with self.assertRaisesRegex(ValueError, "字段必须严格"):
            parse_teacher_audit(json.dumps(data), "问题", "原回答")

    def test_style_examples_require_exact_shape_and_plan_count(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            path.write_text('{"user":"u","assistant":"a","extra":1}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "仅含"):
                load_style_examples(path)

            path.write_text(
                "".join(
                    json.dumps(
                        {"user": f"u{i}", "assistant": f"（点头）a{i}"},
                        ensure_ascii=False,
                    )
                    + "\n"
                    for i in range(9)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "10～20"):
                load_style_examples(path)

            path.write_text(
                "".join(
                    json.dumps(
                        {"user": f"u{i}", "assistant": f"（点头）a{i}"},
                        ensure_ascii=False,
                    )
                    + "\n"
                    for i in range(21)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "10～20"):
                load_style_examples(path)


class PromptLoadingTests(unittest.TestCase):
    def test_accepts_legacy_user_only_records(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text(
                json.dumps({"user": "问题一"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_prompts(path), ["问题一"])

    def test_accepts_datagen_metadata_records(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "sft_0001",
                        "scenario": "daily",
                        "target_goals": ["dialogue_quality"],
                        "user": "问题一",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_prompts(path), ["问题一"])

    def test_rejects_generated_output_records_with_user_field(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft_baseline_outputs.jsonl"
            path.write_text(
                json.dumps(
                    {"user": "问题一", "assistant": "回答一", "attempts": 1},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "datagen prompt"):
                load_prompts(path)

    def test_selects_one_pilot_prompt_per_scenario_in_plan_order(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            scenarios = ["style", "daily", "adversarial", "emotion", "background"]
            records = [
                {
                    "id": f"sft_{index:04d}",
                    "scenario": scenario,
                    "target_goals": ["dialogue_quality"],
                    "user": f"{scenario} 问题",
                }
                for index, scenario in enumerate(scenarios, 1)
            ]
            records.append(
                {
                    "id": "sft_9999",
                    "scenario": "daily",
                    "target_goals": ["dialogue_quality"],
                    "user": "第二条 daily",
                }
            )
            path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )

            selected = select_pilot_records(path)

            self.assertEqual(
                [record["scenario"] for record in selected],
                ["daily", "background", "emotion", "style", "adversarial"],
            )
            self.assertEqual(len(selected), 5)

    def test_pilot_selection_rejects_legacy_records(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text('{"user":"问题"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scenario"):
                select_pilot_records(path)

    def test_rejects_malformed_datagen_metadata_records(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "sft_0001",
                        "scenario": "daily",
                        "target_goals": [],
                        "user": "问题一",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "datagen prompt"):
                load_prompts(path)


class StudentAwarePipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.persona_path = self.root / "persona.json"
        self.persona_path.write_text(
            json.dumps(PERSONA, ensure_ascii=False), encoding="utf-8"
        )
        self.examples_path = self.root / "style_examples.jsonl"
        self.examples_path.write_text(
            "".join(
                json.dumps(
                    {
                        "user": f"示例问题{i}",
                        "assistant": f"（点头）示例回答{i}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for i in range(10)
            ),
            encoding="utf-8",
        )
        self.prompts_path = self.root / "sft_train_prompts.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def write_prompts(self, *prompts: str) -> None:
        self.prompts_path.write_text(
            "".join(
                json.dumps({"user": prompt}, ensure_ascii=False) + "\n"
                for prompt in prompts
            ),
            encoding="utf-8",
        )

    def run_pipeline(
        self, student, teacher, *, restart=False, student_model="student"
    ):
        return run_student_aware_sft(
            persona_path=self.persona_path,
            examples_path=self.examples_path,
            prompts_path=self.prompts_path,
            output_dir=self.root,
            student_model=student_model,
            student_base_url="http://student",
            teacher_model="teacher",
            student_client=student,
            teacher_client=teacher,
            restart=restart,
        )

    def test_writes_three_aligned_outputs_and_ms_swift_messages(self):
        self.write_prompts("问题一", "问题二")
        student = _FakeClient([("（点头）回答一", "stop"), ("（点头）回答二", "stop")])
        teacher = _FakeClient(
            [
                (audit_json("（点头）回答一"), "stop"),
                (
                    audit_json(
                        "（点头）回答二",
                        decision="light_rewrite",
                        improved="（微笑）改进回答二",
                    ),
                    "stop",
                ),
            ]
        )
        outputs = self.run_pipeline(student, teacher)

        baseline = [
            json.loads(line)
            for line in outputs["baseline"].read_text(encoding="utf-8").splitlines()
        ]
        audits = [
            json.loads(line)
            for line in outputs["audit"].read_text(encoding="utf-8").splitlines()
        ]
        train = [
            json.loads(line)
            for line in outputs["train"].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["user"] for record in baseline], ["问题一", "问题二"])
        self.assertEqual([record["decision"] for record in audits], ["keep", "light_rewrite"])
        self.assertEqual(
            [record["messages"][2]["content"] for record in train],
            ["（点头）回答一", "（微笑）改进回答二"],
        )
        self.assertEqual(
            [message["role"] for message in train[0]["messages"]],
            ["system", "user", "assistant"],
        )
        self.assertIn(
            "示例回答9",
            teacher.chat.completions.calls[0]["messages"][0]["content"],
        )
        teacher_call = teacher.chat.completions.calls[0]
        self.assertNotIn("temperature", teacher_call)
        self.assertEqual(teacher_call["max_tokens"], TEACHER_MAX_TOKENS)
        self.assertEqual(
            teacher_call["extra_body"]["thinking"]["type"],
            TEACHER_THINKING_TYPE,
        )
        self.assertEqual(
            teacher_call["extra_body"]["reasoning_effort"],
            TEACHER_REASONING_EFFORT,
        )
        self.assertFalse(
            student.chat.completions.calls[0]["extra_body"]["chat_template_kwargs"][
                "enable_thinking"
            ]
        )
        metadata = json.loads(
            outputs["metadata"].read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["student"]["model"], "student")
        self.assertEqual(
            metadata["student"]["revision"], DEFAULT_STUDENT_REVISION
        )
        self.assertEqual(metadata["teacher"]["model"], "teacher")
        self.assertIsNone(metadata["teacher"]["temperature"])
        self.assertIsNone(TEACHER_TEMPERATURE)
        self.assertIsNone(metadata["teacher"]["top_p"])
        self.assertEqual(metadata["teacher"]["max_tokens"], 4096)
        self.assertEqual(metadata["teacher"]["thinking"], {"type": "enabled"})
        self.assertEqual(metadata["teacher"]["reasoning_effort"], "high")
        self.assertEqual(
            set(metadata["inputs"]),
            {"persona_sha256", "style_examples_sha256", "prompts_sha256"},
        )

    def test_keeps_truncated_student_baseline_for_teacher_correction(self):
        self.write_prompts("问题一")
        truncated = "（点头）这是一条被截断的 Student 回答"
        improved = "（点头）这是 Teacher 改写后的完整简短回答。"
        teacher = _FakeClient(
            [
                (
                    audit_json(
                        truncated, decision="rewrite", improved=improved
                    ),
                    "stop",
                )
            ]
        )
        outputs = self.run_pipeline(
            _FakeClient([(truncated, "length")]),
            teacher,
        )

        baseline = json.loads(outputs["baseline"].read_text(encoding="utf-8"))
        train = json.loads(outputs["train"].read_text(encoding="utf-8"))
        self.assertEqual(baseline["finish_reason"], "length")
        self.assertEqual(train["messages"][2]["content"], improved)
        teacher_message = teacher.chat.completions.calls[0]["messages"][1]["content"]
        self.assertIn("已达到输出 token 上限并被截断", teacher_message)

    def test_teacher_retry_uses_flash_reasoning_configuration(self):
        teacher = _FakeClient([(audit_json("（点头）回答"), "stop")])
        audit, attempts = call_teacher_with_retry(
            teacher,
            "deepseek-v4-flash",
            "system",
            "问题",
            "（点头）回答",
            item_label="样本 1",
        )

        self.assertEqual(attempts, 1)
        self.assertEqual(audit["improved_assistant"], "（点头）回答")
        call = teacher.chat.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertEqual(call["max_tokens"], 4096)
        self.assertNotIn("temperature", call)
        self.assertEqual(call["extra_body"]["thinking"], {"type": "enabled"})
        self.assertEqual(call["extra_body"]["reasoning_effort"], "high")

    def test_teacher_retries_when_final_answer_breaks_format_contract(self):
        baseline = "（点头）原回答"
        teacher = _FakeClient(
            [
                (
                    audit_json(
                        baseline,
                        decision="rewrite",
                        improved="（点头）第一句（挥手）第二句",
                    ),
                    "stop",
                ),
                (
                    audit_json(
                        baseline,
                        decision="rewrite",
                        improved="（点头）合格回答",
                    ),
                    "stop",
                ),
            ]
        )

        audit, attempts = call_teacher_with_retry(
            teacher,
            "teacher",
            "system",
            "问题",
            baseline,
            item_label="样本 1",
        )

        self.assertEqual(attempts, 2)
        self.assertEqual(audit["improved_assistant"], "（点头）合格回答")

    def test_teacher_retries_when_semantic_final_check_is_false(self):
        baseline = "（点头）原回答"
        failed_checks = {
            "persona_and_addressing": True,
            "grounding_and_creativity": True,
            "format": True,
            "directly_answers_user": False,
            "natural_dialogue": True,
        }
        teacher = _FakeClient(
            [
                (audit_json(baseline, final_checks=failed_checks), "stop"),
                (audit_json(baseline), "stop"),
            ]
        )

        _, attempts = call_teacher_with_retry(
            teacher,
            "teacher",
            "system",
            "问题",
            baseline,
            item_label="样本 1",
        )

        self.assertEqual(attempts, 2)

    def test_failure_keeps_aligned_prefix_and_next_run_resumes(self):
        self.write_prompts("问题一", "问题二")
        first_student = _FakeClient(
            [("（点头）回答一", "stop"), ("（点头）回答二", "stop")]
        )
        first_teacher = _FakeClient(
            [
                (audit_json("（点头）回答一"), "stop"),
                ("not json", "stop"),
                ("not json", "stop"),
                ("not json", "stop"),
            ]
        )
        with self.assertRaises(StudentAwareSFTError):
            self.run_pipeline(first_student, first_teacher)

        paths = [
            self.root / "sft_baseline_outputs.jsonl",
            self.root / "sft_teacher_edits.jsonl",
            self.root / "sft_train.jsonl",
        ]
        self.assertEqual([len(path.read_text().splitlines()) for path in paths], [1, 1, 1])

        resumed_student = _FakeClient([("（点头）新回答二", "stop")])
        resumed_teacher = _FakeClient(
            [(audit_json("（点头）新回答二"), "stop")]
        )
        self.run_pipeline(resumed_student, resumed_teacher)
        self.assertEqual([len(path.read_text().splitlines()) for path in paths], [2, 2, 2])
        self.assertEqual(len(resumed_student.chat.completions.calls), 1)

    def test_resume_rejects_changed_generation_configuration(self):
        self.write_prompts("问题一")
        self.run_pipeline(
            _FakeClient([("（点头）回答一", "stop")]),
            _FakeClient([(audit_json("（点头）回答一"), "stop")]),
        )
        student = _FakeClient([])
        teacher = _FakeClient([])
        with self.assertRaisesRegex(StudentAwareSFTError, "生成配置"):
            self.run_pipeline(
                student,
                teacher,
                student_model="different-student",
            )
        self.assertEqual(student.chat.completions.calls, [])

    def test_resume_rejects_changed_style_examples(self):
        self.write_prompts("问题一")
        self.run_pipeline(
            _FakeClient([("（点头）回答一", "stop")]),
            _FakeClient([(audit_json("（点头）回答一"), "stop")]),
        )
        with self.examples_path.open("a", encoding="utf-8") as file:
            file.write('{"user":"新问题","assistant":"（点头）新回答"}\n')
        with self.assertRaisesRegex(StudentAwareSFTError, "输入或生成配置"):
            self.run_pipeline(_FakeClient([]), _FakeClient([]))

    def test_mismatched_existing_bundle_is_rejected_without_calls(self):
        self.write_prompts("问题一")
        (self.root / "sft_baseline_outputs.jsonl").write_text("{}\n", encoding="utf-8")
        student = _FakeClient([])
        teacher = _FakeClient([])
        with self.assertRaisesRegex(StudentAwareSFTError, "部分已有产物"):
            self.run_pipeline(student, teacher)
        self.assertEqual(student.chat.completions.calls, [])

    def test_restart_replaces_existing_progress(self):
        self.write_prompts("问题一")
        for name in (
            "sft_baseline_outputs.jsonl",
            "sft_teacher_edits.jsonl",
            "sft_train.jsonl",
        ):
            (self.root / name).write_text('{"stale":true}\n', encoding="utf-8")
        self.run_pipeline(
            _FakeClient([("（点头）新回答", "stop")]),
            _FakeClient([(audit_json("（点头）新回答"), "stop")]),
            restart=True,
        )
        self.assertNotIn(
            "stale",
            (self.root / "sft_baseline_outputs.jsonl").read_text(encoding="utf-8"),
        )


class PilotPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.persona_path = self.root / "persona.json"
        self.persona_path.write_text(
            json.dumps(PERSONA, ensure_ascii=False), encoding="utf-8"
        )
        self.examples_path = self.root / "style_examples.jsonl"
        self.examples_path.write_text(
            "".join(
                json.dumps(
                    {"user": f"示例问题{i}", "assistant": f"（点头）示例回答{i}"},
                    ensure_ascii=False,
                )
                + "\n"
                for i in range(10)
            ),
            encoding="utf-8",
        )
        self.prompts_path = self.root / "sft_train_prompts.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def write_scenario_prompts(self) -> None:
        scenarios = ["style", "daily", "adversarial", "emotion", "background"]
        self.prompts_path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"sft_{index:04d}",
                        "scenario": scenario,
                        "target_goals": ["dialogue_quality"],
                        "user": f"{scenario} 问题",
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for index, scenario in enumerate(scenarios, 1)
            ),
            encoding="utf-8",
        )

    def test_runs_isolated_five_item_pilot_and_writes_review_artifacts(self):
        self.write_scenario_prompts()
        answers = [f"（点头）回答{i}" for i in range(5)]
        pilot_dir = self.root / "pilot"
        pilot_dir.mkdir()
        (pilot_dir / "pilot_review.md").write_text("旧复核表", encoding="utf-8")
        outputs = run_pilot(
            persona_path=self.persona_path,
            examples_path=self.examples_path,
            prompts_path=self.prompts_path,
            output_dir=pilot_dir,
            student_model="student",
            student_base_url="http://student",
            teacher_model="teacher",
            student_client=_FakeClient([(answer, "stop") for answer in answers]),
            teacher_client=_FakeClient(
                [(audit_json(answer), "stop") for answer in answers]
            ),
        )

        report = json.loads(outputs["report"].read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "automatic_checks_passed")
        self.assertEqual(report["sample_count"], 5)
        self.assertEqual(report["scenario_count"], 5)
        self.assertTrue(all(item["automatic_pass"] for item in report["items"]))
        self.assertIn("[ ] 通过", outputs["review"].read_text(encoding="utf-8"))
        self.assertNotIn("旧复核表", outputs["review"].read_text(encoding="utf-8"))
        self.assertIn("报告 SHA256", outputs["review"].read_text(encoding="utf-8"))
        self.assertEqual(len(outputs["prompts"].read_text().splitlines()), 5)

    def test_report_flags_invalid_final_format_for_manual_review(self):
        prompts = [
            {
                "id": f"sft_{index:04d}",
                "scenario": scenario,
                "target_goals": ["dialogue_quality"],
                "user": f"问题{index}",
            }
            for index, scenario in enumerate(
                ["daily", "background", "emotion", "style", "adversarial"], 1
            )
        ]
        baselines = [
            {
                "user": prompt["user"],
                "assistant": "回答",
                "finish_reason": "stop",
                "attempts": 1,
            }
            for prompt in prompts
        ]
        audits = [
            parse_teacher_audit(audit_json("回答"), prompt["user"], "回答")
            for prompt in prompts
        ]

        report = build_pilot_report(prompts, baselines, audits)

        self.assertEqual(report["status"], "manual_review_required")
        self.assertFalse(report["items"][0]["automatic_checks"]["format_contract"])

    def test_teacher_only_rerun_keeps_frozen_baselines(self):
        self.write_scenario_prompts()
        answers = [f"（点头）baseline {i}" for i in range(5)]
        pilot_dir = self.root / "pilot"
        run_pilot(
            persona_path=self.persona_path,
            examples_path=self.examples_path,
            prompts_path=self.prompts_path,
            output_dir=pilot_dir,
            student_model="student",
            student_base_url="http://student",
            teacher_model="teacher",
            student_client=_FakeClient([(answer, "stop") for answer in answers]),
            teacher_client=_FakeClient(
                [(audit_json(answer), "stop") for answer in answers]
            ),
        )
        frozen_baselines = (pilot_dir / "sft_baseline_outputs.jsonl").read_bytes()
        improved = [f"（微笑）Teacher 新回答 {i}" for i in range(5)]

        outputs = rerun_pilot_teacher(
            persona_path=self.persona_path,
            examples_path=self.examples_path,
            prompts_path=self.prompts_path,
            output_dir=pilot_dir,
            student_model="student",
            student_base_url="http://student",
            teacher_model="teacher",
            teacher_client=_FakeClient(
                [
                    (
                        audit_json(
                            answer, decision="rewrite", improved=new_answer
                        ),
                        "stop",
                    )
                    for answer, new_answer in zip(answers, improved)
                ]
            ),
        )

        self.assertEqual(outputs["baseline"].read_bytes(), frozen_baselines)
        train = [
            json.loads(line)
            for line in outputs["train"].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [record["messages"][2]["content"] for record in train], improved
        )


class CliTests(unittest.TestCase):
    def test_student_retry_exhaustion_is_reported_without_traceback(self):
        error = BaselineGenerationError("Student 连续 3 次生成失败")
        stderr = io.StringIO()
        with patch("roleplay.sft_data.run_student_aware_sft", side_effect=error):
            with patch.object(sys, "argv", ["roleplay-sft-data"]):
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Student 连续 3 次生成失败", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
