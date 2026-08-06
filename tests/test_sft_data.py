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
    call_teacher_with_retry,
    load_style_examples,
    main,
    parse_teacher_audit,
    run_student_aware_sft,
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
        self.assertIn("唯一来源", prompt)
        self.assertIn("最小充分修改", prompt)
        self.assertIn("不知道", prompt)
        self.assertIn("预设", prompt)
        self.assertIn("家庭成员", prompt)
        self.assertIn("格式契约 format", prompt)
        self.assertIn("全角括号", prompt)
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
        student = _FakeClient([("回答一", "stop"), ("回答二", "stop")])
        teacher = _FakeClient(
            [
                (audit_json("回答一"), "stop"),
                (
                    audit_json(
                        "回答二", decision="light_rewrite", improved="改进回答二"
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
            ["回答一", "改进回答二"],
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

    def test_teacher_retry_uses_flash_reasoning_configuration(self):
        teacher = _FakeClient([(audit_json("回答"), "stop")])
        audit, attempts = call_teacher_with_retry(
            teacher,
            "deepseek-v4-flash",
            "system",
            "问题",
            "回答",
            item_label="样本 1",
        )

        self.assertEqual(attempts, 1)
        self.assertEqual(audit["improved_assistant"], "回答")
        call = teacher.chat.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertEqual(call["max_tokens"], 4096)
        self.assertNotIn("temperature", call)
        self.assertEqual(call["extra_body"]["thinking"], {"type": "enabled"})
        self.assertEqual(call["extra_body"]["reasoning_effort"], "high")

    def test_failure_keeps_aligned_prefix_and_next_run_resumes(self):
        self.write_prompts("问题一", "问题二")
        first_student = _FakeClient([("回答一", "stop"), ("回答二", "stop")])
        first_teacher = _FakeClient(
            [
                (audit_json("回答一"), "stop"),
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

        resumed_student = _FakeClient([("新回答二", "stop")])
        resumed_teacher = _FakeClient([(audit_json("新回答二"), "stop")])
        self.run_pipeline(resumed_student, resumed_teacher)
        self.assertEqual([len(path.read_text().splitlines()) for path in paths], [2, 2, 2])
        self.assertEqual(len(resumed_student.chat.completions.calls), 1)

    def test_resume_rejects_changed_generation_configuration(self):
        self.write_prompts("问题一")
        self.run_pipeline(
            _FakeClient([("回答一", "stop")]),
            _FakeClient([(audit_json("回答一"), "stop")]),
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
            _FakeClient([("回答一", "stop")]),
            _FakeClient([(audit_json("回答一"), "stop")]),
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
            _FakeClient([("新回答", "stop")]),
            _FakeClient([(audit_json("新回答"), "stop")]),
            restart=True,
        )
        self.assertNotIn(
            "stale",
            (self.root / "sft_baseline_outputs.jsonl").read_text(encoding="utf-8"),
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
