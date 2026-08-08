"""Unit tests for validated baseline inference."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from roleplay import inference
from roleplay.inference import (
    BaselineGenerationError,
    generate,
    run_baseline,
    validate_answer,
)


PERSONA = {
    "name": "小衣",
    "identity": ["女仆恋人"],
    "personality": ["可爱"],
    "speech_style": ["简洁"],
    "relationships": ["恋人"],
    "facts": [],
    "boundaries": ["不承认是模型"],
}


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
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )
        return SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class ValidationTests(unittest.TestCase):
    def test_accepts_normal_stopped_answer(self):
        self.assertIsNone(validate_answer("今天想和你一起散步。", "stop"))

    def test_rejects_empty_or_incomplete_answer(self):
        self.assertEqual(validate_answer("  \n", "stop"), "回答为空")
        self.assertEqual(validate_answer("还没有说完", "length"), "finish_reason=length")
        self.assertEqual(validate_answer("回答", ""), "finish_reason=missing")

    def test_rejects_obvious_contiguous_repetition(self):
        self.assertIn("连续复读", validate_answer("她她她她她她", "stop"))
        self.assertIn(
            "连续复读",
            validate_answer("我想喝茶。喝茶。喝茶。喝茶。喝茶。", "stop"),
        )

    def test_can_preserve_repetition_for_teacher_correction(self):
        self.assertIsNone(
            validate_answer("她她她她她她", "stop", allow_repeated=True)
        )


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.persona_path = self.tmpdir / "persona.json"
        self.persona_path.write_text(
            json.dumps(PERSONA, ensure_ascii=False), encoding="utf-8"
        )
        self.eval_path = self.tmpdir / "eval.jsonl"
        self.eval_path.write_text(
            json.dumps({"user": "今天怎么样？"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.output_path = self.tmpdir / "baseline.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_generate_passes_sampling_parameters_and_returns_finish_reason(self):
        client = _FakeClient([("正常回答", "stop")])
        answer, finish_reason = generate(client, "model", [{"role": "user", "content": "q"}])

        self.assertEqual((answer, finish_reason), ("正常回答", "stop"))
        call = client.chat.completions.calls[0]
        self.assertEqual(call["max_tokens"], 512)
        self.assertEqual(call["temperature"], 0.7)
        self.assertEqual(call["top_p"], 0.8)
        self.assertEqual(call["presence_penalty"], 0.0)
        self.assertEqual(call["extra_body"]["top_k"], 20)
        self.assertEqual(call["extra_body"]["repetition_penalty"], 1.1)
        self.assertEqual(call["extra_body"]["repetition_context_size"], 64)
        self.assertEqual(call["extra_body"]["presence_context_size"], 64)
        self.assertFalse(
            call["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_generate_passes_fixed_seed_when_requested(self):
        client = _FakeClient([("正常回答", "stop")])

        generate(client, "model", [{"role": "user", "content": "q"}], seed=7)

        self.assertEqual(client.chat.completions.calls[0]["seed"], 7)

    def test_retries_invalid_answers_then_writes_valid_result(self):
        client = _FakeClient(
            [
                ("回答被截断", "length"),
                ("她她她她她她", "stop"),
                ("这次是完整而正常的回答。", "stop"),
            ]
        )

        run_baseline(
            self.persona_path,
            self.eval_path,
            self.output_path,
            "model",
            "http://unused",
            client=client,
        )

        record = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(record["assistant"], "这次是完整而正常的回答。")
        self.assertEqual(record["finish_reason"], "stop")
        self.assertEqual(record["attempts"], 3)
        self.assertEqual(len(client.chat.completions.calls), 3)
        self.assertFalse(self.output_path.with_name("baseline.jsonl.tmp").exists())

    def test_preserves_dev_id_and_fixed_seed(self):
        self.eval_path.write_text(
            json.dumps({"id": "dev_0001", "user": "今天怎么样？"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        client = _FakeClient([("完整回答。", "stop")])

        run_baseline(
            self.persona_path,
            self.eval_path,
            self.output_path,
            "model",
            "http://unused",
            client=client,
            generation_config={"seed": 20260807},
        )

        record = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(record["id"], "dev_0001")
        self.assertEqual(record["seed"], 20260807)

    def test_exhausted_retries_do_not_overwrite_existing_output(self):
        marker = '{"marker":"keep-me"}\n'
        self.output_path.write_text(marker, encoding="utf-8")
        client = _FakeClient(
            [
                RuntimeError("server unavailable"),
                ("", "stop"),
                ("未完成", "length"),
            ]
        )

        with self.assertRaises(BaselineGenerationError):
            run_baseline(
                self.persona_path,
                self.eval_path,
                self.output_path,
                "model",
                "http://unused",
                client=client,
            )

        self.assertEqual(self.output_path.read_text(encoding="utf-8"), marker)
        self.assertFalse(self.output_path.with_name("baseline.jsonl.tmp").exists())


if __name__ == "__main__":
    unittest.main()
