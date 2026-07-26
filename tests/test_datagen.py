"""Unit tests for the data generation module.

Only the deterministic parts are covered here (parsing, distribution, formatting,
retry logic). The real DeepSeek API is never called — a fake OpenAI client is
used to feed canned teacher responses through the orchestration code.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from roleplay import datagen
from roleplay.datagen import (
    BATCH_SIZE,
    MAX_CONSECUTIVE_EMPTY,
    SCENARIOS,
    GenerationShortfallError,
    _batch_sizes,
    _render_examples,
    _scenario_distribution,
    build_scenario_user_prompt,
    build_teacher_system,
    format_prompt_records,
    format_sft_records,
    generate,
    load_examples,
    parse_pairs,
    write_jsonl,
)


PERSONA = {
    "name": "小衣",
    "identity": ["女仆恋人"],
    "personality": ["可爱"],
    "speech_style": ["口头禅"],
    "relationships": ["恋人"],
    "facts": [],
    "boundaries": ["不承认是模型"],
}


class ScenarioDistributionTests(unittest.TestCase):
    def test_total_is_preserved(self):
        for total in (5, 30, 50, 100, 103):
            dist = _scenario_distribution(total)
            self.assertEqual(sum(dist.values()), total)
            self.assertEqual(set(dist), {s["id"] for s in SCENARIOS})

    def test_is_as_even_as_possible(self):
        dist = _scenario_distribution(100)
        self.assertTrue(all(v == 20 for v in dist.values()))

    def test_remainder_goes_to_first_scenarios(self):
        dist = _scenario_distribution(103)
        values = [dist[s["id"]] for s in SCENARIOS]
        self.assertEqual(values.count(21), 3)
        self.assertEqual(values.count(20), 2)

    def test_smaller_than_scenario_count(self):
        dist = _scenario_distribution(3)
        self.assertEqual(sum(dist.values()), 3)
        self.assertGreaterEqual(min(dist.values()), 0)


class BatchSizesTests(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(_batch_sizes(15), [BATCH_SIZE, BATCH_SIZE, BATCH_SIZE])

    def test_with_remainder(self):
        self.assertEqual(_batch_sizes(7), [BATCH_SIZE, 2])

    def test_less_than_batch_size(self):
        self.assertEqual(_batch_sizes(3), [3])

    def test_zero(self):
        self.assertEqual(_batch_sizes(0), [])


class ParsePairsTests(unittest.TestCase):
    def test_clean_json(self):
        raw = json.dumps({"pairs": [{"user": "hi", "assistant": "hello"}]})
        self.assertEqual(parse_pairs(raw), [{"user": "hi", "assistant": "hello"}])

    def test_json_in_code_fence(self):
        raw = '```json\n{"pairs":[{"user":"hi","assistant":"hello"}]}\n```'
        self.assertEqual(parse_pairs(raw), [{"user": "hi", "assistant": "hello"}])

    def test_bare_array(self):
        raw = '[{"user":"a","assistant":"b"}]'
        self.assertEqual(parse_pairs(raw), [{"user": "a", "assistant": "b"}])

    def test_skips_blank_or_missing_fields(self):
        raw = json.dumps(
            {
                "pairs": [
                    {"user": "   ", "assistant": "ok"},
                    {"user": "hi", "assistant": ""},
                    {"user": "ok"},
                    {"user": "good", "assistant": "response"},
                ]
            }
        )
        self.assertEqual(parse_pairs(raw), [{"user": "good", "assistant": "response"}])

    def test_strips_whitespace(self):
        raw = json.dumps({"pairs": [{"user": "  hi  ", "assistant": "  ok  "}]})
        self.assertEqual(parse_pairs(raw), [{"user": "hi", "assistant": "ok"}])

    def test_empty_or_invalid_returns_empty(self):
        self.assertEqual(parse_pairs(""), [])
        self.assertEqual(parse_pairs("not json at all"), [])
        self.assertEqual(parse_pairs("null"), [])
        self.assertEqual(parse_pairs('{"other": 1}'), [])

    def test_non_dict_items_are_skipped(self):
        raw = json.dumps({"pairs": ["not-a-dict", {"user": "ok", "assistant": "fine"}]})
        self.assertEqual(parse_pairs(raw), [{"user": "ok", "assistant": "fine"}])


class PromptBuildingTests(unittest.TestCase):
    def test_teacher_system_includes_persona_and_rules(self):
        text = build_teacher_system("PERSONA_TEXT", "EXAMPLES_TEXT", "小衣")
        self.assertIn("PERSONA_TEXT", text)
        self.assertIn("EXAMPLES_TEXT", text)
        self.assertIn("小衣", text)
        self.assertIn("JSON", text)

    def test_scenario_user_prompt_mentions_offset_and_count(self):
        scenario = SCENARIOS[0]
        prompt = build_scenario_user_prompt(scenario, count=5, offset=10)
        self.assertIn(scenario["name"], prompt)
        self.assertIn("5", prompt)
        self.assertIn("11", prompt)
        self.assertIn("15", prompt)

    def test_render_examples_includes_all(self):
        examples = [{"user": f"u{i}", "assistant": f"a{i}"} for i in range(12)]
        rendered = _render_examples(examples)
        self.assertIn("u0", rendered)
        self.assertIn("u11", rendered)
        self.assertIn("a11", rendered)

    def test_render_examples_empty(self):
        self.assertIn("暂无样例", _render_examples([]))


class FormattingTests(unittest.TestCase):
    def test_sft_records_have_system(self):
        records = format_sft_records(
            [{"user": "q", "assistant": "a"}], system_prompt="SYS"
        )
        self.assertEqual(
            records,
            [
                {
                    "messages": [
                        {"role": "system", "content": "SYS"},
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ],
        )

    def test_prompt_records_drop_assistant(self):
        records = format_prompt_records([{"user": "q", "assistant": "a"}])
        self.assertEqual(records, [{"user": "q"}])


class JsonlTests(unittest.TestCase):
    def test_write_and_load(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            write_jsonl([{"a": 1}, {"b": 2}], path)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line) for line in lines], [{"a": 1}, {"b": 2}])

    def test_load_examples_skips_blank_lines(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            path.write_text(
                '{"user":"a","assistant":"b"}\n\n{"user":"c","assistant":"d"}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                load_examples(path),
                [{"user": "a", "assistant": "b"}, {"user": "c", "assistant": "d"}],
            )


class _FakeCompletions:
    """Records calls and returns canned JSON responses for each batch."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            content = json.dumps({"pairs": [{"user": "fallback", "assistant": "fb"}]})
        else:
            content = self._responses.pop(0)
        choice = SimpleNamespace(message=SimpleNamespace(content=content))
        return SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class GenerateEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.persona_path = self.tmpdir / "persona.json"
        self.persona_path.write_text(json.dumps(PERSONA), encoding="utf-8")
        self.examples_path = self.tmpdir / "examples.jsonl"
        self.examples_path.write_text(
            json.dumps({"user": "示例问题", "assistant": "示例回答"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Avoid real sleeps during empty-retry / inter-batch delays.
        self._sleep_patcher = patch("roleplay.datagen.time.sleep")
        self._sleep_patcher.start()

    def tearDown(self):
        self._sleep_patcher.stop()
        self.tmp.cleanup()

    def _build_responses(self, pairs_per_call: int = 5) -> list[str]:
        """One response per API call; number of calls depends on profile totals."""
        out = []
        for _ in range(200):
            pairs = [
                {"user": f"u_{len(out)}_{i}", "assistant": f"a_{len(out)}_{i}"}
                for i in range(pairs_per_call)
            ]
            out.append(json.dumps({"pairs": pairs}))
        return out

    def test_smoke_profile_writes_three_files(self):
        responses = self._build_responses(pairs_per_call=5)
        client = _FakeClient(responses)
        outputs = generate(
            persona_path=self.persona_path,
            examples_path=self.examples_path,
            profile="smoke",
            output_dir=self.tmpdir,
            client=client,
        )
        for path in outputs.values():
            self.assertTrue(path.exists())

        sft_lines = [json.loads(line) for line in outputs["sft"].read_text(encoding="utf-8").splitlines()]
        rl_lines = [json.loads(line) for line in outputs["rl"].read_text(encoding="utf-8").splitlines()]
        eval_lines = [json.loads(line) for line in outputs["eval"].read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(sft_lines), 100)
        self.assertEqual(len(rl_lines), 30)
        self.assertEqual(len(eval_lines), 50)
        self.assertIn("messages", sft_lines[0])
        self.assertIn("user", rl_lines[0])
        self.assertNotIn("assistant", rl_lines[0])
        self.assertIn("user", eval_lines[0])

    def test_retry_on_empty_response_still_reaches_target(self):
        """Empty first call is retried; partial pair is kept and later batches fill up."""
        empty_first_then_partial = [
            "",
            json.dumps({"pairs": [{"user": "u", "assistant": "a"}]}),
        ]
        rest = self._build_responses(pairs_per_call=5)
        client = _FakeClient(empty_first_then_partial + rest)

        import io
        from contextlib import redirect_stderr

        buf = io.StringIO()
        with redirect_stderr(buf):
            outputs = generate(
                persona_path=self.persona_path,
                examples_path=self.examples_path,
                profile="smoke",
                output_dir=self.tmpdir,
                client=client,
            )
        self.assertIn("重试一次", buf.getvalue())
        sft_n = sum(1 for _ in outputs["sft"].read_text(encoding="utf-8").splitlines())
        rl_n = sum(1 for _ in outputs["rl"].read_text(encoding="utf-8").splitlines())
        eval_n = sum(1 for _ in outputs["eval"].read_text(encoding="utf-8").splitlines())
        self.assertEqual(sft_n, 100)
        self.assertEqual(rl_n, 30)
        self.assertEqual(eval_n, 50)

    def test_partial_valid_response_is_backfilled(self):
        """Teacher returning fewer pairs than requested must still hit profile targets."""
        partial = json.dumps({"pairs": [{"user": "only_one", "assistant": "a"}]})
        rest = self._build_responses(pairs_per_call=5)
        client = _FakeClient([partial] + rest)

        outputs = generate(
            persona_path=self.persona_path,
            examples_path=self.examples_path,
            profile="smoke",
            output_dir=self.tmpdir,
            client=client,
        )
        sft_lines = [
            json.loads(line)
            for line in outputs["sft"].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(sft_lines), 100)
        self.assertEqual(sft_lines[0]["messages"][1]["content"], "only_one")

    def test_shortfall_fails_without_overwriting_existing(self):
        """Exhausted empty retries must fail and leave prior artifacts untouched."""
        sft_path = self.tmpdir / "sft_train.jsonl"
        rl_path = self.tmpdir / "rl_train.jsonl"
        eval_path = self.tmpdir / "eval.jsonl"
        marker = '{"marker":"keep-me"}\n'
        sft_path.write_text(marker, encoding="utf-8")
        rl_path.write_text(marker, encoding="utf-8")
        eval_path.write_text(marker, encoding="utf-8")

        # Each empty batch: first call + one retry = 2 API responses.
        empties = [""] * (MAX_CONSECUTIVE_EMPTY * 2 + 4)
        client = _FakeClient(empties)

        import io
        from contextlib import redirect_stderr

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(GenerationShortfallError):
                generate(
                    persona_path=self.persona_path,
                    examples_path=self.examples_path,
                    profile="smoke",
                    output_dir=self.tmpdir,
                    client=client,
                )

        self.assertEqual(sft_path.read_text(encoding="utf-8"), marker)
        self.assertEqual(rl_path.read_text(encoding="utf-8"), marker)
        self.assertEqual(eval_path.read_text(encoding="utf-8"), marker)
        self.assertFalse((self.tmpdir / "sft_train.jsonl.tmp").exists())

    def test_unknown_profile_raises(self):
        client = _FakeClient([])
        with self.assertRaises(ValueError):
            generate(
                persona_path=self.persona_path,
                examples_path=self.examples_path,
                profile="nope",
                output_dir=self.tmpdir,
                client=client,
            )


if __name__ == "__main__":
    unittest.main()
