"""Unit tests for frozen prompt generation and split isolation."""

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from roleplay.datagen import (
    MAX_CONSECUTIVE_EMPTY,
    PROFILES,
    SCENARIOS,
    GenerationShortfallError,
    _render_examples,
    _scenario_distribution,
    _take_unique_prompts,
    build_scenario_user_prompt,
    build_teacher_system,
    format_prompt_records,
    generate,
    load_examples,
    normalize_prompt,
    parse_prompts,
    write_jsonl,
    write_jsonl_bundle,
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
    def test_total_is_preserved_and_even(self):
        for total in (3, 20, 30, 50, 103):
            distribution = _scenario_distribution(total)
            self.assertEqual(sum(distribution.values()), total)
            self.assertEqual(set(distribution), {scenario["id"] for scenario in SCENARIOS})
            self.assertLessEqual(max(distribution.values()) - min(distribution.values()), 1)

    def test_profile_targets_match_plan(self):
        self.assertEqual(
            PROFILES["smoke"], {"sft": 100, "grpo": 30, "dev": 20, "eval": 50}
        )
        self.assertEqual(
            PROFILES["mvp"], {"sft": 300, "grpo": 100, "dev": 50, "eval": 100}
        )


class ParsePromptTests(unittest.TestCase):
    def test_clean_json_and_code_fence(self):
        self.assertEqual(parse_prompts('{"prompts":["你好","在吗？"]}'), ["你好", "在吗？"])
        self.assertEqual(
            parse_prompts('```json\n{"prompts":["问题"]}\n```'), ["问题"]
        )

    def test_accepts_bare_list_and_user_objects(self):
        self.assertEqual(parse_prompts('["a","b"]'), ["a", "b"])
        self.assertEqual(
            parse_prompts('{"data":[{"user":"a"},{"user":"b"}]}'), ["a", "b"]
        )

    def test_skips_invalid_and_blank_items(self):
        raw = json.dumps({"prompts": [" ", 1, None, {"user": ""}, {"user": " ok "}]})
        self.assertEqual(parse_prompts(raw), ["ok"])
        self.assertEqual(parse_prompts("not-json"), [])
        self.assertEqual(parse_prompts("null"), [])

    def test_rejects_non_list_prompt_collection(self):
        self.assertEqual(parse_prompts('{"prompts":"你好"}'), [])
        self.assertEqual(parse_prompts('{"prompts":{"user":"你好"}}'), [])
        self.assertEqual(parse_prompts('{"data":"你好"}'), [])


class NormalizationTests(unittest.TestCase):
    def test_nfkc_and_whitespace_are_normalized(self):
        self.assertEqual(normalize_prompt("  ＡＢＣ\t 问题\n"), "ABC 问题")

    def test_punctuation_and_case_are_preserved(self):
        self.assertNotEqual(normalize_prompt("Hello?"), normalize_prompt("hello?"))
        self.assertNotEqual(normalize_prompt("你好？"), normalize_prompt("你好。"))

    def test_unique_filter_deduplicates_within_and_across_batches(self):
        seen = {normalize_prompt("已有问题")}
        accepted = _take_unique_prompts(
            [" 新问题 ", "新问题", "已有问题", "另一个问题"], seen, limit=5
        )
        self.assertEqual(accepted, [" 新问题 ", "另一个问题"])
        self.assertEqual(
            _take_unique_prompts(["新问题", "另一个问题"], seen, limit=5), []
        )


class PromptBuildingTests(unittest.TestCase):
    def test_teacher_system_enforces_prompt_only_and_fact_boundary(self):
        text = build_teacher_system("PERSONA_TEXT", "EXAMPLES_TEXT", "小衣")
        self.assertIn("PERSONA_TEXT", text)
        self.assertIn("EXAMPLES_TEXT", text)
        self.assertIn("唯一来源", text)
        self.assertIn("不要生成角色回答", text)
        self.assertIn('"prompts"', text)

    def test_scenario_prompt_mentions_split_offset_and_count(self):
        scenario = SCENARIOS[0]
        prompt = build_scenario_user_prompt(scenario, 5, 10, "DEV")
        self.assertIn("DEV", prompt)
        self.assertIn(scenario["name"], prompt)
        self.assertIn("11", prompt)
        self.assertIn("15", prompt)

    def test_examples_render_all_or_empty_notice(self):
        examples = [{"user": f"u{i}", "assistant": f"a{i}"} for i in range(12)]
        rendered = _render_examples(examples)
        self.assertIn("u0", rendered)
        self.assertIn("a11", rendered)
        self.assertIn("暂无样例", _render_examples([]))


class JsonlTests(unittest.TestCase):
    def test_format_prompt_records_has_no_answer(self):
        self.assertEqual(format_prompt_records(["问题"]), [{"user": "问题"}])

    def test_write_and_load(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            write_jsonl([{"a": 1}, {"b": 2}], path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records, [{"a": 1}, {"b": 2}])

    def test_bundle_stages_all_files_before_publish(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {root / "a.jsonl": [{"a": 1}], root / "b.jsonl": [{"b": 2}]}
            write_jsonl_bundle(paths)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertFalse(any(root.glob("*.tmp")))
            self.assertFalse(any(root.glob("*.bak")))

    def test_bundle_rolls_back_partially_published_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.jsonl"
            second = root / "b.jsonl"
            first.write_text('{"old":"a"}\n', encoding="utf-8")
            second.write_text('{"old":"b"}\n', encoding="utf-8")
            original_replace = Path.replace

            def interrupt_second_publish(source, target):
                target = Path(target)
                if source.name == "b.jsonl.tmp" and target.name == "b.jsonl":
                    raise KeyboardInterrupt
                return original_replace(source, target)

            with patch("pathlib.Path.replace", new=interrupt_second_publish):
                with self.assertRaises(KeyboardInterrupt):
                    write_jsonl_bundle(
                        {first: [{"new": "a"}], second: [{"new": "b"}]}
                    )

            self.assertEqual(first.read_text(encoding="utf-8"), '{"old":"a"}\n')
            self.assertEqual(second.read_text(encoding="utf-8"), '{"old":"b"}\n')
            self.assertFalse(any(root.glob("*.tmp")))
            self.assertFalse(any(root.glob("*.bak")))

    def test_load_examples_skips_blank_lines(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "style_examples.jsonl"
            path.write_text(
                '{"user":"a","assistant":"b"}\n\n{"user":"c","assistant":"d"}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                load_examples(path),
                [{"user": "a", "assistant": "b"}, {"user": "c", "assistant": "d"}],
            )


class _FakeCompletions:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.fallback_index = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            content = self._responses.pop(0)
        else:
            prompts = [
                f"fallback_{self.fallback_index + index}" for index in range(5)
            ]
            self.fallback_index += 5
            content = json.dumps({"prompts": prompts})
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
        self.examples_path = self.tmpdir / "style_examples.jsonl"
        self.examples_path.write_text(
            json.dumps(
                {"user": "示例问题", "assistant": "示例回答"}, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
        self.sleep_patcher = patch("roleplay.datagen.time.sleep")
        self.sleep_patcher.start()

    def tearDown(self):
        self.sleep_patcher.stop()
        self.tmp.cleanup()

    def _responses(self, count: int = 200) -> list[str]:
        return [
            json.dumps(
                {"prompts": [f"prompt_{batch}_{index}" for index in range(5)]}
            )
            for batch in range(count)
        ]

    def _generate(self, responses: list[str] | None = None) -> dict[str, Path]:
        return generate(
            persona_path=self.persona_path,
            examples_path=self.examples_path,
            profile="smoke",
            output_dir=self.tmpdir,
            client=_FakeClient(responses if responses is not None else self._responses()),
        )

    def test_smoke_writes_four_isolated_prompt_files(self):
        outputs = self._generate()
        self.assertEqual(set(outputs), {"sft", "rl", "dev", "eval"})

        expected = {"sft": 100, "rl": 30, "dev": 20, "eval": 50}
        all_keys: set[str] = set()
        for name, path in outputs.items():
            records = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), expected[name])
            self.assertTrue(all(set(record) == {"user"} for record in records))
            keys = {normalize_prompt(record["user"]) for record in records}
            self.assertEqual(len(keys), expected[name])
            self.assertTrue(all_keys.isdisjoint(keys))
            all_keys.update(keys)

    def test_duplicate_only_batch_is_retried_and_backfilled(self):
        duplicate = json.dumps({"prompts": ["same"] * 5})
        responses = [duplicate, duplicate] + self._responses()
        with redirect_stderr(io.StringIO()) as stderr:
            outputs = self._generate(responses)
        self.assertIn("无新增有效 Prompt", stderr.getvalue())
        self.assertEqual(
            len(outputs["sft"].read_text(encoding="utf-8").splitlines()), 100
        )

    def test_normalized_cross_split_duplicates_are_backfilled(self):
        first = json.dumps({"prompts": ["ＡＢＣ", "p1", "p2", "p3", "p4"]})
        later = json.dumps({"prompts": ["ABC", "p5", "p6", "p7", "p8"]})
        outputs = self._generate([first, later] + self._responses())
        records = []
        for path in outputs.values():
            records.extend(
                json.loads(line)["user"]
                for line in path.read_text(encoding="utf-8").splitlines()
            )
        keys = [normalize_prompt(prompt) for prompt in records]
        self.assertEqual(len(keys), len(set(keys)))

    def test_shortfall_does_not_overwrite_existing_bundle(self):
        paths = [
            self.tmpdir / "sft_train_prompts.jsonl",
            self.tmpdir / "rl_train.jsonl",
            self.tmpdir / "dev.jsonl",
            self.tmpdir / "eval.jsonl",
        ]
        marker = '{"marker":"keep-me"}\n'
        for path in paths:
            path.write_text(marker, encoding="utf-8")

        empties = [""] * (MAX_CONSECUTIVE_EMPTY * 2)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(GenerationShortfallError):
                self._generate(empties)

        for path in paths:
            self.assertEqual(path.read_text(encoding="utf-8"), marker)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            generate(
                self.persona_path,
                self.examples_path,
                "invalid",
                self.tmpdir,
                client=_FakeClient([]),
            )


if __name__ == "__main__":
    unittest.main()
