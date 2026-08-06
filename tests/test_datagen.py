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
    DEEPSEEK_MODEL,
    MAX_CONSECUTIVE_EMPTY,
    MVP_TARGETS,
    PROMPT_GENERATION_MAX_TOKENS,
    PROMPT_GENERATION_TEMPERATURE,
    PROMPT_GENERATION_THINKING_TYPE,
    SCENARIOS,
    GenerationShortfallError,
    _call_teacher,
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
    save_input_snapshot,
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

    def test_mvp_targets_match_plan(self):
        expected = {"sft": 50, "grpo": 20, "dev": 10, "eval": 20}
        self.assertEqual(MVP_TARGETS, expected)


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
                "\n".join(
                    json.dumps(
                        {"user": f"u{i}", "assistant": f"（点头）a{i}"}
                    )
                    for i in range(10)
                )
                + "\n\n",
                encoding="utf-8",
            )
            self.assertEqual(len(load_examples(path)), 10)

    def test_load_examples_enforces_plan_maximum(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "style_examples.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {"user": f"u{i}", "assistant": f"（点头）a{i}"}
                    )
                    + "\n"
                    for i in range(21)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "10～20"):
                load_examples(path)

    def test_save_input_snapshot_validates_and_copies_exact_inputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            persona_path = root / "persona.json"
            persona_path.write_text(
                json.dumps(PERSONA, ensure_ascii=False), encoding="utf-8"
            )
            examples_path = root / "style_examples.jsonl"
            examples_path.write_text(
                "".join(
                    json.dumps(
                        {"user": f"u{i}", "assistant": f"（点头）a{i}"},
                        ensure_ascii=False,
                    )
                    + "\n"
                    for i in range(10)
                ),
                encoding="utf-8",
            )
            output_dir = root / "run"
            paths = save_input_snapshot(persona_path, examples_path, output_dir)

            self.assertEqual(paths["persona"].read_bytes(), persona_path.read_bytes())
            self.assertEqual(
                paths["style_examples"].read_bytes(), examples_path.read_bytes()
            )
            self.assertIn(
                "全角括号", paths["system_prompt"].read_text(encoding="utf-8")
            )
            self.assertTrue(paths["manifest"].is_file())

    def test_load_examples_rejects_invalid_format_and_duplicate_users(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "style_examples.jsonl"
            records = [
                {"user": f"u{i}", "assistant": f"（点头）a{i}"}
                for i in range(10)
            ]
            records[3]["assistant"] = "没有动作括号"
            path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "必须遵循"):
                load_examples(path)

            records[3]["assistant"] = "（点头）恢复格式"
            records[3]["user"] = records[2]["user"]
            path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "重复"):
                load_examples(path)


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
            output_dir=self.tmpdir,
            client=_FakeClient(
                responses if responses is not None else self._responses()
            ),
        )

    def test_writes_four_isolated_prompt_files(self):
        outputs = self._generate()
        self.assertEqual(set(outputs), {"sft", "rl", "dev", "eval"})

        expected = {"sft": 50, "rl": 20, "dev": 10, "eval": 20}
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

        self.assertEqual(
            (self.tmpdir / "inputs/persona.json").read_bytes(),
            self.persona_path.read_bytes(),
        )
        self.assertEqual(
            (self.tmpdir / "inputs/style_examples.jsonl").read_bytes(),
            self.examples_path.read_bytes(),
        )
        system_prompt = (self.tmpdir / "system_prompt.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("全角括号", system_prompt)
        manifest = json.loads(
            (self.tmpdir / "input_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(manifest), {"persona", "style_examples", "system_prompt"}
        )

    def test_duplicate_only_batch_is_retried_and_backfilled(self):
        duplicate = json.dumps({"prompts": ["same"] * 5})
        responses = [duplicate, duplicate] + self._responses()
        with redirect_stderr(io.StringIO()) as stderr:
            outputs = self._generate(responses)
        self.assertIn("无新增有效 Prompt", stderr.getvalue())
        self.assertEqual(
            len(outputs["sft"].read_text(encoding="utf-8").splitlines()), 50
        )

    def test_prompt_generation_uses_flash_sampling_and_disabled_thinking(self):
        client = _FakeClient([json.dumps({"prompts": ["问题"]})])
        _call_teacher(client, DEEPSEEK_MODEL, "system", "user", max_tokens=123)

        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertEqual(call["temperature"], PROMPT_GENERATION_TEMPERATURE)
        self.assertEqual(call["max_tokens"], 123)
        self.assertEqual(
            call["extra_body"]["thinking"]["type"],
            PROMPT_GENERATION_THINKING_TYPE,
        )
        self.assertEqual(PROMPT_GENERATION_MAX_TOKENS, 2048)

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


if __name__ == "__main__":
    unittest.main()
