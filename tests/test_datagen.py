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
    BACKFILL_BATCH_SIZE,
    DEFAULT_SPLIT_SEED,
    DEEPSEEK_MODEL,
    GOALS,
    MAX_CONSECUTIVE_EMPTY,
    MVP_TARGETS,
    PROMPT_GENERATION_MAX_TOKENS,
    PROMPT_GENERATION_REASONING_EFFORT,
    PROMPT_GENERATION_TEMPERATURE,
    PROMPT_GENERATION_THINKING_TYPE,
    SCENARIO_OVERSAMPLE_COUNT,
    SCENARIOS,
    GenerationContext,
    GenerationShortfallError,
    _call_teacher,
    _candidate_pool_distribution,
    _render_examples,
    _scenario_distribution,
    _split_scenario_distributions,
    _take_unique_prompts,
    build_scenario_user_prompt,
    build_teacher_system,
    format_prompt_records,
    generate,
    load_examples,
    main,
    normalize_prompt,
    parse_prompts,
    save_input_snapshot,
    split_candidate_pool,
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

MORGANA_CONTEXT = GenerationContext(
    persona_text="PERSONA_TEXT",
    examples_text="EXAMPLES_TEXT",
    persona_name="摩尔加纳",
    user_name="莲",
    role_self_references=("吾辈",),
)


class CliTests(unittest.TestCase):
    def test_output_dir_is_required_to_protect_frozen_runs(self):
        stderr = io.StringIO()
        with patch("sys.argv", ["roleplay-datagen"]), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--output-dir", stderr.getvalue())


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

    def test_scenarios_have_known_target_goals(self):
        goal_ids = {goal["id"] for goal in GOALS}
        self.assertEqual(
            tuple(goal["id"] for goal in GOALS),
            (
                "generation_stability",
                "character_consistency",
                "dialogue_quality",
            ),
        )
        covered_goals: set[str] = set()
        for scenario in SCENARIOS:
            self.assertTrue(scenario["target_goals"])
            self.assertLessEqual(set(scenario["target_goals"]), goal_ids)
            covered_goals.update(scenario["target_goals"])
        self.assertEqual(covered_goals, goal_ids)

    def test_candidate_pool_combines_split_scenario_targets(self):
        split_distributions = _split_scenario_distributions()
        pool_distribution = _candidate_pool_distribution()

        self.assertEqual(sum(pool_distribution.values()), sum(MVP_TARGETS.values()))
        for scenario in SCENARIOS:
            scenario_id = scenario["id"]
            self.assertEqual(
                pool_distribution[scenario_id],
                sum(
                    distribution[scenario_id]
                    for distribution in split_distributions.values()
                ),
            )

    def test_seeded_candidate_pool_split_is_reproducible(self):
        pool = []
        for scenario in SCENARIOS:
            for index in range(_candidate_pool_distribution()[scenario["id"]]):
                pool.append(
                    {
                        "user": f"{scenario['id']}_{index}",
                        "scenario": scenario["id"],
                        "target_goals": list(scenario["target_goals"]),
                    }
                )

        first = split_candidate_pool(pool, split_seed=11)
        second = split_candidate_pool(pool, split_seed=11)
        third = split_candidate_pool(pool, split_seed=12)

        self.assertEqual(first, second)
        self.assertNotEqual(
            [record["user"] for record in first["sft"]],
            [record["user"] for record in third["sft"]],
        )
        for split, total in MVP_TARGETS.items():
            self.assertEqual(len(first[split]), total)
            scenario_counts = {
                scenario["id"]: sum(
                    1 for record in first[split]
                    if record["scenario"] == scenario["id"]
                )
                for scenario in SCENARIOS
            }
            self.assertEqual(scenario_counts, _scenario_distribution(total))

    def test_candidate_pool_split_rejects_scenario_count_mismatch(self):
        pool = []
        for scenario in SCENARIOS:
            for index in range(_candidate_pool_distribution()[scenario["id"]]):
                pool.append(
                    {
                        "user": f"{scenario['id']}_{index}",
                        "scenario": scenario["id"],
                        "target_goals": list(scenario["target_goals"]),
                    }
                )
        pool.append(
            {
                "user": "daily_extra",
                "scenario": "daily",
                "target_goals": ["dialogue_quality"],
            }
        )

        with self.assertRaisesRegex(GenerationShortfallError, "数量不匹配"):
            split_candidate_pool(pool)


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
        self.assertEqual(
            parse_prompts('{"prompts":[{"user_input":"a"},{"prompt":"b"}]}'),
            ["a", "b"],
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

    def test_unique_filter_rejects_explicit_goal_metadata_leaks(self):
        seen: set[str] = set()
        accepted = _take_unique_prompts(
            ["请测试角色一致性", "像平常那样跟我聊两句"], seen, limit=5
        )
        self.assertEqual(accepted, ["像平常那样跟我聊两句"])

    def test_unique_filter_rejects_speaker_perspective_inversion(self):
        seen: set[str] = set()
        accepted = _take_unique_prompts(
            [
                "吾辈觉得还是直接问莲比较好。",
                "莲，你为什么总摸我的头？",
                "莲蹲下来问我是不是生气了。",
                "摩尔加纳，你为什么总爱自称吾辈？",
            ],
            seen,
            limit=4,
            context=MORGANA_CONTEXT,
        )
        self.assertEqual(accepted, ["摩尔加纳，你为什么总爱自称吾辈？"])

    def test_unique_filter_rejects_long_near_duplicates(self):
        seen: set[str] = set()
        existing = ["今天天气特别好，我们一起去附近的公园散步怎么样？"]
        accepted = _take_unique_prompts(
            [
                "今天天气特别好，我们一起去附近的公园走走怎么样？",
                "我把钥匙忘在家里了，你能帮我看看窗台吗？",
            ],
            seen,
            limit=2,
            existing_prompts=existing,
        )
        self.assertEqual(accepted, ["我把钥匙忘在家里了，你能帮我看看窗台吗？"])


class PromptBuildingTests(unittest.TestCase):
    def test_teacher_system_enforces_prompt_only_and_core_boundaries(self):
        text = build_teacher_system(
            "PERSONA_TEXT", "EXAMPLES_TEXT", "摩尔加纳", "莲", ("吾辈",)
        )
        self.assertIn("PERSONA_TEXT", text)
        self.assertIn("EXAMPLES_TEXT", text)
        self.assertIn("允许角色在不冲突的前提下自然发挥", text)
        self.assertIn("角色扮演目标", text)
        self.assertIn("不得写入用户 Prompt", text)
        self.assertIn("不要生成角色回答", text)
        self.assertIn("说话人始终是莲", text)
        self.assertIn("用户不得用“吾辈”自称", text)
        self.assertIn('"prompts"', text)

    def test_scenario_prompt_mentions_split_offset_and_count(self):
        scenario = SCENARIOS[0]
        prompt = build_scenario_user_prompt(scenario, 5, 10, "DEV")
        self.assertIn("DEV", prompt)
        self.assertIn(scenario["name"], prompt)
        self.assertIn("主要覆盖目标", prompt)
        self.assertIn("不得写入用户 Prompt", prompt)
        self.assertIn("11", prompt)
        self.assertIn("15", prompt)

    def test_backfill_prompt_lists_accepted_prompts_to_avoid(self):
        prompt = build_scenario_user_prompt(
            SCENARIOS[0],
            5,
            18,
            "POOL",
            accepted_prompts=["已经存在的问题"],
        )
        self.assertIn("定向补充候选", prompt)
        self.assertIn("已经存在的问题", prompt)
        self.assertIn("不要复述、同义改写", prompt)

    def test_examples_render_all_or_empty_notice(self):
        examples = [{"user": f"u{i}", "assistant": f"a{i}"} for i in range(12)]
        rendered = _render_examples(examples)
        self.assertIn("u0", rendered)
        self.assertIn("a11", rendered)
        self.assertIn("暂无样例", _render_examples([]))


class JsonlTests(unittest.TestCase):
    def test_format_prompt_records_has_metadata_and_no_answer(self):
        records = format_prompt_records(
            [
                {
                    "user": "问题",
                    "scenario": "daily",
                    "target_goals": ["dialogue_quality"],
                }
            ],
            "sft",
        )
        self.assertEqual(
            records,
            [
                {
                    "id": "sft_0001",
                    "scenario": "daily",
                    "target_goals": ["dialogue_quality"],
                    "user": "问题",
                }
            ],
        )
        self.assertNotIn("assistant", records[0])

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
                "鼓励根据场景适度穿插",
                paths["system_prompt"].read_text(encoding="utf-8"),
            )
            self.assertTrue(paths["manifest"].is_file())

    def test_load_examples_allows_free_format_but_rejects_duplicate_users(self):
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
            self.assertEqual(len(load_examples(path)), 10)

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
                f"fallback_{self.fallback_index + index}" for index in range(25)
            ]
            self.fallback_index += 25
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
                {"prompts": [f"prompt_{batch}_{index}" for index in range(25)]}
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
        id_prefixes = {"sft": "sft", "rl": "grpo", "dev": "dev", "eval": "eval"}
        target_names = {"sft": "sft", "rl": "grpo", "dev": "dev", "eval": "eval"}
        scenario_ids = {scenario["id"] for scenario in SCENARIOS}
        goal_ids = {goal["id"] for goal in GOALS}
        all_keys: set[str] = set()
        for name, path in outputs.items():
            records = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), expected[name])
            self.assertTrue(
                all(
                    set(record) == {"id", "scenario", "target_goals", "user"}
                    for record in records
                )
            )
            self.assertEqual(
                [record["id"] for record in records],
                [
                    f"{id_prefixes[name]}_{index:04d}"
                    for index in range(1, expected[name] + 1)
                ],
            )
            self.assertTrue(all(record["scenario"] in scenario_ids for record in records))
            self.assertTrue(
                all(
                    record["target_goals"]
                    and set(record["target_goals"]).issubset(goal_ids)
                    for record in records
                )
            )
            keys = {normalize_prompt(record["user"]) for record in records}
            self.assertEqual(len(keys), expected[name])
            self.assertTrue(all_keys.isdisjoint(keys))
            all_keys.update(keys)
            scenario_counts = {
                scenario_id: sum(
                    1 for record in records if record["scenario"] == scenario_id
                )
                for scenario_id in scenario_ids
            }
            self.assertEqual(
                scenario_counts,
                _scenario_distribution(MVP_TARGETS[target_names[name]]),
            )

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
        self.assertIn("鼓励根据场景适度穿插", system_prompt)
        manifest = json.loads(
            (self.tmpdir / "input_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "stage",
                "persona",
                "style_examples",
                "system_prompt",
                "data",
                "split",
                "prompt_generation",
            },
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["data"]["targets"], MVP_TARGETS)
        self.assertEqual(
            manifest["data"]["candidate_pool_size"], sum(MVP_TARGETS.values())
        )
        self.assertEqual(
            manifest["data"]["initial_candidate_target"],
            sum(MVP_TARGETS.values())
            + len(SCENARIOS) * SCENARIO_OVERSAMPLE_COUNT,
        )
        self.assertEqual(
            manifest["data"]["scenario_targets"]["candidate_pool"],
            _candidate_pool_distribution(),
        )
        self.assertEqual(manifest["split"]["seed"], DEFAULT_SPLIT_SEED)
        self.assertEqual(
            manifest["split"]["method"],
            "shared_candidate_pool_then_seeded_per_scenario_split",
        )
        self.assertEqual(manifest["prompt_generation"]["model"], DEEPSEEK_MODEL)
        self.assertEqual(
            manifest["prompt_generation"]["temperature"],
            PROMPT_GENERATION_TEMPERATURE,
        )
        self.assertEqual(
            manifest["prompt_generation"]["thinking"],
            {"type": PROMPT_GENERATION_THINKING_TYPE},
        )
        self.assertEqual(
            manifest["prompt_generation"]["reasoning_effort"],
            PROMPT_GENERATION_REASONING_EFFORT,
        )
        self.assertEqual(
            manifest["prompt_generation"]["strategy"],
            {
                "mode": "scenario_batch_oversample_filter_backfill",
                "oversample_per_scenario": SCENARIO_OVERSAMPLE_COUNT,
                "backfill_batch_size": BACKFILL_BATCH_SIZE,
                "quality_filters": [
                    "goal_metadata_leak",
                    "speaker_perspective",
                    "missing_context",
                    "exact_duplicate",
                    "near_duplicate",
                ],
            },
        )

    def test_initial_generation_is_one_oversampled_batch_per_scenario(self):
        client = _FakeClient(self._responses())
        generate(
            persona_path=self.persona_path,
            examples_path=self.examples_path,
            output_dir=self.tmpdir,
            client=client,
        )

        calls = client.chat.completions.calls
        self.assertEqual(len(calls), len(SCENARIOS))
        for call in calls:
            user_prompt = call["messages"][1]["content"]
            self.assertIn(
                f"生成 {20 + SCENARIO_OVERSAMPLE_COUNT} 条", user_prompt
            )
            self.assertIn("首轮过采样候选", user_prompt)

    def test_duplicate_only_batch_is_retried_and_backfilled(self):
        duplicate = json.dumps({"prompts": ["same"] * 5})
        responses = [duplicate, duplicate] + self._responses()
        with redirect_stderr(io.StringIO()) as stderr:
            outputs = self._generate(responses)
        self.assertIn("无新增有效 Prompt", stderr.getvalue())
        self.assertEqual(
            len(outputs["sft"].read_text(encoding="utf-8").splitlines()), 50
        )

    def test_prompt_generation_uses_flash_deep_thinking_without_sampling(self):
        client = _FakeClient([json.dumps({"prompts": ["问题"]})])
        _call_teacher(client, DEEPSEEK_MODEL, "system", "user", max_tokens=123)

        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertIsNone(PROMPT_GENERATION_TEMPERATURE)
        self.assertNotIn("temperature", call)
        self.assertEqual(call["max_tokens"], 123)
        self.assertEqual(
            call["extra_body"]["thinking"]["type"],
            PROMPT_GENERATION_THINKING_TYPE,
        )
        self.assertEqual(
            call["extra_body"]["reasoning_effort"],
            PROMPT_GENERATION_REASONING_EFFORT,
        )
        self.assertEqual(PROMPT_GENERATION_MAX_TOKENS, 8192)

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
