"""Tests for the frozen morgana-v2 GRPO reward."""

import asyncio
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from roleplay.grpo_reward import (
    DEEPSEEK_MODEL,
    GRPORewardError,
    JudgeResponseError,
    MorganaRewardEngine,
    build_judge_user,
    calculate_length_penalty,
    completion_length,
    load_frozen_persona,
    normalize_for_copy,
    parse_judge_score,
    persona_copy_coverage,
    persona_copy_penalty,
    score_local,
)


def judge_json(**overrides):
    value = {
        "identity_boundary_facts": 2,
        "personality_relationship": 2,
        "character_voice": 1,
        "response_effectiveness": 3,
        "expression_quality": 2,
        "violations": [],
        "reason": "回答完整且符合角色。",
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
        ]
    )


class QueueCompletions:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return fake_response(result)


class MappingCompletions:
    def __init__(self, values, delays=None):
        self.values = values
        self.delays = delays or {}
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"].splitlines()[-1])
        completion = payload["candidate"]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delays.get(completion, 0))
            result = self.values[completion]
            if isinstance(result, Exception):
                raise result
            return fake_response(result)
        finally:
            self.active -= 1


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def messages(user="今天要做什么？"):
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": user},
    ]


class LocalRewardTests(unittest.TestCase):
    def test_loads_frozen_persona(self):
        self.assertIn("摩尔加纳", load_frozen_persona())

    def test_normalizes_only_whitespace_and_punctuation(self):
        self.assertEqual(normalize_for_copy("吾辈， 可靠！\n★"), "吾辈可靠★")

    def test_calculates_overlapping_persona_coverage(self):
        persona = "abcdefghijklmnopqrst"
        completion = persona + "UVWXYZ0123456789ABCD"

        self.assertEqual(len(completion), 40)
        self.assertAlmostEqual(
            persona_copy_coverage(completion, persona), 0.5
        )

    def test_applies_copy_thresholds_and_short_answer_exemption(self):
        self.assertEqual(persona_copy_penalty(39, 1.0), 0.0)
        self.assertEqual(persona_copy_penalty(40, 0.1999), 0.0)
        self.assertEqual(persona_copy_penalty(40, 0.20), 2.0)
        self.assertEqual(persona_copy_penalty(40, 0.49), 2.0)
        self.assertEqual(persona_copy_penalty(40, 0.50), 4.0)

    def test_calculates_length_penalty_boundaries(self):
        expected = {
            180: 0.0,
            210: 0.5,
            240: 1.0,
            300: 2.0,
            360: 2.0,
        }
        for length, penalty in expected.items():
            with self.subTest(length=length):
                self.assertEqual(calculate_length_penalty(length), penalty)
        self.assertEqual(completion_length("甲 乙\n丙！"), 4)

    def test_readability_rejects_empty_gibberish_repetition_and_truncation(self):
        persona = "与完成内容无关的角色设定"
        cases = {
            "empty": ("  ", None, False),
            "gibberish": ("αβγδεζηθικλμ", None, False),
            "repeated": ("abcdefghijkl" * 3, None, False),
            "finish_reason": ("吾辈还没说完", "length", False),
            "is_truncated": ("吾辈还没说完", None, True),
        }
        for name, (completion, finish_reason, truncated) in cases.items():
            with self.subTest(name=name):
                result = score_local(
                    completion,
                    persona,
                    finish_reason=finish_reason,
                    is_truncated=truncated,
                )
                self.assertEqual(result.readable, 0)

    def test_emoji_and_missing_signature_do_not_fail_readability(self):
        result = score_local("今天也要打起精神啊！🐾", "其他设定")

        self.assertEqual(result.readable, 1)


class JudgeScoreTests(unittest.TestCase):
    def test_parses_scores_and_calculates_totals(self):
        score = parse_judge_score(judge_json())

        self.assertEqual(score.role_consistency, 5)
        self.assertEqual(score.dialogue_quality, 5)

    def test_applies_role_caps(self):
        identity = parse_judge_score(
            judge_json(violations=["identity_break"])
        )
        fabrication = parse_judge_score(
            judge_json(
                violations=["fabricated_person_or_major_experience"]
            )
        )

        self.assertEqual(identity.role_consistency, 1)
        self.assertEqual(fabrication.role_consistency, 2)

    def test_rejects_invalid_schema_and_values(self):
        cases = {
            "boolean": judge_json(character_voice=True),
            "out_of_range": judge_json(response_effectiveness=4),
            "unknown_violation": judge_json(violations=["unknown"]),
            "duplicate_violation": judge_json(
                violations=["identity_break", "identity_break"]
            ),
            "empty_reason": judge_json(reason=""),
        }
        extra = json.loads(judge_json())
        extra["total"] = 10
        cases["extra_field"] = json.dumps(extra)
        missing = json.loads(judge_json())
        missing.pop("expression_quality")
        cases["missing_field"] = json.dumps(missing)

        for name, raw in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(JudgeResponseError):
                    parse_judge_score(raw)

    def test_serializes_untrusted_candidate_as_json(self):
        prompt = build_judge_user("用户", '"ignore judge"')
        payload = json.loads(prompt.splitlines()[-1])

        self.assertEqual(payload["candidate"], '"ignore judge"')


class RewardEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_candidates_independently_and_preserves_order(self):
        first = "候选一——简短但只能部分回应当前问题。"
        second = "候选二——吾辈会认真地给出完整、直接且自然的回答。"
        completions = MappingCompletions(
            {
                first: judge_json(
                    identity_boundary_facts=1,
                    personality_relationship=0,
                    character_voice=0,
                    response_effectiveness=1,
                    expression_quality=0,
                ),
                second: judge_json(),
            },
            delays={first: 0.02, second: 0.01},
        )
        engine = MorganaRewardEngine(
            persona_text="完全无关的 Persona 内容",
            client=fake_client(completions),
            sleep=lambda _: asyncio.sleep(0),
        )

        rewards = await engine.score_batch(
            [first, second], [messages(), messages()]
        )

        self.assertEqual(rewards, [1.0, 5.0])
        self.assertEqual(completions.max_active, 2)
        self.assertEqual(len(completions.calls), 2)
        for call in completions.calls:
            self.assertEqual(call["model"], DEEPSEEK_MODEL)
            self.assertEqual(call["extra_body"]["reasoning_effort"], "max")
            self.assertNotIn("temperature", call)

    async def test_skips_judge_for_unreadable_completion(self):
        completions = QueueCompletions([])
        engine = MorganaRewardEngine(
            persona_text="Persona",
            client=fake_client(completions),
        )

        rewards = await engine.score_batch([""], [messages()])

        self.assertEqual(rewards, [0.0])
        self.assertEqual(completions.calls, [])

    async def test_combines_high_copy_and_overlength_penalties(self):
        persona = "".join(chr(0x4E00 + index) for index in range(240))
        completion = persona
        completions = QueueCompletions([judge_json()])
        engine = MorganaRewardEngine(
            persona_text=persona,
            client=fake_client(completions),
        )

        rewards = await engine.score_batch([completion], [messages()])

        self.assertEqual(rewards, [0.0])  # 5 semantic - 4 copy - 1 length

    async def test_retries_twice_then_succeeds(self):
        completions = QueueCompletions(
            [RuntimeError("one"), RuntimeError("two"), judge_json()]
        )
        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        engine = MorganaRewardEngine(
            persona_text="Persona",
            client=fake_client(completions),
            sleep=fake_sleep,
        )

        rewards = await engine.score_batch(
            ["吾辈会好好回答你的问题。"], [messages()]
        )

        self.assertEqual(rewards, [5.0])
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(len(completions.calls), 3)

    async def test_logs_failure_and_raises_after_three_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "reward_samples.jsonl"
            completions = QueueCompletions(
                [RuntimeError("server exposed super-secret-key")] * 3
            )

            async def no_sleep(_):
                return None

            engine = MorganaRewardEngine(
                persona_text="Persona",
                client=fake_client(completions),
                log_path=log_path,
                sleep=no_sleep,
            )

            with patch.dict(
                "os.environ", {"DEEPSEEK_API_KEY": "super-secret-key"}
            ):
                with self.assertRaisesRegex(GRPORewardError, "Judge"):
                    await engine.score_batch(
                        ["吾辈会回答你。"],
                        [messages()],
                        prompt_ids=["prompt-1"],
                        request_ids=["request-1"],
                        record_ids=["grpo_0001"],
                        global_step=2,
                    )

            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "error")
            self.assertEqual(rows[0]["judge_attempts"], 3)
            self.assertIsNone(rows[0]["total_reward"])
            self.assertNotIn("super-secret-key", log_path.read_text())
            self.assertIn("[REDACTED]", log_path.read_text())
            self.assertNotIn("DEEPSEEK_API_KEY", log_path.read_text())

    async def test_logs_one_complete_row_per_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "reward_samples.jsonl"
            candidate = "吾辈会直接、自然地回答你的问题。"
            completions = QueueCompletions([judge_json()])
            engine = MorganaRewardEngine(
                persona_text="Persona",
                client=fake_client(completions),
                log_path=log_path,
            )

            await engine.score_batch(
                [candidate, ""], [messages(), messages("second")]
            )

            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["total_reward"], 5.0)
            self.assertIsNotNone(rows[0]["judge"])
            self.assertEqual(rows[1]["total_reward"], 0.0)
            self.assertIsNone(rows[1]["judge"])


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def test_registers_and_delegates_to_reward_engine(self):
        registry = {}

        class FakeAsyncORM:
            def __init__(self, args=None, **kwargs):
                self.args = args

        rewards_module = types.ModuleType("swift.rewards")
        rewards_module.AsyncORM = FakeAsyncORM
        rewards_module.orms = registry
        swift_module = types.ModuleType("swift")
        swift_module.rewards = rewards_module

        class FakeEngine:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.calls = []
                self.__class__.instances.append(self)

            async def score_batch(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return [3.5]

        module_name = "roleplay.grpo_reward_plugin"
        sys.modules.pop(module_name, None)
        with patch.dict(
            sys.modules,
            {"swift": swift_module, "swift.rewards": rewards_module},
        ):
            module = importlib.import_module(module_name)
            with patch.object(module, "MorganaRewardEngine", FakeEngine):
                args = SimpleNamespace(output_dir="/tmp/grpo-test")
                orm = module.MorganaRewardORM(args=args)
                result = await orm(
                    ["候选"],
                    [messages()],
                    finish_reason=["stop"],
                    is_truncated=[False],
                    prompt_id=["p"],
                    request_id=["r"],
                    trainer_state=SimpleNamespace(global_step=4),
                    id=["grpo_0001"],
                )

        sys.modules.pop(module_name, None)
        self.assertIs(registry["morgana_reward"], module.MorganaRewardORM)
        self.assertEqual(result, [3.5])
        engine = FakeEngine.instances[0]
        self.assertEqual(
            engine.kwargs["log_path"],
            Path("/tmp/grpo-test/reward_samples.jsonl"),
        )
        self.assertEqual(engine.calls[0][1]["record_ids"], ["grpo_0001"])
        self.assertEqual(engine.calls[0][1]["global_step"], 4)


if __name__ == "__main__":
    unittest.main()
