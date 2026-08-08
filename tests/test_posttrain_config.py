"""Tests for frozen Mac/MLX configuration and assistant masking."""

import json
import tempfile
import unittest
from pathlib import Path

from roleplay.posttrain_config import (
    ConfigurationError,
    assistant_loss_bounds,
    load_grpo_config,
    load_sft_config,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, **_kwargs):
        size = sum(len(item["content"]) for item in messages)
        if add_generation_prompt:
            size += 3
        else:
            size += len(messages)
        return list(range(size))


class ConfigTests(unittest.TestCase):
    def test_primary_configs_validate_exact_training_shape(self):
        sft = load_sft_config(ROOT / "configs/morgana_v1_sft_mlx.json")
        grpo = load_grpo_config(ROOT / "configs/morgana_v1_grpo_mlx.json")
        self.assertEqual(sft["microbatches"], 150)
        self.assertEqual(sft["optimizer_steps"], 15)
        self.assertTrue(sft["mask_prompt"])
        self.assertEqual(grpo["generations_per_prompt"], 4)
        self.assertEqual(grpo["beta"], 0)

    def test_rejects_inconsistent_steps_and_nonzero_beta(self):
        cases = [
            ("configs/morgana_v1_sft_mlx.json", load_sft_config, "optimizer_steps", 14),
            ("configs/morgana_v1_grpo_mlx.json", load_grpo_config, "beta", 0.1),
        ]
        for source, loader, field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                data = json.loads((ROOT / source).read_text())
                data[field] = value
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ConfigurationError):
                    loader(path)

    def test_assistant_mask_covers_only_final_completion(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        start, end = assistant_loss_bounds(FakeTokenizer(), messages)
        expected_start = len("system") + len("question") + 3
        expected_end = len("system") + len("question") + len("answer") + 3
        self.assertEqual((start, end), (expected_start, expected_end))
        self.assertGreater(end, start)

    def test_assistant_mask_rejects_non_assistant_last_message(self):
        with self.assertRaisesRegex(ConfigurationError, "assistant"):
            assistant_loss_bounds(
                FakeTokenizer(), [{"role": "user", "content": "question"}]
            )
