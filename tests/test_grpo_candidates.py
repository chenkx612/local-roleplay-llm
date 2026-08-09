"""Tests for local GRPO reward-calibration candidate preparation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from roleplay.grpo_candidates import (
    CandidateGenerationError,
    load_prompts,
    mlx_weight_key,
    require_hash,
)


class AdapterKeyConversionTests(unittest.TestCase):
    def test_maps_and_transposes_peft_lora_keys(self) -> None:
        prefix = "base_model.model.model.language_model.layers.3."
        self.assertEqual(
            mlx_weight_key(prefix + "self_attn.q_proj.lora_A.weight"),
            (
                "language_model.model.layers.3.self_attn.q_proj.lora_a",
                True,
            ),
        )
        self.assertEqual(
            mlx_weight_key(prefix + "mlp.down_proj.lora_B.weight"),
            ("language_model.model.layers.3.mlp.down_proj.lora_b", True),
        )

    def test_rejects_unknown_peft_key(self) -> None:
        with self.assertRaises(CandidateGenerationError):
            mlx_weight_key("model.layers.0.self_attn.q_proj.weight")


class FrozenInputTests(unittest.TestCase):
    def test_requires_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "input.txt"
            path.write_text("frozen", encoding="utf-8")
            expected = hashlib.sha256(b"frozen").hexdigest()
            require_hash(path, expected)
            with self.assertRaisesRegex(CandidateGenerationError, "哈希不匹配"):
                require_hash(path, "0" * 64)

    def test_loads_requested_prompts_in_requested_order(self) -> None:
        rows = [
            {"id": "p1", "user": "一"},
            {"id": "p2", "user": "二"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompts.jsonl"
            content = "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows
            )
            path.write_text(content, encoding="utf-8")
            expected = hashlib.sha256(content.encode()).hexdigest()
            from unittest.mock import patch

            with patch("roleplay.grpo_candidates.PROMPTS_SHA256", expected):
                loaded = load_prompts(path, ["p2", "p1"])
            self.assertEqual([row["id"] for row in loaded], ["p2", "p1"])

    def test_rejects_duplicate_or_missing_prompt_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompts.jsonl"
            content = json.dumps({"id": "p1", "user": "一"}) + "\n"
            path.write_text(content, encoding="utf-8")
            expected = hashlib.sha256(content.encode()).hexdigest()
            from unittest.mock import patch

            with patch("roleplay.grpo_candidates.PROMPTS_SHA256", expected):
                with self.assertRaisesRegex(CandidateGenerationError, "不得重复"):
                    load_prompts(path, ["p1", "p1"])
                with self.assertRaisesRegex(CandidateGenerationError, "找不到"):
                    load_prompts(path, ["missing"])


if __name__ == "__main__":
    unittest.main()
