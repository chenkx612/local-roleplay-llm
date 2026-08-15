"""Tests for the unified pipeline command and architectural boundaries."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roleplay import cli
from roleplay.evaluation.cli import compare_outputs, inspect_outputs


def output_row(seed: int, record_id: str, assistant: str) -> dict:
    return {
        "seed": seed,
        "id": record_id,
        "scenario": "daily",
        "target_goals": ["role_consistency"],
        "user": "你好",
        "assistant": assistant,
        "finish_reason": "stop",
        "attempts": 1,
    }


class UnifiedCommandTests(unittest.TestCase):
    def test_forwards_nested_training_arguments(self):
        calls = []

        def command(arguments):
            calls.append(arguments)
            return 7

        with patch("roleplay.cli._resolve", return_value=command) as resolve:
            result = cli.main(
                ["train", "sft", "run", "--output-root", "output/test"]
            )

        self.assertEqual(result, 7)
        self.assertEqual(
            resolve.call_args.args[0],
            ("roleplay.stage2_sft", "main"),
        )
        self.assertEqual(
            calls,
            [["run", "--output-root", "output/test"]],
        )

    def test_prefixes_unified_evaluation_subcommand(self):
        calls = []

        def command(arguments):
            calls.append(arguments)

        with patch("roleplay.cli._resolve", return_value=command):
            self.assertEqual(
                cli.main(["eval", "inspect", "--input", "outputs.jsonl"]),
                0,
            )

        self.assertEqual(
            calls,
            [["inspect", "--input", "outputs.jsonl"]],
        )


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_stages_do_not_depend_on_other_stage_implementations(self):
        source_root = Path(__file__).resolve().parents[1] / "src/roleplay"
        violations = []
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                if node.module.startswith("roleplay.stage"):
                    violations.append(f"{path.name}:{node.lineno}:{node.module}")
        self.assertEqual(violations, [])

    def test_modules_do_not_import_private_cross_module_names(self):
        source_root = Path(__file__).resolve().parents[1] / "src/roleplay"
        violations = []
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                for alias in node.names:
                    if alias.name.startswith("_"):
                        violations.append(
                            f"{path.name}:{node.lineno}:{alias.name}"
                        )
        self.assertEqual(violations, [])


class EvaluationCommandTests(unittest.TestCase):
    @staticmethod
    def write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows
            ),
            encoding="utf-8",
        )

    def test_inspects_complete_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "outputs.jsonl"
            self.write_rows(
                path,
                [output_row(11, "dev_0001", "（抬眼）你好。")],
            )

            result = inspect_outputs(path)

        self.assertEqual(result["evaluation"]["overall"]["records"], 1)
        self.assertEqual(
            result["evaluation"]["overall"]["strict_format_rate"], 1.0
        )

    def test_compares_aligned_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            rows = [output_row(11, "dev_0001", "（抬眼）你好。")]
            self.write_rows(baseline, rows)
            self.write_rows(candidate, rows)

            result = compare_outputs(baseline, candidate)

        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(
            result["summaries"]["candidate"]["overall"]["records"], 1
        )

    def test_rejects_misaligned_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            self.write_rows(
                baseline,
                [output_row(11, "dev_0001", "（抬眼）你好。")],
            )
            self.write_rows(
                candidate,
                [output_row(12, "dev_0001", "（抬眼）你好。")],
            )

            with self.assertRaisesRegex(ValueError, "网格不一致"):
                compare_outputs(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
