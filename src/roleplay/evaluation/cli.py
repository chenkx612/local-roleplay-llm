"""Command-line entry point for the unified mechanical evaluation layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from roleplay.core.artifacts import read_jsonl, write_json_atomic
from roleplay.sft_eval import evaluate_core_behavior_gate, summarize_outputs


def _grid_axes(rows: Sequence[dict[str, Any]]) -> tuple[list[int], list[str]]:
    seeds = sorted(
        {
            row.get("seed")
            for row in rows
            if isinstance(row.get("seed"), int)
            and not isinstance(row.get("seed"), bool)
        }
    )
    ids = sorted(
        {
            row.get("id")
            for row in rows
            if isinstance(row.get("id"), str) and row["id"].strip()
        }
    )
    if not seeds or not ids:
        raise ValueError("输出必须包含非空的整数 seed 和字符串 id")
    return seeds, ids


def inspect_outputs(path: Path) -> dict[str, Any]:
    """Summarize one complete output grid using the unified checks."""
    rows = read_jsonl(path)
    seeds, ids = _grid_axes(rows)
    return {
        "schema_version": 1,
        "source": str(path),
        "evaluation": summarize_outputs(rows, seeds, ids),
    }


def compare_outputs(baseline: Path, candidate: Path) -> dict[str, Any]:
    """Evaluate one aligned candidate grid against its baseline."""
    baseline_rows = read_jsonl(baseline)
    candidate_rows = read_jsonl(candidate)
    seeds, ids = _grid_axes(baseline_rows)
    candidate_axes = _grid_axes(candidate_rows)
    if candidate_axes != (seeds, ids):
        raise ValueError("baseline 与 candidate 的 seed/id 网格不一致")
    return {
        "schema_version": 1,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "gate": evaluate_core_behavior_gate(
            baseline_rows,
            candidate_rows,
            seeds,
            ids,
        ),
        "summaries": {
            "baseline": summarize_outputs(baseline_rows, seeds, ids),
            "candidate": summarize_outputs(candidate_rows, seeds, ids),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一检查或比较角色扮演模型输出"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="检查一个完整输出网格"
    )
    inspect_parser.add_argument("--input", type=Path, required=True)
    inspect_parser.add_argument("--output", type=Path)
    compare_parser = subparsers.add_parser(
        "compare", help="对齐比较 baseline 与 candidate"
    )
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_outputs(args.input)
        else:
            result = compare_outputs(args.baseline, args.candidate)
        if args.output is not None:
            write_json_atomic(args.output, result)
            print(f"评测结果: {args.output}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

