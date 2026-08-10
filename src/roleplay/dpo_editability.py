"""Assess whether SFT outputs can become excellent through minimal edits."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from roleplay.dpo_data import (
    DPODataError,
    OUTPUT_ROOT,
    PERSONA_PATH,
    ROOT,
    STYLE_EXAMPLES_PATH,
    SYSTEM_PROMPT_PATH,
    _validate_candidate_rows,
    is_stable_candidate,
    load_jsonl,
    load_prompts,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_SOURCE_RUN = OUTPUT_ROOT / "20260810-dpo-data-3"
DEFAULT_EXPERIMENT_DIR = OUTPUT_ROOT / "20260811-dpo-editability-1"
DEFAULT_TRAINING_OUTPUT = (
    ROOT / "data/runs/morgana-v2/dpo_train_editability17.jsonl"
)
ORDER_SEED = 20260811
MIN_SIMILARITY = 0.65
MIN_LENGTH_RATIO = 0.80
MAX_LENGTH_RATIO = 1.20
MIN_SUCCESSFUL_PAIRS = 20
STRONG_SUCCESSFUL_PAIRS = 30
EDITABILITY_TRAIN_PAIR_COUNT = 17
SCORE_DIMENSIONS = (
    "generation_stability",
    "role_consistency",
    "dialogue_quality",
)
TARGET_PREFERENCES = {
    "role_naturalness": "role_consistency",
    "emotion_attunement": "dialogue_quality",
    "expression_naturalness": "dialogue_quality",
    "dialogue_continuation": "dialogue_quality",
}
BLOCKING_REASONS = frozenset(
    {
        "no_qualified_source",
        "multiple_defects",
        "requires_large_rewrite",
        "no_clear_subjective_upgrade",
    }
)

CONTRACT = {
    "schema_version": 1,
    "reviewer": "codex",
    "minimum_similarity": MIN_SIMILARITY,
    "length_ratio": [MIN_LENGTH_RATIO, MAX_LENGTH_RATIO],
    "source_quality": {
        "generation_stability": 8,
        "target_dimension": 7,
        "other_subjective_dimension": 8,
    },
    "chosen_quality": {dimension: 8 for dimension in SCORE_DIMENSIONS},
    "minimum_successful_pairs": MIN_SUCCESSFUL_PAIRS,
    "strong_successful_pairs": STRONG_SUCCESSFUL_PAIRS,
    "target_preferences": TARGET_PREFERENCES,
}


class DPOEditabilityError(RuntimeError):
    """Raised when the editability experiment contract is violated."""


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_review_artifacts(
    prompts: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    *,
    order_seed: int = ORDER_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build an anonymous packet containing every stable candidate."""
    rng = random.Random(order_seed)
    packet_items = []
    key_items = []
    result_items = []
    for index, prompt in enumerate(prompts, 1):
        stable = [
            row
            for row in candidates
            if row["prompt_id"] == prompt["id"] and is_stable_candidate(row)
        ]
        if not stable:
            raise DPOEditabilityError(f"{prompt['id']} 没有稳定候选")
        rng.shuffle(stable)
        labels = {
            chr(ord("A") + label_index): row
            for label_index, row in enumerate(stable)
        }
        review_id = f"editability-review-{index:02d}"
        packet_items.append(
            {
                "review_id": review_id,
                "prompt_id": prompt["id"],
                "scenario": prompt["scenario"],
                "user": prompt["user"],
                "answers": {
                    label: row["assistant"] for label, row in labels.items()
                },
            }
        )
        key_items.append(
            {
                "review_id": review_id,
                "prompt_id": prompt["id"],
                "labels": {
                    label: {
                        "candidate_id": row["candidate_id"],
                        "assistant": row["assistant"],
                    }
                    for label, row in labels.items()
                },
            }
        )
        result_items.append(
            {
                "review_id": review_id,
                "decision": None,
                "source_label": None,
                "source_scores": None,
                "target_preference": None,
                "improved_assistant": None,
                "improved_scores": None,
                "changes": "",
                "blocking_reasons": [],
                "notes": "",
            }
        )
    packet = {
        "schema_version": 1,
        "reviewer": "codex",
        "instructions": (
            "为每题选择最接近优秀答案的稳定候选。只有候选本身已合格、"
            "仅有一个主观偏好弱点，且能通过局部编辑达到 8/8/8 时，才标记 "
            "successful_minimal_edit；否则标记 "
            "not_editable。不得修复硬错误或同时改动多个能力。"
        ),
        "contract": CONTRACT,
        "references": {
            "persona": str(PERSONA_PATH.relative_to(ROOT)),
            "style_examples": str(STYLE_EXAMPLES_PATH.relative_to(ROOT)),
            "system_prompt": str(SYSTEM_PROMPT_PATH.relative_to(ROOT)),
        },
        "items": packet_items,
    }
    key = {"schema_version": 1, "order_seed": order_seed, "items": key_items}
    results = {"schema_version": 1, "reviewer": "codex", "results": result_items}
    return packet, key, results


def prepare_experiment(
    *,
    source_run: Path = DEFAULT_SOURCE_RUN,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
) -> Path:
    """Create a frozen anonymous Codex packet from an existing candidate run."""
    source_run = source_run.resolve()
    experiment_dir = experiment_dir.resolve()
    candidates_path = source_run / "candidates.jsonl"
    source_summary_path = source_run / "run_summary.json"
    if not candidates_path.is_file() or not source_summary_path.is_file():
        raise DPOEditabilityError(f"候选 run 不完整: {source_run}")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    expected_hash = source_summary.get("artifacts", {}).get(
        "candidates.jsonl", {}
    ).get("sha256")
    candidates_hash = sha256_file(candidates_path)
    if expected_hash != candidates_hash:
        raise DPOEditabilityError("候选文件与来源 run 的冻结哈希不一致")
    prompts = load_prompts()
    candidates = load_jsonl(candidates_path)
    _validate_candidate_rows(candidates, prompts)
    packet, key, results = build_review_artifacts(prompts, candidates)
    summary_path = experiment_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("contract") != CONTRACT
            or summary.get("source_candidates_sha256") != candidates_hash
        ):
            raise DPOEditabilityError("现有判定实验与当前契约或候选不一致")
        print(f"判定实验已存在（{summary['status']}）: {experiment_dir}")
        return experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=False)
    packet_path = experiment_dir / "editability_packet.json"
    key_path = experiment_dir / "editability_key.json"
    results_path = experiment_dir / "editability_results.json"
    write_json_atomic(packet_path, packet)
    write_json_atomic(key_path, key)
    write_json_atomic(results_path, results)
    summary = {
        "schema_version": 1,
        "status": "awaiting_codex_review",
        "contract": CONTRACT,
        "source_run": str(source_run),
        "source_candidates_sha256": candidates_hash,
        "counts": {
            "prompts": len(prompts),
            "stable_candidates": sum(map(is_stable_candidate, candidates)),
        },
        "artifacts": {
            path.name: _artifact(path, experiment_dir)
            for path in (packet_path, key_path)
        },
    }
    write_json_atomic(summary_path, summary)
    print(f"Codex editability packet: {packet_path}")
    return experiment_dir


def _validate_scores(value: Any, review_id: str, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(SCORE_DIMENSIONS):
        raise DPOEditabilityError(f"{review_id} {label} 评分维度不完整")
    if any(
        isinstance(score, bool)
        or not isinstance(score, int)
        or not 0 <= score <= 10
        for score in value.values()
    ):
        raise DPOEditabilityError(f"{review_id} {label} 评分必须是 0～10 整数")
    return value


def evaluate_result(
    result: dict[str, Any], labels: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Validate one Codex decision and calculate objective edit distances."""
    expected_fields = {
        "review_id",
        "decision",
        "source_label",
        "source_scores",
        "target_preference",
        "improved_assistant",
        "improved_scores",
        "changes",
        "blocking_reasons",
        "notes",
    }
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise DPOEditabilityError("Codex 判定字段不正确")
    review_id = result["review_id"]
    if not isinstance(review_id, str) or result["source_label"] not in labels:
        raise DPOEditabilityError(f"{review_id} source_label 无效")
    source_scores = _validate_scores(
        result["source_scores"], review_id, "source"
    )
    source = labels[result["source_label"]]["assistant"].strip()
    decision = result["decision"]
    blocking = result["blocking_reasons"]
    notes = result["notes"]
    if (
        not isinstance(blocking, list)
        or len(blocking) != len(set(blocking))
        or any(reason not in BLOCKING_REASONS for reason in blocking)
        or not isinstance(notes, str)
        or not notes.strip()
        or not isinstance(result["changes"], str)
    ):
        raise DPOEditabilityError(f"{review_id} reasons/notes/changes 无效")
    if decision == "not_editable":
        if (
            result["target_preference"] is not None
            or result["improved_assistant"] is not None
            or result["improved_scores"] is not None
            or result["changes"]
            or not blocking
        ):
            raise DPOEditabilityError(f"{review_id} not_editable 语义不一致")
        return {**result, "notes": notes.strip()}
    if decision != "successful_minimal_edit":
        raise DPOEditabilityError(f"{review_id} decision 无效")
    target = result["target_preference"]
    improved = result["improved_assistant"]
    if (
        target not in TARGET_PREFERENCES
        or not isinstance(improved, str)
        or not improved.strip()
        or not result["changes"].strip()
        or blocking
    ):
        raise DPOEditabilityError(
            f"{review_id} successful_minimal_edit 语义不一致"
        )
    improved = improved.strip()
    improved_scores = _validate_scores(
        result["improved_scores"], review_id, "improved"
    )
    target_dimension = TARGET_PREFERENCES[target]
    other_dimension = (
        "dialogue_quality"
        if target_dimension == "role_consistency"
        else "role_consistency"
    )
    if (
        source_scores["generation_stability"] < 8
        or source_scores[target_dimension] != 7
        or source_scores[other_dimension] < 8
    ):
        raise DPOEditabilityError(
            f"{review_id} source 不是合格单缺陷硬负样本"
        )
    if any(improved_scores[name] < 8 for name in SCORE_DIMENSIONS):
        raise DPOEditabilityError(f"{review_id} improved 未达到 8/8/8")
    if (
        improved_scores[target_dimension] < source_scores[target_dimension] + 1
        or improved_scores[other_dimension] != source_scores[other_dimension]
        or improved_scores["generation_stability"]
        != source_scores["generation_stability"]
    ):
        raise DPOEditabilityError(
            f"{review_id} 编辑不满足单一目标维度改进"
        )
    if source == improved or not is_stable_candidate(
        {"assistant": improved, "finish_reason": "stop"}
    ):
        raise DPOEditabilityError(f"{review_id} improved 不稳定或未发生修改")
    similarity = SequenceMatcher(
        None, source, improved, autojunk=False
    ).ratio()
    length_ratio = len(improved) / len(source)
    if similarity < MIN_SIMILARITY:
        raise DPOEditabilityError(
            f"{review_id} 相似度 {similarity:.3f} 低于 {MIN_SIMILARITY:.2f}"
        )
    if not MIN_LENGTH_RATIO <= length_ratio <= MAX_LENGTH_RATIO:
        raise DPOEditabilityError(
            f"{review_id} 长度比 {length_ratio:.3f} 超出允许范围"
        )
    return {
        **result,
        "improved_assistant": improved,
        "changes": result["changes"].strip(),
        "notes": notes.strip(),
        "similarity": similarity,
        "length_ratio": length_ratio,
    }


def finalize_experiment(*, experiment_dir: Path) -> tuple[Path, Path]:
    """Validate all Codex edits and write an inspectable experiment report."""
    experiment_dir = experiment_dir.resolve()
    summary_path = experiment_dir / "run_summary.json"
    if not summary_path.is_file():
        raise DPOEditabilityError(f"缺少实验摘要: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract") != CONTRACT:
        raise DPOEditabilityError("判定实验契约不一致")
    for name in ("editability_packet.json", "editability_key.json"):
        path = experiment_dir / name
        if summary.get("artifacts", {}).get(name, {}).get("sha256") != sha256_file(
            path
        ):
            raise DPOEditabilityError(f"{name} 在评审期间发生变化")
    packet = json.loads(
        (experiment_dir / "editability_packet.json").read_text(encoding="utf-8")
    )
    key = json.loads(
        (experiment_dir / "editability_key.json").read_text(encoding="utf-8")
    )
    submitted = json.loads(
        (experiment_dir / "editability_results.json").read_text(encoding="utf-8")
    )
    packet_by_id = {row["review_id"]: row for row in packet.get("items", [])}
    key_by_id = {row["review_id"]: row for row in key.get("items", [])}
    result_by_id = {
        row.get("review_id"): row
        for row in submitted.get("results", [])
        if isinstance(row, dict)
    }
    if (
        len(packet_by_id) != 40
        or set(packet_by_id) != set(key_by_id)
        or set(packet_by_id) != set(result_by_id)
    ):
        raise DPOEditabilityError("packet/key/results 未完整对齐 40 条 Prompt")
    evaluated = []
    pairs = []
    for review_id, packet_row in packet_by_id.items():
        labels = key_by_id[review_id]["labels"]
        if set(labels) != set(packet_row["answers"]):
            raise DPOEditabilityError(f"{review_id} 匿名映射不一致")
        parsed = evaluate_result(result_by_id[review_id], labels)
        item = {
            **parsed,
            "prompt_id": packet_row["prompt_id"],
            "scenario": packet_row["scenario"],
            "source_candidate_id": labels[parsed["source_label"]]["candidate_id"],
        }
        evaluated.append(item)
        if parsed["decision"] == "successful_minimal_edit":
            pairs.append(
                {
                    "review_id": review_id,
                    "prompt_id": packet_row["prompt_id"],
                    "scenario": packet_row["scenario"],
                    "user": packet_row["user"],
                    "source_candidate_id": item["source_candidate_id"],
                    "target_preference": parsed["target_preference"],
                    "rejected_response": labels[parsed["source_label"]]["assistant"],
                    "chosen_response": parsed["improved_assistant"],
                    "source_scores": parsed["source_scores"],
                    "chosen_scores": parsed["improved_scores"],
                    "similarity": parsed["similarity"],
                    "length_ratio": parsed["length_ratio"],
                    "changes": parsed["changes"],
                }
            )
    success_count = len(pairs)
    outcome = (
        "strong_sft_foundation"
        if success_count >= STRONG_SUCCESSFUL_PAIRS
        else "viable_sft_foundation"
        if success_count >= MIN_SUCCESSFUL_PAIRS
        else "insufficient_sft_foundation"
    )
    scenario_counts = Counter(row["scenario"] for row in pairs)
    target_counts = Counter(row["target_preference"] for row in pairs)
    blocking_counts = Counter(
        reason
        for row in evaluated
        for reason in row["blocking_reasons"]
    )
    report = {
        "schema_version": 1,
        "outcome": outcome,
        "contract": CONTRACT,
        "counts": {
            "prompts": len(evaluated),
            "successful_minimal_edits": success_count,
            "not_editable": len(evaluated) - success_count,
        },
        "successful_by_scenario": dict(sorted(scenario_counts.items())),
        "successful_by_target": dict(sorted(target_counts.items())),
        "blocking_reasons": dict(sorted(blocking_counts.items())),
        "mean_similarity": (
            sum(row["similarity"] for row in pairs) / success_count
            if pairs
            else None
        ),
        "mean_length_ratio": (
            sum(row["length_ratio"] for row in pairs) / success_count
            if pairs
            else None
        ),
        "items": evaluated,
    }
    report_path = experiment_dir / "editability_report.json"
    pairs_path = experiment_dir / "editability_pairs.jsonl"
    if report_path.exists() or pairs_path.exists():
        raise DPOEditabilityError("判定实验报告已存在，拒绝覆盖")
    write_jsonl_atomic(pairs_path, pairs)
    write_json_atomic(report_path, report)
    summary["status"] = "completed"
    summary["result"] = {
        "outcome": outcome,
        "successful_minimal_edits": success_count,
        "report_sha256": sha256_file(report_path),
        "pairs_sha256": sha256_file(pairs_path),
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"outcome={outcome}")
    return report_path, pairs_path


def export_training_data(
    *,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    output_path: Path = DEFAULT_TRAINING_OUTPUT,
    system_prompt_path: Path = SYSTEM_PROMPT_PATH,
) -> Path:
    """Export the verified minimal-edit pairs in ms-swift DPO format."""
    experiment_dir = experiment_dir.resolve()
    output_path = output_path.resolve()
    summary_path = experiment_dir / "run_summary.json"
    report_path = experiment_dir / "editability_report.json"
    pairs_path = experiment_dir / "editability_pairs.jsonl"
    if not all(path.is_file() for path in (summary_path, report_path, pairs_path)):
        raise DPOEditabilityError("判定实验尚未完成，不能导出训练集")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = summary.get("result", {})
    if (
        summary.get("status") != "completed"
        or report.get("contract") != CONTRACT
        or result.get("report_sha256") != sha256_file(report_path)
        or result.get("pairs_sha256") != sha256_file(pairs_path)
    ):
        raise DPOEditabilityError("判定实验报告或 pair 哈希不一致")
    pairs = load_jsonl(pairs_path)
    successful_ids = {
        row["review_id"]
        for row in report.get("items", [])
        if row.get("decision") == "successful_minimal_edit"
    }
    pair_ids = [row.get("review_id") for row in pairs]
    if (
        len(pairs) != EDITABILITY_TRAIN_PAIR_COUNT
        or report.get("counts", {}).get("successful_minimal_edits")
        != EDITABILITY_TRAIN_PAIR_COUNT
        or len(pair_ids) != len(set(pair_ids))
        or set(pair_ids) != successful_ids
    ):
        raise DPOEditabilityError("训练 pair 与 17 条判定成功结果不一致")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise DPOEditabilityError("system prompt 为空")
    required_fields = {
        "review_id",
        "prompt_id",
        "scenario",
        "user",
        "source_candidate_id",
        "target_preference",
        "rejected_response",
        "chosen_response",
        "source_scores",
        "chosen_scores",
        "similarity",
        "length_ratio",
        "changes",
    }
    training_rows = []
    for index, pair in enumerate(pairs, 1):
        if set(pair) != required_fields:
            raise DPOEditabilityError(f"第 {index} 条训练 pair 字段不正确")
        chosen = pair["chosen_response"]
        rejected = pair["rejected_response"]
        user = pair["user"]
        if (
            not all(
                isinstance(value, str) and value.strip()
                for value in (user, chosen, rejected)
            )
            or chosen == rejected
            or pair["similarity"] < MIN_SIMILARITY
            or not MIN_LENGTH_RATIO
            <= pair["length_ratio"]
            <= MAX_LENGTH_RATIO
            or any(
                pair.get("chosen_scores", {}).get(dimension, -1) < 8
                for dimension in SCORE_DIMENSIONS
            )
        ):
            raise DPOEditabilityError(f"第 {index} 条训练 pair 不满足判定合同")
        training_rows.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user.strip()},
                    {"role": "assistant", "content": chosen.strip()},
                ],
                "rejected_response": rejected.strip(),
            }
        )
    if output_path.exists():
        if load_jsonl(output_path) != training_rows:
            raise DPOEditabilityError(f"拒绝覆盖不同的训练集: {output_path}")
        print(f"DPO 训练集已存在: {output_path}")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_path, training_rows)
    print(f"DPO 训练集: {output_path}")
    print(f"pairs={len(training_rows)} sha256={sha256_file(output_path)}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 SFT chosen 最小编辑判定实验"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="生成匿名 Codex 最小编辑评审包"
    )
    prepare.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    prepare.add_argument(
        "--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR
    )
    finalize = subparsers.add_parser(
        "finalize", help="校验编辑并生成判定报告"
    )
    finalize.add_argument("--experiment-dir", type=Path, required=True)
    export = subparsers.add_parser(
        "export-training", help="导出 17 条冻结 DPO 训练集"
    )
    export.add_argument(
        "--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR
    )
    export.add_argument("--output", type=Path, default=DEFAULT_TRAINING_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            prepare_experiment(
                source_run=args.source_run,
                experiment_dir=args.experiment_dir,
            )
        elif args.command == "finalize":
            report, pairs = finalize_experiment(
                experiment_dir=args.experiment_dir
            )
            print(f"Report: {report}")
            print(f"Pairs: {pairs}")
        else:
            export_training_data(
                experiment_dir=args.experiment_dir,
                output_path=args.output,
            )
    except (
        DPODataError,
        DPOEditabilityError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"DPO editability failed: {exc}") from None


if __name__ == "__main__":
    main()
