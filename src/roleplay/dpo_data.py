"""Prepare and freeze the second morgana-v2 DPO preference dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from roleplay.grpo_candidates import (
    BASE_MODEL,
    BASE_REVISION,
    BASE_SNAPSHOT,
    MAX_TOKENS,
    REPETITION_CONTEXT_SIZE,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    _generate_one,
    _verify_loaded_adapter,
    convert_peft_adapter_to_mlx,
)
from roleplay.sft_eval import (
    has_gibberish,
    has_repeated_span,
    has_unclosed_brackets,
    normalize_empty_think_wrapper,
)


ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = ROOT / "data/runs/morgana-v2/dpo_prompts_v2.jsonl"
PERSONA_PATH = ROOT / "data/runs/morgana-v2/inputs/persona.json"
STYLE_EXAMPLES_PATH = ROOT / "data/runs/morgana-v2/inputs/style_examples.jsonl"
SYSTEM_PROMPT_PATH = ROOT / "data/runs/morgana-v2/system_prompt.txt"
SFT_ADAPTER_PATH = ROOT / "output/morgana-v2/stage2-sft/4/adapter"
OUTPUT_ROOT = ROOT / "output/morgana-v2/stage3-dpo/data"
DEFAULT_RUN_ID = "20260810-dpo-data-2"
DEFAULT_TRAIN_OUTPUT = ROOT / "data/runs/morgana-v2/dpo_train_v2.jsonl"
DEFAULT_AUDIT_OUTPUT = ROOT / "data/runs/morgana-v2/dpo_train_audit_v2.json"

FROZEN_HASHES = {
    PROMPTS_PATH: "f60f27d858a1fb333d14f3c79d10d45086642a57475bfcdb816f15745bc1c0c7",
    PERSONA_PATH: "42010082a1db9afcbf15cfed077dd59d4c7a0a0d8f44510292f00fe7ef87a10a",
    STYLE_EXAMPLES_PATH: "b8fa53f494d3202fe80aad90be4e2db098f56ac080bc152644ba3661e6104250",
    SYSTEM_PROMPT_PATH: "d88993aaa1178ced740f6b54530a27e5fcdb2486a66d8b460367e842b53ee112",
    SFT_ADAPTER_PATH / "adapter_model.safetensors": (
        "617e6e00535fa356272d32fb16d8fe8d451a9c3cfd2f766f56af02cdf2f9b78d"
    ),
    SFT_ADAPTER_PATH / "adapter_config.json": (
        "67f3ab10168164cc014c7f9c8720984760b1d94be33fde9abf89fb3004a886dc"
    ),
    SFT_ADAPTER_PATH / "additional_config.json": (
        "2b7ed6cc0ca6c21dc39bf80fd3351f5f87c462df23a237dd6ad20473eb9a33a2"
    ),
}

CANDIDATES_PER_PROMPT = 2
BASE_SEED = 20260810
REVIEW_ORDER_SEED = 20260810
MIN_FINAL_PAIRS = 30
EXPECTED_PROMPT_COUNT = 40
EXPECTED_SCENARIOS = frozenset(
    {"daily", "background", "emotion", "style", "adversarial"}
)
PREFERENCE_REASONS = frozenset(
    {
        "character_consistency",
        "expression_naturalness",
        "emotion_response",
        "dialogue_continuation",
    }
)
CODEX_DECISIONS = frozenset(
    {"clear_preference", "no_clear_preference", "teacher_edit"}
)
PROMPT_ID_PATTERN = re.compile(r"dpo2_(\d{4})\Z")
MIN_TEACHER_SIMILARITY = 0.60
MIN_TEACHER_LENGTH_RATIO = 0.70
MAX_TEACHER_LENGTH_RATIO = 1.30

GENERATION = {
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "repetition_penalty": REPETITION_PENALTY,
    "repetition_context_size": REPETITION_CONTEXT_SIZE,
    "enable_thinking": False,
}

CONTRACT = {
    "schema_version": 2,
    "reviewer": "codex_artifact_handoff",
    "candidates_per_prompt": CANDIDATES_PER_PROMPT,
    "candidate_rounds": 1,
    "generation": GENERATION,
    "expected_prompts": EXPECTED_PROMPT_COUNT,
    "minimum_final_pairs": MIN_FINAL_PAIRS,
    "teacher_pair_fraction_max": "1/3",
    "teacher_edit": {
        "minimum_similarity": MIN_TEACHER_SIMILARITY,
        "length_ratio": [MIN_TEACHER_LENGTH_RATIO, MAX_TEACHER_LENGTH_RATIO],
    },
}


class DPODataError(RuntimeError):
    """Raised when the frozen DPO data contract cannot be honored."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise DPODataError(f"缺少冻结文件: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise DPODataError(f"冻结文件哈希不匹配: {path}，{actual} != {expected}")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DPODataError(f"{path}:{line_number} 不是有效 JSON") from exc
            if not isinstance(row, dict):
                raise DPODataError(f"{path}:{line_number} 不是对象")
            rows.append(row)
    return rows


def _canonical_user(text: str) -> str:
    return " ".join(text.strip().split())


def load_prompts(path: Path = PROMPTS_PATH) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if len(rows) != EXPECTED_PROMPT_COUNT:
        raise DPODataError(
            f"DPO v2 Prompt 必须为 {EXPECTED_PROMPT_COUNT} 条，实际 {len(rows)}"
        )
    ids: set[str] = set()
    users: set[str] = set()
    scenarios: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        if set(row) != {"id", "scenario", "target_goals", "user"}:
            raise DPODataError(f"DPO v2 Prompt 第 {index} 条字段不正确")
        prompt_id = row["id"]
        match = (
            PROMPT_ID_PATTERN.fullmatch(prompt_id)
            if isinstance(prompt_id, str)
            else None
        )
        if not match or int(match.group(1)) != index:
            raise DPODataError(f"DPO v2 Prompt 第 {index} 条 id 不正确")
        user = row["user"]
        scenario = row["scenario"]
        if not isinstance(user, str) or not user.strip():
            raise DPODataError(f"DPO v2 Prompt 第 {index} 条 user 无效")
        if scenario not in EXPECTED_SCENARIOS:
            raise DPODataError(f"DPO v2 Prompt 第 {index} 条 scenario 无效")
        if row["target_goals"] != [
            "generation_stability",
            "character_consistency",
            "dialogue_quality",
        ]:
            raise DPODataError(f"DPO v2 Prompt 第 {index} 条 target_goals 无效")
        normalized = _canonical_user(user)
        if prompt_id in ids or normalized in users:
            raise DPODataError("DPO v2 Prompt 包含重复 id 或 user")
        ids.add(prompt_id)
        users.add(normalized)
        scenarios[scenario] += 1
    if set(scenarios) != EXPECTED_SCENARIOS or set(scenarios.values()) != {8}:
        raise DPODataError(f"DPO v2 Prompt 场景必须各 8 条: {dict(scenarios)}")
    return rows


def validate_cross_split_uniqueness(
    prompts: Sequence[dict[str, Any]],
    other_paths: Sequence[Path] | None = None,
) -> None:
    paths = other_paths or (
        ROOT / "data/runs/morgana-v2/sft_train_prompts.jsonl",
        ROOT / "data/runs/morgana-v2/dev.jsonl",
        ROOT / "data/runs/morgana-v2/eval.jsonl",
    )
    dpo_users = {_canonical_user(row["user"]) for row in prompts}
    for path in paths:
        for row in load_jsonl(path):
            user = row.get("user")
            if isinstance(user, str) and _canonical_user(user) in dpo_users:
                raise DPODataError(
                    f"DPO v2 Prompt 与其他 split 重复: {path} user={user}"
                )


def candidate_seed(prompt_id: str, candidate_index: int) -> int:
    match = PROMPT_ID_PATTERN.fullmatch(prompt_id)
    if not match or not 1 <= candidate_index <= CANDIDATES_PER_PROMPT:
        raise ValueError("候选 seed 输入无效")
    prompt_index = int(match.group(1)) - 1
    return BASE_SEED + prompt_index * 100 + candidate_index - 1


def candidate_id(prompt_id: str, candidate_index: int) -> str:
    return f"{prompt_id}-c{candidate_index}"


def is_stable_candidate(record: dict[str, Any]) -> bool:
    answer = record.get("assistant")
    return bool(
        isinstance(answer, str)
        and answer.strip()
        and record.get("finish_reason") == "stop"
        and not has_gibberish(answer)
        and not has_repeated_span(answer)
        and not has_unclosed_brackets(answer)
    )


def _validate_candidate_rows(
    rows: Sequence[dict[str, Any]], prompts: Sequence[dict[str, Any]]
) -> None:
    prompt_by_id = {row["id"]: row for row in prompts}
    seen: set[str] = set()
    required = {
        "candidate_id",
        "prompt_id",
        "candidate_index",
        "seed",
        "assistant",
        "raw_assistant",
        "finish_reason",
        "source",
        "base_model",
        "base_revision",
        "adapter_sha256",
        "generation",
    }
    for row in rows:
        if set(row) != required:
            raise DPODataError("候选记录字段不正确")
        prompt_id = row["prompt_id"]
        if prompt_id not in prompt_by_id:
            raise DPODataError(f"候选引用未知 Prompt: {prompt_id}")
        expected_id = candidate_id(prompt_id, row["candidate_index"])
        expected_seed = candidate_seed(prompt_id, row["candidate_index"])
        if row["candidate_id"] != expected_id or row["seed"] != expected_seed:
            raise DPODataError(f"候选 id/seed 不正确: {row.get('candidate_id')}")
        if (
            row["source"] != "sft_candidate"
            or row["base_model"] != BASE_MODEL
            or row["base_revision"] != BASE_REVISION
            or row["adapter_sha256"]
            != FROZEN_HASHES[SFT_ADAPTER_PATH / "adapter_model.safetensors"]
            or row["generation"] != GENERATION
            or not isinstance(row["assistant"], str)
            or not isinstance(row["raw_assistant"], str)
            or row["assistant"]
            != normalize_empty_think_wrapper(row["raw_assistant"])
            or not isinstance(row["finish_reason"], str)
        ):
            raise DPODataError(f"候选内容或冻结配置不正确: {expected_id}")
        if expected_id in seen:
            raise DPODataError(f"候选 id 重复: {expected_id}")
        seen.add(expected_id)
    if len(rows) != len(prompts) * CANDIDATES_PER_PROMPT:
        raise DPODataError("候选总数不正确")
    counts = Counter(row["prompt_id"] for row in rows)
    if set(counts) != set(prompt_by_id) or set(counts.values()) != {
        CANDIDATES_PER_PROMPT
    }:
        raise DPODataError("每条 Prompt 必须恰好包含 2 个候选")


def candidate_set_sha256(candidates: Sequence[dict[str, Any]]) -> str:
    payload = [
        {"candidate_id": row["candidate_id"], "assistant": row["assistant"]}
        for row in candidates
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_candidates(
    *,
    prompts: Sequence[dict[str, Any]],
    base_model_path: Path,
    mlx_adapter_path: Path,
    system_prompt: str,
) -> list[dict[str, Any]]:
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise DPODataError("本地候选生成需要 mlx-lm") from exc
    model, tokenizer = load(
        str(base_model_path), adapter_path=str(mlx_adapter_path)
    )
    _verify_loaded_adapter(model, mlx_adapter_path)
    records: list[dict[str, Any]] = []
    total = len(prompts) * CANDIDATES_PER_PROMPT
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt["user"]},
        ]
        for index in range(1, CANDIDATES_PER_PROMPT + 1):
            seed = candidate_seed(prompt["id"], index)
            raw, finish_reason = _generate_one(model, tokenizer, messages, seed)
            assistant = normalize_empty_think_wrapper(raw)
            record = {
                "candidate_id": candidate_id(prompt["id"], index),
                "prompt_id": prompt["id"],
                "candidate_index": index,
                "seed": seed,
                "assistant": assistant,
                "raw_assistant": raw,
                "finish_reason": finish_reason,
                "source": "sft_candidate",
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "adapter_sha256": FROZEN_HASHES[
                    SFT_ADAPTER_PATH / "adapter_model.safetensors"
                ],
                "generation": GENERATION,
            }
            records.append(record)
            print(f"[{len(records)}/{total}] {record['candidate_id']}: {finish_reason}")
    return records


def _candidate_map(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in rows}


def build_codex_review_artifacts(
    *,
    prompts: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    order_seed: int = REVIEW_ORDER_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    rng = random.Random(order_seed)
    packet_items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for prompt in prompts:
        stable = [
            row
            for row in candidates
            if row["prompt_id"] == prompt["id"] and is_stable_candidate(row)
        ]
        if len(stable) != CANDIDATES_PER_PROMPT:
            unresolved.append(prompt["id"])
            continue
        rng.shuffle(stable)
        labels = {"A": stable[0], "B": stable[1]}
        review_id = f"dpo2-review-{len(packet_items) + 1:02d}"
        packet_items.append(
            {
                "review_id": review_id,
                "prompt_id": prompt["id"],
                "scenario": prompt["scenario"],
                "user": prompt["user"],
                "answer_a": labels["A"]["assistant"],
                "answer_b": labels["B"]["assistant"],
            }
        )
        key_items.append(
            {
                "review_id": review_id,
                "prompt_id": prompt["id"],
                "labels": {
                    label: {
                        "assistant": row["assistant"],
                        "source": row["source"],
                        "source_id": row["candidate_id"],
                    }
                    for label, row in labels.items()
                },
                "candidate_set_sha256": candidate_set_sha256(stable),
            }
        )
    packet = {
        "schema_version": 2,
        "reviewer": "codex",
        "instructions": (
            "逐条匿名裁决。两条均合格且存在清晰主观偏好时使用 "
            "clear_preference；平局、实质权衡或仅有硬规则差异时使用 "
            "no_clear_preference；两条都不适合作为 chosen 时，仅可对较好一条"
            "做最小修改并使用 teacher_edit。"
        ),
        "references": {
            "persona": str(PERSONA_PATH.relative_to(ROOT)),
            "style_examples": str(STYLE_EXAMPLES_PATH.relative_to(ROOT)),
            "system_prompt": str(SYSTEM_PROMPT_PATH.relative_to(ROOT)),
        },
        "allowed_preference_reasons": sorted(PREFERENCE_REASONS),
        "items": packet_items,
    }
    key = {
        "schema_version": 2,
        "order_seed": order_seed,
        "items": key_items,
    }
    results = {
        "schema_version": 2,
        "reviewer": "codex",
        "results": [
            {
                "review_id": row["review_id"],
                "decision": None,
                "source_label": None,
                "preference_reasons": [],
                "material_tradeoff": None,
                "hard_rule_only": None,
                "improved_assistant": None,
                "changes": "",
                "notes": "",
            }
            for row in packet_items
        ],
    }
    return packet, key, results, unresolved


def _validate_teacher_edit(source: str, improved: str, review_id: str) -> None:
    candidate = {"assistant": improved, "finish_reason": "stop"}
    if not is_stable_candidate(candidate):
        raise DPODataError(f"{review_id} Teacher 修改未通过稳定性检查")
    if improved.strip() == source.strip():
        raise DPODataError(f"{review_id} Teacher 修改不得与来源相同")
    source_length = len(source.strip())
    improved_length = len(improved.strip())
    ratio = improved_length / source_length if source_length else 0.0
    if not MIN_TEACHER_LENGTH_RATIO <= ratio <= MAX_TEACHER_LENGTH_RATIO:
        raise DPODataError(f"{review_id} Teacher 修改长度变化超过 30%")
    similarity = SequenceMatcher(
        None, source.strip(), improved.strip(), autojunk=False
    ).ratio()
    if similarity < MIN_TEACHER_SIMILARITY:
        raise DPODataError(
            f"{review_id} Teacher 修改相似度 {similarity:.3f} 低于 "
            f"{MIN_TEACHER_SIMILARITY:.2f}"
        )


def parse_codex_result(
    result: Any,
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise DPODataError("Codex 结果必须是对象")
    review_id = result.get("review_id")
    expected = {
        "review_id",
        "decision",
        "source_label",
        "preference_reasons",
        "material_tradeoff",
        "hard_rule_only",
        "improved_assistant",
        "changes",
        "notes",
    }
    if set(result) != expected or not isinstance(review_id, str):
        raise DPODataError(f"{review_id} Codex 结果字段不正确")
    decision = result["decision"]
    source_label = result["source_label"]
    reasons = result["preference_reasons"]
    tradeoff = result["material_tradeoff"]
    hard_rule_only = result["hard_rule_only"]
    improved = result["improved_assistant"]
    changes = result["changes"]
    notes = result["notes"]
    if decision not in CODEX_DECISIONS:
        raise DPODataError(f"{review_id} decision 无效")
    if (
        not isinstance(reasons, list)
        or any(reason not in PREFERENCE_REASONS for reason in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise DPODataError(f"{review_id} preference_reasons 无效")
    if type(tradeoff) is not bool or type(hard_rule_only) is not bool:
        raise DPODataError(f"{review_id} tradeoff/hard_rule_only 无效")
    if not isinstance(changes, str) or not isinstance(notes, str) or not notes.strip():
        raise DPODataError(f"{review_id} changes/notes 无效")
    if decision == "clear_preference":
        if (
            source_label not in labels
            or not reasons
            or tradeoff
            or hard_rule_only
            or improved is not None
            or changes
        ):
            raise DPODataError(f"{review_id} clear_preference 语义不一致")
    elif decision == "no_clear_preference":
        if source_label is not None or improved is not None or changes:
            raise DPODataError(f"{review_id} no_clear_preference 语义不一致")
    else:
        if (
            source_label not in labels
            or not reasons
            or tradeoff
            or hard_rule_only
            or not isinstance(improved, str)
            or not improved.strip()
            or not changes.strip()
        ):
            raise DPODataError(f"{review_id} teacher_edit 语义不一致")
        _validate_teacher_edit(
            labels[source_label]["assistant"], improved.strip(), review_id
        )
    return {
        **result,
        "improved_assistant": improved.strip() if isinstance(improved, str) else None,
        "changes": changes.strip(),
        "notes": notes.strip(),
    }


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _summary_contract() -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "input_hashes": {
            str(path.relative_to(ROOT)): expected
            for path, expected in FROZEN_HASHES.items()
        },
    }


def _load_or_create_summary(run_dir: Path, run_id: str) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    expected = _summary_contract()
    if path.is_file():
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("contract") != expected["contract"] or summary.get(
            "input_hashes"
        ) != expected["input_hashes"]:
            raise DPODataError("现有 run 与冻结 DPO v2 配置或输入不一致")
        return summary
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run": {
            "id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        **expected,
        "status": "preparing",
        "artifacts": {},
    }
    write_json_atomic(path, summary)
    return summary


def prepare_run(
    *,
    run_id: str = DEFAULT_RUN_ID,
    output_root: Path = OUTPUT_ROOT,
    base_model_path: Path = BASE_SNAPSHOT,
) -> Path:
    for path, expected in FROZEN_HASHES.items():
        require_hash(path, expected)
    prompts = load_prompts()
    validate_cross_split_uniqueness(prompts)
    if not base_model_path.is_dir():
        raise DPODataError(f"缺少本地 MLX 基座: {base_model_path}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise DPODataError("run-id 只能包含字母、数字、点、下划线和连字符")
    run_dir = output_root / run_id
    summary = _load_or_create_summary(run_dir, run_id)
    if summary.get("status") in {
        "awaiting_codex_review",
        "insufficient_candidates",
        "ready_for_dpo",
    }:
        print(f"DPO v2 run 已存在（{summary['status']}）: {run_dir}")
        return run_dir

    candidates_path = run_dir / "candidates.jsonl"
    candidates = load_jsonl(candidates_path)
    if not candidates:
        mlx_adapter = convert_peft_adapter_to_mlx(
            SFT_ADAPTER_PATH, run_dir / "mlx-sft-adapter"
        )
        candidates = generate_candidates(
            prompts=prompts,
            base_model_path=base_model_path,
            mlx_adapter_path=mlx_adapter,
            system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        )
        write_jsonl_atomic(candidates_path, candidates)
    _validate_candidate_rows(candidates, prompts)

    packet, key, results, unresolved = build_codex_review_artifacts(
        prompts=prompts,
        candidates=candidates,
    )
    packet_path = run_dir / "codex_review_packet.json"
    key_path = run_dir / "codex_review_key.json"
    results_path = run_dir / "codex_review_results.json"
    write_json_atomic(packet_path, packet)
    write_json_atomic(key_path, key)
    if not results_path.is_file():
        write_json_atomic(results_path, results)
    pair_count = len(packet["items"])
    summary.update(
        {
            "status": (
                "awaiting_codex_review"
                if pair_count >= MIN_FINAL_PAIRS
                else "insufficient_candidates"
            ),
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "prompts": len(prompts),
                "candidates": len(candidates),
                "stable_candidates": sum(map(is_stable_candidate, candidates)),
                "codex_review_items": pair_count,
                "unresolved": len(unresolved),
            },
            "unresolved_prompt_ids": unresolved,
        }
    )
    artifact_paths = candidates_path, packet_path, key_path, results_path
    summary["artifacts"] = {
        path.name: _artifact(path, run_dir) for path in artifact_paths
    }
    write_json_atomic(run_dir / "run_summary.json", summary)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"status={summary['status']}")
    print(f"Codex review packet: {packet_path}")
    return run_dir


def _load_finalization_inputs(
    run_dir: Path, summary: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    candidates_path = run_dir / "candidates.jsonl"
    artifact_names = (
        "candidates.jsonl",
        "codex_review_packet.json",
        "codex_review_key.json",
    )
    for name in artifact_names:
        path = run_dir / name
        artifact = summary.get("artifacts", {}).get(name, {})
        if artifact.get("sha256") != sha256_file(path):
            raise DPODataError(f"{name} 在 Codex 裁决前后发生变化")
    prompts = load_prompts()
    candidates = load_jsonl(candidates_path)
    _validate_candidate_rows(candidates, prompts)
    packet = json.loads(
        (run_dir / "codex_review_packet.json").read_text(encoding="utf-8")
    )
    key = json.loads(
        (run_dir / "codex_review_key.json").read_text(encoding="utf-8")
    )
    submitted = json.loads(
        (run_dir / "codex_review_results.json").read_text(encoding="utf-8")
    )
    packet_items = packet.get("items")
    key_items = key.get("items")
    results = submitted.get("results")
    if not all(isinstance(value, list) for value in (packet_items, key_items, results)):
        raise DPODataError("Codex packet/key/results 格式无效")
    try:
        packet_by_id = {row["review_id"]: row for row in packet_items}
        key_by_id = {row["review_id"]: row for row in key_items}
    except (KeyError, TypeError) as exc:
        raise DPODataError("Codex packet/key 缺少 review_id") from exc
    result_by_id = {
        row.get("review_id"): row for row in results if isinstance(row, dict)
    }
    if (
        not packet_by_id
        or set(packet_by_id) != set(key_by_id)
        or set(result_by_id) != set(packet_by_id)
        or len(packet_by_id) != len(packet_items)
        or len(key_by_id) != len(key_items)
        or len(result_by_id) != len(results)
    ):
        raise DPODataError("Codex packet/key/results 未完整对齐")
    for review_id, packet_row in packet_by_id.items():
        key_row = key_by_id[review_id]
        labels = key_row.get("labels")
        if (
            key_row.get("prompt_id") != packet_row.get("prompt_id")
            or not isinstance(labels, dict)
            or set(labels) != {"A", "B"}
            or labels["A"].get("assistant") != packet_row.get("answer_a")
            or labels["B"].get("assistant") != packet_row.get("answer_b")
        ):
            raise DPODataError(f"{review_id} packet/key A/B 映射不一致")
        result_by_id[review_id] = parse_codex_result(
            result_by_id[review_id], labels
        )
    return candidates, packet_by_id, key_by_id, result_by_id


def finalize_run(
    *,
    run_dir: Path,
    train_output: Path = DEFAULT_TRAIN_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
) -> tuple[Path, Path]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise DPODataError(f"缺少 run_summary.json: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = _summary_contract()
    if summary.get("contract") != expected["contract"] or summary.get(
        "input_hashes"
    ) != expected["input_hashes"]:
        raise DPODataError("run 配置或输入哈希与当前冻结契约不一致")
    if train_output.exists() or audit_output.exists():
        raise DPODataError("DPO v2 训练集或审计文件已存在，拒绝覆盖")

    candidates, packet_by_id, key_by_id, result_by_id = _load_finalization_inputs(
        run_dir, summary
    )
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    train_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    teacher_pairs = 0
    for review_id, packet_row in packet_by_id.items():
        result = result_by_id[review_id]
        decision = result["decision"]
        audit_row: dict[str, Any] = {
            "review_id": review_id,
            "prompt_id": packet_row["prompt_id"],
            "decision": decision,
            "included": decision != "no_clear_preference",
            "preference_reasons": result["preference_reasons"],
            "material_tradeoff": result["material_tradeoff"],
            "hard_rule_only": result["hard_rule_only"],
            "notes": result["notes"],
        }
        if decision == "no_clear_preference":
            audit_rows.append(audit_row)
            continue
        labels = key_by_id[review_id]["labels"]
        source_label = result["source_label"]
        source = labels[source_label]
        if decision == "clear_preference":
            rejected_label = "B" if source_label == "A" else "A"
            chosen = source
            rejected = labels[rejected_label]
        else:
            teacher_pairs += 1
            chosen = {
                "assistant": result["improved_assistant"],
                "source": "codex_teacher_edit",
                "source_id": f"codex:{packet_row['prompt_id']}",
            }
            rejected = source
            audit_row["changes"] = result["changes"]
        train_rows.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": packet_row["user"]},
                    {"role": "assistant", "content": chosen["assistant"]},
                ],
                "rejected_response": rejected["assistant"],
            }
        )
        audit_row.update(
            {
                "chosen_source": chosen["source"],
                "chosen_source_id": chosen["source_id"],
                "rejected_source": rejected["source"],
                "rejected_source_id": rejected["source_id"],
            }
        )
        audit_rows.append(audit_row)
    reviewed_prompt_ids = {
        row["prompt_id"] for row in packet_by_id.values()
    }
    filtered_rows = []
    for prompt in load_prompts():
        if prompt["id"] in reviewed_prompt_ids:
            continue
        prompt_candidates = [
            row for row in candidates if row["prompt_id"] == prompt["id"]
        ]
        filtered_rows.append(
            {
                "review_id": None,
                "prompt_id": prompt["id"],
                "decision": "filtered_before_codex",
                "included": False,
                "preference_reasons": [],
                "material_tradeoff": False,
                "hard_rule_only": False,
                "filter_reason": "fewer_than_two_stable_candidates",
                "unstable_candidate_ids": [
                    row["candidate_id"]
                    for row in prompt_candidates
                    if not is_stable_candidate(row)
                ],
                "notes": "候选未全部通过本地稳定性检查。",
            }
        )
    audit_rows.extend(filtered_rows)
    audit_rows.sort(key=lambda row: row["prompt_id"])
    if len(train_rows) < MIN_FINAL_PAIRS:
        raise DPODataError(
            f"有效 Codex 偏好对至少需要 {MIN_FINAL_PAIRS} 条，实际 {len(train_rows)}"
        )
    if teacher_pairs * 3 > len(train_rows):
        raise DPODataError(
            f"Teacher 修改参与 {teacher_pairs}/{len(train_rows)} 对，超过三分之一"
        )

    write_jsonl_atomic(train_output, train_rows)
    audit = {
        "schema_version": 2,
        "source_run": str(run_dir),
        "reviewer": "codex",
        "pairs": len(train_rows),
        "teacher_involved_pairs": teacher_pairs,
        "filtered_before_codex": len(filtered_rows),
        "discarded_pairs": sum(not row["included"] for row in audit_rows),
        "codex_review_results_sha256": sha256_file(
            run_dir / "codex_review_results.json"
        ),
        "train_sha256": sha256_file(train_output),
        "items": audit_rows,
    }
    write_json_atomic(audit_output, audit)
    summary["status"] = "ready_for_dpo"
    summary.setdefault("artifacts", {})["codex_review_results.json"] = _artifact(
        run_dir / "codex_review_results.json", run_dir
    )
    summary["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["final_dataset"] = {
        "pairs": len(train_rows),
        "teacher_involved_pairs": teacher_pairs,
        "train_path": str(train_output),
        "train_sha256": audit["train_sha256"],
        "audit_path": str(audit_output),
        "audit_sha256": sha256_file(audit_output),
    }
    write_json_atomic(summary_path, summary)
    return train_output, audit_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备 morgana-v2 第二轮 DPO 偏好数据")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="生成双候选并产出匿名 Codex 裁决包"
    )
    prepare.add_argument("--run-id", default=DEFAULT_RUN_ID)
    prepare.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    prepare.add_argument("--base-model", type=Path, default=BASE_SNAPSHOT)
    finalize = subparsers.add_parser(
        "finalize", help="校验 Codex 裁决并冻结 DPO v2 训练集"
    )
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    finalize.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_run(
                run_id=args.run_id,
                output_root=args.output_root,
                base_model_path=args.base_model,
            )
        else:
            train, audit = finalize_run(
                run_dir=args.run_dir,
                train_output=args.train_output,
                audit_output=args.audit_output,
            )
            print(f"DPO v2 train: {train}")
            print(f"DPO v2 audit: {audit}")
    except (DPODataError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
