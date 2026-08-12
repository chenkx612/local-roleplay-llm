"""Validate the frozen targeted Prompt splits for post-GRPO DPO."""

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
from roleplay.grpo_rule_reward import (
    RewardConstraints,
    score_completion,
)
from roleplay.sft_eval import normalize_empty_think_wrapper


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data/runs/morgana-v2/post_grpo_dpo_prompts.jsonl"
HOLDOUT_PATH = ROOT / "data/runs/morgana-v2/post_grpo_dpo_holdout.jsonl"
MANIFEST_PATH = (
    ROOT / "data/runs/morgana-v2/post_grpo_dpo_prompt_manifest.json"
)
SYSTEM_PROMPT_PATH = ROOT / "data/runs/morgana-v2/system_prompt.txt"
GRPO_ADAPTER_PATH = (
    ROOT / "output/morgana-v2/stage4-grpo/20260812-2144/adapter"
)
OUTPUT_ROOT = ROOT / "output/morgana-v2/post-grpo-dpo/data"
DEFAULT_TRAIN_OUTPUT = (
    ROOT / "data/runs/morgana-v2/post_grpo_dpo_train.jsonl"
)
DEFAULT_AUDIT_OUTPUT = (
    ROOT / "data/runs/morgana-v2/post_grpo_dpo_train_audit.json"
)

SYSTEM_PROMPT_SHA256 = (
    "d88993aaa1178ced740f6b54530a27e5fcdb2486a66d8b460367e842b53ee112"
)
GRPO_ADAPTER_HASHES = {
    "adapter_model.safetensors": (
        "89ca4fa213ea16eeee002b088ea46b3012b6311b02f09b2b27b0937ab6dcd30f"
    ),
    "adapter_config.json": (
        "67f3ab10168164cc014c7f9c8720984760b1d94be33fde9abf89fb3004a886dc"
    ),
    "additional_config.json": (
        "2b7ed6cc0ca6c21dc39bf80fd3351f5f87c462df23a237dd6ad20473eb9a33a2"
    ),
}

TARGET_ISSUES = (
    "fabricated_background",
    "perspective_shift",
    "emotion_response",
)
ALLOWED_SCENARIOS = frozenset(
    {"background", "adversarial", "style", "daily", "emotion"}
)
PROMPT_FIELDS = frozenset(
    {"id", "target_issue", "scenario", "user", "preference_criteria"}
)
TRAIN_COUNT = 30
TRAIN_PER_ISSUE = 10
HOLDOUT_COUNT = 9
HOLDOUT_PER_ISSUE = 3
CANDIDATES_PER_PROMPT = 6
BASE_SEED = 20260812
REVIEW_ORDER_SEED = 20260812
MIN_PAIRS_PER_ISSUE = 6
MIN_FINAL_PAIRS = 18
MAX_FINAL_PAIRS = 30
MAX_TEACHER_PAIR_FRACTION = 0.25
MIN_CHOSEN_NON_TARGET_QUALITY = 8
MIN_REJECTED_NON_TARGET_QUALITY = 7
MIN_TEACHER_SIMILARITY = 0.50
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
CANDIDATE_REWARD_CONSTRAINTS = RewardConstraints(
    min_actions=0,
    max_actions=2,
    min_sentences=1,
    max_sentences=4,
    min_chars=30,
    max_chars=90,
    min_signatures=0,
    max_signatures=1,
)
TARGET_STATUSES = frozenset({"pass", "fail", "ambiguous"})
DECISIONS = frozenset({"native_pair", "teacher_chosen", "no_pair"})
OFF_TARGET_ISSUES = frozenset(
    {
        "fabricated_background",
        "perspective_shift",
        "emotion_response",
        "wrong_self_reference",
        "role_break",
        "does_not_answer",
        "servile_submission",
        "romanticization",
        "unnatural_expression",
        "template_overuse",
    }
)


class PostGRPODPODataError(RuntimeError):
    """Raised when the targeted Prompt contract is violated."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one JSONL file as objects."""
    if not path.is_file():
        raise PostGRPODPODataError(f"缺少 Prompt 文件: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PostGRPODPODataError(
                f"{path}:{line_number} 不是有效 JSON"
            ) from exc
        if not isinstance(row, dict):
            raise PostGRPODPODataError(f"{path}:{line_number} 不是对象")
        rows.append(row)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    """Write one JSON value atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(
    path: Path, rows: Iterable[dict[str, Any]], *, refuse_overwrite: bool = False
) -> None:
    """Write JSONL atomically, optionally preserving an existing artifact."""
    if refuse_overwrite and path.exists():
        raise PostGRPODPODataError(f"拒绝覆盖已有文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _canonical_user(text: str) -> str:
    return " ".join(text.strip().split())


def _validate_split(
    path: Path,
    *,
    id_prefix: str,
    expected_count: int,
    expected_per_issue: int,
) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if len(rows) != expected_count:
        raise PostGRPODPODataError(
            f"{path.name} 应有 {expected_count} 条，实际 {len(rows)}"
        )
    users: set[str] = set()
    issue_counts: Counter[str] = Counter()
    pattern = re.compile(rf"{re.escape(id_prefix)}(\d{{4}})\Z")
    for index, row in enumerate(rows, 1):
        if set(row) != PROMPT_FIELDS:
            raise PostGRPODPODataError(
                f"{path.name} 第 {index} 条字段不正确"
            )
        prompt_id = row["id"]
        match = pattern.fullmatch(prompt_id) if isinstance(prompt_id, str) else None
        if not match or int(match.group(1)) != index:
            raise PostGRPODPODataError(
                f"{path.name} 第 {index} 条 id 不连续"
            )
        issue = row["target_issue"]
        if issue not in TARGET_ISSUES:
            raise PostGRPODPODataError(f"{prompt_id} target_issue 无效")
        if row["scenario"] not in ALLOWED_SCENARIOS:
            raise PostGRPODPODataError(f"{prompt_id} scenario 无效")
        user = row["user"]
        criteria = row["preference_criteria"]
        if (
            not isinstance(user, str)
            or not user.strip()
            or not isinstance(criteria, str)
            or not criteria.strip()
        ):
            raise PostGRPODPODataError(f"{prompt_id} 文本字段为空")
        normalized = _canonical_user(user)
        if normalized in users:
            raise PostGRPODPODataError(f"{path.name} 包含重复 user")
        users.add(normalized)
        issue_counts[issue] += 1
    expected_counts = {
        issue: expected_per_issue for issue in TARGET_ISSUES
    }
    if dict(issue_counts) != expected_counts:
        raise PostGRPODPODataError(
            f"{path.name} 目标分布不正确: {dict(issue_counts)}"
        )
    return rows


def _row_users(row: dict[str, Any]) -> Iterable[str]:
    user = row.get("user")
    if isinstance(user, str):
        yield user
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                yield message["content"]


def _validate_isolation(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    root: Path,
) -> None:
    new_users: dict[str, str] = {}
    for row in [*train_rows, *holdout_rows]:
        normalized = _canonical_user(row["user"])
        if normalized in new_users:
            raise PostGRPODPODataError(
                f"新 Prompt 重复: {new_users[normalized]} / {row['id']}"
            )
        new_users[normalized] = row["id"]

    data_root = root / "data/runs/morgana-v2"
    excluded = {
        TRAIN_PATH.name,
        HOLDOUT_PATH.name,
        DEFAULT_TRAIN_OUTPUT.name,
    }
    comparison_paths = [
        path for path in data_root.glob("*.jsonl") if path.name not in excluded
    ]
    comparison_paths.append(root / "data/style_examples.jsonl")
    for path in comparison_paths:
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            for user in _row_users(row):
                normalized = _canonical_user(user)
                if normalized in new_users:
                    raise PostGRPODPODataError(
                        f"{new_users[normalized]} 与已有 split 重复: {path}"
                    )


def validate_prompt_data(root: Path = ROOT) -> dict[str, Any]:
    """Validate frozen files, balanced targets, and cross-split isolation."""
    train_path = root / TRAIN_PATH.relative_to(ROOT)
    holdout_path = root / HOLDOUT_PATH.relative_to(ROOT)
    manifest_path = root / MANIFEST_PATH.relative_to(ROOT)
    if not manifest_path.is_file():
        raise PostGRPODPODataError(f"缺少 Prompt manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PostGRPODPODataError("Prompt manifest 不是有效 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PostGRPODPODataError("Prompt manifest schema_version 无效")
    if manifest.get("objective") != list(TARGET_ISSUES):
        raise PostGRPODPODataError("Prompt manifest 目标不正确")

    train_rows = _validate_split(
        train_path,
        id_prefix="post_dpo_",
        expected_count=TRAIN_COUNT,
        expected_per_issue=TRAIN_PER_ISSUE,
    )
    holdout_rows = _validate_split(
        holdout_path,
        id_prefix="post_dpo_dev_",
        expected_count=HOLDOUT_COUNT,
        expected_per_issue=HOLDOUT_PER_ISSUE,
    )
    for key, path, count, per_issue in (
        ("training_prompts", train_path, TRAIN_COUNT, TRAIN_PER_ISSUE),
        ("holdout_prompts", holdout_path, HOLDOUT_COUNT, HOLDOUT_PER_ISSUE),
    ):
        entry = manifest.get(key)
        expected_path = str(path.relative_to(root))
        if not isinstance(entry, dict) or entry != {
            "path": expected_path,
            "records": count,
            "per_issue": per_issue,
            "sha256": sha256_file(path),
        }:
            raise PostGRPODPODataError(f"Prompt manifest {key} 不匹配")
    _validate_isolation(train_rows, holdout_rows, root)
    return {
        "training_prompts": len(train_rows),
        "holdout_prompts": len(holdout_rows),
        "training_per_issue": TRAIN_PER_ISSUE,
        "holdout_per_issue": HOLDOUT_PER_ISSUE,
        "target_issues": list(TARGET_ISSUES),
    }


def validate_grpo_adapter(adapter_dir: Path = GRPO_ADAPTER_PATH) -> dict[str, str]:
    """Require the exact accepted Stage 4 GRPO adapter."""
    result: dict[str, str] = {}
    for name, expected in GRPO_ADAPTER_HASHES.items():
        path = adapter_dir / name
        if not path.is_file():
            raise PostGRPODPODataError(f"缺少 GRPO adapter 文件: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PostGRPODPODataError(
                f"GRPO adapter 哈希不匹配: {path}，{actual} != {expected}"
            )
        result[name] = actual
    return result


def candidate_seed(prompt_id: str, candidate_index: int) -> int:
    """Return one deterministic, non-overlapping candidate seed."""
    match = re.fullmatch(r"post_dpo_(\d{4})", prompt_id)
    if not match or not 1 <= candidate_index <= CANDIDATES_PER_PROMPT:
        raise ValueError("候选 seed 输入无效")
    prompt_index = int(match.group(1)) - 1
    return BASE_SEED + prompt_index * 100 + candidate_index - 1


def candidate_id(prompt_id: str, candidate_index: int) -> str:
    """Return the stable candidate identifier for one Prompt sample."""
    candidate_seed(prompt_id, candidate_index)
    return f"{prompt_id}-c{candidate_index}"


def _reward_v2(
    assistant: str, finish_reason: str
) -> tuple[dict[str, Any], bool]:
    components = score_completion(
        assistant,
        CANDIDATE_REWARD_CONSTRAINTS,
        finish_reason=finish_reason,
        is_truncated=finish_reason in {"length", "max_tokens"},
    ).as_log_dict()
    return components, not components["hard_invalid_reasons"]


def generate_candidates(
    *,
    prompts: Sequence[dict[str, Any]],
    base_model_path: Path,
    mlx_adapter_path: Path,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Load the GRPO policy once and sample six candidates per Prompt."""
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise PostGRPODPODataError("本地候选生成需要 mlx-lm") from exc
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
            raw, finish_reason = _generate_one(
                model, tokenizer, messages, seed
            )
            assistant = normalize_empty_think_wrapper(raw)
            reward, eligible = _reward_v2(assistant, finish_reason)
            record = {
                "candidate_id": candidate_id(prompt["id"], index),
                "prompt_id": prompt["id"],
                "target_issue": prompt["target_issue"],
                "scenario": prompt["scenario"],
                "preference_criteria": prompt["preference_criteria"],
                "candidate_index": index,
                "seed": seed,
                "messages": messages,
                "raw_assistant": raw,
                "assistant": assistant,
                "finish_reason": finish_reason,
                "is_truncated": finish_reason in {"length", "max_tokens"},
                "source": "grpo_candidate",
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "adapter_sha256": GRPO_ADAPTER_HASHES[
                    "adapter_model.safetensors"
                ],
                "generation": GENERATION,
                "reward_v2": reward,
                "eligible_for_review": eligible,
            }
            records.append(record)
            print(
                f"[{len(records)}/{total}] {record['candidate_id']}: "
                f"{finish_reason}, eligible={eligible}"
            )
    return records


def validate_candidate_rows(
    rows: Sequence[dict[str, Any]],
    prompts: Sequence[dict[str, Any]],
    system_prompt: str,
) -> None:
    """Require exactly 180 reproducible GRPO candidate records."""
    prompt_by_id = {row["id"]: row for row in prompts}
    required = {
        "candidate_id",
        "prompt_id",
        "target_issue",
        "scenario",
        "preference_criteria",
        "candidate_index",
        "seed",
        "messages",
        "raw_assistant",
        "assistant",
        "finish_reason",
        "is_truncated",
        "source",
        "base_model",
        "base_revision",
        "adapter_sha256",
        "generation",
        "reward_v2",
        "eligible_for_review",
    }
    seen: set[str] = set()
    seeds: set[int] = set()
    for row in rows:
        if set(row) != required:
            raise PostGRPODPODataError("候选记录字段不正确")
        prompt = prompt_by_id.get(row["prompt_id"])
        index = row["candidate_index"]
        if prompt is None or not isinstance(index, int):
            raise PostGRPODPODataError("候选引用未知 Prompt 或序号无效")
        expected_id = candidate_id(prompt["id"], index)
        expected_seed = candidate_seed(prompt["id"], index)
        if row["candidate_id"] != expected_id or row["seed"] != expected_seed:
            raise PostGRPODPODataError(f"候选 id/seed 不正确: {expected_id}")
        expected_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt["user"]},
        ]
        if (
            row["target_issue"] != prompt["target_issue"]
            or row["scenario"] != prompt["scenario"]
            or row["preference_criteria"] != prompt["preference_criteria"]
            or row["messages"] != expected_messages
            or row["source"] != "grpo_candidate"
            or row["base_model"] != BASE_MODEL
            or row["base_revision"] != BASE_REVISION
            or row["adapter_sha256"]
            != GRPO_ADAPTER_HASHES["adapter_model.safetensors"]
            or row["generation"] != GENERATION
            or not isinstance(row["raw_assistant"], str)
            or row["assistant"]
            != normalize_empty_think_wrapper(row["raw_assistant"])
            or not isinstance(row["finish_reason"], str)
        ):
            raise PostGRPODPODataError(f"候选冻结内容不正确: {expected_id}")
        expected_reward, eligible = _reward_v2(
            row["assistant"], row["finish_reason"]
        )
        if (
            row["reward_v2"] != expected_reward
            or row["eligible_for_review"] is not eligible
            or row["is_truncated"]
            is not (row["finish_reason"] in {"length", "max_tokens"})
        ):
            raise PostGRPODPODataError(f"候选 Reward v2 不正确: {expected_id}")
        if expected_id in seen or expected_seed in seeds:
            raise PostGRPODPODataError("候选 id 或 seed 重复")
        seen.add(expected_id)
        seeds.add(expected_seed)
    if len(rows) != TRAIN_COUNT * CANDIDATES_PER_PROMPT:
        raise PostGRPODPODataError(
            f"候选应为 {TRAIN_COUNT * CANDIDATES_PER_PROMPT} 条，"
            f"实际 {len(rows)}"
        )
    counts = Counter(row["prompt_id"] for row in rows)
    if set(counts) != set(prompt_by_id) or set(counts.values()) != {
        CANDIDATES_PER_PROMPT
    }:
        raise PostGRPODPODataError("每条 Prompt 必须恰好包含6个候选")


def _candidate_set_sha256(candidates: Sequence[dict[str, Any]]) -> str:
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


def build_review_artifacts(
    prompts: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    """Build one anonymous semantic review packet from distinct candidates."""
    rng = random.Random(REVIEW_ORDER_SEED)
    packet_items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for prompt in prompts:
        reviewable = [
            row
            for row in candidates
            if row["prompt_id"] == prompt["id"]
            and row["eligible_for_review"]
            and row.get("duplicate_of_candidate_id") is None
        ]
        eligible: list[dict[str, Any]] = []
        seen_answers: set[str] = set()
        for row in reviewable:
            canonical = " ".join(row["assistant"].strip().split())
            if canonical in seen_answers:
                continue
            seen_answers.add(canonical)
            eligible.append(row)
        if len(eligible) < 2:
            unresolved.append(prompt["id"])
            continue
        rng.shuffle(eligible)
        labels = {
            chr(ord("A") + index): row
            for index, row in enumerate(eligible)
        }
        review_id = f"post-dpo-review-{int(prompt['id'][-4:]):04d}"
        packet_items.append(
            {
                "review_id": review_id,
                "prompt_id": prompt["id"],
                "target_issue": prompt["target_issue"],
                "scenario": prompt["scenario"],
                "user": prompt["user"],
                "preference_criteria": prompt["preference_criteria"],
                "answers": {
                    label: row["assistant"] for label, row in labels.items()
                },
            }
        )
        key_items.append(
            {
                "review_id": review_id,
                "prompt_id": prompt["id"],
                "target_issue": prompt["target_issue"],
                "labels": {
                    label: {
                        "candidate_id": row["candidate_id"],
                        "assistant": row["assistant"],
                        "reward_v2_total": row["reward_v2"]["total_reward"],
                    }
                    for label, row in labels.items()
                },
                "candidate_set_sha256": _candidate_set_sha256(eligible),
            }
        )
    packet = {
        "schema_version": 1,
        "reviewer": "codex",
        "instructions": (
            "逐题评价每个候选在唯一 target_issue 上是 pass、fail 或 ambiguous，"
            "并评价排除该目标后的非目标质量。chosen 必须 pass 且无非目标问题；"
            "rejected 必须是整体质量较高但明确 fail 的真实 GRPO 候选。"
        ),
        "target_statuses": sorted(TARGET_STATUSES),
        "decisions": sorted(DECISIONS),
        "allowed_off_target_issues": sorted(OFF_TARGET_ISSUES),
        "quality_thresholds": {
            "chosen": MIN_CHOSEN_NON_TARGET_QUALITY,
            "rejected": MIN_REJECTED_NON_TARGET_QUALITY,
        },
        "items": packet_items,
    }
    key = {
        "schema_version": 1,
        "order_seed": REVIEW_ORDER_SEED,
        "items": key_items,
    }
    results = {
        "schema_version": 1,
        "reviewer": "codex",
        "results": [
            {
                "review_id": row["review_id"],
                "decision": None,
                "chosen_label": None,
                "rejected_label": None,
                "candidate_reviews": {},
                "teacher_assistant": None,
                "teacher_target_status": None,
                "teacher_non_target_quality": None,
                "teacher_off_target_issues": None,
                "teacher_evidence": "",
                "teacher_changes": "",
                "notes": "",
            }
            for row in packet_items
        ],
    }
    return packet, key, results, unresolved


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _default_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M")


def prepare_run(
    *,
    run_id: str | None = None,
    output_root: Path = OUTPUT_ROOT,
    base_model_path: Path = BASE_SNAPSHOT,
) -> Path:
    """Validate frozen inputs, sample 180 candidates, and build review files."""
    validate_prompt_data()
    validate_grpo_adapter()
    if sha256_file(SYSTEM_PROMPT_PATH) != SYSTEM_PROMPT_SHA256:
        raise PostGRPODPODataError("system prompt 哈希不匹配")
    if not base_model_path.is_dir():
        raise PostGRPODPODataError(f"缺少本地 MLX 基座: {base_model_path}")
    run_id = _default_run_id() if run_id is None else run_id
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise PostGRPODPODataError("run-id 只能包含字母、数字、点、下划线和连字符")
    run_dir = output_root.resolve() / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PostGRPODPODataError(f"运行目录已存在: {run_dir}") from exc
    summary_path = run_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "post_grpo_dpo_data",
        "status": "sampling",
        "run": {
            "id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "contract": {
            "prompts": TRAIN_COUNT,
            "candidates_per_prompt": CANDIDATES_PER_PROMPT,
            "candidate_rounds": 1,
            "generation": GENERATION,
            "base_seed": BASE_SEED,
            "minimum_pairs_per_issue": MIN_PAIRS_PER_ISSUE,
            "teacher_pair_fraction_max": MAX_TEACHER_PAIR_FRACTION,
        },
        "inputs": {
            str(TRAIN_PATH.relative_to(ROOT)): sha256_file(TRAIN_PATH),
            str(HOLDOUT_PATH.relative_to(ROOT)): sha256_file(HOLDOUT_PATH),
            str(SYSTEM_PROMPT_PATH.relative_to(ROOT)): SYSTEM_PROMPT_SHA256,
            str(GRPO_ADAPTER_PATH.relative_to(ROOT)): GRPO_ADAPTER_HASHES,
        },
        "artifacts": {},
    }
    write_json_atomic(summary_path, summary)
    try:
        prompts = _validate_split(
            TRAIN_PATH,
            id_prefix="post_dpo_",
            expected_count=TRAIN_COUNT,
            expected_per_issue=TRAIN_PER_ISSUE,
        )
        mlx_adapter = convert_peft_adapter_to_mlx(
            GRPO_ADAPTER_PATH,
            run_dir / "mlx-grpo-adapter",
            expected_sha256=GRPO_ADAPTER_HASHES[
                "adapter_model.safetensors"
            ],
        )
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        candidates = generate_candidates(
            prompts=prompts,
            base_model_path=base_model_path,
            mlx_adapter_path=mlx_adapter,
            system_prompt=system_prompt,
        )
        candidates_path = run_dir / "candidates.jsonl"
        write_jsonl_atomic(candidates_path, candidates)
        validate_candidate_rows(candidates, prompts, system_prompt)
        packet, key, results, unresolved = build_review_artifacts(
            prompts, candidates
        )
        packet_path = run_dir / "review_packet.json"
        key_path = run_dir / "review_key.json"
        results_path = run_dir / "review_results.json"
        write_json_atomic(packet_path, packet)
        write_json_atomic(key_path, key)
        write_json_atomic(results_path, results)
        artifact_paths = candidates_path, packet_path, key_path
        summary.update(
            {
                "status": "awaiting_codex_review",
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "counts": {
                    "prompts": len(prompts),
                    "candidates": len(candidates),
                    "eligible_candidates": sum(
                        row["eligible_for_review"] for row in candidates
                    ),
                    "review_items": len(packet["items"]),
                    "unresolved_prompts": len(unresolved),
                },
                "unresolved_prompt_ids": unresolved,
                "artifacts": {
                    path.name: _artifact(path, run_dir)
                    for path in artifact_paths
                },
            }
        )
        write_json_atomic(summary_path, summary)
        return run_dir
    except Exception as exc:
        summary["status"] = "sampling_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(summary_path, summary)
        raise


def _validate_candidate_review(
    value: Any,
    *,
    review_id: str,
    label: str,
    target_issue: str,
) -> dict[str, Any]:
    expected = {
        "target_status",
        "non_target_quality",
        "off_target_issues",
        "evidence",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise PostGRPODPODataError(
            f"{review_id} {label} candidate review 字段不正确"
        )
    status = value["target_status"]
    quality = value["non_target_quality"]
    issues = value["off_target_issues"]
    evidence = value["evidence"]
    if status not in TARGET_STATUSES:
        raise PostGRPODPODataError(f"{review_id} {label} target_status 无效")
    if isinstance(quality, bool) or not isinstance(quality, int) or not 0 <= quality <= 10:
        raise PostGRPODPODataError(
            f"{review_id} {label} non_target_quality 无效"
        )
    if (
        not isinstance(issues, list)
        or len(issues) != len(set(issues))
        or any(issue not in OFF_TARGET_ISSUES for issue in issues)
        or target_issue in issues
    ):
        raise PostGRPODPODataError(
            f"{review_id} {label} off_target_issues 无效"
        )
    if not isinstance(evidence, str) or not evidence.strip():
        raise PostGRPODPODataError(f"{review_id} {label} evidence 为空")
    return {
        **value,
        "off_target_issues": sorted(issues),
        "evidence": evidence.strip(),
    }


def _native_chosen_labels(
    reviews: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        label
        for label, review in reviews.items()
        if review["target_status"] == "pass"
        and review["non_target_quality"] >= MIN_CHOSEN_NON_TARGET_QUALITY
        and not review["off_target_issues"]
    ]


def _rejected_labels(
    reviews: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        label
        for label, review in reviews.items()
        if review["target_status"] == "fail"
        and review["non_target_quality"] >= MIN_REJECTED_NON_TARGET_QUALITY
        and not review["off_target_issues"]
    ]


def _best_native_chosen(
    labels: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
) -> str | None:
    eligible = _native_chosen_labels(reviews)
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda label: (
            -reviews[label]["non_target_quality"],
            -labels[label]["reward_v2_total"],
            label,
        ),
    )[0]


def _best_rejected(
    labels: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    *,
    chosen_label: str | None,
    chosen_reward: float | None = None,
) -> str | None:
    eligible = _rejected_labels(reviews)
    if not eligible:
        return None
    if chosen_label is None and chosen_reward is None:
        return sorted(
            eligible,
            key=lambda label: (
                -reviews[label]["non_target_quality"],
                -labels[label]["reward_v2_total"],
                label,
            ),
        )[0]
    if chosen_label is not None:
        chosen_reward = labels[chosen_label]["reward_v2_total"]
    assert chosen_reward is not None
    return sorted(
        eligible,
        key=lambda label: (
            -reviews[label]["non_target_quality"],
            abs(labels[label]["reward_v2_total"] - chosen_reward),
            label,
        ),
    )[0]


def _validate_teacher_answer(
    *,
    review_id: str,
    rejected: str,
    teacher: str,
) -> dict[str, Any]:
    if not teacher.strip() or teacher.strip() == rejected.strip():
        raise PostGRPODPODataError(f"{review_id} Teacher chosen 无效")
    ratio = len(teacher.strip()) / len(rejected.strip()) if rejected.strip() else 0.0
    similarity = SequenceMatcher(
        None, rejected.strip(), teacher.strip(), autojunk=False
    ).ratio()
    if not MIN_TEACHER_LENGTH_RATIO <= ratio <= MAX_TEACHER_LENGTH_RATIO:
        raise PostGRPODPODataError(
            f"{review_id} Teacher chosen 长度比 {ratio:.3f} 越界"
        )
    if similarity < MIN_TEACHER_SIMILARITY:
        raise PostGRPODPODataError(
            f"{review_id} Teacher chosen 相似度 {similarity:.3f} 过低"
        )
    reward, eligible = _reward_v2(teacher.strip(), "stop")
    if not eligible:
        raise PostGRPODPODataError(
            f"{review_id} Teacher chosen 未通过 Reward v2 硬过滤"
        )
    return {
        "assistant": teacher.strip(),
        "length_ratio": ratio,
        "similarity": similarity,
        "reward_v2": reward,
    }


def parse_review_result(
    result: Any,
    packet_item: dict[str, Any],
    key_item: dict[str, Any],
) -> dict[str, Any]:
    """Validate one semantic adjudication and deterministic pair selection."""
    review_id = packet_item["review_id"]
    expected = {
        "review_id",
        "decision",
        "chosen_label",
        "rejected_label",
        "candidate_reviews",
        "teacher_assistant",
        "teacher_target_status",
        "teacher_non_target_quality",
        "teacher_off_target_issues",
        "teacher_evidence",
        "teacher_changes",
        "notes",
    }
    if (
        not isinstance(result, dict)
        or set(result) != expected
        or result.get("review_id") != review_id
    ):
        raise PostGRPODPODataError(f"{review_id} 裁决字段不正确")
    labels = key_item["labels"]
    raw_reviews = result["candidate_reviews"]
    if not isinstance(raw_reviews, dict) or set(raw_reviews) != set(labels):
        raise PostGRPODPODataError(
            f"{review_id} candidate_reviews 未完整覆盖候选"
        )
    target_issue = packet_item["target_issue"]
    reviews = {
        label: _validate_candidate_review(
            raw_reviews[label],
            review_id=review_id,
            label=label,
            target_issue=target_issue,
        )
        for label in labels
    }
    decision = result["decision"]
    chosen_label = result["chosen_label"]
    rejected_label = result["rejected_label"]
    notes = result["notes"]
    if decision not in DECISIONS or not isinstance(notes, str) or not notes.strip():
        raise PostGRPODPODataError(f"{review_id} decision/notes 无效")
    best_chosen = _best_native_chosen(labels, reviews)
    best_rejected = _best_rejected(
        labels, reviews, chosen_label=best_chosen
    )
    teacher_fields = (
        result["teacher_assistant"],
        result["teacher_target_status"],
        result["teacher_non_target_quality"],
        result["teacher_off_target_issues"],
        result["teacher_evidence"],
        result["teacher_changes"],
    )
    teacher: dict[str, Any] | None = None
    if decision == "native_pair":
        if (
            chosen_label != best_chosen
            or rejected_label is None
            or chosen_label == rejected_label
            or any(value is not None and value != "" for value in teacher_fields)
        ):
            raise PostGRPODPODataError(
                f"{review_id} native_pair 选择或 Teacher 字段不正确"
            )
        expected_rejected = _best_rejected(
            labels, reviews, chosen_label=chosen_label
        )
        if rejected_label != expected_rejected:
            raise PostGRPODPODataError(
                f"{review_id} 未选择最佳 hard negative"
            )
    elif decision == "teacher_chosen":
        if (
            best_chosen is not None
            or chosen_label is not None
            or rejected_label is None
            or rejected_label not in _rejected_labels(reviews)
            or not isinstance(result["teacher_assistant"], str)
            or result["teacher_target_status"] != "pass"
            or isinstance(result["teacher_non_target_quality"], bool)
            or not isinstance(result["teacher_non_target_quality"], int)
            or result["teacher_non_target_quality"]
            < MIN_CHOSEN_NON_TARGET_QUALITY
            or result["teacher_off_target_issues"] != []
            or not isinstance(result["teacher_evidence"], str)
            or not result["teacher_evidence"].strip()
            or not isinstance(result["teacher_changes"], str)
            or not result["teacher_changes"].strip()
        ):
            raise PostGRPODPODataError(
                f"{review_id} teacher_chosen 字段或选择不正确"
            )
        teacher = _validate_teacher_answer(
            review_id=review_id,
            rejected=labels[rejected_label]["assistant"],
            teacher=result["teacher_assistant"],
        )
        expected_rejected = _best_rejected(
            labels,
            reviews,
            chosen_label=None,
            chosen_reward=teacher["reward_v2"]["total_reward"],
        )
        if rejected_label != expected_rejected:
            raise PostGRPODPODataError(
                f"{review_id} 未选择最接近 Teacher chosen 的 hard negative"
            )
        teacher.update(
            {
                "target_status": "pass",
                "non_target_quality": result[
                    "teacher_non_target_quality"
                ],
                "off_target_issues": [],
                "evidence": result["teacher_evidence"].strip(),
                "changes": result["teacher_changes"].strip(),
            }
        )
    else:
        if (
            chosen_label is not None
            or rejected_label is not None
            or any(value is not None and value != "" for value in teacher_fields)
            or (best_chosen is not None and best_rejected is not None)
        ):
            raise PostGRPODPODataError(f"{review_id} no_pair 语义不一致")
    return {
        **result,
        "candidate_reviews": reviews,
        "teacher": teacher,
        "notes": notes.strip(),
    }


def _load_finalization_inputs(
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise PostGRPODPODataError(f"缺少 run_summary.json: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_codex_review":
        raise PostGRPODPODataError(
            f"run 状态不能 finalize: {summary.get('status')}"
        )
    for name in ("candidates.jsonl", "review_packet.json", "review_key.json"):
        path = run_dir / name
        metadata = summary.get("artifacts", {}).get(name, {})
        if not path.is_file() or metadata.get("sha256") != sha256_file(path):
            raise PostGRPODPODataError(f"冻结 run 产物发生变化: {name}")
    stage = summary.get("stage")
    if stage == "post_grpo_dpo_sampling":
        from roleplay.post_grpo_dpo_sampling import (
            PostGRPODPOSamplingError,
            validate_expansion_prompt_data,
            validate_sampling_candidate_rows,
        )

        try:
            prompts = validate_expansion_prompt_data(ROOT)
        except PostGRPODPOSamplingError as exc:
            raise PostGRPODPODataError(str(exc)) from exc
        candidate_validator = validate_sampling_candidate_rows
    elif stage in {None, "post_grpo_dpo_data"}:
        prompts = _validate_split(
            TRAIN_PATH,
            id_prefix="post_dpo_",
            expected_count=TRAIN_COUNT,
            expected_per_issue=TRAIN_PER_ISSUE,
        )
        candidate_validator = validate_candidate_rows
    else:
        raise PostGRPODPODataError(f"不支持的 run stage: {stage}")
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    candidates = load_jsonl(run_dir / "candidates.jsonl")
    try:
        candidate_validator(candidates, prompts, system_prompt)
    except RuntimeError as exc:
        if isinstance(exc, PostGRPODPODataError):
            raise
        raise PostGRPODPODataError(str(exc)) from exc
    packet = json.loads((run_dir / "review_packet.json").read_text(encoding="utf-8"))
    key = json.loads((run_dir / "review_key.json").read_text(encoding="utf-8"))
    submitted = json.loads(
        (run_dir / "review_results.json").read_text(encoding="utf-8")
    )
    packet_items = packet.get("items")
    key_items = key.get("items")
    result_items = submitted.get("results")
    if not all(
        isinstance(value, list)
        for value in (packet_items, key_items, result_items)
    ):
        raise PostGRPODPODataError("review packet/key/results 结构不正确")
    packet_by_id = {row.get("review_id"): row for row in packet_items}
    key_by_id = {row.get("review_id"): row for row in key_items}
    result_by_id = {row.get("review_id"): row for row in result_items}
    if (
        None in packet_by_id
        or set(packet_by_id) != set(key_by_id)
        or set(packet_by_id) != set(result_by_id)
        or len(packet_by_id) != len(packet_items)
        or len(key_by_id) != len(key_items)
        or len(result_by_id) != len(result_items)
    ):
        raise PostGRPODPODataError("review packet/key/results 未完整唯一对齐")
    parsed: dict[str, dict[str, Any]] = {}
    for review_id, packet_item in packet_by_id.items():
        key_item = key_by_id[review_id]
        labels = key_item.get("labels")
        if (
            key_item.get("prompt_id") != packet_item.get("prompt_id")
            or key_item.get("target_issue") != packet_item.get("target_issue")
            or not isinstance(labels, dict)
            or set(labels) != set(packet_item.get("answers", {}))
            or any(
                labels[label].get("assistant")
                != packet_item["answers"].get(label)
                for label in labels
            )
        ):
            raise PostGRPODPODataError(f"{review_id} packet/key 映射不一致")
        parsed[review_id] = parse_review_result(
            result_by_id[review_id], packet_item, key_item
        )
    return summary, prompts, candidates, packet_by_id, key_by_id, parsed


def finalize_run(
    *,
    run_dir: Path,
    train_output: Path = DEFAULT_TRAIN_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
) -> tuple[str, Path | None, Path]:
    """Validate Codex adjudication and export at most one pair per Prompt."""
    run_dir = run_dir.resolve()
    summary, prompts, candidates, packet_by_id, key_by_id, results = (
        _load_finalization_inputs(run_dir)
    )
    expansion_run = summary.get("stage") == "post_grpo_dpo_sampling"
    if expansion_run and (
        train_output.resolve() == DEFAULT_TRAIN_OUTPUT.resolve()
        or audit_output.resolve() == DEFAULT_AUDIT_OUTPUT.resolve()
    ):
        raise PostGRPODPODataError(
            "扩展采样必须显式指定新的 --train-output 和 --audit-output"
        )
    if expansion_run:
        from roleplay.post_grpo_dpo_sampling import (
            MAX_FINAL_PAIRS as run_max_pairs,
            MIN_FINAL_PAIRS as run_min_pairs,
            MIN_PAIRS_PER_ISSUE as run_min_per_issue,
        )
    else:
        run_min_pairs = MIN_FINAL_PAIRS
        run_max_pairs = MAX_FINAL_PAIRS
        run_min_per_issue = MIN_PAIRS_PER_ISSUE
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    train_rows: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    teacher_pairs = 0
    for review_id, packet_item in packet_by_id.items():
        result = results[review_id]
        decision = result["decision"]
        labels = key_by_id[review_id]["labels"]
        audit_item: dict[str, Any] = {
            "review_id": review_id,
            "prompt_id": packet_item["prompt_id"],
            "target_issue": packet_item["target_issue"],
            "decision": decision,
            "included": decision != "no_pair",
            "candidate_reviews": result["candidate_reviews"],
            "notes": result["notes"],
        }
        if decision == "no_pair":
            audit_items.append(audit_item)
            continue
        rejected_label = result["rejected_label"]
        rejected = labels[rejected_label]
        if decision == "native_pair":
            chosen_label = result["chosen_label"]
            chosen = labels[chosen_label]
            chosen_assistant = chosen["assistant"]
            chosen_source = "grpo_candidate"
            chosen_source_id = chosen["candidate_id"]
            chosen_reward = chosen["reward_v2_total"]
            audit_item["chosen_label"] = chosen_label
        else:
            teacher_pairs += 1
            teacher = result["teacher"]
            chosen_assistant = teacher["assistant"]
            chosen_source = "codex_teacher_edit"
            chosen_source_id = f"codex:{packet_item['prompt_id']}"
            chosen_reward = teacher["reward_v2"]["total_reward"]
            audit_item["teacher"] = teacher
        train_rows.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": packet_item["user"]},
                    {"role": "assistant", "content": chosen_assistant},
                ],
                "rejected_response": rejected["assistant"],
            }
        )
        audit_item.update(
            {
                "chosen_source": chosen_source,
                "chosen_source_id": chosen_source_id,
                "rejected_label": rejected_label,
                "rejected_source": "grpo_candidate",
                "rejected_source_id": rejected["candidate_id"],
                "chosen_reward_v2_total": chosen_reward,
                "rejected_reward_v2_total": rejected[
                    "reward_v2_total"
                ],
                "reward_v2_gap": chosen_reward
                - rejected["reward_v2_total"],
            }
        )
        audit_items.append(audit_item)

    reviewed_prompt_ids = {
        item["prompt_id"] for item in packet_by_id.values()
    }
    for prompt in prompts:
        if prompt["id"] in reviewed_prompt_ids:
            continue
        prompt_candidates = [
            row for row in candidates if row["prompt_id"] == prompt["id"]
        ]
        audit_items.append(
            {
                "review_id": None,
                "prompt_id": prompt["id"],
                "target_issue": prompt["target_issue"],
                "decision": "filtered_before_review",
                "included": False,
                "eligible_candidates": sum(
                    row["eligible_for_review"] for row in prompt_candidates
                ),
                "notes": "少于两个不同候选通过硬过滤并进入裁决。",
            }
        )
    audit_items.sort(key=lambda row: row["prompt_id"])
    if teacher_pairs * 4 > len(train_rows):
        raise PostGRPODPODataError(
            f"Teacher pair 为 {teacher_pairs}/{len(train_rows)}，超过25%"
        )
    issue_counts = Counter(
        row["target_issue"] for row in audit_items if row["included"]
    )
    ready = (
        run_min_pairs <= len(train_rows) <= run_max_pairs
        and all(
            issue_counts[issue] >= run_min_per_issue
            for issue in TARGET_ISSUES
        )
    )
    status = "ready_for_dpo" if ready else "insufficient_pairs"
    audit: dict[str, Any] = {
        "schema_version": 2 if expansion_run else 1,
        "source_run": str(run_dir),
        "status": status,
        "pairs": len(train_rows),
        "pairs_by_issue": {
            issue: issue_counts[issue] for issue in TARGET_ISSUES
        },
        "teacher_pairs": teacher_pairs,
        "readiness_contract": {
            "minimum_pairs": run_min_pairs,
            "maximum_pairs": run_max_pairs,
            "minimum_pairs_per_issue": run_min_per_issue,
            "teacher_pair_fraction_max": MAX_TEACHER_PAIR_FRACTION,
        },
        "candidate_sha256": sha256_file(run_dir / "candidates.jsonl"),
        "review_results_sha256": sha256_file(
            run_dir / "review_results.json"
        ),
        "items": audit_items,
    }
    final_train_path: Path | None = None
    if ready:
        if train_output.exists() or audit_output.exists():
            raise PostGRPODPODataError(
                "最终 DPO 训练集或审计文件已存在，拒绝覆盖"
            )
        write_jsonl_atomic(train_output, train_rows, refuse_overwrite=True)
        final_train_path = train_output
        audit["train_path"] = str(train_output)
        audit["train_sha256"] = sha256_file(train_output)
        write_json_atomic(audit_output, audit)
    pair_audit_path = run_dir / "pair_audit.json"
    if pair_audit_path.exists():
        raise PostGRPODPODataError("run 已存在 pair_audit.json，拒绝重复 finalize")
    write_json_atomic(pair_audit_path, audit)
    summary.update(
        {
            "status": status,
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "pair_summary": {
                "pairs": len(train_rows),
                "pairs_by_issue": audit["pairs_by_issue"],
                "teacher_pairs": teacher_pairs,
            },
        }
    )
    summary.setdefault("artifacts", {})["review_results.json"] = _artifact(
        run_dir / "review_results.json", run_dir
    )
    summary["artifacts"]["pair_audit.json"] = _artifact(
        pair_audit_path, run_dir
    )
    if ready:
        summary["final_dataset"] = {
            "train_path": str(train_output),
            "train_sha256": audit["train_sha256"],
            "audit_path": str(audit_output),
            "audit_sha256": sha256_file(audit_output),
        }
    write_json_atomic(run_dir / "run_summary.json", summary)
    return status, final_train_path, pair_audit_path


def main(argv: list[str] | None = None) -> int:
    """Validate, prepare, or finalize post-GRPO DPO preference data."""
    parser = argparse.ArgumentParser(
        description="准备 morgana-v2 post-GRPO DPO 定向偏好数据"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="校验冻结 Prompt 数据")
    prepare = subparsers.add_parser(
        "prepare", help="从 GRPO adapter 采样并生成 Codex 裁决包"
    )
    prepare.add_argument("--run-id")
    prepare.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    prepare.add_argument("--base-model", type=Path, default=BASE_SNAPSHOT)
    finalize = subparsers.add_parser(
        "finalize", help="校验裁决并导出 DPO pair"
    )
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument(
        "--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT
    )
    finalize.add_argument(
        "--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_prompt_data()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "prepare":
            run_dir = prepare_run(
                run_id=args.run_id,
                output_root=args.output_root,
                base_model_path=args.base_model,
            )
            print(f"post-GRPO DPO 数据 run: {run_dir}")
            print(f"请填写: {run_dir / 'review_results.json'}")
        else:
            status, train_path, audit_path = finalize_run(
                run_dir=args.run_dir,
                train_output=args.train_output,
                audit_output=args.audit_output,
            )
            print(f"status={status}")
            if train_path is not None:
                print(f"DPO train: {train_path}")
            print(f"pair audit: {audit_path}")
    except (OSError, PostGRPODPODataError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
