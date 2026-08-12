"""Sample post-GRPO DPO candidates on AutoDL and exchange review bundles."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

from roleplay.post_grpo_dpo_data import (
    GRPO_ADAPTER_HASHES,
    PostGRPODPODataError,
    SYSTEM_PROMPT_SHA256,
    TARGET_ISSUES,
    _canonical_user,
    _reward_v2,
    _row_users,
    _validate_split,
    build_review_artifacts,
    load_jsonl,
)
from roleplay.sft_eval import normalize_empty_think_wrapper
from roleplay.stage2_sft import (
    DEFAULT_GITHUB_REPOSITORY,
    PINNED_PACKAGES,
    Stage2SFTError,
    capture_environment,
    configure_huggingface_environment,
    create_exclusive_directory,
    generate_run_id,
    git_context,
    sha256_file,
    validate_pinned_packages,
    write_json_atomic,
)


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "965dcc54bc9c0591873df0e9869c056a54d323d1"
EXPANSION_PROMPTS_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_prompts_expansion.jsonl"
)
SAMPLING_MANIFEST_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_sampling_manifest.json"
)
EXISTING_PROMPTS_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_prompts.jsonl"
)
HOLDOUT_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_holdout.jsonl"
)
SYSTEM_PROMPT_RELATIVE_PATH = Path("data/runs/morgana-v2/system_prompt.txt")
GRPO_ADAPTER_RELATIVE_PATH = Path(
    "output/morgana-v2/stage4-grpo/20260812-2144/adapter"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "output/morgana-v2/post-grpo-dpo/sampling"
)

EXPANSION_PROMPTS_SHA256 = (
    "c2e323a40ea5c16c74c7e8727a61ae35e66eb4a3bec9eeb70290360f46ff65ed"
)
EXISTING_PROMPTS_SHA256 = (
    "cadd7c4168a746f20d7593756b32b301302ce0909a406a93e0c59cb8d8e7dae8"
)
HOLDOUT_SHA256 = (
    "59e043347e1b1436f75ed357e5edd10adef1ad2e749003dea5e195b3ec2bedde"
)
PROMPT_COUNT = 60
PROMPTS_PER_ISSUE = 20
CANDIDATES_PER_PROMPT = 8
TOTAL_CANDIDATES = PROMPT_COUNT * CANDIDATES_PER_PROMPT
BASE_BATCH_SEED = 20260813
MIN_FINAL_PAIRS = 40
MAX_FINAL_PAIRS = 60
MIN_PAIRS_PER_ISSUE = 12
MAX_TEACHER_PAIR_FRACTION = 0.25
GENERATION = {
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.45,
    "enable_thinking": False,
}
EXPECTED_ARCHIVE_FILES = frozenset(
    {
        "run_summary.json",
        "sampling.log",
        "candidates.jsonl",
        "review_packet.json",
        "review_key.json",
        "review_results.json",
    }
)


class PostGRPODPOSamplingError(RuntimeError):
    """Raised when the frozen AutoDL sampling contract is violated."""


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[2]


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostGRPODPOSamplingError(
            f"{label} 不是有效 JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise PostGRPODPOSamplingError(f"{label} 必须是 JSON 对象: {path}")
    return value


def _validate_frozen_file(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise PostGRPODPOSamplingError(f"缺少冻结文件: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise PostGRPODPOSamplingError(f"冻结文件哈希不匹配: {path}")
    return actual


def validate_expansion_prompt_data(
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate the 60 new prompts, manifest, and historical isolation."""
    root = repository_root() if root is None else root.resolve()
    prompt_path = root / EXPANSION_PROMPTS_RELATIVE_PATH
    manifest_path = root / SAMPLING_MANIFEST_RELATIVE_PATH
    existing_path = root / EXISTING_PROMPTS_RELATIVE_PATH
    holdout_path = root / HOLDOUT_RELATIVE_PATH
    system_path = root / SYSTEM_PROMPT_RELATIVE_PATH
    manifest = _read_json_object(manifest_path, "sampling manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "post_grpo_dpo_sampling"
        or manifest.get("objective") != list(TARGET_ISSUES)
    ):
        raise PostGRPODPOSamplingError("sampling manifest 头部不正确")
    policy = manifest.get("policy")
    expected_policy = {
        "one_target_issue_per_prompt": True,
        "holdout_must_not_enter_sampling": True,
        "existing_prompts_must_not_enter_sampling": True,
        "candidate_rounds": 1,
        "candidates_per_prompt": CANDIDATES_PER_PROMPT,
        "resampling": False,
    }
    if policy != expected_policy:
        raise PostGRPODPOSamplingError("sampling manifest policy 不正确")

    rows = _validate_split(
        prompt_path,
        id_prefix="post_dpo_exp_",
        expected_count=PROMPT_COUNT,
        expected_per_issue=PROMPTS_PER_ISSUE,
    )
    expected_entries = {
        "expansion_prompts": {
            "path": str(EXPANSION_PROMPTS_RELATIVE_PATH),
            "records": PROMPT_COUNT,
            "per_issue": PROMPTS_PER_ISSUE,
            "sha256": EXPANSION_PROMPTS_SHA256,
        },
        "excluded_existing_prompts": {
            "path": str(EXISTING_PROMPTS_RELATIVE_PATH),
            "records": 30,
            "sha256": EXISTING_PROMPTS_SHA256,
        },
        "excluded_holdout_prompts": {
            "path": str(HOLDOUT_RELATIVE_PATH),
            "records": 9,
            "sha256": HOLDOUT_SHA256,
        },
        "system_prompt": {
            "path": str(SYSTEM_PROMPT_RELATIVE_PATH),
            "sha256": SYSTEM_PROMPT_SHA256,
        },
    }
    for key, expected in expected_entries.items():
        if manifest.get(key) != expected:
            raise PostGRPODPOSamplingError(
                f"sampling manifest {key} 不匹配"
            )
    _validate_frozen_file(prompt_path, EXPANSION_PROMPTS_SHA256)
    _validate_frozen_file(existing_path, EXISTING_PROMPTS_SHA256)
    _validate_frozen_file(holdout_path, HOLDOUT_SHA256)
    _validate_frozen_file(system_path, SYSTEM_PROMPT_SHA256)

    expansion_users = {
        _canonical_user(row["user"]): row["id"] for row in rows
    }
    comparison_paths = [
        path
        for path in (root / "data/runs/morgana-v2").glob("*.jsonl")
        if path.resolve() != prompt_path.resolve()
    ]
    comparison_paths.append(root / "data/style_examples.jsonl")
    for path in comparison_paths:
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            for user in _row_users(row):
                prompt_id = expansion_users.get(_canonical_user(user))
                if prompt_id is not None:
                    raise PostGRPODPOSamplingError(
                        f"{prompt_id} 与已有 split 重复: {path}"
                    )
    return rows


def validate_grpo_adapter(adapter_dir: Path) -> dict[str, str]:
    """Require the exact accepted 20260812-2144 GRPO adapter."""
    return {
        name: _validate_frozen_file(adapter_dir / name, expected)
        for name, expected in GRPO_ADAPTER_HASHES.items()
    }


def validate_model_revision() -> str:
    """Resolve the frozen Hugging Face model revision before loading."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise PostGRPODPOSamplingError(
            "缺少模型校验依赖: huggingface_hub"
        ) from exc
    actual = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    if actual != MODEL_REVISION:
        raise PostGRPODPOSamplingError(
            f"模型 revision 不匹配: {actual}"
        )
    return actual


def batch_seed(prompt_id: str) -> int:
    """Return the deterministic seed for one eight-candidate batch."""
    match = re.fullmatch(r"post_dpo_exp_(\d{4})", prompt_id)
    if not match:
        raise ValueError(f"扩展 Prompt id 无效: {prompt_id}")
    index = int(match.group(1))
    if not 1 <= index <= PROMPT_COUNT:
        raise ValueError(f"扩展 Prompt id 越界: {prompt_id}")
    return BASE_BATCH_SEED + index - 1


def candidate_id(prompt_id: str, batch_position: int) -> str:
    """Return the stable candidate id for one batch position."""
    batch_seed(prompt_id)
    if not 1 <= batch_position <= CANDIDATES_PER_PROMPT:
        raise ValueError("候选批次位置无效")
    return f"{prompt_id}-c{batch_position}"


def _canonical_assistant(text: str) -> str:
    return " ".join(text.strip().split())


class TransformersSamplingRuntime:
    """One loaded 4-bit Transformers engine used for all prompt batches."""

    def __init__(self, adapter_dir: Path):
        try:
            import torch
            from peft import PeftModel
            from swift import (
                InferRequest,
                RequestConfig,
                TransformersEngine,
                get_model_processor,
                get_template,
            )
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise PostGRPODPOSamplingError(
                f"缺少采样依赖: {exc.name}"
            ) from exc
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model, processor = get_model_processor(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.float32,
            quantization_config=quantization,
            use_hf=True,
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        self._torch = torch
        self._model = model
        self._processor = processor
        self._infer_request = InferRequest
        self._request_config = RequestConfig(
            max_tokens=GENERATION["max_tokens"],
            temperature=GENERATION["temperature"],
            top_p=GENERATION["top_p"],
            top_k=GENERATION["top_k"],
            repetition_penalty=GENERATION["repetition_penalty"],
        )
        self._engine = TransformersEngine(
            model,
            template=get_template(processor, enable_thinking=False),
            max_batch_size=CANDIDATES_PER_PROMPT,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        count: int,
        seed: int,
    ) -> list[tuple[str, str]]:
        random.seed(seed)
        self._torch.manual_seed(seed)
        self._torch.cuda.manual_seed_all(seed)
        requests = [
            self._infer_request(messages=messages) for _ in range(count)
        ]
        responses = self._engine.infer(
            requests, request_config=self._request_config
        )
        if len(responses) != count:
            raise PostGRPODPOSamplingError(
                f"批量推理应返回 {count} 条，实际 {len(responses)}"
            )
        generated: list[tuple[str, str]] = []
        for response in responses:
            choices = getattr(response, "choices", None)
            if not isinstance(choices, Sequence) or len(choices) != 1:
                raise PostGRPODPOSamplingError("批量推理响应 choices 不正确")
            choice = choices[0]
            raw = getattr(getattr(choice, "message", None), "content", None)
            finish_reason = getattr(choice, "finish_reason", None)
            if not isinstance(raw, str) or not isinstance(finish_reason, str):
                raise PostGRPODPOSamplingError("批量推理响应内容不正确")
            generated.append((raw, finish_reason))
        return generated

    def close(self) -> None:
        del self._engine, self._model, self._processor
        gc.collect()
        self._torch.cuda.empty_cache()


def load_sampling_runtime(adapter_dir: Path) -> TransformersSamplingRuntime:
    """Load one sampling runtime; separated for deterministic fake tests."""
    return TransformersSamplingRuntime(adapter_dir)


def build_candidate_records(
    prompt: dict[str, Any],
    messages: list[dict[str, str]],
    responses: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Convert one eight-response batch into auditable candidate records."""
    if len(responses) != CANDIDATES_PER_PROMPT:
        raise PostGRPODPOSamplingError("每题必须返回8个候选")
    seed = batch_seed(prompt["id"])
    records: list[dict[str, Any]] = []
    first_by_text: dict[str, str] = {}
    for position, response in enumerate(responses, 1):
        raw, finish_reason = response
        if not isinstance(raw, str) or not isinstance(finish_reason, str):
            raise PostGRPODPOSamplingError("候选响应类型不正确")
        assistant = normalize_empty_think_wrapper(raw)
        reward, hard_valid = _reward_v2(assistant, finish_reason)
        current_id = candidate_id(prompt["id"], position)
        duplicate_of: str | None = None
        if hard_valid:
            canonical = _canonical_assistant(assistant)
            duplicate_of = first_by_text.get(canonical)
            if duplicate_of is None:
                first_by_text[canonical] = current_id
        exclusion_reasons = (
            list(reward["hard_invalid_reasons"])
            if not hard_valid
            else (["duplicate_candidate"] if duplicate_of else [])
        )
        records.append(
            {
                "candidate_id": current_id,
                "prompt_id": prompt["id"],
                "target_issue": prompt["target_issue"],
                "scenario": prompt["scenario"],
                "preference_criteria": prompt["preference_criteria"],
                "batch_seed": seed,
                "batch_position": position,
                "messages": messages,
                "raw_assistant": raw,
                "assistant": assistant,
                "finish_reason": finish_reason,
                "is_truncated": finish_reason in {"length", "max_tokens"},
                "source": "grpo_candidate",
                "base_model": MODEL_ID,
                "base_revision": MODEL_REVISION,
                "source_adapter": {
                    "path": str(GRPO_ADAPTER_RELATIVE_PATH),
                    "sha256": GRPO_ADAPTER_HASHES,
                },
                "generation": GENERATION,
                "reward_v2": reward,
                "hard_valid": hard_valid,
                "duplicate_of_candidate_id": duplicate_of,
                "review_exclusion_reasons": exclusion_reasons,
                "eligible_for_review": hard_valid and duplicate_of is None,
            }
        )
    return records


def validate_sampling_candidate_rows(
    rows: Sequence[dict[str, Any]],
    prompts: Sequence[dict[str, Any]],
    system_prompt: str,
) -> None:
    """Require exactly 480 frozen, batch-reproducible candidate rows."""
    required = {
        "candidate_id",
        "prompt_id",
        "target_issue",
        "scenario",
        "preference_criteria",
        "batch_seed",
        "batch_position",
        "messages",
        "raw_assistant",
        "assistant",
        "finish_reason",
        "is_truncated",
        "source",
        "base_model",
        "base_revision",
        "source_adapter",
        "generation",
        "reward_v2",
        "hard_valid",
        "duplicate_of_candidate_id",
        "review_exclusion_reasons",
        "eligible_for_review",
    }
    prompt_by_id = {row["id"]: row for row in prompts}
    seen_ids: set[str] = set()
    seeds_by_prompt: dict[str, int] = {}
    rows_by_prompt: dict[str, list[dict[str, Any]]] = {
        prompt_id: [] for prompt_id in prompt_by_id
    }
    for row in rows:
        if set(row) != required:
            raise PostGRPODPOSamplingError("采样候选字段不正确")
        prompt = prompt_by_id.get(row["prompt_id"])
        position = row["batch_position"]
        if prompt is None or isinstance(position, bool) or not isinstance(
            position, int
        ):
            raise PostGRPODPOSamplingError("候选引用未知 Prompt 或位置无效")
        expected_id = candidate_id(prompt["id"], position)
        expected_seed = batch_seed(prompt["id"])
        expected_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt["user"]},
        ]
        if (
            row["candidate_id"] != expected_id
            or row["batch_seed"] != expected_seed
            or row["target_issue"] != prompt["target_issue"]
            or row["scenario"] != prompt["scenario"]
            or row["preference_criteria"] != prompt["preference_criteria"]
            or row["messages"] != expected_messages
            or row["assistant"]
            != normalize_empty_think_wrapper(row["raw_assistant"])
            or row["source"] != "grpo_candidate"
            or row["base_model"] != MODEL_ID
            or row["base_revision"] != MODEL_REVISION
            or row["source_adapter"]
            != {
                "path": str(GRPO_ADAPTER_RELATIVE_PATH),
                "sha256": GRPO_ADAPTER_HASHES,
            }
            or row["generation"] != GENERATION
        ):
            raise PostGRPODPOSamplingError(
                f"候选冻结内容不正确: {expected_id}"
            )
        reward, hard_valid = _reward_v2(
            row["assistant"], row["finish_reason"]
        )
        if row["reward_v2"] != reward or row["hard_valid"] is not hard_valid:
            raise PostGRPODPOSamplingError(
                f"候选 Reward v2 不正确: {expected_id}"
            )
        if row["is_truncated"] is not (
            row["finish_reason"] in {"length", "max_tokens"}
        ):
            raise PostGRPODPOSamplingError(
                f"候选截断状态不正确: {expected_id}"
            )
        if expected_id in seen_ids:
            raise PostGRPODPOSamplingError("候选 id 重复")
        seen_ids.add(expected_id)
        if prompt["id"] in seeds_by_prompt and (
            seeds_by_prompt[prompt["id"]] != expected_seed
        ):
            raise PostGRPODPOSamplingError("同题 batch seed 不一致")
        seeds_by_prompt[prompt["id"]] = expected_seed
        rows_by_prompt[prompt["id"]].append(row)

    if len(rows) != TOTAL_CANDIDATES:
        raise PostGRPODPOSamplingError(
            f"候选应为 {TOTAL_CANDIDATES} 条，实际 {len(rows)}"
        )
    if len(set(seeds_by_prompt.values())) != PROMPT_COUNT:
        raise PostGRPODPOSamplingError("不同 Prompt 的 batch seed 必须唯一")
    for prompt_id, prompt_rows in rows_by_prompt.items():
        if len(prompt_rows) != CANDIDATES_PER_PROMPT:
            raise PostGRPODPOSamplingError(f"{prompt_id} 候选数不是8")
        ordered = sorted(prompt_rows, key=lambda row: row["batch_position"])
        first_by_text: dict[str, str] = {}
        for row in ordered:
            hard_valid = row["hard_valid"]
            duplicate_of = None
            if hard_valid:
                canonical = _canonical_assistant(row["assistant"])
                duplicate_of = first_by_text.get(canonical)
                if duplicate_of is None:
                    first_by_text[canonical] = row["candidate_id"]
            expected_reasons = (
                list(row["reward_v2"]["hard_invalid_reasons"])
                if not hard_valid
                else (["duplicate_candidate"] if duplicate_of else [])
            )
            if (
                row["duplicate_of_candidate_id"] != duplicate_of
                or row["review_exclusion_reasons"] != expected_reasons
                or row["eligible_for_review"]
                is not (hard_valid and duplicate_of is None)
            ):
                raise PostGRPODPOSamplingError(
                    f"候选过滤状态不正确: {row['candidate_id']}"
                )


def _append_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressLogger:
    """Mirror flushed progress messages to the terminal and sampling.log."""

    def __init__(self, path: Path):
        self._output = path.open("x", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._output.write(message + "\n")
        self._output.flush()

    def close(self) -> None:
        self._output.close()


def validate_archive_contract(run_dir: Path) -> None:
    """Require an exact complete sampling archive before publishing."""
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != EXPECTED_ARCHIVE_FILES:
        raise PostGRPODPOSamplingError(
            "采样归档目录不匹配: "
            f"missing={sorted(EXPECTED_ARCHIVE_FILES - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_ARCHIVE_FILES)}"
        )
    summary = _read_json_object(run_dir / "run_summary.json", "run summary")
    if (
        summary.get("stage") != "post_grpo_dpo_sampling"
        or summary.get("status") != "awaiting_codex_review"
        or summary.get("counts", {}).get("candidates") != TOTAL_CANDIDATES
    ):
        raise PostGRPODPOSamplingError("采样 run 尚未完整完成")
    for name in EXPECTED_ARCHIVE_FILES - {"run_summary.json"}:
        if summary.get("artifacts", {}).get(name) != _artifact(
            run_dir / name, run_dir
        ):
            raise PostGRPODPOSamplingError(f"采样产物元数据不匹配: {name}")


def run_sampling(output_root: Path | None = None) -> Path:
    """Run one frozen 60-prompt, eight-candidate AutoDL sampling pass."""
    repo_dir = repository_root()
    output_root = (
        repo_dir / DEFAULT_OUTPUT_RELATIVE_PATH
        if output_root is None
        else output_root.resolve()
    )
    run_id = generate_run_id()
    run_dir = output_root / run_id
    work_dir = output_root / ".work" / run_id
    if run_dir.exists() or work_dir.exists():
        raise PostGRPODPOSamplingError(f"运行目录已存在: {run_id}")
    create_exclusive_directory(work_dir)
    summary_path = work_dir / "run_summary.json"
    log_path = work_dir / "sampling.log"
    candidates_path = work_dir / "candidates.jsonl"
    started = time.monotonic()
    summary: dict[str, Any] = {
        "schema_version": 2,
        "stage": "post_grpo_dpo_sampling",
        "status": "starting",
        "run": {
            "id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "contract": {
            "prompts": PROMPT_COUNT,
            "candidates_per_prompt": CANDIDATES_PER_PROMPT,
            "candidate_rounds": 1,
            "base_batch_seed": BASE_BATCH_SEED,
            "generation": GENERATION,
            "resampling": False,
            "minimum_final_pairs": MIN_FINAL_PAIRS,
            "maximum_final_pairs": MAX_FINAL_PAIRS,
            "minimum_pairs_per_issue": MIN_PAIRS_PER_ISSUE,
            "teacher_pair_fraction_max": MAX_TEACHER_PAIR_FRACTION,
        },
        "progress": {
            "completed_prompts": 0,
            "completed_candidates": 0,
            "eligible_candidates": 0,
            "last_prompt_id": None,
        },
        "artifacts": {},
    }
    write_json_atomic(summary_path, summary)
    logger = ProgressLogger(log_path)
    runtime: Any | None = None
    try:
        logger.log("[1/4] 校验环境和冻结输入")
        huggingface = configure_huggingface_environment()
        environment, torch_module = capture_environment()
        packages = validate_pinned_packages(PINNED_PACKAGES)
        summary["run"].update(git_context(repo_dir))
        prompts = validate_expansion_prompt_data(repo_dir)
        system_path = repo_dir / SYSTEM_PROMPT_RELATIVE_PATH
        system_prompt = system_path.read_text(encoding="utf-8")
        adapter_dir = repo_dir / GRPO_ADAPTER_RELATIVE_PATH
        adapter_hashes = validate_grpo_adapter(adapter_dir)
        validate_model_revision()
        summary.update(
            {
                "status": "loading_model",
                "environment": environment,
                "huggingface": huggingface,
                "packages": packages,
                "model": {"name": MODEL_ID, "revision": MODEL_REVISION},
                "inputs": {
                    str(EXPANSION_PROMPTS_RELATIVE_PATH): {
                        "records": len(prompts),
                        "sha256": EXPANSION_PROMPTS_SHA256,
                    },
                    str(SAMPLING_MANIFEST_RELATIVE_PATH): {
                        "sha256": sha256_file(
                            repo_dir / SAMPLING_MANIFEST_RELATIVE_PATH
                        )
                    },
                    str(EXISTING_PROMPTS_RELATIVE_PATH): {
                        "records": 30,
                        "sha256": EXISTING_PROMPTS_SHA256,
                        "sampled": False,
                    },
                    str(HOLDOUT_RELATIVE_PATH): {
                        "records": 9,
                        "sha256": HOLDOUT_SHA256,
                        "sampled": False,
                    },
                    str(SYSTEM_PROMPT_RELATIVE_PATH): {
                        "sha256": SYSTEM_PROMPT_SHA256
                    },
                    str(GRPO_ADAPTER_RELATIVE_PATH): adapter_hashes,
                },
            }
        )
        write_json_atomic(summary_path, summary)
        del torch_module

        logger.log("[2/4] 加载 4-bit Base 与 20260812-2144 GRPO adapter")
        runtime = load_sampling_runtime(adapter_dir)
        summary["status"] = "sampling"
        write_json_atomic(summary_path, summary)
        sampling_started = time.monotonic()
        all_candidates: list[dict[str, Any]] = []
        eligible_count = 0
        for prompt_index, prompt in enumerate(prompts, 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt["user"]},
            ]
            responses = runtime.generate(
                messages,
                CANDIDATES_PER_PROMPT,
                batch_seed(prompt["id"]),
            )
            prompt_candidates = build_candidate_records(
                prompt, messages, responses
            )
            _append_jsonl_rows(candidates_path, prompt_candidates)
            all_candidates.extend(prompt_candidates)
            eligible_count += sum(
                row["eligible_for_review"] for row in prompt_candidates
            )
            sampling_elapsed = time.monotonic() - sampling_started
            eta = (
                sampling_elapsed / prompt_index * (PROMPT_COUNT - prompt_index)
            )
            completed_candidates = prompt_index * CANDIDATES_PER_PROMPT
            total_elapsed = time.monotonic() - started
            summary["progress"] = {
                "completed_prompts": prompt_index,
                "completed_candidates": completed_candidates,
                "eligible_candidates": eligible_count,
                "last_prompt_id": prompt["id"],
                "elapsed_seconds": round(total_elapsed, 3),
                "eta_seconds": round(eta, 3),
            }
            write_json_atomic(summary_path, summary)
            logger.log(
                f"[3/4] Prompt {prompt_index}/{PROMPT_COUNT} | "
                f"candidates {completed_candidates}/{TOTAL_CANDIDATES} | "
                f"eligible {eligible_count} | "
                f"elapsed {_format_duration(total_elapsed)} | "
                f"ETA {_format_duration(eta)}"
            )

        runtime.close()
        runtime = None
        validate_sampling_candidate_rows(
            all_candidates, prompts, system_prompt
        )
        logger.log("[4/4] 校验候选、生成匿名裁决包和归档")
        packet, key, results, unresolved = build_review_artifacts(
            prompts, all_candidates
        )
        packet_path = work_dir / "review_packet.json"
        key_path = work_dir / "review_key.json"
        results_path = work_dir / "review_results.json"
        write_json_atomic(packet_path, packet)
        write_json_atomic(key_path, key)
        write_json_atomic(results_path, results)
        logger.log(
            f"采样完成：480 candidates，{eligible_count} eligible，"
            f"{len(packet['items'])} review items。"
        )
        logger.close()
        summary.update(
            {
                "status": "awaiting_codex_review",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "counts": {
                    "prompts": len(prompts),
                    "candidates": len(all_candidates),
                    "eligible_candidates": eligible_count,
                    "review_items": len(packet["items"]),
                    "unresolved_prompts": len(unresolved),
                },
                "unresolved_prompt_ids": unresolved,
            }
        )
        artifact_paths = {
            "sampling.log": log_path,
            "candidates.jsonl": candidates_path,
            "review_packet.json": packet_path,
            "review_key.json": key_path,
            "review_results.json": results_path,
        }
        summary["artifacts"] = {
            name: _artifact(path, work_dir)
            for name, path in artifact_paths.items()
        }
        write_json_atomic(summary_path, summary)
        validate_archive_contract(work_dir)
        work_dir.replace(run_dir)
        return run_dir
    except BaseException as exc:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass
        try:
            logger.log(f"采样失败：{type(exc).__name__}: {exc}")
            logger.close()
        except (OSError, ValueError):
            pass
        summary["status"] = "sampling_failed"
        summary["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        summary["retained_work_dir"] = str(work_dir)
        write_json_atomic(summary_path, summary)
        raise


def _bundle_paths(output_dir: Path, run_id: str) -> tuple[Path, Path, str]:
    tag = f"morgana-v2-post-grpo-dpo-sampling-{run_id}"
    return (
        output_dir / f"{tag}.tar.gz",
        output_dir / f"{tag}.manifest.json",
        tag,
    )


def create_release_bundle(
    run_dir: Path, output_dir: Path
) -> tuple[Path, Path, str]:
    """Create or reuse one exact, content-addressed sampling bundle."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    validate_archive_contract(run_dir)
    summary = _read_json_object(run_dir / "run_summary.json", "run summary")
    run_id = summary.get("run", {}).get("id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise PostGRPODPOSamplingError("run summary 缺少有效 run.id")
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise PostGRPODPOSamplingError("发布包目录不能位于 run 目录内")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, manifest_path, tag = _bundle_paths(output_dir, run_id)
    contents = {
        name: _artifact(run_dir / name, run_dir)
        for name in sorted(EXPECTED_ARCHIVE_FILES)
    }
    if bundle.exists() or manifest_path.exists():
        if not bundle.is_file() or not manifest_path.is_file():
            raise PostGRPODPOSamplingError("发布包或 manifest 不完整")
        manifest = _read_json_object(manifest_path, "release manifest")
        if (
            manifest.get("contents") != contents
            or manifest.get("bundle", {}).get("sha256")
            != sha256_file(bundle)
        ):
            raise PostGRPODPOSamplingError("现有发布包与 run 不一致")
        return bundle, manifest_path, tag
    temporary = output_dir / f".{bundle.name}.{uuid4().hex}.tmp"
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name in sorted(EXPECTED_ARCHIVE_FILES):
                archive.add(run_dir / name, arcname=f"{run_id}/{name}")
        temporary.replace(bundle)
        manifest = {
            "schema_version": 1,
            "stage": "post_grpo_dpo_sampling",
            "run_id": run_id,
            "run_status": summary["status"],
            "source_commit": summary.get("run", {}).get("commit"),
            "github_release_tag": tag,
            "bundle": {
                "file": bundle.name,
                "bytes": bundle.stat().st_size,
                "sha256": sha256_file(bundle),
            },
            "contents": contents,
        }
        write_json_atomic(manifest_path, manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        bundle.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return bundle, manifest_path, tag


def publish_run(
    run_dir: Path,
    output_dir: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> tuple[Path, Path, str]:
    """Publish one validated sampling bundle to GitHub Releases."""
    bundle, manifest_path, tag = create_release_bundle(run_dir, output_dir)
    manifest = _read_json_object(manifest_path, "release manifest")
    command = [
        "gh",
        "release",
        "create",
        tag,
        str(bundle),
        str(manifest_path),
        "--repo",
        repository,
        "--title",
        f"morgana-v2 post-GRPO DPO sampling {manifest['run_id']}",
        "--notes",
        "480条 post-GRPO DPO 候选与本地 Codex 裁决材料。",
    ]
    source_commit = manifest.get("source_commit")
    if isinstance(source_commit, str) and source_commit:
        command.extend(["--target", source_commit])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise PostGRPODPOSamplingError("未安装 GitHub CLI：gh") from exc
    except subprocess.CalledProcessError as exc:
        raise PostGRPODPOSamplingError(
            "GitHub Release 上传失败；本地发布包已保留"
        ) from exc
    return bundle, manifest_path, tag


def format_download_command(
    tag: str, repository: str = DEFAULT_GITHUB_REPOSITORY
) -> str:
    command = ["roleplay-post-grpo-dpo-sampling", "download", "--tag", tag]
    if repository != DEFAULT_GITHUB_REPOSITORY:
        command.extend(["--repo", repository])
    return shlex.join(command)


def extract_release_bundle(
    bundle: Path, manifest_path: Path, output_root: Path
) -> Path:
    """Verify and safely extract one downloaded sampling release."""
    bundle = bundle.resolve()
    manifest = _read_json_object(manifest_path, "release manifest")
    if manifest.get("stage") != "post_grpo_dpo_sampling":
        raise PostGRPODPOSamplingError("manifest stage 不正确")
    if manifest.get("bundle", {}).get("sha256") != sha256_file(bundle):
        raise PostGRPODPOSamplingError("下载包 SHA-256 与 manifest 不匹配")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise PostGRPODPOSamplingError("manifest 缺少有效 run_id")
    contents = manifest.get("contents")
    if not isinstance(contents, dict) or set(contents) != EXPECTED_ARCHIVE_FILES:
        raise PostGRPODPOSamplingError("manifest 文件清单不满足归档契约")
    expected_names = {f"{run_id}/{name}" for name in contents}
    output_root = output_root.resolve()
    destination = output_root / run_id
    if destination.exists():
        raise PostGRPODPOSamplingError(
            f"本地 run 目录已存在，拒绝覆盖: {destination}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".download-{uuid4().hex}"
    staging.mkdir()
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != expected_names or not all(
                member.isfile() for member in members
            ):
                raise PostGRPODPOSamplingError(
                    "发布包内容与 manifest 不一致"
                )
            for member in members:
                target = (staging / member.name).resolve()
                try:
                    target.relative_to(staging)
                except ValueError as exc:
                    raise PostGRPODPOSamplingError(
                        "发布包包含不安全路径"
                    ) from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise PostGRPODPOSamplingError(
                        f"无法读取发布包文件: {member.name}"
                    )
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        staged_run = staging / run_id
        for relative, metadata in contents.items():
            path = staged_run / relative
            if (
                not path.is_file()
                or path.stat().st_size != metadata.get("bytes")
                or sha256_file(path) != metadata.get("sha256")
            ):
                raise PostGRPODPOSamplingError(
                    f"解包文件校验失败: {relative}"
                )
        staged_run.replace(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def download_release(
    tag: str,
    output_root: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> Path:
    """Download, validate, and extract one sampling GitHub Release."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise PostGRPODPOSamplingError("release tag 格式无效")
    with tempfile.TemporaryDirectory() as temporary:
        download_dir = Path(temporary)
        try:
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    tag,
                    "--repo",
                    repository,
                    "--dir",
                    str(download_dir),
                ],
                check=True,
            )
        except FileNotFoundError as exc:
            raise PostGRPODPOSamplingError("未安装 GitHub CLI：gh") from exc
        except subprocess.CalledProcessError as exc:
            raise PostGRPODPOSamplingError(
                f"GitHub Release 下载失败: {tag}"
            ) from exc
        bundle = download_dir / f"{tag}.tar.gz"
        manifest = download_dir / f"{tag}.manifest.json"
        if not bundle.is_file() or not manifest.is_file():
            raise PostGRPODPOSamplingError("Release 缺少发布包或 manifest")
        return extract_release_bundle(bundle, manifest, output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 AutoDL 批量采样 post-GRPO DPO 候选"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="采样60条 Prompt × 每题8候选（实时显示进度）"
    )
    run_parser.add_argument("--output-root", type=Path)
    publish_parser = subparsers.add_parser(
        "publish", help="打包并上传到 GitHub Release"
    )
    publish_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    publish_parser.add_argument("--repo", default=DEFAULT_GITHUB_REPOSITORY)
    download_parser = subparsers.add_parser(
        "download", help="从 GitHub Release 下载并校验解包"
    )
    download_parser.add_argument("--tag", required=True)
    download_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_RELATIVE_PATH
    )
    download_parser.add_argument("--repo", default=DEFAULT_GITHUB_REPOSITORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_dir = run_sampling(args.output_root)
            print(f"采样 run: {run_dir}")
            print("ready_to_publish=True")
        elif args.command == "publish":
            bundle, manifest, tag = publish_run(
                args.run_dir, args.output_dir, args.repo
            )
            print(f"GitHub Release: {tag}")
            print(f"发布包: {bundle}")
            print(f"清单: {manifest}")
            print(f"本地下载命令: {format_download_command(tag, args.repo)}")
        else:
            run_dir = download_release(args.tag, args.output_root, args.repo)
            print(f"本地采样目录: {run_dir}")
            print("请交给 Codex 填写 review_results.json 后运行 finalize。")
    except (
        OSError,
        PostGRPODPODataError,
        PostGRPODPOSamplingError,
        Stage2SFTError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
