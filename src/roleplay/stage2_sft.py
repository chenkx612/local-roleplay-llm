"""Run the frozen morgana-v2 Stage 2 SFT workflow on AutoDL."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roleplay.core.artifacts import (
    read_jsonl as _read_jsonl,
    repository_root as _repository_root,
    sha256_file,
    write_json_atomic,
    write_jsonl_exclusive,
)
from roleplay.core.release import (
    ReleaseSpec,
    create_release_bundle as _create_release_bundle,
    download_release as _download_release,
    extract_release_bundle as _extract_release_bundle,
    format_download_command as _format_download_command,
    publish_release as _publish_release,
)
from roleplay.core.runtime import (
    ADAPTER_FILES,
    DEFAULT_GITHUB_REPOSITORY,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_HF_HOME,
    DISABLED_ACCELERATION_PACKAGES,
    EXPECTED_TRAINING_PRECISION,
    PINNED_PACKAGES,
    capture_environment as _capture_environment,
    configure_huggingface_environment as _configure_huggingface_environment,
    create_exclusive_directory as _create_exclusive_directory,
    ensure_clean_tracked_status as _ensure_clean_tracked_status,
    find_final_adapter as _find_final_adapter,
    generate_run_id,
    git_context as _git_context,
    normalized_package_version,
    read_metric,
    run_logged as _run_logged,
    training_progress,
    validate_effective_training_args as _validate_effective_training_args,
    validate_environment_snapshot as _validate_environment_snapshot,
    validate_pinned_packages as _validate_pinned_packages,
)
from roleplay.experiments.morgana_v2 import MODEL_ID, MODEL_REVISION
from roleplay.sft_eval import (
    EVALUATION_SEEDS,
    PRIMARY_EVALUATION_SEED,
    build_manual_review,
    empty_manual_review_results,
    evaluate_core_behavior_gate,
    evaluate_manual_review,
    normalize_empty_think_wrapper,
)


CONFIG_RELATIVE_PATH = Path("configs/morgana_v2_sft.yaml")
DEFAULT_OUTPUT_RELATIVE_PATH = Path("output/morgana-v2/stage2-sft")
MIN_GPU_MEMORY_GIB = 20.0

EXPECTED_INPUTS = {
    "data/runs/morgana-v2/sft_train.jsonl": {
        "records": 62,
        "sha256": "c1ec8824db45db98f0e82547938a67e652fe75b759278d98aef5d0552daab142",
    },
    "data/runs/morgana-v2/sft_targeted_additions.jsonl": {
        "records": 12,
        "sha256": "c299a2f9de1542340d25f92c9947cc8fb172aabb4b4668428b4aa5c9cbcd5a60",
    },
    "data/runs/morgana-v2/dev.jsonl": {
        "records": 10,
        "sha256": "74cf6d05921155cec5c070ca8a611c7a8e6751b00ca0b77a6f4e9085aeeecb22",
    },
}

EXPECTED_FILE_HASHES = {
    "data/runs/morgana-v2/inputs/persona.json": (
        "42010082a1db9afcbf15cfed077dd59d4c7a0a0d8f44510292f00fe7ef87a10a"
    ),
    "data/runs/morgana-v2/inputs/style_examples.jsonl": (
        "b8fa53f494d3202fe80aad90be4e2db098f56ac080bc152644ba3661e6104250"
    ),
    "data/runs/morgana-v2/system_prompt.txt": (
        "d88993aaa1178ced740f6b54530a27e5fcdb2486a66d8b460367e842b53ee112"
    ),
    "data/runs/morgana-v2/base_dev_outputs.jsonl": (
        "e3e1f00d904f8f5f27da7c8f86d68c43900a2fab498f668fe9c33b6f4b8335ac"
    ),
    "data/runs/morgana-v2/base_generation_meta.json": (
        "7f947cf8e7ce8beefe7aad5cf6665fc33c8f9f9215a17829e473966635bbc245"
    ),
}

EXPECTED_ARCHIVE_FILES = frozenset(
    {
        "run_summary.json",
        "training_config.yaml",
        *(f"adapter/{name}" for name in ADAPTER_FILES),
        "hf_base_dev_outputs.jsonl",
        "dev_outputs.jsonl",
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
    }
)
LOCAL_ONLY_ARCHIVE_FILES = frozenset({"train.log"})


class Stage2SFTError(RuntimeError):
    """Raised when the frozen Stage 2 execution contract is violated."""


RELEASE_SPEC = ReleaseSpec(
    tag_prefix="morgana-v2-stage2-sft",
    cli_name="roleplay-stage2-sft",
    title="morgana-v2 Stage 2 SFT {run_id}",
    notes="Stage 2 SFT 精简归档；人工复核在下载到本地后完成。",
    expected_files=EXPECTED_ARCHIVE_FILES,
    contract_label="",
    default_repository=DEFAULT_GITHUB_REPOSITORY,
)


def configure_huggingface_environment(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    return _configure_huggingface_environment(environment)


def repository_root() -> Path:
    return _repository_root()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path)
    except ValueError as exc:
        raise Stage2SFTError(str(exc)) from exc


def validate_environment_snapshot(snapshot: dict[str, Any]) -> None:
    _validate_environment_snapshot(
        snapshot,
        error_type=Stage2SFTError,
        min_gpu_memory_gib=MIN_GPU_MEMORY_GIB,
    )


def capture_environment() -> tuple[dict[str, Any], Any]:
    return _capture_environment(
        error_type=Stage2SFTError,
        run_command=subprocess.run,
        min_gpu_memory_gib=MIN_GPU_MEMORY_GIB,
    )


def _format_environment_status(snapshot: dict[str, Any]) -> str:
    """Format the validated environment snapshot for the startup log."""
    return (
        f"GPU {snapshot['gpu']} | "
        f"显存 {snapshot['gpu_memory_gib']:.1f} GiB"
    )


def validate_pinned_packages(
    required_packages: Mapping[str, str] | None = None,
    disabled_packages: Sequence[str] | None = None,
) -> dict[str, str]:
    return _validate_pinned_packages(
        PINNED_PACKAGES if required_packages is None else required_packages,
        (
            DISABLED_ACCELERATION_PACKAGES
            if disabled_packages is None
            else disabled_packages
        ),
        error_type=Stage2SFTError,
        version_lookup=importlib.metadata.version,
    )


def ensure_clean_tracked_status(status: str) -> None:
    _ensure_clean_tracked_status(status, error_type=Stage2SFTError)


def git_context(repo_dir: Path) -> dict[str, str]:
    return _git_context(
        repo_dir,
        error_type=Stage2SFTError,
        run_command=subprocess.run,
    )


def create_exclusive_directory(path: Path) -> Path:
    return _create_exclusive_directory(path, error_type=Stage2SFTError)


def validate_file_manifest(
    base_dir: Path, manifest: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Validate file hashes and optional JSONL record counts."""
    validated: dict[str, dict[str, Any]] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for relative_path, expected in manifest.items():
        path = base_dir / relative_path
        if not path.is_file():
            raise Stage2SFTError(f"缺少冻结输入: {relative_path}")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise Stage2SFTError(f"冻结输入哈希不匹配: {relative_path}")
        details: dict[str, Any] = {"sha256": digest}
        if "records" in expected:
            rows = read_jsonl(path)
            if len(rows) != expected["records"]:
                raise Stage2SFTError(
                    f"冻结输入数量不匹配: {relative_path}: {len(rows)}"
                )
            details["records"] = len(rows)
            loaded[relative_path] = rows
        validated[relative_path] = details
    return validated, loaded


def validate_frozen_inputs(
    repo_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate every frozen Stage 2 input and its semantic shape."""
    validated, loaded = validate_file_manifest(repo_dir, EXPECTED_INPUTS)
    hash_manifest = {
        path: {"sha256": digest}
        for path, digest in EXPECTED_FILE_HASHES.items()
    }
    hash_validated, _ = validate_file_manifest(repo_dir, hash_manifest)
    validated.update(hash_validated)

    sft_rows = loaded["data/runs/morgana-v2/sft_train.jsonl"]
    for index, row in enumerate(sft_rows):
        if set(row) != {"messages"}:
            raise Stage2SFTError(f"SFT 第 {index} 条字段不正确")
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) != 3:
            raise Stage2SFTError(f"SFT 第 {index} 条 messages 不正确")
        if [message.get("role") for message in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise Stage2SFTError(f"SFT 第 {index} 条角色顺序不正确")

    assistant_targets = [row["messages"][-1]["content"] for row in sft_rows]
    signature_count = sum("吾辈" in text for text in assistant_targets)
    wrong_count = sum(
        any(alias in text for alias in ("本大爷", "本喵"))
        for text in assistant_targets
    )
    if signature_count != len(sft_rows) or wrong_count:
        raise Stage2SFTError(
            f"训练标签角色信号不正确: 吾辈={signature_count}, 错误自称={wrong_count}"
        )

    dev_rows = loaded["data/runs/morgana-v2/dev.jsonl"]
    if len({row.get("id") for row in dev_rows}) != 10:
        raise Stage2SFTError("Dev ID 不完整或重复")
    return validated, sft_rows, dev_rows


def validate_training_config(
    train_config: dict[str, Any], record_count: int
) -> int:
    """Validate the frozen pure-FP32 setup and return planned steps."""
    expected = {
        **EXPECTED_TRAINING_PRECISION,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 3,
    }
    mismatches = [
        f"{name}={train_config.get(name)!r} (预期 {value!r})"
        for name, value in expected.items()
        if train_config.get(name) != value
    ]
    if mismatches:
        raise Stage2SFTError(
            "冻结训练配置不正确: " + ", ".join(mismatches)
        )

    planned_optimizer_steps = (
        math.ceil(
            math.ceil(
                record_count / train_config["per_device_train_batch_size"]
            )
            / train_config["gradient_accumulation_steps"]
        )
        * train_config["num_train_epochs"]
    )
    if planned_optimizer_steps != 48:
        raise Stage2SFTError(
            "冻结训练配置不再产生预期的 48 steps: "
            f"{planned_optimizer_steps}"
        )
    return planned_optimizer_steps


def validate_effective_training_args(output_dir: Path) -> dict[str, Any]:
    return _validate_effective_training_args(
        output_dir,
        error_type=Stage2SFTError,
        expected_precision=EXPECTED_TRAINING_PRECISION,
    )


def run_logged(
    command: list[str],
    log_path: Path,
    repo_dir: Path,
    environment: dict[str, str],
    expected_steps: int,
) -> float:
    return _run_logged(
        command,
        log_path,
        repo_dir,
        environment,
        expected_steps,
        error_type=Stage2SFTError,
        progress_parser=training_progress,
        popen_factory=subprocess.Popen,
    )


def find_final_adapter(output_dir: Path) -> Path:
    return _find_final_adapter(output_dir, error_type=Stage2SFTError)


def inspect_adapter_update(adapter_dir: Path) -> dict[str, Any]:
    """Require all LoRA-B tensors to be finite and nonzero."""
    import numpy as np
    from safetensors import safe_open

    weights_path = adapter_dir / "adapter_model.safetensors"
    tensor_count = 0
    lora_b_tensors = 0
    nonzero_lora_b_tensors = 0
    nonzero_elements = 0
    max_abs = 0.0
    with safe_open(weights_path, framework="np") as weights:
        for key in weights.keys():
            tensor = weights.get_tensor(key)
            tensor_count += 1
            if not np.isfinite(tensor).all():
                raise Stage2SFTError(f"adapter 包含非有限值: {key}")
            if tensor.size:
                max_abs = max(max_abs, float(np.abs(tensor).max()))
            if ".lora_B." in key:
                lora_b_tensors += 1
                nonzero = int(np.count_nonzero(tensor))
                nonzero_elements += nonzero
                nonzero_lora_b_tensors += int(nonzero > 0)
    result = {
        "tensor_count": tensor_count,
        "lora_b_tensors": lora_b_tensors,
        "nonzero_lora_b_tensors": nonzero_lora_b_tensors,
        "nonzero_lora_b_elements": nonzero_elements,
        "max_abs": max_abs,
    }
    if not lora_b_tensors or nonzero_lora_b_tensors != lora_b_tensors:
        raise Stage2SFTError(f"LoRA-B 未全部更新: {result}")
    return result


def validate_archive_contract(run_dir: Path) -> None:
    """Require the successful run to contain exactly the core artifacts."""
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != EXPECTED_ARCHIVE_FILES:
        raise Stage2SFTError(
            "归档目录契约不匹配: "
            f"missing={sorted(EXPECTED_ARCHIVE_FILES - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_ARCHIVE_FILES)}"
        )


def prune_run_artifacts(run_dir: Path) -> list[str]:
    """Remove known legacy files that should stay local, never in GitHub."""
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise Stage2SFTError(f"缺少 run_summary.json: {run_dir}")
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    successful_archive = EXPECTED_ARCHIVE_FILES <= actual
    allowed = (
        EXPECTED_ARCHIVE_FILES | LOCAL_ONLY_ARCHIVE_FILES
        if successful_archive
        else {"run_summary.json"} | LOCAL_ONLY_ARCHIVE_FILES
    )
    unexpected = actual - allowed
    if unexpected:
        raise Stage2SFTError(
            "发现未知文件，拒绝自动精简: " + ", ".join(sorted(unexpected))
        )
    if not successful_archive and actual - LOCAL_ONLY_ARCHIVE_FILES != {
        "run_summary.json"
    }:
        raise Stage2SFTError("失败 run 目录不满足最小归档契约")

    removed = sorted(actual & LOCAL_ONLY_ARCHIVE_FILES)
    if not removed:
        if successful_archive:
            validate_archive_contract(run_dir)
        return []

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = summary.get("artifacts")
    if isinstance(artifacts, dict):
        for relative_name in removed:
            artifacts.pop(relative_name, None)
    summary["pruned_local_only_artifacts"] = removed
    for relative_name in removed:
        (run_dir / relative_name).unlink()
    write_json_atomic(summary_path, summary)
    if successful_archive:
        validate_archive_contract(run_dir)
    return removed


def _release_metadata(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def create_release_bundle(
    run_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Create one exact release bundle without overwriting prior artifacts."""
    bundle, manifest, _ = _create_release_bundle(
        run_dir,
        output_dir,
        spec=RELEASE_SPEC,
        error_type=Stage2SFTError,
        validate_archive=validate_archive_contract,
        metadata_builder=_release_metadata,
        reuse_existing=False,
    )
    return bundle, manifest


def publish_run(
    run_dir: Path,
    output_dir: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> tuple[Path, Path, str]:
    """Prune, bundle, and upload one run before local manual review."""
    prune_run_artifacts(run_dir.resolve())
    return _publish_release(
        run_dir,
        output_dir,
        repository,
        spec=RELEASE_SPEC,
        error_type=Stage2SFTError,
        validate_archive=validate_archive_contract,
        run_command=subprocess.run,
        metadata_builder=_release_metadata,
    )


def format_download_command(
    tag: str, repository: str = DEFAULT_GITHUB_REPOSITORY
) -> str:
    """Format the local command for downloading a published release."""
    return _format_download_command(tag, repository, spec=RELEASE_SPEC)


def extract_release_bundle(
    bundle_path: Path, manifest_path: Path, output_root: Path
) -> Path:
    """Verify and atomically extract one downloaded release bundle."""
    return _extract_release_bundle(
        bundle_path,
        manifest_path,
        output_root,
        spec=RELEASE_SPEC,
        error_type=Stage2SFTError,
    )


def download_release(
    tag: str,
    output_root: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> Path:
    """Download, verify, and extract one GitHub Release locally."""
    return _download_release(
        tag,
        output_root,
        repository,
        spec=RELEASE_SPEC,
        error_type=Stage2SFTError,
        run_command=subprocess.run,
    )


def _records_from_responses(
    responses: Any, seed: int, dev_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(responses) != len(dev_rows) or len(dev_rows) != 10:
        raise Stage2SFTError("推理输出数量不正确")
    records: list[dict[str, Any]] = []
    for source, response in zip(dev_rows, responses, strict=True):
        choice = response.choices[0]
        raw_assistant = choice.message.content
        if not isinstance(raw_assistant, str) or not raw_assistant.strip():
            raise Stage2SFTError(f"推理输出为空: {source['id']}")
        assistant = normalize_empty_think_wrapper(raw_assistant)
        if not assistant:
            raise Stage2SFTError(f"规范化后输出为空: {source['id']}")
        records.append(
            {
                "seed": seed,
                "id": source["id"],
                "scenario": source["scenario"],
                "target_goals": source["target_goals"],
                "user": source["user"],
                "assistant": assistant,
                "raw_assistant": raw_assistant,
                "finish_reason": choice.finish_reason,
                "attempts": 1,
            }
        )
    return records


def _archive_metadata(path: Path, run_dir: Path) -> tuple[str, dict[str, Any]]:
    return (
        str(path.relative_to(run_dir)),
        {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
    )


def _record_failure(
    summary: dict[str, Any],
    summary_path: Path,
    status: str,
    error: BaseException,
    work_root: Path,
) -> None:
    summary["status"] = status
    summary["error"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    summary["retained_work_dir"] = str(work_root)
    write_json_atomic(summary_path, summary)


def run_stage2(output_root: Path | None = None) -> Path:
    """Execute the complete frozen Stage 2 SFT run and return its archive."""
    print("[1/5] 检查运行环境...")
    huggingface_environment = configure_huggingface_environment()
    repo_dir = repository_root()
    output_root = (
        output_root.resolve()
        if output_root is not None
        else repo_dir / DEFAULT_OUTPUT_RELATIVE_PATH
    )

    environment_snapshot, torch = capture_environment()
    installed_versions = validate_pinned_packages()
    git = git_context(repo_dir)
    print(
        "[1/5] 环境正常 | "
        + _format_environment_status(environment_snapshot)
    )

    run_id = generate_run_id()
    run_dir = create_exclusive_directory(output_root / run_id)
    work_root = create_exclusive_directory(output_root / ".work" / run_id)
    summary_path = run_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 2,
        "status": "initialized",
        "run": {
            "id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": "https://github.com/chenkx612/local-roleplay-llm.git",
            **git,
        },
        "environment": {
            "platform": "AutoDL",
            **environment_snapshot,
            "packages": installed_versions,
            "huggingface": huggingface_environment,
        },
    }
    write_json_atomic(summary_path, summary)
    print(f"Run 目录: {run_dir}")
    print("[2/5] 校验冻结输入与训练配置...")

    phase = "input_validation_failed"
    train_log = work_root / "train.log"
    try:
        import yaml
        from huggingface_hub import HfApi
        from peft import PeftModel
        from swift import (
            InferRequest,
            RequestConfig,
            TransformersEngine,
            get_model_processor,
            get_template,
        )
        from transformers import AutoProcessor, BitsAndBytesConfig

        validated_inputs, sft_rows, dev_rows = validate_frozen_inputs(repo_dir)
        assistant_targets = [
            row["messages"][-1]["content"] for row in sft_rows
        ]
        config_path = repo_dir / CONFIG_RELATIVE_PATH
        train_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(train_config, dict):
            raise Stage2SFTError("训练配置不是 YAML 对象")
        planned_optimizer_steps = validate_training_config(
            train_config, len(sft_rows)
        )

        effective_config_path = work_root / "training_config.yaml"
        effective_config_path.write_text(
            yaml.safe_dump(
                train_config, sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )

        model_info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
        if model_info.sha != MODEL_REVISION:
            raise Stage2SFTError(
                f"模型 revision 不匹配: {model_info.sha}"
            )
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        token_lengths: list[int] = []
        for index, row in enumerate(sft_rows):
            try:
                token_ids = processor.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                token_ids = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            if isinstance(token_ids, dict):
                token_ids = token_ids["input_ids"]
            length = (
                len(token_ids[0])
                if token_ids and isinstance(token_ids[0], list)
                else len(token_ids)
            )
            if length > train_config["max_length"]:
                raise Stage2SFTError(
                    f"SFT 第 {index} 条超过 max_length: {length}"
                )
            token_lengths.append(length)
        del processor, tokenizer

        summary["model"] = {"name": MODEL_ID, "revision": MODEL_REVISION}
        summary["inputs"] = {
            "files": validated_inputs,
            "max_sft_tokens": max(token_lengths),
            "max_length": train_config["max_length"],
            "assistant_targets": {
                "records": len(assistant_targets),
                "signature_self_reference_count": sum(
                    "吾辈" in text for text in assistant_targets
                ),
                "wrong_self_reference_count": sum(
                    any(alias in text for alias in ("本大爷", "本喵"))
                    for text in assistant_targets
                ),
                "average_characters": round(
                    sum(map(len, assistant_targets))
                    / len(assistant_targets),
                    2,
                ),
            },
            "planned_optimizer_steps": planned_optimizer_steps,
        }
        summary["status"] = "inputs_validated"
        write_json_atomic(summary_path, summary)
        print(
            "[2/5] 输入正常 | "
            f"训练 {len(sft_rows)} 条 | Dev {len(dev_rows)} 条 | "
            f"计划 {planned_optimizer_steps} 步"
        )

        phase = "training_failed"
        train_dir = work_root / "full"
        train_environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
        train_command = [
            "swift",
            "sft",
            str(effective_config_path),
            "--output_dir",
            str(train_dir),
            "--add_version",
            "false",
        ]
        print(f"[3/5] 开始训练 | 详细日志: {train_log}")
        train_duration_seconds = run_logged(
            train_command,
            train_log,
            repo_dir,
            train_environment,
            planned_optimizer_steps,
        )

        effective_precision = validate_effective_training_args(train_dir)
        summary["training"] = {
            "command": train_command,
            "epochs": train_config["num_train_epochs"],
            "expected_optimizer_steps": planned_optimizer_steps,
            "duration_seconds": round(train_duration_seconds, 3),
            "effective_precision": effective_precision,
        }
        summary["status"] = "training_completed"
        write_json_atomic(summary_path, summary)
        print(f"[3/5] 训练完成 | 用时 {train_duration_seconds / 60:.1f} 分钟")

        full_adapter = find_final_adapter(train_dir)
        full_losses = read_metric(train_dir, "loss")
        full_grad_norms = read_metric(train_dir, "grad_norm")
        if not full_losses or not all(
            math.isfinite(value) and value > 0 for value in full_losses
        ):
            raise Stage2SFTError(f"loss 无效: {full_losses}")
        if not full_grad_norms or not all(
            math.isfinite(value) and value > 0 for value in full_grad_norms
        ):
            raise Stage2SFTError(f"grad_norm 无效: {full_grad_norms}")
        if len(full_grad_norms) != planned_optimizer_steps:
            raise Stage2SFTError(
                "optimizer step 数不正确: "
                f"{len(full_grad_norms)} != {planned_optimizer_steps}"
            )
        adapter_update = inspect_adapter_update(full_adapter)
        adapter_weights = full_adapter / "adapter_model.safetensors"
        summary["training"].update(
            {
                "optimizer_steps": len(full_grad_norms),
                "losses": full_losses,
                "grad_norms": full_grad_norms,
                "adapter_update": adapter_update,
                "adapter_sha256": sha256_file(adapter_weights),
                "adapter_archive": "adapter",
            }
        )
        summary["status"] = "training_validated"
        write_json_atomic(summary_path, summary)

        phase = "inference_or_evaluation_failed"
        print("[4/5] 加载 Adapter，运行 Base/SFT Dev 对照评估...")
        system_prompt = (
            repo_dir / "data/runs/morgana-v2/system_prompt.txt"
        ).read_text(encoding="utf-8")
        stage1_meta = json.loads(
            (
                repo_dir
                / "data/runs/morgana-v2/base_generation_meta.json"
            ).read_text(encoding="utf-8")
        )
        request_config = RequestConfig(
            max_tokens=512,
            temperature=0.6,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.45,
        )
        requests = [
            InferRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["user"]},
                ]
            )
            for row in dev_rows
        ]
        expected_ids = [row["id"] for row in dev_rows]

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model, processor = get_model_processor(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.float32,
            quantization_config=quantization_config,
            use_hf=True,
        )
        base_engine = TransformersEngine(
            model, template=get_template(processor, enable_thinking=False)
        )
        base_records: list[dict[str, Any]] = []
        for seed in EVALUATION_SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            responses = base_engine.infer(
                requests, request_config=request_config
            )
            base_records.extend(
                _records_from_responses(responses, seed, dev_rows)
            )
        del base_engine
        gc.collect()
        torch.cuda.empty_cache()

        model = PeftModel.from_pretrained(model, full_adapter)
        sft_engine = TransformersEngine(
            model, template=get_template(processor, enable_thinking=False)
        )
        sft_records: list[dict[str, Any]] = []
        for seed in EVALUATION_SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            responses = sft_engine.infer(
                requests, request_config=request_config
            )
            sft_records.extend(
                _records_from_responses(responses, seed, dev_rows)
            )
        del sft_engine, model, processor
        gc.collect()
        torch.cuda.empty_cache()

        core_behavior_gate = evaluate_core_behavior_gate(
            base_records, sft_records, EVALUATION_SEEDS, expected_ids
        )
        manual_packet, manual_answer_key = build_manual_review(
            base_records, sft_records, expected_ids
        )
        manual_results = empty_manual_review_results(manual_packet)
        generated_paths = {
            "hf_base_dev_outputs.jsonl": work_root
            / "hf_base_dev_outputs.jsonl",
            "dev_outputs.jsonl": work_root / "dev_outputs.jsonl",
            "manual_review_packet.json": work_root
            / "manual_review_packet.json",
            "manual_review_answer_key.json": work_root
            / "manual_review_answer_key.json",
            "manual_review_results.json": work_root
            / "manual_review_results.json",
        }
        write_jsonl_exclusive(
            generated_paths["hf_base_dev_outputs.jsonl"], base_records
        )
        write_jsonl_exclusive(
            generated_paths["dev_outputs.jsonl"], sft_records
        )
        write_json_atomic(
            generated_paths["manual_review_packet.json"], manual_packet
        )
        write_json_atomic(
            generated_paths["manual_review_answer_key.json"],
            manual_answer_key,
        )
        write_json_atomic(
            generated_paths["manual_review_results.json"], manual_results
        )

        base_metrics = core_behavior_gate["base"]
        sft_metrics = core_behavior_gate["sft"]
        technical_gate = {
            "training_gradients_finite": all(
                math.isfinite(value) and value > 0
                for value in full_grad_norms
            ),
            "optimizer_step_count_expected": (
                len(full_grad_norms) == planned_optimizer_steps
            ),
            "adapter_updated": (
                adapter_update["nonzero_lora_b_tensors"]
                == adapter_update["lora_b_tensors"]
            ),
            "adapter_reloaded_and_generated": (
                sft_metrics["overall"]["nonempty_count"] == 10
            ),
            "base_dev_complete": (
                base_metrics["overall"]["nonempty_count"] == 10
            ),
            "sft_dev_complete": (
                sft_metrics["overall"]["nonempty_count"] == 10
            ),
            "outputs_aligned": core_behavior_gate["checks"][
                "complete_and_aligned"
            ],
        }
        summary["evaluation_seeds"] = list(EVALUATION_SEEDS)
        summary["inference"] = {
            "backend": "ms-swift TransformersEngine",
            "generation": {
                "max_tokens": 512,
                "temperature": 0.6,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.45,
                "enable_thinking": False,
            },
            "rng_reset": (
                "Base 和 SFT 在每个 seed 推理前分别重置 Python/Torch/CUDA RNG"
            ),
            "normalization": (
                "仅移除开头的空 <think></think> wrapper；保留 raw_assistant"
            ),
            "stage1_reference": stage1_meta["model"]
            | {"generation": stage1_meta["generation"]},
            "stage1_backend_difference": (
                "阶段一使用 MLX；本次为保证 Base/SFT 公平对照改用同一 "
                "TransformersEngine。TransformersEngine 不支持 presence_penalty "
                "和 context-size 参数，其他采样参数保持一致。"
            ),
            "hf_base": {
                **base_metrics,
                "file": "hf_base_dev_outputs.jsonl",
                "sha256": sha256_file(
                    generated_paths["hf_base_dev_outputs.jsonl"]
                ),
            },
            "sft": {
                **sft_metrics,
                "file": "dev_outputs.jsonl",
                "sha256": sha256_file(generated_paths["dev_outputs.jsonl"]),
            },
        }
        summary["technical_gate"] = technical_gate
        summary["technically_valid"] = all(technical_gate.values())
        summary["core_behavior_gate"] = core_behavior_gate
        summary["manual_review"] = {
            "status": (
                "awaiting_manual_review"
                if core_behavior_gate["passed"]
                else "not_started_stability_failed"
            ),
            "primary_seed": PRIMARY_EVALUATION_SEED,
            "packet": "manual_review_packet.json",
            "answer_key": "manual_review_answer_key.json",
            "results": "manual_review_results.json",
        }
        summary["ready_for_grpo"] = False
        phase = "technical_validation_failed"
        if not summary["technically_valid"]:
            raise Stage2SFTError(f"技术门槛失败: {technical_gate}")
        summary["status"] = (
            "awaiting_manual_review"
            if core_behavior_gate["passed"]
            else "stability_failed"
        )
        write_json_atomic(summary_path, summary)
        print(
            "[4/5] 自动评估完成 | "
            f"技术门槛通过 | 稳定性门槛"
            f"{'通过' if core_behavior_gate['passed'] else '未通过'}"
        )

        phase = "archive_failed"
        print("[5/5] 归档核心产物...")
        shutil.copy2(effective_config_path, run_dir / "training_config.yaml")
        for name, source in generated_paths.items():
            shutil.copy2(source, run_dir / name)
        adapter_archive = run_dir / "adapter"
        adapter_archive.mkdir()
        for name in ADAPTER_FILES:
            source = full_adapter / name
            if not source.is_file():
                raise Stage2SFTError(f"adapter 缺少文件: {name}")
            shutil.copy2(source, adapter_archive / name)

        artifact_paths = [
            run_dir / "training_config.yaml",
            *(adapter_archive / name for name in ADAPTER_FILES),
            *(run_dir / name for name in generated_paths),
        ]
        summary["artifacts"] = dict(
            _archive_metadata(path, run_dir) for path in artifact_paths
        )
        summary["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary.pop("error", None)
        summary.pop("retained_work_dir", None)
        write_json_atomic(summary_path, summary)
        validate_archive_contract(run_dir)
        shutil.rmtree(work_root)
        print(f"Stage 2 核心产物已归档: {run_dir}")
        if summary["status"] == "stability_failed":
            print("生成稳定性门槛失败；可 publish 到本地检查，禁止进入 GRPO。")
        else:
            print("请运行 publish 上传；人工复核在本地下载后完成。")
        return run_dir
    except BaseException as exc:
        _record_failure(summary, summary_path, phase, exc, work_root)
        raise


def review_run(run_dir: Path) -> dict[str, Any]:
    """Evaluate one completed manual-review artifact exactly once."""
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise Stage2SFTError(f"缺少 run_summary.json: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "manual_reviewed_at_utc" in summary:
        raise Stage2SFTError("该 run 已完成人工复核，拒绝重复提交")
    required = (
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            raise Stage2SFTError(f"缺少人工复核文件: {name}")

    submitted = json.loads(
        (run_dir / "manual_review_results.json").read_text(encoding="utf-8")
    )
    if not submitted.get("results"):
        print("尚未提交人工复核；run_summary 保持不变。")
        return summary

    packet = json.loads(
        (run_dir / "manual_review_packet.json").read_text(encoding="utf-8")
    )
    answer_key = json.loads(
        (run_dir / "manual_review_answer_key.json").read_text(encoding="utf-8")
    )
    manual_gate = evaluate_manual_review(packet, answer_key, submitted)
    summary["manual_review"] = {
        **summary["manual_review"],
        "status": "passed" if manual_gate["passed"] else "failed",
        "gate": manual_gate,
    }
    summary["ready_for_grpo"] = bool(
        summary["technically_valid"]
        and summary["core_behavior_gate"]["passed"]
        and manual_gate["passed"]
    )
    if not summary["core_behavior_gate"]["passed"]:
        summary["status"] = "stability_failed"
    else:
        summary["status"] = (
            "ready_for_grpo"
            if summary["ready_for_grpo"]
            else "manual_failed"
        )
    results_path = run_dir / "manual_review_results.json"
    summary["artifacts"]["manual_review_results.json"] = {
        "bytes": results_path.stat().st_size,
        "sha256": sha256_file(results_path),
    }
    summary["manual_reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary["manual_review"], ensure_ascii=False, indent=2))
    print(f"ready_for_grpo={summary['ready_for_grpo']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 2 command-line parser."""
    parser = argparse.ArgumentParser(
        description="在 AutoDL 上运行 morgana-v2 Stage 2 SFT"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行完整 SFT 与 Dev 评测")
    run_parser.add_argument(
        "--output-root",
        type=Path,
        help="run 目录父路径；默认 output/morgana-v2/stage2-sft",
    )
    review_parser = subparsers.add_parser(
        "review", help="提交并验证人工复核结果"
    )
    review_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser = subparsers.add_parser(
        "publish", help="精简、打包并上传到 GitHub Release"
    )
    publish_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser.add_argument(
        "--output-dir", type=Path, default=Path("dist")
    )
    publish_parser.add_argument(
        "--repo", default=DEFAULT_GITHUB_REPOSITORY
    )
    download_parser = subparsers.add_parser(
        "download", help="从 GitHub Release 下载并校验解包"
    )
    download_parser.add_argument("--tag", required=True)
    download_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_RELATIVE_PATH,
    )
    download_parser.add_argument(
        "--repo", default=DEFAULT_GITHUB_REPOSITORY
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested AutoDL Stage 2 command."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_stage2(args.output_root)
        elif args.command == "review":
            review_run(args.run_dir)
        elif args.command == "publish":
            bundle_path, manifest_path, tag = publish_run(
                args.run_dir, args.output_dir, args.repo
            )
            print(f"GitHub Release: {tag}")
            print(f"发布包: {bundle_path}")
            print(f"清单: {manifest_path}")
            print(f"本地下载命令: {format_download_command(tag, args.repo)}")
            print("下载完成后再完成人工复核。")
        elif args.command == "download":
            run_dir = download_release(args.tag, args.output_root, args.repo)
            print(f"本地 run 目录: {run_dir}")
            print("请填写 manual_review_results.json 后运行 review。")
    except (Stage2SFTError, subprocess.CalledProcessError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
