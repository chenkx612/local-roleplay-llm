"""Run the frozen morgana-v2 Stage 2 SFT workflow on AutoDL."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from roleplay.sft_eval import (
    EVALUATION_SEEDS,
    PRIMARY_EVALUATION_SEED,
    build_manual_review,
    empty_manual_review_results,
    evaluate_core_behavior_gate,
    evaluate_manual_review,
    normalize_empty_think_wrapper,
)


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "965dcc54bc9c0591873df0e9869c056a54d323d1"
CONFIG_RELATIVE_PATH = Path("configs/morgana_v2_sft.yaml")
DEFAULT_OUTPUT_RELATIVE_PATH = Path("output/morgana-v2/stage2-sft")
MIN_GPU_MEMORY_GIB = 20.0
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_HOME = "/root/autodl-tmp/huggingface"
DEFAULT_GITHUB_REPOSITORY = "chenkx612/local-roleplay-llm"
CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")

EXPECTED_TRAINING_PRECISION = {
    "torch_dtype": "float32",
    "bnb_4bit_compute_dtype": "float32",
    "lora_dtype": "float32",
    "fp16": False,
    "bf16": False,
}

PINNED_PACKAGES = {
    "ms-swift": "4.4.1",
    "datasets": "4.8.4",
    "transformers": "5.12.1",
    "peft": "0.19.1",
    "bitsandbytes": "0.49.2",
    "qwen-vl-utils": "0.0.14",
}

DISABLED_ACCELERATION_PACKAGES = (
    "flash-linear-attention",
    "causal-conv1d",
)

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

ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "additional_config.json",
)

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


def configure_huggingface_environment(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set AutoDL-friendly Hugging Face defaults without overriding the user."""
    target = os.environ if environment is None else environment
    target.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    target.setdefault("HF_HOME", DEFAULT_HF_HOME)
    target.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    target.setdefault("TQDM_DISABLE", "1")
    return {
        "endpoint": target["HF_ENDPOINT"],
        "cache_home": target["HF_HOME"],
    }


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and require every nonblank row to be an object."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Stage2SFTError(f"{path}:{line_number} 不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise Stage2SFTError(f"{path}:{line_number} 不是对象")
        rows.append(value)
    return rows


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a UTF-8 JSON object in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL without replacing an existing artifact."""
    with path.open("x", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_package_version(version: str) -> str:
    """Strip a local wheel suffix such as ``+cu128``."""
    return version.split("+", 1)[0]


def validate_environment_snapshot(snapshot: dict[str, Any]) -> None:
    """Validate a dependency-free snapshot of the AutoDL runtime."""
    checks = {
        "Linux": snapshot.get("platform") == "Linux",
        "Python 3.12": tuple(snapshot.get("python_version", ())) == (3, 12),
        "PyTorch 2.8.0": (
            normalized_package_version(str(snapshot.get("pytorch", "")))
            == "2.8.0"
        ),
        "CUDA 12.8": snapshot.get("cuda") == "12.8",
        "CUDA available": snapshot.get("cuda_available") is True,
        "one GPU": snapshot.get("gpu_count") == 1,
        f"GPU memory >= {MIN_GPU_MEMORY_GIB:.0f} GiB": (
            isinstance(snapshot.get("gpu_memory_gib"), (int, float))
            and snapshot["gpu_memory_gib"] >= MIN_GPU_MEMORY_GIB
        ),
        "CXX11 ABI": snapshot.get("cxx11_abi") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise Stage2SFTError("AutoDL 环境不满足要求: " + ", ".join(failed))


def capture_environment() -> tuple[dict[str, Any], Any]:
    """Import torch lazily, capture runtime details, and validate them."""
    try:
        import torch
    except ImportError as exc:
        raise Stage2SFTError(
            "缺少 PyTorch；请选择 docs/AUTODL.md 指定的 PyTorch 2.8.0 基础镜像"
        ) from exc

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_count else None
    gpu_memory_gib = (
        torch.cuda.get_device_properties(0).total_memory / 2**30
        if gpu_count
        else 0.0
    )
    try:
        nvidia_smi_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage2SFTError("无法读取 nvidia-smi") from exc

    snapshot = {
        "platform": platform.system(),
        "python_version": list(sys.version_info[:2]),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu": gpu_name,
        "gpu_memory_gib": round(gpu_memory_gib, 2),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "nvidia_smi_query": nvidia_smi_query,
    }
    validate_environment_snapshot(snapshot)
    return snapshot, torch


def _format_environment_status(snapshot: dict[str, Any]) -> str:
    """Format the validated environment snapshot for the startup log."""
    return (
        f"GPU {snapshot['gpu']} | "
        f"显存 {snapshot['gpu_memory_gib']:.1f} GiB"
    )


def validate_pinned_packages() -> dict[str, str]:
    """Require the direct training dependencies to match the frozen versions."""
    disabled = {}
    for name in DISABLED_ACCELERATION_PACKAGES:
        try:
            disabled[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if disabled:
        packages = " ".join(DISABLED_ACCELERATION_PACKAGES)
        raise Stage2SFTError(
            f"检测到禁用的可选加速依赖: {disabled}；"
            f"请运行 python -m pip uninstall -y {packages}"
        )

    installed: dict[str, str] = {}
    missing: list[str] = []
    for name in PINNED_PACKAGES:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise Stage2SFTError("缺少固定依赖: " + ", ".join(missing))
    mismatches = {
        name: version
        for name, version in installed.items()
        if normalized_package_version(version) != PINNED_PACKAGES[name]
    }
    if mismatches:
        raise Stage2SFTError(f"固定依赖版本不匹配: {mismatches}")
    return installed


def ensure_clean_tracked_status(status: str) -> None:
    """Reject tracked changes while allowing ignored and untracked files."""
    if status.strip():
        raise Stage2SFTError("仓库包含 tracked 未提交修改，拒绝开始训练")


def git_context(repo_dir: Path) -> dict[str, str]:
    """Return the exact clean Git commit and branch used by a run."""
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        ensure_clean_tracked_status(status)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage2SFTError("无法读取 Git checkout") from exc
    return {"commit": commit, "branch": branch or "detached"}


def create_exclusive_directory(path: Path) -> Path:
    """Create a new run directory and never reuse existing output."""
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise Stage2SFTError(f"运行目录已存在: {path}") from exc
    return path


def generate_run_id(created_at: datetime | None = None) -> str:
    """Return a readable China Standard Time run identifier, precise to minutes."""
    timestamp = created_at or datetime.now(timezone.utc)
    return timestamp.astimezone(CHINA_TIMEZONE).strftime("%Y%m%d-%H%M")


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
    """Require ms-swift's resolved arguments to remain pure FP32."""
    args_path = output_dir / "args.json"
    if not args_path.is_file():
        raise Stage2SFTError(f"训练输出缺少 args.json: {args_path}")
    try:
        args = json.loads(args_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Stage2SFTError(f"训练 args.json 无效: {args_path}") from exc
    if not isinstance(args, dict):
        raise Stage2SFTError(f"训练 args.json 不是对象: {args_path}")

    effective = {
        name: args.get(name) for name in EXPECTED_TRAINING_PRECISION
    }
    mismatches = [
        f"{name}={effective[name]!r} (预期 {value!r})"
        for name, value in EXPECTED_TRAINING_PRECISION.items()
        if effective[name] != value
    ]
    if mismatches:
        raise Stage2SFTError(
            "ms-swift 实际训练精度不正确: " + ", ".join(mismatches)
        )
    return effective


def training_progress(line: str) -> dict[str, str] | None:
    """Extract the few training metrics useful for a concise status update."""
    fields: dict[str, str] = {}
    for name in (
        "loss",
        "grad_norm",
        "global_step/max_steps",
        "remaining_time",
    ):
        pattern = rf"['\"]{re.escape(name)}['\"]:\s*['\"]([^'\"]+)"
        match = re.search(pattern, line)
        if match:
            fields[name] = match.group(1)
    if set(fields) != {
        "loss",
        "grad_norm",
        "global_step/max_steps",
        "remaining_time",
    }:
        return None
    return fields


def run_logged(
    command: list[str],
    log_path: Path,
    repo_dir: Path,
    environment: dict[str, str],
    expected_steps: int,
) -> float:
    """Run a command, preserve all output, and print concise progress."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress_interval = max(1, math.ceil(expected_steps / 5))
    last_reported_step = 0
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=repo_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise Stage2SFTError("训练子进程没有 stdout")
        for line in process.stdout:
            log_file.write(line)
            progress = training_progress(line)
            if progress is None:
                continue
            step_text = progress["global_step/max_steps"]
            step, total = (int(value) for value in step_text.split("/", 1))
            should_report = (
                step == 1
                or step == total
                or step - last_reported_step >= progress_interval
            )
            if should_report:
                print(
                    "  训练进度 "
                    f"{step}/{total} | loss {progress['loss']} | "
                    f"grad {progress['grad_norm']} | "
                    f"预计剩余 {progress['remaining_time']}"
                )
                last_reported_step = step
        return_code = process.wait()
    if return_code != 0:
        raise Stage2SFTError(
            f"训练子进程退出码 {return_code}；完整日志: {log_path}"
        )
    return time.monotonic() - started


def find_final_adapter(output_dir: Path) -> Path:
    """Find and validate the newest final/checkpoint PEFT adapter."""
    from peft import PeftConfig

    candidates: list[tuple[int, int, Path]] = []
    for config_path in output_dir.rglob("adapter_config.json"):
        directory = config_path.parent
        if (directory / "adapter_model.safetensors").is_file():
            match = re.search(r"checkpoint-(\d+)$", directory.name)
            step = int(match.group(1)) if match else -1
            candidates.append((step, config_path.stat().st_mtime_ns, directory))
    if not candidates:
        raise Stage2SFTError(f"没有在 {output_dir} 找到 adapter")
    adapter_dir = max(candidates)[2]
    PeftConfig.from_pretrained(adapter_dir)
    return adapter_dir


def read_metric(output_dir: Path, metric: str) -> list[float]:
    """Read the longest metric series emitted by ms-swift."""
    candidates: list[list[float]] = []
    for path in output_dir.rglob("*.jsonl"):
        values: list[float] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get(metric) if isinstance(row, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if values:
            candidates.append(values)
    return max(candidates, key=len, default=[])


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


def create_release_bundle(run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create one GitHub Release bundle without adding model files to Git."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    validate_archive_contract(run_dir)

    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = summary.get("run", {}).get("id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise Stage2SFTError("run_summary.json 缺少可用于发布包名称的 run.id")

    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise Stage2SFTError("发布包目录不能位于 run 目录内")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"morgana-v2-stage2-sft-{run_id}"
    bundle_path = output_dir / f"{stem}.tar.gz"
    manifest_path = output_dir / f"{stem}.manifest.json"
    existing = [path for path in (bundle_path, manifest_path) if path.exists()]
    if existing:
        raise Stage2SFTError(
            "发布产物已存在，拒绝覆盖: " + ", ".join(map(str, existing))
        )

    temporary_bundle = output_dir / f".{bundle_path.name}.{uuid4().hex}.tmp"
    try:
        with tarfile.open(temporary_bundle, "w:gz") as archive:
            for relative_name in sorted(EXPECTED_ARCHIVE_FILES):
                archive.add(
                    run_dir / relative_name,
                    arcname=f"{run_id}/{relative_name}",
                    recursive=False,
                )
        temporary_bundle.replace(bundle_path)
        contents = dict(
            _archive_metadata(run_dir / name, run_dir)
            for name in sorted(EXPECTED_ARCHIVE_FILES)
        )
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "run_status": summary.get("status"),
            "source_commit": summary.get("run", {}).get("commit"),
            "github_release_tag": stem,
            "bundle": {
                "file": bundle_path.name,
                "bytes": bundle_path.stat().st_size,
                "sha256": sha256_file(bundle_path),
            },
            "contents": contents,
        }
        write_json_atomic(manifest_path, manifest)
    except BaseException:
        temporary_bundle.unlink(missing_ok=True)
        bundle_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return bundle_path, manifest_path


def _reuse_release_bundle(
    run_dir: Path, output_dir: Path
) -> tuple[Path, Path] | None:
    """Return an existing bundle only when it still matches the run exactly."""
    summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    run_id = summary.get("run", {}).get("id")
    if not isinstance(run_id, str):
        return None
    stem = f"morgana-v2-stage2-sft-{run_id}"
    bundle_path = output_dir / f"{stem}.tar.gz"
    manifest_path = output_dir / f"{stem}.manifest.json"
    if not bundle_path.exists() and not manifest_path.exists():
        return None
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise Stage2SFTError("发布包或清单不完整，拒绝复用")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_contents = dict(
        _archive_metadata(run_dir / name, run_dir)
        for name in sorted(EXPECTED_ARCHIVE_FILES)
    )
    if (
        manifest.get("run_id") != run_id
        or manifest.get("contents") != expected_contents
        or manifest.get("bundle", {}).get("sha256")
        != sha256_file(bundle_path)
    ):
        raise Stage2SFTError("现有发布包与 run 目录不一致，拒绝复用")
    return bundle_path, manifest_path


def publish_run(
    run_dir: Path,
    output_dir: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> tuple[Path, Path, str]:
    """Prune, bundle, and upload one run before local manual review."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    prune_run_artifacts(run_dir)
    existing = _reuse_release_bundle(run_dir, output_dir)
    if existing is None:
        bundle_path, manifest_path = create_release_bundle(run_dir, output_dir)
    else:
        bundle_path, manifest_path = existing
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tag = manifest["github_release_tag"]
    command = [
        "gh",
        "release",
        "create",
        tag,
        str(bundle_path),
        str(manifest_path),
        "--repo",
        repository,
        "--title",
        f"morgana-v2 Stage 2 SFT {manifest['run_id']}",
        "--notes",
        "Stage 2 SFT 精简归档；人工复核在下载到本地后完成。",
    ]
    source_commit = manifest.get("source_commit")
    if isinstance(source_commit, str) and source_commit:
        command.extend(["--target", source_commit])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise Stage2SFTError("未安装 GitHub CLI：gh") from exc
    except subprocess.CalledProcessError as exc:
        raise Stage2SFTError(
            "GitHub Release 上传失败；本地发布包已保留，可修复后重试"
        ) from exc
    return bundle_path, manifest_path, tag


def format_download_command(
    tag: str, repository: str = DEFAULT_GITHUB_REPOSITORY
) -> str:
    """Format the local command for downloading a published release."""
    command = ["roleplay-stage2-sft", "download", "--tag", tag]
    if repository != DEFAULT_GITHUB_REPOSITORY:
        command.extend(["--repo", repository])
    return shlex.join(command)


def extract_release_bundle(
    bundle_path: Path, manifest_path: Path, output_root: Path
) -> Path:
    """Verify and atomically extract one downloaded release bundle."""
    bundle_path = bundle_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle", {}).get("sha256") != sha256_file(bundle_path):
        raise Stage2SFTError("下载包 SHA-256 与 manifest 不匹配")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise Stage2SFTError("manifest 缺少有效 run_id")
    if set(manifest.get("contents", {})) != EXPECTED_ARCHIVE_FILES:
        raise Stage2SFTError("manifest 文件清单不满足归档契约")
    expected_names = {
        f"{run_id}/{name}" for name in manifest.get("contents", {})
    }
    output_root = output_root.resolve()
    destination = output_root / run_id
    if destination.exists():
        raise Stage2SFTError(f"本地 run 目录已存在，拒绝覆盖: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / f".download-{uuid4().hex}"
    staging_root.mkdir()
    try:
        with tarfile.open(bundle_path, "r:gz") as archive:
            members = archive.getmembers()
            actual_names = {member.name for member in members}
            if actual_names != expected_names or not all(
                member.isfile() for member in members
            ):
                raise Stage2SFTError("发布包内容与 manifest 不一致")
            for member in members:
                target = (staging_root / member.name).resolve()
                try:
                    target.relative_to(staging_root)
                except ValueError as exc:
                    raise Stage2SFTError("发布包包含不安全路径") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise Stage2SFTError(f"无法读取发布包文件: {member.name}")
                with source, target.open("xb") as output_file:
                    shutil.copyfileobj(source, output_file)
        staged_run = staging_root / run_id
        for relative_name, metadata in manifest["contents"].items():
            path = staged_run / relative_name
            if (
                not path.is_file()
                or path.stat().st_size != metadata["bytes"]
                or sha256_file(path) != metadata["sha256"]
            ):
                raise Stage2SFTError(f"解包文件校验失败: {relative_name}")
        staged_run.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return destination


def download_release(
    tag: str,
    output_root: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> Path:
    """Download, verify, and extract one GitHub Release locally."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise Stage2SFTError("release tag 格式无效")
    with tempfile.TemporaryDirectory() as temporary:
        download_dir = Path(temporary)
        command = [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            repository,
            "--dir",
            str(download_dir),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise Stage2SFTError("未安装 GitHub CLI：gh") from exc
        except subprocess.CalledProcessError as exc:
            raise Stage2SFTError(f"GitHub Release 下载失败: {tag}") from exc
        bundle_path = download_dir / f"{tag}.tar.gz"
        manifest_path = download_dir / f"{tag}.manifest.json"
        if not bundle_path.is_file() or not manifest_path.is_file():
            raise Stage2SFTError("Release 缺少发布包或 manifest")
        return extract_release_bundle(bundle_path, manifest_path, output_root)


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
