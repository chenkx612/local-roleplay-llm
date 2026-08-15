"""Shared local/AutoDL runtime mechanics for training workflows."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "additional_config.json",
)
DEFAULT_GITHUB_REPOSITORY = "chenkx612/local-roleplay-llm"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_HOME = "/root/autodl-tmp/huggingface"
CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")
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
EXPECTED_TRAINING_PRECISION = {
    "torch_dtype": "float32",
    "bnb_4bit_compute_dtype": "float32",
    "lora_dtype": "float32",
    "fp16": False,
    "bf16": False,
}


def configure_huggingface_environment(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set AutoDL-friendly Hugging Face defaults without overriding the user."""
    target = os.environ if environment is None else environment
    target.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    target.setdefault("HF_HOME", DEFAULT_HF_HOME)
    target.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    target.setdefault("TQDM_DISABLE", "1")
    return {"endpoint": target["HF_ENDPOINT"], "cache_home": target["HF_HOME"]}


def normalized_package_version(version: str) -> str:
    """Strip a local wheel suffix such as ``+cu128``."""
    return version.split("+", 1)[0]


def validate_environment_snapshot(
    snapshot: dict[str, Any],
    *,
    error_type: type[Exception],
    min_gpu_memory_gib: float = 20.0,
) -> None:
    """Validate the frozen, dependency-free AutoDL runtime snapshot."""
    checks = {
        "Linux": snapshot.get("platform") == "Linux",
        "Python 3.12": tuple(snapshot.get("python_version", ())) == (3, 12),
        "PyTorch 2.8.0": normalized_package_version(
            str(snapshot.get("pytorch", ""))
        )
        == "2.8.0",
        "CUDA 12.8": snapshot.get("cuda") == "12.8",
        "CUDA available": snapshot.get("cuda_available") is True,
        "one GPU": snapshot.get("gpu_count") == 1,
        f"GPU memory >= {min_gpu_memory_gib:.0f} GiB": (
            isinstance(snapshot.get("gpu_memory_gib"), (int, float))
            and snapshot["gpu_memory_gib"] >= min_gpu_memory_gib
        ),
        "CXX11 ABI": snapshot.get("cxx11_abi") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise error_type("AutoDL 环境不满足要求: " + ", ".join(failed))


def capture_environment(
    *,
    error_type: type[Exception],
    run_command: Callable[..., Any] = subprocess.run,
    min_gpu_memory_gib: float = 20.0,
) -> tuple[dict[str, Any], Any]:
    """Import torch lazily, capture runtime details, and validate them."""
    try:
        import torch
    except ImportError as exc:
        raise error_type(
            "缺少 PyTorch；请选择 docs/AUTODL.md 指定的 "
            "PyTorch 2.8.0 基础镜像"
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
        nvidia_smi_query = run_command(
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
        raise error_type("无法读取 nvidia-smi") from exc

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
    validate_environment_snapshot(
        snapshot,
        error_type=error_type,
        min_gpu_memory_gib=min_gpu_memory_gib,
    )
    return snapshot, torch


def validate_pinned_packages(
    required_packages: Mapping[str, str],
    disabled_packages: Sequence[str],
    *,
    error_type: type[Exception],
    version_lookup: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    """Require one stage's dependencies to match its frozen versions."""
    disabled = {}
    for name in disabled_packages:
        try:
            disabled[name] = version_lookup(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if disabled:
        packages = " ".join(disabled_packages)
        raise error_type(
            f"检测到禁用的可选加速依赖: {disabled}；"
            f"请运行 python -m pip uninstall -y {packages}"
        )

    installed: dict[str, str] = {}
    missing: list[str] = []
    for name in required_packages:
        try:
            installed[name] = version_lookup(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise error_type("缺少固定依赖: " + ", ".join(missing))
    mismatches = {
        name: version
        for name, version in installed.items()
        if normalized_package_version(version) != required_packages[name]
    }
    if mismatches:
        raise error_type(f"固定依赖版本不匹配: {mismatches}")
    return installed


def ensure_clean_tracked_status(
    status: str, *, error_type: type[Exception]
) -> None:
    """Reject tracked changes while allowing ignored and untracked files."""
    if status.strip():
        raise error_type("仓库包含 tracked 未提交修改，拒绝开始训练")


def git_context(
    repo_dir: Path,
    *,
    error_type: type[Exception],
    run_command: Callable[..., Any] = subprocess.run,
) -> dict[str, str]:
    """Return the exact clean Git commit and branch used by a run."""
    try:
        status = run_command(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        ensure_clean_tracked_status(status, error_type=error_type)
        commit = run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        branch = run_command(
            ["git", "branch", "--show-current"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise error_type("无法读取 Git checkout") from exc
    return {"commit": commit, "branch": branch or "detached"}


def create_exclusive_directory(
    path: Path, *, error_type: type[Exception]
) -> Path:
    """Create a new run directory and never reuse existing output."""
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise error_type(f"运行目录已存在: {path}") from exc
    return path


def generate_run_id(created_at: datetime | None = None) -> str:
    """Return a readable China Standard Time run identifier."""
    timestamp = created_at or datetime.now(timezone.utc)
    return timestamp.astimezone(CHINA_TIMEZONE).strftime("%Y%m%d-%H%M")


def training_progress(line: str) -> dict[str, str] | None:
    """Extract the few training metrics useful for a concise status update."""
    fields: dict[str, str] = {}
    for name in ("loss", "grad_norm", "global_step/max_steps", "remaining_time"):
        pattern = rf"['\"]{re.escape(name)}['\"]:\s*['\"]([^'\"]+)"
        match = re.search(pattern, line)
        if match:
            fields[name] = match.group(1)
    required = {"loss", "grad_norm", "global_step/max_steps", "remaining_time"}
    return fields if set(fields) == required else None


def run_logged(
    command: list[str],
    log_path: Path,
    repo_dir: Path,
    environment: dict[str, str],
    expected_steps: int,
    *,
    error_type: type[Exception],
    progress_parser: Callable[[str], dict[str, str] | None] = training_progress,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> float:
    """Run a command, preserve all output, and print concise progress."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress_interval = max(1, math.ceil(expected_steps / 5))
    last_reported_step = 0
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log_file.flush()
        process = popen_factory(
            command,
            cwd=repo_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise error_type("训练子进程没有 stdout")
        for line in process.stdout:
            log_file.write(line)
            progress = progress_parser(line)
            if progress is None:
                continue
            step, total = (
                int(value)
                for value in progress["global_step/max_steps"].split("/", 1)
            )
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
        raise error_type(
            f"训练子进程退出码 {return_code}；完整日志: {log_path}"
        )
    return time.monotonic() - started


def find_final_adapter(
    output_dir: Path, *, error_type: type[Exception]
) -> Path:
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
        raise error_type(f"没有在 {output_dir} 找到 adapter")
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


def validate_effective_training_args(
    output_dir: Path,
    *,
    error_type: type[Exception],
    expected_precision: Mapping[str, Any] = EXPECTED_TRAINING_PRECISION,
) -> dict[str, Any]:
    """Require ms-swift's resolved arguments to retain the frozen precision."""
    args_path = output_dir / "args.json"
    if not args_path.is_file():
        raise error_type(f"训练输出缺少 args.json: {args_path}")
    try:
        args = json.loads(args_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise error_type(f"训练 args.json 无效: {args_path}") from exc
    if not isinstance(args, dict):
        raise error_type(f"训练 args.json 不是对象: {args_path}")
    effective = {name: args.get(name) for name in expected_precision}
    mismatches = [
        f"{name}={effective[name]!r} (预期 {value!r})"
        for name, value in expected_precision.items()
        if effective[name] != value
    ]
    if mismatches:
        raise error_type(
            "ms-swift 实际训练精度不正确: " + ", ".join(mismatches)
        )
    return effective
