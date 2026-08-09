"""Run and publish the frozen morgana-v2 Stage 3 GRPO workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from roleplay.stage2_sft import (
    ADAPTER_FILES,
    DEFAULT_GITHUB_REPOSITORY,
    Stage2SFTError,
    capture_environment,
    configure_huggingface_environment,
    create_exclusive_directory,
    find_final_adapter,
    generate_run_id,
    git_context,
    read_jsonl,
    run_logged,
    sha256_file,
    validate_pinned_packages,
    write_json_atomic,
    write_jsonl_exclusive,
)


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "965dcc54bc9c0591873df0e9869c056a54d323d1"
CONFIG_RELATIVE_PATH = Path("configs/morgana_v2_grpo.yaml")
PROMPTS_RELATIVE_PATH = Path("data/runs/morgana-v2/rl_train.jsonl")
SYSTEM_PROMPT_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/system_prompt.txt"
)
SFT_ADAPTER_RELATIVE_PATH = Path(
    "output/morgana-v2/stage2-sft/final/adapter"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path("output/morgana-v2/stage3-grpo")

PROMPTS_SHA256 = "b36b4f01f232901ab0b5f6011fa64b66f48e02c75b6b0050035e4caf703e7231"
SYSTEM_PROMPT_SHA256 = (
    "d88993aaa1178ced740f6b54530a27e5fcdb2486a66d8b460367e842b53ee112"
)
SFT_ADAPTER_HASHES = {
    "adapter_model.safetensors": (
        "617e6e00535fa356272d32fb16d8fe8d451a9c3cfd2f766f56af02cdf2f9b78d"
    ),
    "adapter_config.json": (
        "67f3ab10168164cc014c7f9c8720984760b1d94be33fde9abf89fb3004a886dc"
    ),
    "additional_config.json": (
        "2b7ed6cc0ca6c21dc39bf80fd3351f5f87c462df23a237dd6ad20473eb9a33a2"
    ),
}

EXPECTED_CONFIG = {
    "rlhf_type": "grpo",
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "use_hf": True,
    "split_dataset_ratio": 0.0,
    "tuner_type": "lora",
    "adapters": [str(SFT_ADAPTER_RELATIVE_PATH)],
    "ref_adapters": [str(SFT_ADAPTER_RELATIVE_PATH)],
    "torch_dtype": "float32",
    "fp16": False,
    "bf16": False,
    "quant_method": "bnb",
    "quant_bits": 4,
    "bnb_4bit_compute_dtype": "float32",
    "lora_dtype": "float32",
    "max_completion_length": 256,
    "num_generations": 4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1,
    "learning_rate": 1.0e-6,
    "enable_thinking": False,
    "use_vllm": False,
    "external_plugins": ["src/roleplay/grpo_reward_plugin.py"],
    "reward_funcs": ["morgana_reward"],
}

EXPECTED_ARCHIVE_FILES = frozenset(
    {
        "run_summary.json",
        "training_config.yaml",
        "train.jsonl",
        "train.log",
        "reward_samples.jsonl",
        *(f"adapter/{name}" for name in ADAPTER_FILES),
    }
)


class Stage3GRPOError(RuntimeError):
    """Raised when the frozen Stage 3 execution contract is violated."""


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[2]


def validate_frozen_file(path: Path, expected_sha256: str) -> str:
    """Require one frozen file and return its verified digest."""
    if not path.is_file():
        raise Stage3GRPOError(f"缺少冻结文件: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise Stage3GRPOError(f"冻结文件哈希不匹配: {path}")
    return digest


def validate_sft_adapter(adapter_dir: Path) -> dict[str, str]:
    """Require the exact accepted Stage 2 adapter at the fixed path."""
    validated: dict[str, str] = {}
    for name, expected in SFT_ADAPTER_HASHES.items():
        validated[name] = validate_frozen_file(adapter_dir / name, expected)
    return validated


def prepare_training_rows(
    prompts: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    """Convert frozen user prompts to ms-swift conversation records."""
    if len(prompts) != 20:
        raise Stage3GRPOError(f"GRPO Prompt 数量必须为 20，实际 {len(prompts)}")
    expected_fields = {"id", "scenario", "target_goals", "user"}
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(prompts):
        if set(row) != expected_fields:
            raise Stage3GRPOError(f"GRPO Prompt {index} 字段不正确")
        record_id = row["id"]
        user = row["user"]
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in ids
            or not isinstance(user, str)
            or not user.strip()
        ):
            raise Stage3GRPOError(f"GRPO Prompt {index} id/user 不正确")
        ids.add(record_id)
        records.append(
            {
                "id": record_id,
                "prompt_id": record_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
            }
        )
    return records


def validate_training_config(config: dict[str, Any]) -> None:
    """Require the single frozen GRPO configuration."""
    mismatches = [
        f"{name}={config.get(name)!r} (预期 {expected!r})"
        for name, expected in EXPECTED_CONFIG.items()
        if config.get(name) != expected
    ]
    if config.get("dataset") != "__GRPO_DATA__":
        mismatches.append("dataset 占位符不正确")
    if config.get("output_dir") != "__GRPO_OUTPUT__":
        mismatches.append("output_dir 占位符不正确")
    if mismatches:
        raise Stage3GRPOError("GRPO 冻结配置不正确: " + ", ".join(mismatches))


def validate_reward_samples(
    path: Path, expected_record_ids: set[str]
) -> dict[str, Any]:
    """Require successful reward rows covering every frozen prompt."""
    if not path.is_file():
        raise Stage3GRPOError(f"缺少奖励日志: {path}")
    rows = read_jsonl(path)
    if not rows:
        raise Stage3GRPOError("奖励日志为空")
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        raise Stage3GRPOError(f"奖励日志包含 {len(errors)} 条失败")
    rewards = [row.get("total_reward") for row in rows]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in rewards
    ):
        raise Stage3GRPOError("奖励日志缺少有效 total_reward")
    counts = Counter(row.get("record_id") for row in rows)
    missing = sorted(expected_record_ids - set(counts))
    insufficient = sorted(
        record_id
        for record_id in expected_record_ids
        if counts[record_id] < 4
    )
    if missing or insufficient:
        raise Stage3GRPOError(
            f"奖励覆盖不完整: missing={missing}, insufficient={insufficient}"
        )
    numeric = [float(value) for value in rewards]
    return {
        "rows": len(rows),
        "minimum": min(numeric),
        "mean": mean(numeric),
        "maximum": max(numeric),
        "record_counts": dict(sorted(counts.items())),
    }


def inspect_adapter_change(source_dir: Path, trained_dir: Path) -> dict[str, Any]:
    """Require finite trained LoRA tensors with at least one changed value."""
    try:
        import numpy as np
        from safetensors import safe_open
    except ImportError as exc:
        raise Stage3GRPOError("缺少 numpy 或 safetensors") from exc

    source_path = source_dir / "adapter_model.safetensors"
    trained_path = trained_dir / "adapter_model.safetensors"
    changed_tensors = 0
    changed_elements = 0
    max_abs_delta = 0.0
    with safe_open(source_path, framework="np") as source, safe_open(
        trained_path, framework="np"
    ) as trained:
        source_keys = set(source.keys())
        trained_keys = set(trained.keys())
        if source_keys != trained_keys:
            raise Stage3GRPOError("训练后 adapter tensor 集合发生变化")
        for key in sorted(source_keys):
            before = source.get_tensor(key)
            after = trained.get_tensor(key)
            if before.shape != after.shape or not np.isfinite(after).all():
                raise Stage3GRPOError(f"训练后 adapter tensor 无效: {key}")
            delta = np.abs(after - before)
            changed = int(np.count_nonzero(delta))
            changed_elements += changed
            changed_tensors += int(changed > 0)
            if delta.size:
                max_abs_delta = max(max_abs_delta, float(delta.max()))
    if not changed_tensors:
        raise Stage3GRPOError("训练后 adapter 没有发生变化")
    return {
        "changed_tensors": changed_tensors,
        "changed_elements": changed_elements,
        "max_abs_delta": max_abs_delta,
    }


def _artifact_metadata(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_archive_contract(run_dir: Path) -> None:
    """Require exactly the small Stage 3 publication artifact set."""
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != EXPECTED_ARCHIVE_FILES:
        raise Stage3GRPOError(
            "GRPO 归档目录不匹配: "
            f"missing={sorted(EXPECTED_ARCHIVE_FILES - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_ARCHIVE_FILES)}"
        )
    summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "grpo_trained":
        raise Stage3GRPOError("GRPO run 未成功，不能发布")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise Stage3GRPOError("缺少 PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage3GRPOError(f"YAML 不是对象: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def run_stage3(output_root: Path | None = None) -> Path:
    """Execute the frozen GRPO training run and return its artifact directory."""
    repo_dir = repository_root()
    output_root = (
        (repo_dir / DEFAULT_OUTPUT_RELATIVE_PATH)
        if output_root is None
        else output_root.resolve()
    )
    run_id = generate_run_id()
    run_dir = output_root / run_id
    work_dir = output_root / ".work" / run_id
    if run_dir.exists() or work_dir.exists():
        raise Stage3GRPOError(f"运行目录已存在: {run_id}")
    create_exclusive_directory(work_dir)
    summary_path = work_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage3_grpo",
        "status": "starting",
        "run": {"id": run_id},
    }
    write_json_atomic(summary_path, summary)

    try:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise Stage3GRPOError("缺少 DEEPSEEK_API_KEY")
        huggingface = configure_huggingface_environment()
        environment, torch_module = capture_environment()
        packages = validate_pinned_packages()
        git = git_context(repo_dir)
        summary["run"].update(git)
        summary["environment"] = environment
        summary["huggingface"] = huggingface
        summary["packages"] = packages
        del torch_module

        prompts_path = repo_dir / PROMPTS_RELATIVE_PATH
        system_path = repo_dir / SYSTEM_PROMPT_RELATIVE_PATH
        adapter_dir = repo_dir / SFT_ADAPTER_RELATIVE_PATH
        inputs = {
            str(PROMPTS_RELATIVE_PATH): validate_frozen_file(
                prompts_path, PROMPTS_SHA256
            ),
            str(SYSTEM_PROMPT_RELATIVE_PATH): validate_frozen_file(
                system_path, SYSTEM_PROMPT_SHA256
            ),
            str(SFT_ADAPTER_RELATIVE_PATH): validate_sft_adapter(adapter_dir),
        }
        prompts = read_jsonl(prompts_path)
        rows = prepare_training_rows(
            prompts, system_path.read_text(encoding="utf-8")
        )
        train_data_path = work_dir / "train.jsonl"
        write_jsonl_exclusive(train_data_path, rows)

        source_config_path = repo_dir / CONFIG_RELATIVE_PATH
        config = _load_yaml(source_config_path)
        validate_training_config(config)
        train_output = work_dir / "swift-output"
        config["dataset"] = str(train_data_path)
        config["output_dir"] = str(train_output)
        effective_config_path = work_dir / "training_config.yaml"
        _write_yaml(effective_config_path, config)

        summary["inputs"] = inputs
        summary["training"] = {
            "config_source": str(CONFIG_RELATIVE_PATH),
            "config_sha256": sha256_file(source_config_path),
            "planned_optimizer_steps": 20,
        }
        summary["status"] = "training"
        write_json_atomic(summary_path, summary)

        train_log = work_dir / "train.log"
        train_environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
        command = ["swift", "rlhf", str(effective_config_path)]
        duration = run_logged(
            command,
            train_log,
            repo_dir,
            train_environment,
            expected_steps=20,
        )

        reward_path = train_output / "reward_samples.jsonl"
        reward = validate_reward_samples(
            reward_path, {row["id"] for row in prompts}
        )
        final_adapter = find_final_adapter(train_output)
        adapter_update = inspect_adapter_change(adapter_dir, final_adapter)
        args_path = train_output / "args.json"
        if not args_path.is_file():
            raise Stage3GRPOError("训练输出缺少 args.json")

        archive_stage = work_dir / "archive"
        archive_stage.mkdir()
        shutil.copy2(
            effective_config_path, archive_stage / "training_config.yaml"
        )
        shutil.copy2(train_data_path, archive_stage / "train.jsonl")
        shutil.copy2(train_log, archive_stage / "train.log")
        shutil.copy2(reward_path, archive_stage / "reward_samples.jsonl")
        published_adapter = archive_stage / "adapter"
        published_adapter.mkdir()
        for name in ADAPTER_FILES:
            source = final_adapter / name
            if not source.is_file():
                raise Stage3GRPOError(f"训练 adapter 缺少 {name}")
            shutil.copy2(source, published_adapter / name)

        summary["status"] = "grpo_trained"
        summary["technically_valid"] = True
        summary["ready_to_publish"] = True
        summary["training"].update(
            {
                "duration_seconds": duration,
                "resolved_args_sha256": sha256_file(args_path),
                "adapter_update": adapter_update,
            }
        )
        summary["reward"] = reward
        summary["artifacts"] = {
            relative: _artifact_metadata(
                archive_stage / relative, archive_stage
            )
            for relative in sorted(EXPECTED_ARCHIVE_FILES - {"run_summary.json"})
        }
        write_json_atomic(archive_stage / "run_summary.json", summary)
        validate_archive_contract(archive_stage)
        archive_stage.replace(run_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        return run_dir
    except Exception as exc:
        summary["status"] = "training_failed"
        summary["technically_valid"] = False
        summary["ready_to_publish"] = False
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        summary["retained_work_dir"] = str(work_dir)
        write_json_atomic(summary_path, summary)
        if not run_dir.exists():
            create_exclusive_directory(run_dir)
        shutil.copy2(summary_path, run_dir / "run_summary.json")
        raise


def _bundle_paths(output_dir: Path, run_id: str) -> tuple[Path, Path, str]:
    tag = f"morgana-v2-stage3-grpo-{run_id}"
    return output_dir / f"{tag}.tar.gz", output_dir / f"{tag}.manifest.json", tag


def create_release_bundle(run_dir: Path, output_dir: Path) -> tuple[Path, Path, str]:
    """Create or safely reuse one verified Stage 3 release bundle."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    validate_archive_contract(run_dir)
    summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    run_id = summary.get("run", {}).get("id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise Stage3GRPOError("run_summary.json 缺少有效 run.id")
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise Stage3GRPOError("发布包目录不能位于 run 目录内")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path, manifest_path, tag = _bundle_paths(output_dir, run_id)

    contents = {
        name: _artifact_metadata(run_dir / name, run_dir)
        for name in sorted(EXPECTED_ARCHIVE_FILES)
    }
    if bundle_path.exists() or manifest_path.exists():
        if not bundle_path.is_file() or not manifest_path.is_file():
            raise Stage3GRPOError("发布包或 manifest 不完整")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("contents") != contents
            or manifest.get("bundle", {}).get("sha256")
            != sha256_file(bundle_path)
        ):
            raise Stage3GRPOError("现有发布包与 run 不一致")
        return bundle_path, manifest_path, tag

    temporary = output_dir / f".{bundle_path.name}.{uuid4().hex}.tmp"
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name in sorted(EXPECTED_ARCHIVE_FILES):
                archive.add(
                    run_dir / name,
                    arcname=f"{run_id}/{name}",
                    recursive=False,
                )
        temporary.replace(bundle_path)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "run_status": summary["status"],
            "source_commit": summary.get("run", {}).get("commit"),
            "github_release_tag": tag,
            "bundle": {
                "file": bundle_path.name,
                "bytes": bundle_path.stat().st_size,
                "sha256": sha256_file(bundle_path),
            },
            "contents": contents,
        }
        write_json_atomic(manifest_path, manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        bundle_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return bundle_path, manifest_path, tag


def publish_run(
    run_dir: Path,
    output_dir: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> tuple[Path, Path, str]:
    """Bundle and upload one successful Stage 3 run."""
    bundle_path, manifest_path, tag = create_release_bundle(
        run_dir, output_dir
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        f"morgana-v2 Stage 3 GRPO {manifest['run_id']}",
        "--notes",
        "Stage 3 GRPO 训练产物、奖励日志和 adapter。",
    ]
    source_commit = manifest.get("source_commit")
    if isinstance(source_commit, str) and source_commit:
        command.extend(["--target", source_commit])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise Stage3GRPOError("未安装 GitHub CLI：gh") from exc
    except subprocess.CalledProcessError as exc:
        raise Stage3GRPOError(
            "GitHub Release 上传失败；本地发布包已保留"
        ) from exc
    return bundle_path, manifest_path, tag


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 3 command-line parser."""
    parser = argparse.ArgumentParser(
        description="在 AutoDL 上运行 morgana-v2 Stage 3 GRPO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行完整 GRPO 训练")
    run_parser.add_argument("--output-root", type=Path)
    publish_parser = subparsers.add_parser(
        "publish", help="打包并上传到 GitHub Release"
    )
    publish_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser.add_argument(
        "--output-dir", type=Path, default=Path("dist")
    )
    publish_parser.add_argument("--repo", default=DEFAULT_GITHUB_REPOSITORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested AutoDL Stage 3 command."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_dir = run_stage3(args.output_root)
            print(f"GRPO run: {run_dir}")
            print("ready_to_publish=True")
        elif args.command == "publish":
            bundle, manifest, tag = publish_run(
                args.run_dir, args.output_dir, args.repo
            )
            print(f"GitHub Release: {tag}")
            print(f"发布包: {bundle}")
            print(f"清单: {manifest}")
    except (
        Stage2SFTError,
        Stage3GRPOError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
