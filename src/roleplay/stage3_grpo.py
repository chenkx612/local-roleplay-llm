"""Run and publish the frozen morgana-v2 Stage 3 GRPO workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from roleplay.core.adapters import inspect_adapter_change as _inspect_adapter_change
from roleplay.core.artifacts import (
    read_jsonl as _read_jsonl,
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
    PINNED_PACKAGES,
    capture_environment as _capture_environment,
    configure_huggingface_environment,
    create_exclusive_directory as _create_exclusive_directory,
    find_final_adapter as _find_final_adapter,
    generate_run_id,
    git_context as _git_context,
    run_logged as _run_logged,
    validate_pinned_packages as _validate_pinned_packages,
)
from roleplay.evaluation.comparison import generate_adapter_review_artifacts
from roleplay.experiments.morgana_v2 import (
    DEV_RELATIVE_PATH,
    DEV_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLING_CONFIG as STAGE3_SAMPLING_CONFIG,
    SFT_ADAPTER_HASHES,
    SFT_ADAPTER_RELATIVE_PATH,
    SYSTEM_PROMPT_RELATIVE_PATH,
    SYSTEM_PROMPT_SHA256,
)
from roleplay.sft_eval import evaluate_manual_review


CONFIG_RELATIVE_PATH = Path("configs/morgana_v2_grpo.yaml")
PROMPTS_RELATIVE_PATH = Path("data/runs/morgana-v2/rl_train.jsonl")
DEFAULT_OUTPUT_RELATIVE_PATH = Path("output/morgana-v2/stage3-grpo")

STAGE3_PINNED_PACKAGES = {
    **PINNED_PACKAGES,
    "flash-linear-attention": "0.4.2",
    "msgspec": "0.21.1",
}
STAGE3_DISABLED_ACCELERATION_PACKAGES = ("causal-conv1d",)

PROMPTS_SHA256 = "b36b4f01f232901ab0b5f6011fa64b66f48e02c75b6b0050035e4caf703e7231"

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
    **STAGE3_SAMPLING_CONFIG,
    "enable_thinking": False,
    "packing": False,
    "padding_free": False,
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
        "sft_dev_outputs.jsonl",
        "grpo_dev_outputs.jsonl",
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
        *(f"adapter/{name}" for name in ADAPTER_FILES),
    }
)


class Stage3GRPOError(RuntimeError):
    """Raised when the frozen Stage 3 execution contract is violated."""


RELEASE_SPEC = ReleaseSpec(
    tag_prefix="morgana-v2-stage3-grpo",
    cli_name="roleplay-stage3-grpo",
    title="morgana-v2 Stage 3 GRPO {run_id}",
    notes="Stage 3 GRPO 训练产物、奖励日志和 adapter。",
    expected_files=EXPECTED_ARCHIVE_FILES,
    contract_label="GRPO ",
    default_repository=DEFAULT_GITHUB_REPOSITORY,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path)
    except ValueError as exc:
        raise Stage3GRPOError(str(exc)) from exc


def capture_environment() -> tuple[dict[str, Any], Any]:
    return _capture_environment(
        error_type=Stage3GRPOError,
        run_command=subprocess.run,
    )


def create_exclusive_directory(path: Path) -> Path:
    return _create_exclusive_directory(path, error_type=Stage3GRPOError)


def find_final_adapter(output_dir: Path) -> Path:
    return _find_final_adapter(output_dir, error_type=Stage3GRPOError)


def git_context(repo_dir: Path) -> dict[str, str]:
    return _git_context(
        repo_dir,
        error_type=Stage3GRPOError,
        run_command=subprocess.run,
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
        error_type=Stage3GRPOError,
    )


def validate_pinned_packages(
    required_packages: dict[str, str] = PINNED_PACKAGES,
    disabled_packages: tuple[str, ...] = ("flash-linear-attention", "causal-conv1d"),
) -> dict[str, str]:
    return _validate_pinned_packages(
        required_packages,
        disabled_packages,
        error_type=Stage3GRPOError,
    )


def inspect_adapter_change(
    source_dir: Path, trained_dir: Path
) -> dict[str, Any]:
    return _inspect_adapter_change(
        source_dir,
        trained_dir,
        error_type=Stage3GRPOError,
    )


def generate_dev_review_artifacts(
    repo_dir: Path,
    sft_adapter: Path,
    grpo_adapter: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return generate_adapter_review_artifacts(
        repo_dir,
        sft_adapter,
        grpo_adapter,
        output_dir,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dev_relative_path=DEV_RELATIVE_PATH,
        dev_sha256=DEV_SHA256,
        system_prompt_relative_path=SYSTEM_PROMPT_RELATIVE_PATH,
        sampling_config=STAGE3_SAMPLING_CONFIG,
        error_type=Stage3GRPOError,
    )


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


def _evaluate_grpo_manual_review(
    packet: dict[str, Any],
    answer_key: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the generic A/B gate and expose Stage 3 terminology."""
    generic_key = {
        **answer_key,
        "answers": [
            {
                "review_id": row["review_id"],
                "id": row["id"],
                "base_label": row["sft_label"],
                "sft_label": row["grpo_label"],
            }
            for row in answer_key["answers"]
        ],
    }
    result = evaluate_manual_review(packet, generic_key, results)
    renamed = {
        key: value
        for key, value in result.items()
        if not key.startswith("sft_") and key != "mean_scores"
    }
    return {
        **renamed,
        "checks": {
            key.replace("sft_", "grpo_", 1): value
            for key, value in result["checks"].items()
        },
        "grpo_wins": result["sft_wins"],
        "grpo_clear_losses": result["sft_clear_losses"],
        "grpo_severe_issue_ids": result["sft_severe_issue_ids"],
        "mean_scores": {
            "sft": result["mean_scores"]["base"],
            "grpo": result["mean_scores"]["sft"],
        },
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
    if summary.get("status") not in {
        "awaiting_manual_review",
        "automatic_review_failed",
    }:
        raise Stage3GRPOError("GRPO run 未完成自动复核，不能发布")


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
        packages = validate_pinned_packages(
            STAGE3_PINNED_PACKAGES,
            STAGE3_DISABLED_ACCELERATION_PACKAGES,
        )
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
            str(DEV_RELATIVE_PATH): validate_frozen_file(
                repo_dir / DEV_RELATIVE_PATH, DEV_SHA256
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

        review = generate_dev_review_artifacts(
            repo_dir, adapter_dir, final_adapter, work_dir
        )
        automatic = review["automatic_gate"]

        archive_stage = work_dir / "archive"
        archive_stage.mkdir()
        shutil.copy2(
            effective_config_path, archive_stage / "training_config.yaml"
        )
        shutil.copy2(train_data_path, archive_stage / "train.jsonl")
        shutil.copy2(train_log, archive_stage / "train.log")
        shutil.copy2(reward_path, archive_stage / "reward_samples.jsonl")
        for name, source in review["paths"].items():
            shutil.copy2(source, archive_stage / name)
        published_adapter = archive_stage / "adapter"
        published_adapter.mkdir()
        for name in ADAPTER_FILES:
            source = final_adapter / name
            if not source.is_file():
                raise Stage3GRPOError(f"训练 adapter 缺少 {name}")
            shutil.copy2(source, published_adapter / name)

        summary["status"] = (
            "awaiting_manual_review"
            if automatic["passed"]
            else "automatic_review_failed"
        )
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
        summary["automatic_review"] = {
            "passed": automatic["passed"],
            "checks": automatic["checks"],
            "sft": automatic["base"],
            "grpo": automatic["sft"],
        }
        summary["manual_review"] = {
            "status": (
                "awaiting_manual_review"
                if automatic["passed"]
                else "not_started_automatic_failed"
            ),
            "packet": "manual_review_packet.json",
            "answer_key": "manual_review_answer_key.json",
            "results": "manual_review_results.json",
        }
        summary["ready_for_eval"] = False
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


def create_release_bundle(
    run_dir: Path, output_dir: Path
) -> tuple[Path, Path, str]:
    """Create or safely reuse one verified Stage 3 release bundle."""
    return _create_release_bundle(
        run_dir,
        output_dir,
        spec=RELEASE_SPEC,
        error_type=Stage3GRPOError,
        validate_archive=validate_archive_contract,
    )


def publish_run(
    run_dir: Path,
    output_dir: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> tuple[Path, Path, str]:
    """Bundle and upload one successful Stage 3 run."""
    return _publish_release(
        run_dir,
        output_dir,
        repository,
        spec=RELEASE_SPEC,
        error_type=Stage3GRPOError,
        validate_archive=validate_archive_contract,
        run_command=subprocess.run,
    )


def format_download_command(
    tag: str, repository: str = DEFAULT_GITHUB_REPOSITORY
) -> str:
    """Format the copyable local download command."""
    return _format_download_command(tag, repository, spec=RELEASE_SPEC)


def extract_release_bundle(
    bundle_path: Path, manifest_path: Path, output_root: Path
) -> Path:
    """Verify and atomically extract one Stage 3 release bundle."""
    return _extract_release_bundle(
        bundle_path,
        manifest_path,
        output_root,
        spec=RELEASE_SPEC,
        error_type=Stage3GRPOError,
    )


def download_release(
    tag: str,
    output_root: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> Path:
    """Download, verify, and extract one Stage 3 GitHub Release."""
    return _download_release(
        tag,
        output_root,
        repository,
        spec=RELEASE_SPEC,
        error_type=Stage3GRPOError,
        run_command=subprocess.run,
    )


def review_run(run_dir: Path) -> dict[str, Any]:
    """Validate and record one completed local GRPO manual review."""
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise Stage3GRPOError(f"缺少 run_summary.json: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "manual_reviewed_at_utc" in summary:
        raise Stage3GRPOError("该 run 已完成人工复核，拒绝重复提交")
    required = (
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            raise Stage3GRPOError(f"缺少人工复核文件: {name}")
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
    try:
        gate = _evaluate_grpo_manual_review(packet, answer_key, submitted)
    except ValueError as exc:
        raise Stage3GRPOError(str(exc)) from exc
    automatic_passed = bool(summary.get("automatic_review", {}).get("passed"))
    summary["manual_review"] = {
        **summary.get("manual_review", {}),
        "status": "passed" if gate["passed"] else "failed",
        "gate": gate,
    }
    summary["ready_for_eval"] = automatic_passed and gate["passed"]
    summary["status"] = (
        "ready_for_eval" if summary["ready_for_eval"] else "grpo_failed"
    )
    result_path = run_dir / "manual_review_results.json"
    summary.setdefault("artifacts", {})["manual_review_results.json"] = (
        _artifact_metadata(result_path, run_dir)
    )
    summary["manual_reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary["manual_review"], ensure_ascii=False, indent=2))
    print(f"ready_for_eval={summary['ready_for_eval']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 3 command-line parser."""
    parser = argparse.ArgumentParser(
        description="在 AutoDL 上运行 morgana-v2 Stage 3 GRPO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行完整 GRPO 训练")
    run_parser.add_argument("--output-root", type=Path)
    review_parser = subparsers.add_parser(
        "review", help="提交并验证人工复核结果"
    )
    review_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser = subparsers.add_parser(
        "publish", help="打包并上传到 GitHub Release"
    )
    publish_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser.add_argument(
        "--output-dir", type=Path, default=Path("dist")
    )
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
    """Run the requested AutoDL Stage 3 command."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_dir = run_stage3(args.output_root)
            print(f"GRPO run: {run_dir}")
            print("ready_to_publish=True")
        elif args.command == "review":
            review_run(args.run_dir)
        elif args.command == "publish":
            bundle, manifest, tag = publish_run(
                args.run_dir, args.output_dir, args.repo
            )
            print(f"GitHub Release: {tag}")
            print(f"发布包: {bundle}")
            print(f"清单: {manifest}")
            print(f"本地下载命令: {format_download_command(tag, args.repo)}")
        elif args.command == "download":
            run_dir = download_release(args.tag, args.output_root, args.repo)
            print(f"本地 run 目录: {run_dir}")
            print("请填写 manual_review_results.json 后运行 review。")
    except (
        Stage3GRPOError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
