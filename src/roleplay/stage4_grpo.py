"""Run, publish, download, and review morgana-v2 Stage 4 rule GRPO."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence
from uuid import uuid4

from roleplay.grpo_rule_reward import (
    ACTION_POLICY_COUNTS,
    ACTION_POLICIES,
    PROMPTS_SHA256,
    RuleRewardError,
    reward_spec,
    score_completion,
)
from roleplay.sft_eval import (
    EVALUATION_SEEDS,
    summarize_outputs,
    validate_aligned_outputs,
)
from roleplay.stage2_sft import (
    ADAPTER_FILES,
    DEFAULT_GITHUB_REPOSITORY,
    PINNED_PACKAGES,
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
from roleplay.stage3_grpo import (
    DEV_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    SFT_ADAPTER_HASHES,
    STAGE3_SAMPLING_CONFIG,
    SYSTEM_PROMPT_SHA256,
    Stage3GRPOError,
    generate_dev_review_artifacts,
    inspect_adapter_change,
)


CONFIG_RELATIVE_PATH = Path("configs/morgana_v2_stage4_grpo.yaml")
CONFIG_SHA256 = "50762d76e7a3526b8f110a4ec36665fbbd9c3de3afdfe09149a08c804fbd9437"
PROMPTS_RELATIVE_PATH = Path("data/runs/morgana-v2/rule_grpo_train.jsonl")
DEV_RELATIVE_PATH = Path("data/runs/morgana-v2/dev.jsonl")
SYSTEM_PROMPT_RELATIVE_PATH = Path("data/runs/morgana-v2/system_prompt.txt")
SFT_ADAPTER_RELATIVE_PATH = Path(
    "output/morgana-v2/stage2-sft/final/adapter"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path("output/morgana-v2/stage4-grpo")

STAGE4_PINNED_PACKAGES = {
    **PINNED_PACKAGES,
    "flash-linear-attention": "0.4.2",
    "msgspec": "0.21.1",
}
STAGE4_DISABLED_ACCELERATION_PACKAGES = ("causal-conv1d",)

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
    "num_generations": 8,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 1,
    "learning_rate": 1.0e-6,
    "beta": 0.1,
    **STAGE3_SAMPLING_CONFIG,
    "temperature": 0.8,
    "top_p": 0.9,
    "enable_thinking": False,
    "packing": False,
    "padding_free": False,
    "use_vllm": False,
    "external_plugins": ["src/roleplay/grpo_rule_reward_plugin.py"],
    "reward_funcs": ["morgana_rule_reward"],
}

SEVERE_ISSUE_CODES = frozenset(
    {
        "unreadable",
        "role_break",
        "perspective_shift",
        "fabricated_background",
        "servile_submission",
        "romanticization",
    }
)

EXPECTED_ARCHIVE_FILES = frozenset(
    {
        "run_summary.json",
        "training_config.yaml",
        "train.jsonl",
        "train.log",
        "reward_spec.json",
        "reward_samples.jsonl",
        "rule_dev_scores.jsonl",
        "sft_dev_outputs.jsonl",
        "grpo_dev_outputs.jsonl",
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
        *(f"adapter/{name}" for name in ADAPTER_FILES),
    }
)

_DEV_ACTION_POLICY = {
    "adversarial": "encouraged",
    "daily": "encouraged",
    "style": "encouraged",
    "background": "optional",
    "emotion": "optional",
}

REWARD_COMPONENT_FIELDS = frozenset(
    {
        "action_policy",
        "normalized_length",
        "length_score",
        "signature_count",
        "signature_score",
        "wrong_self_references",
        "wrong_self_penalty",
        "action",
        "action_score",
        "format_reasons",
        "format_penalty",
        "hard_invalid_reasons",
        "raw_reward",
        "total_reward",
    }
)
ACTION_ANALYSIS_FIELDS = frozenset(
    {
        "segments",
        "count",
        "total_action_length",
        "action_ratio",
        "minimum_dialogue_gap",
        "unbalanced",
        "nested",
        "overlong",
        "over_ratio",
        "dense_pair",
        "invalid_content",
    }
)


class Stage4GRPOError(RuntimeError):
    """Raised when the frozen Stage 4 execution contract is violated."""


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[2]


def validate_frozen_file(path: Path, expected_sha256: str) -> str:
    """Require one frozen file and return its verified digest."""
    if not path.is_file():
        raise Stage4GRPOError(f"缺少冻结文件: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise Stage4GRPOError(f"冻结文件哈希不匹配: {path}")
    return digest


def validate_sft_adapter(adapter_dir: Path) -> dict[str, str]:
    """Require the exact accepted Stage 2 adapter at the fixed path."""
    validated = {}
    for name, expected in SFT_ADAPTER_HASHES.items():
        validated[name] = validate_frozen_file(adapter_dir / name, expected)
    return validated


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _row_user(row: dict[str, Any]) -> str | None:
    user = row.get("user")
    if isinstance(user, str):
        return user
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                return message["content"]
    return None


def validate_prompt_isolation(
    repo_dir: Path,
    prompts: Sequence[dict[str, Any]],
) -> None:
    """Reject exact normalized overlap with every other v2 JSONL split."""
    current = {_normalize_prompt(row["user"]) for row in prompts}
    if len(current) != len(prompts):
        raise Stage4GRPOError("规则 GRPO Prompt 内部存在重复")
    overlaps = []
    data_dir = repo_dir / "data/runs/morgana-v2"
    for path in sorted(data_dir.glob("*.jsonl")):
        if path == repo_dir / PROMPTS_RELATIVE_PATH:
            continue
        for row in read_jsonl(path):
            user = _row_user(row)
            if isinstance(user, str) and _normalize_prompt(user) in current:
                overlaps.append(f"{path.name}:{user}")
    if overlaps:
        raise Stage4GRPOError(
            "规则 GRPO Prompt 与其他 split 重复: " + "; ".join(overlaps)
        )


def prepare_training_rows(
    prompts: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    """Validate frozen rule prompts and create ms-swift conversations."""
    if len(prompts) != 20:
        raise Stage4GRPOError(f"规则 Prompt 数量必须为 20，实际 {len(prompts)}")
    expected_fields = {
        "id",
        "scenario",
        "user",
        "target_rules",
        "reward_policy",
    }
    ids = set()
    counts = {name: 0 for name in ACTION_POLICIES}
    records = []
    for index, row in enumerate(prompts):
        if set(row) != expected_fields:
            raise Stage4GRPOError(f"规则 Prompt {index} 字段不正确")
        record_id = row["id"]
        policy = row["reward_policy"]
        if (
            not isinstance(record_id, str)
            or not re.fullmatch(r"rule_grpo_\d{4}", record_id)
            or record_id in ids
            or not isinstance(row["scenario"], str)
            or not row["scenario"].strip()
            or not isinstance(row["user"], str)
            or not row["user"].strip()
            or not isinstance(row["target_rules"], list)
            or not row["target_rules"]
            or not isinstance(policy, dict)
            or set(policy) != {"action"}
            or policy["action"] not in ACTION_POLICIES
        ):
            raise Stage4GRPOError(f"规则 Prompt {index} 内容不正确")
        ids.add(record_id)
        counts[policy["action"]] += 1
        records.append(
            {
                "id": record_id,
                "prompt_id": record_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["user"]},
                ],
            }
        )
    if counts != ACTION_POLICY_COUNTS:
        raise Stage4GRPOError(
            f"动作策略分布不正确: {counts} != {ACTION_POLICY_COUNTS}"
        )
    return records


def validate_training_config(config: dict[str, Any]) -> None:
    """Require the single frozen Stage 4 GRPO configuration."""
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
        raise Stage4GRPOError(
            "Stage 4 GRPO 冻结配置不正确: " + ", ".join(mismatches)
        )


def validate_reward_samples(
    path: Path,
    expected_record_ids: set[str],
) -> dict[str, Any]:
    """Require eight successful, component-complete rewards per prompt."""
    if not path.is_file():
        raise Stage4GRPOError(f"缺少奖励日志: {path}")
    rows = read_jsonl(path)
    if not rows:
        raise Stage4GRPOError("奖励日志为空")
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        raise Stage4GRPOError(f"奖励日志包含 {len(errors)} 条失败")
    for row in rows:
        reward = row.get("total_reward")
        components = row.get("components")
        action = (
            components.get("action") if isinstance(components, dict) else None
        )
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not isinstance(components, dict)
            or set(components) != REWARD_COMPONENT_FIELDS
            or not isinstance(action, dict)
            or set(action) != ACTION_ANALYSIS_FIELDS
            or not isinstance(components["wrong_self_references"], list)
            or not isinstance(components["format_reasons"], list)
            or not isinstance(components["hard_invalid_reasons"], list)
            or not isinstance(action["segments"], list)
            or components.get("total_reward") != reward
        ):
            raise Stage4GRPOError("奖励日志缺少有效子分或 total_reward")
    counts = Counter(row.get("record_id") for row in rows)
    unexpected = sorted(set(counts) - expected_record_ids, key=str)
    invalid_counts = {
        record_id: counts[record_id]
        for record_id in sorted(expected_record_ids)
        if counts[record_id] != 8
    }
    if unexpected or invalid_counts:
        raise Stage4GRPOError(
            "奖励覆盖不完整: "
            f"unexpected={unexpected}, invalid_counts={invalid_counts}"
        )
    rewards = [float(row["total_reward"]) for row in rows]
    return {
        "rows": len(rows),
        "minimum": min(rewards),
        "mean": mean(rewards),
        "maximum": max(rewards),
        "record_counts": dict(sorted(counts.items())),
    }


def _dev_policy(row: dict[str, Any]) -> str:
    try:
        return _DEV_ACTION_POLICY[row["scenario"]]
    except KeyError as exc:
        raise Stage4GRPOError(
            f"Dev 场景缺少动作策略: {row.get('scenario')!r}"
        ) from exc


def evaluate_rule_dev(
    sft_rows: Sequence[dict[str, Any]],
    grpo_rows: Sequence[dict[str, Any]],
    expected_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score aligned Dev outputs and apply the frozen automatic gate."""
    validate_aligned_outputs(
        sft_rows, grpo_rows, EVALUATION_SEEDS, expected_ids
    )
    sft_summary = summarize_outputs(sft_rows, EVALUATION_SEEDS, expected_ids)
    grpo_summary = summarize_outputs(grpo_rows, EVALUATION_SEEDS, expected_ids)
    sft_index = {(row["seed"], row["id"]): row for row in sft_rows}
    grpo_index = {(row["seed"], row["id"]): row for row in grpo_rows}
    score_rows = []
    sft_rewards = []
    grpo_rewards = []
    grpo_hard_invalid = []
    sft_wrong_count = 0
    grpo_wrong_count = 0
    wins = 0
    for seed in EVALUATION_SEEDS:
        for record_id in expected_ids:
            sft = sft_index[(seed, record_id)]
            grpo = grpo_index[(seed, record_id)]
            policy = _dev_policy(sft)
            sft_score = score_completion(
                sft["assistant"], policy, finish_reason=sft.get("finish_reason")
            )
            grpo_score = score_completion(
                grpo["assistant"], policy, finish_reason=grpo.get("finish_reason")
            )
            delta = grpo_score.total_reward - sft_score.total_reward
            wins += int(delta > 0)
            sft_rewards.append(sft_score.total_reward)
            grpo_rewards.append(grpo_score.total_reward)
            sft_wrong_count += int(bool(sft_score.wrong_self_references))
            grpo_wrong_count += int(bool(grpo_score.wrong_self_references))
            if grpo_score.hard_invalid_reasons:
                grpo_hard_invalid.append(f"{seed}:{record_id}")
            score_rows.append(
                {
                    "schema_version": 1,
                    "seed": seed,
                    "id": record_id,
                    "action_policy": policy,
                    "sft": sft_score.as_log_dict(),
                    "grpo": grpo_score.as_log_dict(),
                    "reward_delta": delta,
                    "winner": "grpo" if delta > 0 else "sft" if delta < 0 else "tie",
                }
            )
    sft_mean = mean(sft_rewards)
    grpo_mean = mean(grpo_rewards)
    mean_delta = grpo_mean - sft_mean
    sft_overall = sft_summary["overall"]
    grpo_overall = grpo_summary["overall"]
    expected_count = len(EVALUATION_SEEDS) * len(expected_ids)
    checks = {
        "complete_and_normal_stop": (
            grpo_overall["records"] == expected_count
            and grpo_overall["nonempty_count"] == expected_count
            and grpo_overall["stop_count"] == expected_count
        ),
        "no_hard_invalid_outputs": not grpo_hard_invalid,
        "degeneration_count_not_higher": (
            grpo_overall["degeneration_count"]
            <= sft_overall["degeneration_count"]
        ),
        "wrong_self_reference_count_not_higher": (
            grpo_wrong_count <= sft_wrong_count
        ),
        "rule_reward_mean_delta_at_least_0_3": mean_delta >= 0.3,
        "rule_reward_wins_at_least_6": wins >= 6,
    }
    return (
        {
            "passed": all(checks.values()),
            "checks": checks,
            "sft": {
                "mean_rule_reward": sft_mean,
                "wrong_self_reference_count": sft_wrong_count,
                "generation": sft_summary,
            },
            "grpo": {
                "mean_rule_reward": grpo_mean,
                "wrong_self_reference_count": grpo_wrong_count,
                "hard_invalid_ids": grpo_hard_invalid,
                "generation": grpo_summary,
            },
            "mean_rule_reward_delta": mean_delta,
            "grpo_rule_wins": wins,
            "reviewed_pairs": len(score_rows),
        },
        score_rows,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise Stage4GRPOError("缺少 PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage4GRPOError(f"YAML 不是对象: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _artifact_metadata(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def specialize_manual_review_artifacts(paths: dict[str, Path]) -> None:
    """Expose the frozen severe-issue vocabulary in downloaded review files."""
    packet_path = paths["manual_review_packet.json"]
    results_path = paths["manual_review_results.json"]
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["severe_issue_codes"] = sorted(SEVERE_ISSUE_CODES)
    packet["severe_issue_examples"] = sorted(SEVERE_ISSUE_CODES)
    write_json_atomic(packet_path, packet)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    instructions = results.setdefault("instructions", {})
    instructions["severe_issues"] = (
        "A/B 各填写严重问题代码列表，仅允许 severe_issue_codes 中的值，无则为空列表"
    )
    results["severe_issue_codes"] = sorted(SEVERE_ISSUE_CODES)
    write_json_atomic(results_path, results)


def validate_archive_contract(run_dir: Path) -> None:
    """Require exactly the Stage 4 publication artifact set."""
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != EXPECTED_ARCHIVE_FILES:
        raise Stage4GRPOError(
            "Stage 4 GRPO 归档目录不匹配: "
            f"missing={sorted(EXPECTED_ARCHIVE_FILES - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_ARCHIVE_FILES)}"
        )
    summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") not in {
        "awaiting_manual_review",
        "automatic_review_failed",
        "ready_for_eval",
        "grpo_failed",
    }:
        raise Stage4GRPOError("Stage 4 GRPO run 未完成，不能归档")


def run_stage4(output_root: Path | None = None) -> Path:
    """Execute the frozen Stage 4 rule GRPO workflow."""
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
        raise Stage4GRPOError(f"运行目录已存在: {run_id}")
    create_exclusive_directory(work_dir)
    summary_path = work_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage4_rule_grpo",
        "status": "starting",
        "run": {"id": run_id},
    }
    write_json_atomic(summary_path, summary)
    try:
        huggingface = configure_huggingface_environment()
        environment, torch_module = capture_environment()
        packages = validate_pinned_packages(
            STAGE4_PINNED_PACKAGES,
            STAGE4_DISABLED_ACCELERATION_PACKAGES,
        )
        summary["run"].update(git_context(repo_dir))
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
        validate_prompt_isolation(repo_dir, prompts)
        rows = prepare_training_rows(
            prompts, system_path.read_text(encoding="utf-8")
        )
        train_data_path = work_dir / "train.jsonl"
        write_jsonl_exclusive(train_data_path, rows)
        reward_spec_path = work_dir / "reward_spec.json"
        write_json_atomic(reward_spec_path, reward_spec())

        source_config_path = repo_dir / CONFIG_RELATIVE_PATH
        config_digest = validate_frozen_file(source_config_path, CONFIG_SHA256)
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
            "config_sha256": config_digest,
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
        duration = run_logged(
            ["swift", "rlhf", str(effective_config_path)],
            train_log,
            repo_dir,
            train_environment,
            expected_steps=20,
        )

        reward_path = train_output / "reward_samples.jsonl"
        reward = validate_reward_samples(
            reward_path,
            {row["id"] for row in prompts},
        )
        final_adapter = find_final_adapter(train_output)
        adapter_update = inspect_adapter_change(adapter_dir, final_adapter)
        args_path = train_output / "args.json"
        if not args_path.is_file():
            raise Stage4GRPOError("训练输出缺少 args.json")

        generated = generate_dev_review_artifacts(
            repo_dir, adapter_dir, final_adapter, work_dir
        )
        specialize_manual_review_artifacts(generated["paths"])
        sft_dev = read_jsonl(generated["paths"]["sft_dev_outputs.jsonl"])
        grpo_dev = read_jsonl(generated["paths"]["grpo_dev_outputs.jsonl"])
        dev_rows = read_jsonl(repo_dir / DEV_RELATIVE_PATH)
        automatic, dev_scores = evaluate_rule_dev(
            sft_dev, grpo_dev, [row["id"] for row in dev_rows]
        )
        rule_dev_path = work_dir / "rule_dev_scores.jsonl"
        write_jsonl_exclusive(rule_dev_path, dev_scores)

        archive_stage = work_dir / "archive"
        archive_stage.mkdir()
        direct_artifacts = {
            "training_config.yaml": effective_config_path,
            "train.jsonl": train_data_path,
            "train.log": train_log,
            "reward_spec.json": reward_spec_path,
            "reward_samples.jsonl": reward_path,
            "rule_dev_scores.jsonl": rule_dev_path,
            **generated["paths"],
        }
        for name, source in direct_artifacts.items():
            shutil.copy2(source, archive_stage / name)
        published_adapter = archive_stage / "adapter"
        published_adapter.mkdir()
        for name in ADAPTER_FILES:
            source = final_adapter / name
            if not source.is_file():
                raise Stage4GRPOError(f"训练 adapter 缺少 {name}")
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
        summary["automatic_review"] = automatic
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
            relative: _artifact_metadata(archive_stage / relative, archive_stage)
            for relative in sorted(
                EXPECTED_ARCHIVE_FILES - {"run_summary.json"}
            )
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
    tag = f"morgana-v2-stage4-grpo-{run_id}"
    return output_dir / f"{tag}.tar.gz", output_dir / f"{tag}.manifest.json", tag


def create_release_bundle(
    run_dir: Path, output_dir: Path
) -> tuple[Path, Path, str]:
    """Create or safely reuse one verified Stage 4 release bundle."""
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
        raise Stage4GRPOError("run_summary.json 缺少有效 run.id")
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise Stage4GRPOError("发布包目录不能位于 run 目录内")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path, manifest_path, tag = _bundle_paths(output_dir, run_id)
    contents = {
        name: _artifact_metadata(run_dir / name, run_dir)
        for name in sorted(EXPECTED_ARCHIVE_FILES)
    }
    if bundle_path.exists() or manifest_path.exists():
        if not bundle_path.is_file() or not manifest_path.is_file():
            raise Stage4GRPOError("发布包或 manifest 不完整")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("contents") != contents
            or manifest.get("bundle", {}).get("sha256")
            != sha256_file(bundle_path)
        ):
            raise Stage4GRPOError("现有发布包与 run 不一致")
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
            "stage": "stage4_rule_grpo",
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
    """Bundle and upload one completed Stage 4 run."""
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
        f"morgana-v2 Stage 4 rule GRPO {manifest['run_id']}",
        "--notes",
        "Stage 4 规则型 GRPO 训练产物、本地奖励日志和 adapter。",
    ]
    source_commit = manifest.get("source_commit")
    if isinstance(source_commit, str) and source_commit:
        command.extend(["--target", source_commit])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise Stage4GRPOError("未安装 GitHub CLI：gh") from exc
    except subprocess.CalledProcessError as exc:
        raise Stage4GRPOError(
            "GitHub Release 上传失败；本地发布包已保留"
        ) from exc
    return bundle_path, manifest_path, tag


def format_download_command(
    tag: str, repository: str = DEFAULT_GITHUB_REPOSITORY
) -> str:
    command = ["roleplay-stage4-grpo", "download", "--tag", tag]
    if repository != DEFAULT_GITHUB_REPOSITORY:
        command.extend(["--repo", repository])
    return shlex.join(command)


def extract_release_bundle(
    bundle_path: Path, manifest_path: Path, output_root: Path
) -> Path:
    """Verify and atomically extract one Stage 4 release bundle."""
    bundle_path = bundle_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle", {}).get("sha256") != sha256_file(bundle_path):
        raise Stage4GRPOError("下载包 SHA-256 与 manifest 不匹配")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise Stage4GRPOError("manifest 缺少有效 run_id")
    contents = manifest.get("contents")
    if (
        not isinstance(contents, dict)
        or set(contents) != EXPECTED_ARCHIVE_FILES
        or any(not isinstance(value, dict) for value in contents.values())
    ):
        raise Stage4GRPOError("manifest 文件清单不满足 Stage 4 归档契约")
    expected_names = {f"{run_id}/{name}" for name in contents}
    output_root = output_root.resolve()
    destination = output_root / run_id
    if destination.exists():
        raise Stage4GRPOError(f"本地 run 目录已存在，拒绝覆盖: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / f".download-{uuid4().hex}"
    staging_root.mkdir()
    try:
        with tarfile.open(bundle_path, "r:gz") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != expected_names or not all(
                member.isfile() for member in members
            ):
                raise Stage4GRPOError("发布包内容与 manifest 不一致")
            for member in members:
                target = (staging_root / member.name).resolve()
                try:
                    target.relative_to(staging_root)
                except ValueError as exc:
                    raise Stage4GRPOError("发布包包含不安全路径") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise Stage4GRPOError(f"无法读取发布包文件: {member.name}")
                with source, target.open("xb") as output_file:
                    shutil.copyfileobj(source, output_file)
        staged_run = staging_root / run_id
        for relative, metadata in contents.items():
            path = staged_run / relative
            if (
                not path.is_file()
                or path.stat().st_size != metadata.get("bytes")
                or sha256_file(path) != metadata.get("sha256")
            ):
                raise Stage4GRPOError(f"解包文件校验失败: {relative}")
        staged_run.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return destination


def download_release(
    tag: str,
    output_root: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> Path:
    """Download, verify, and extract one Stage 4 GitHub Release."""
    if not re.fullmatch(r"morgana-v2-stage4-grpo-[A-Za-z0-9._-]+", tag):
        raise Stage4GRPOError("Stage 4 release tag 格式无效")
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
            raise Stage4GRPOError("未安装 GitHub CLI：gh") from exc
        except subprocess.CalledProcessError as exc:
            raise Stage4GRPOError(f"GitHub Release 下载失败: {tag}") from exc
        bundle = download_dir / f"{tag}.tar.gz"
        manifest = download_dir / f"{tag}.manifest.json"
        if not bundle.is_file() or not manifest.is_file():
            raise Stage4GRPOError("Release 缺少发布包或 manifest")
        return extract_release_bundle(bundle, manifest, output_root)


def evaluate_stage4_manual_review(
    packet: dict[str, Any],
    answer_key: dict[str, Any],
    submitted: dict[str, Any],
) -> dict[str, Any]:
    """Validate the review and block only new severe GRPO regressions."""
    items = packet.get("items")
    answers = answer_key.get("answers")
    results = submitted.get("results")
    if not all(isinstance(value, list) for value in (items, answers, results)):
        raise Stage4GRPOError("人工复核文件结构不正确")
    item_ids = {item.get("review_id") for item in items}
    answer_by_id = {answer.get("review_id"): answer for answer in answers}
    result_by_id = {result.get("review_id"): result for result in results}
    if (
        None in item_ids
        or len(item_ids) != len(items)
        or set(answer_by_id) != item_ids
        or set(result_by_id) != item_ids
        or len(answer_by_id) != len(answers)
        or len(result_by_id) != len(results)
    ):
        raise Stage4GRPOError("人工复核结果未完整且唯一地对齐")
    wins = 0
    clear_losses = 0
    new_severe = []
    score_totals = {
        model: {
            dimension: 0
            for dimension in (
                "generation_stability",
                "role_consistency",
                "dialogue_quality",
            )
        }
        for model in ("sft", "grpo")
    }
    for review_id in item_ids:
        result = result_by_id[review_id]
        if set(result) != {
            "review_id",
            "winner",
            "clearly_worse",
            "scores",
            "severe_issues",
        }:
            raise Stage4GRPOError(f"{review_id} 人工结果字段不正确")
        if result["winner"] not in {"A", "B", "tie"}:
            raise Stage4GRPOError(f"{review_id} winner 不正确")
        if result["clearly_worse"] not in {"A", "B", None}:
            raise Stage4GRPOError(f"{review_id} clearly_worse 不正确")
        scores = result["scores"]
        severe = result["severe_issues"]
        dimensions = set(score_totals["sft"])
        if not isinstance(scores, dict) or set(scores) != {"A", "B"}:
            raise Stage4GRPOError(f"{review_id} scores 不正确")
        if not isinstance(severe, dict) or set(severe) != {"A", "B"}:
            raise Stage4GRPOError(f"{review_id} severe_issues 不正确")
        for label in ("A", "B"):
            if (
                not isinstance(scores[label], dict)
                or set(scores[label]) != dimensions
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= 10
                    for value in scores[label].values()
                )
            ):
                raise Stage4GRPOError(f"{review_id} {label} scores 不正确")
            codes = severe[label]
            if (
                not isinstance(codes, list)
                or any(not isinstance(code, str) for code in codes)
                or len(codes) != len(set(codes))
                or set(codes) - SEVERE_ISSUE_CODES
            ):
                raise Stage4GRPOError(
                    f"{review_id} {label} severe_issues 包含未知或重复代码"
                )
        mapping = answer_by_id[review_id]
        sft_label = mapping.get("sft_label")
        grpo_label = mapping.get("grpo_label")
        if {sft_label, grpo_label} != {"A", "B"}:
            raise Stage4GRPOError(f"{review_id} answer key 不正确")
        wins += int(result["winner"] == grpo_label)
        clear_losses += int(result["clearly_worse"] == grpo_label)
        new_codes = sorted(set(severe[grpo_label]) - set(severe[sft_label]))
        if new_codes:
            new_severe.append({"review_id": review_id, "codes": new_codes})
        for dimension in dimensions:
            score_totals["sft"][dimension] += scores[sft_label][dimension]
            score_totals["grpo"][dimension] += scores[grpo_label][dimension]
    mean_scores = {
        model: {
            dimension: value / len(results)
            for dimension, value in totals.items()
        }
        for model, totals in score_totals.items()
    }
    checks = {"grpo_has_no_new_severe_issues": not new_severe}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reviewed_pairs": len(results),
        "grpo_wins": wins,
        "grpo_clear_losses": clear_losses,
        "new_grpo_severe_issues": new_severe,
        "mean_scores": mean_scores,
    }


def review_run(run_dir: Path) -> dict[str, Any]:
    """Validate and record one completed Stage 4 manual review."""
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise Stage4GRPOError(f"缺少 run_summary.json: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "manual_reviewed_at_utc" in summary:
        raise Stage4GRPOError("该 run 已完成人工复核，拒绝重复提交")
    for name in (
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
    ):
        if not (run_dir / name).is_file():
            raise Stage4GRPOError(f"缺少人工复核文件: {name}")
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
    gate = evaluate_stage4_manual_review(packet, answer_key, submitted)
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
    """Build the Stage 4 command-line parser."""
    parser = argparse.ArgumentParser(
        description="运行 morgana-v2 Stage 4 规则型 GRPO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行完整规则型 GRPO")
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
    review_parser = subparsers.add_parser(
        "review", help="提交并验证人工复核结果"
    )
    review_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested Stage 4 command."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_dir = run_stage4(args.output_root)
            print(f"Stage 4 GRPO run: {run_dir}")
            print("ready_to_publish=True")
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
        elif args.command == "review":
            review_run(args.run_dir)
    except (
        RuleRewardError,
        Stage2SFTError,
        Stage3GRPOError,
        Stage4GRPOError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
