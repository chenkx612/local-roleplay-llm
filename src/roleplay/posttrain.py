"""Mac/MLX post-training command line for the morgana-v1 learning loop."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    environment_snapshot,
    new_run_id,
    read_json,
    read_jsonl,
    sha256_file,
)
from .grpo import group_should_skip
from .mlx_backend import (
    MLXRuntimeError,
    adapter_reload_smoke,
    apply_grpo_group_update,
    configure_grpo_memory_strategy,
    generate_candidate_with_logprobs,
    generate_grid,
    gradient_precheck,
    load_base,
    make_temporary_mlx_dataset,
    prepare_trainable_model,
    require_mlx,
    save_adapter,
    train_sft,
)
from .persona import load_persona, render_persona_prompt
from .posttrain_config import (
    ConfigurationError,
    load_grpo_config,
    load_sft_config,
)
from .reward import (
    build_judge_messages,
    judge_group_with_retry,
    load_reward_sources,
    score_group,
)
from .sft_eval import (
    build_manual_review,
    empty_manual_review_results,
    evaluate_manual_review,
    evaluate_relative_behavior_gate,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_CONFIG = ROOT / "configs/morgana_v1_sft_mlx.json"
DEFAULT_GRPO_CONFIG = ROOT / "configs/morgana_v1_grpo_mlx.json"
OUTPUT_ROOT = ROOT / "output/morgana-v1/posttrain"
FROZEN_HASHES = {
    "data/runs/morgana-v1/sft_train.jsonl": "277323c097305ebbee7bfa93cb27c34247800f2dec835f6d052ede7c2b178a7a",
    "data/runs/morgana-v1/rl_train.jsonl": "e3e2c530516c6e5079bb970ffc8aaea553ce881480e8286d14fec4653edb1039",
    "data/runs/morgana-v1/dev.jsonl": "cbce0b38bb6f8b8cbef0bc45fd52b5a8212a66445569dfe0a8c7e8e88f63ddc6",
    "data/runs/morgana-v1/eval.jsonl": "e6148daa90bde920fea7a4a455ddc228ae9518f134113f3c73caadc476f480cc",
    "data/runs/morgana-v1/inputs/persona.json": "8b4be4ac72b0f90ff2bd875fe318319ce1498cd11cd503e39f782ca14b46ae90",
    "data/runs/morgana-v1/inputs/style_examples.jsonl": "fb0b748cbb44d31df93561515daebd8b598bb0352236c46500491dde9b9c084a",
}
MEMORY_STRATEGIES = (
    {"name": "baseline", "grad_checkpoint": False, "clear_cache_each_candidate": False, "train_layers": 24},
    {"name": "grad_checkpoint", "grad_checkpoint": True, "clear_cache_each_candidate": False, "train_layers": 24},
    {"name": "grad_checkpoint_clear_cache", "grad_checkpoint": True, "clear_cache_each_candidate": True, "train_layers": 24},
    {"name": "last_8_layers", "grad_checkpoint": True, "clear_cache_each_candidate": True, "train_layers": 8},
)


class PosttrainError(RuntimeError):
    """Raised when a stage gate or immutable input contract is not met."""


def resolve_run_dir(run_id: str | None, *, create: bool) -> Path:
    selected = run_id or new_run_id()
    if Path(selected).name != selected or selected in {".", ".."}:
        raise PosttrainError("run-id 只能是单个安全目录名")
    run_dir = OUTPUT_ROOT / selected
    if create:
        for child in ("sft", "grpo", "evaluation"):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
    elif not run_dir.is_dir():
        raise PosttrainError(f"run 不存在: {run_dir}")
    return run_dir


def _relative_path(config_value: str) -> Path:
    path = Path(config_value)
    return path if path.is_absolute() else ROOT / path


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tokenizer_for_doctor(config: dict[str, Any]) -> Any:
    try:
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import load_tokenizer
    except ImportError as exc:
        raise MLXRuntimeError("doctor 缺少 huggingface-hub 或 mlx-lm") from exc
    snapshot = Path(snapshot_download(
        config["model"], revision=config["model_revision"],
        allow_patterns=["*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"],
    ))
    if snapshot.name != config["model_revision"]:
        raise PosttrainError(
            f"下载模型 snapshot={snapshot.name}，不等于冻结 revision={config['model_revision']}"
        )
    return load_tokenizer(snapshot)


def _default_doctor_probe(config: dict[str, Any]) -> dict[str, Any]:
    """Probe Metal/revision/tokens in a child so a Metal abort is reportable."""
    script = r'''
import json
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_tokenizer

model, revision, dataset, limit = sys.argv[1:]
snapshot = Path(snapshot_download(
    model,
    revision=revision,
    allow_patterns=["*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"],
))
tokenizer = load_tokenizer(snapshot)
rows = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
lengths = [len(tokenizer.apply_chat_template(row["messages"], return_dict=False)) for row in rows]
print("ROLEPLAY_PROBE=" + json.dumps({
    "metal_available": bool(mx.metal.is_available()),
    "device": mx.device_info(),
    "snapshot_revision": snapshot.name,
    "records": len(lengths),
    "maximum": max(lengths),
    "limit": int(limit),
    "all_within_limit": max(lengths) <= int(limit),
}))
'''
    result = subprocess.run(
        [
            sys.executable, "-c", script, config["model"],
            config["model_revision"], str(_relative_path(config["dataset"])),
            str(config["max_seq_length"]),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    marker = "ROLEPLAY_PROBE="
    line = next(
        (item for item in reversed(result.stdout.splitlines()) if item.startswith(marker)),
        None,
    )
    if result.returncode != 0 or line is None:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
        raise MLXRuntimeError(f"MLX doctor 子进程失败: {detail[-1000:]}")
    return json.loads(line[len(marker):])


def run_doctor(
    run_dir: Path, sft_config_path: Path = DEFAULT_SFT_CONFIG,
    *, tokenizer_factory: Callable[[dict[str, Any]], Any] = _tokenizer_for_doctor,
) -> dict[str, Any]:
    config = load_sft_config(sft_config_path)
    checks: dict[str, bool] = {}
    details = environment_snapshot()
    checks["python_3_13"] = sys.version_info[:2] == (3, 13)
    checks["apple_silicon"] = platform.system() == "Darwin" and platform.machine() == "arm64"
    versions = {"mlx": _version("mlx"), "mlx_lm": _version("mlx-lm")}
    checks["mlx_0_32_0"] = versions["mlx"] == "0.32.0"
    checks["mlx_lm_0_31_3"] = versions["mlx_lm"] == "0.31.3"
    probe = None
    using_default_probe = tokenizer_factory is _tokenizer_for_doctor
    if versions["mlx"] and versions["mlx_lm"] and using_default_probe:
        try:
            probe = _default_doctor_probe(config)
            details["mlx_device"] = probe["device"]
        except Exception as exc:
            details["mlx_probe_error"] = str(exc)
    checks["metal_available"] = bool(probe and probe["metal_available"])
    hashes = {}
    for relative, expected in FROZEN_HASHES.items():
        actual = sha256_file(ROOT / relative)
        hashes[relative] = {"expected": expected, "actual": actual, "passed": actual == expected}
    checks["frozen_input_hashes"] = all(item["passed"] for item in hashes.values())
    token_audit: dict[str, Any] = {"checked": False}
    try:
        if using_default_probe:
            if probe is None:
                raise MLXRuntimeError(
                    details.get("mlx_probe_error", "MLX doctor probe 未返回结果")
                )
            token_audit = {
                "checked": True, "records": probe["records"],
                "maximum": probe["maximum"], "limit": probe["limit"],
                "all_within_limit": probe["all_within_limit"],
            }
            snapshot_revision = probe["snapshot_revision"]
        else:
            tokenizer = tokenizer_factory(config)
            rows = read_jsonl(_relative_path(config["dataset"]))
            lengths = [
                len(tokenizer.apply_chat_template(row["messages"], return_dict=False))
                for row in rows
            ]
            token_audit = {
                "checked": True, "records": len(lengths), "maximum": max(lengths),
                "limit": config["max_seq_length"],
                "all_within_limit": max(lengths) <= config["max_seq_length"],
            }
            snapshot_revision = config["model_revision"]
        checks["model_revision"] = snapshot_revision == config["model_revision"]
        checks["token_lengths"] = token_audit["all_within_limit"]
    except Exception as exc:
        token_audit = {"checked": False, "error": str(exc)}
        checks["model_revision"] = False
        checks["token_lengths"] = False
    report = {
        "schema_version": 1, "stage": "doctor", "passed": all(checks.values()),
        "checks": checks, "versions": versions, "environment": details,
        "config_sha256": sha256_file(sft_config_path),
        "model": {"name": config["model"], "revision": config["model_revision"]},
        "input_hashes": hashes, "token_audit": token_audit,
    }
    atomic_write_json(run_dir / "environment.json", details)
    atomic_write_json(run_dir / "doctor.json", report)
    return report


def _require_doctor(run_dir: Path, config_path: Path) -> None:
    report_path = run_dir / "doctor.json"
    report = read_json(report_path) if report_path.exists() else run_doctor(run_dir, config_path)
    if report.get("config_sha256") != sha256_file(config_path):
        raise PosttrainError("doctor 使用的 SFT 配置与当前命令不一致")
    if not report.get("passed"):
        raise PosttrainError(f"doctor 未通过；详见 {report_path}")


def run_sft_stage(run_dir: Path, config_path: Path = DEFAULT_SFT_CONFIG) -> dict[str, Any]:
    _require_doctor(run_dir, config_path)
    config = load_sft_config(config_path)
    stage = run_dir / "sft"
    atomic_write_json(stage / "config.json", config)
    rows = read_jsonl(_relative_path(config["dataset"]))
    if len(rows) != 50:
        raise PosttrainError("冻结 SFT 数据必须恰好 50 条")
    temporary, temporary_path = make_temporary_mlx_dataset(rows, stage)
    try:
        snapshot = {
            "source": config["dataset"], "source_sha256": sha256_file(_relative_path(config["dataset"])),
            "temporary_train": str(temporary_path / "train.jsonl"), "records": len(rows),
        }
        atomic_write_json(stage / "input_snapshot.json", snapshot)
        runtime_rows = read_jsonl(temporary_path / "train.jsonl")
        precheck = gradient_precheck(config, runtime_rows)
        atomic_write_json(stage / "gradient_precheck.json", precheck)
        if not precheck["passed"]:
            raise PosttrainError("SFT 梯度预检超过内存门槛")
        technical = train_sft(
            config, runtime_rows, stage / "adapter", stage / "metrics.jsonl"
        )
    finally:
        temporary.cleanup()
    persona = render_persona_prompt(load_persona(_relative_path(config["persona"])))
    dev = read_jsonl(_relative_path(config["dev_dataset"]))
    base_rows = generate_grid(config, dev, persona, stage / "base_dev_outputs.jsonl")
    sft_rows = generate_grid(
        config, dev, persona, stage / "sft_dev_outputs.jsonl", adapter_path=stage / "adapter"
    )
    smoke = adapter_reload_smoke(config, stage / "adapter", persona, dev[0]["user"])
    technical["adapter_reload_smoke"] = smoke
    technical["passed"] = technical["passed"] and smoke["passed"]
    atomic_write_json(stage / "technical_gate.json", technical)
    ids = [record["id"] for record in dev]
    automatic = evaluate_relative_behavior_gate(
        base_rows, sft_rows, tuple(config["generation"]["seeds"]), ids
    )
    atomic_write_json(stage / "automatic_gate.json", automatic)
    packet, key = build_manual_review(base_rows, sft_rows, ids)
    atomic_write_json(stage / "manual_review_packet.json", packet)
    atomic_write_json(stage / "manual_review_answer_key.json", key)
    atomic_write_json(stage / "manual_review_results.json", empty_manual_review_results(packet))
    summary = {
        "stage": "sft", "technical_gate": technical["passed"],
        "relative_behavior_gate": automatic["passed"], "manual_gate": False,
        "ready_for_grpo": False, "status": "awaiting_manual_review" if automatic["passed"] else "behavior_failed",
    }
    atomic_write_json(stage / "summary.json", summary)
    return summary


def gate_sft(run_dir: Path, results_path: Path | None = None) -> dict[str, Any]:
    stage = run_dir / "sft"
    packet = read_json(stage / "manual_review_packet.json")
    key = read_json(stage / "manual_review_answer_key.json")
    results = read_json(results_path or stage / "manual_review_results.json")
    manual = evaluate_manual_review(packet, key, results)
    technical = read_json(stage / "technical_gate.json")
    automatic = read_json(stage / "automatic_gate.json")
    ready = bool(technical["passed"] and automatic["passed"] and manual["passed"])
    atomic_write_json(stage / "manual_gate.json", manual)
    summary = {
        "stage": "sft", "technical_gate": technical["passed"],
        "relative_behavior_gate": automatic["passed"], "manual_gate": manual["passed"],
        "ready_for_grpo": ready, "status": "ready_for_grpo" if ready else "gate_failed",
    }
    atomic_write_json(stage / "summary.json", summary)
    return summary


def _require_sft_ready(run_dir: Path) -> None:
    summary = read_json(run_dir / "sft/summary.json")
    if summary.get("ready_for_grpo") is not True:
        raise PosttrainError("SFT 三层门槛尚未通过，拒绝进入奖励审计或 GRPO")


def _judge_client(config: dict[str, Any]) -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise PosttrainError("缺少 DEEPSEEK_API_KEY")
    return OpenAI(base_url=config["judge_base_url"], api_key=key)


def _collect_reward_group(
    model: Any, tokenizer: Any, config: dict[str, Any], system_prompt: str,
    prompt_record: dict[str, Any], group_index: int, client: OpenAI,
    sources: tuple[list[str], list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    for candidate_index in range(config["generations_per_prompt"]):
        generated = generate_candidate_with_logprobs(
            model, tokenizer,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_record["user"]}],
            config, config["seed"] + group_index * 100 + candidate_index,
        )
        candidates.append({
            "candidate_id": f"{prompt_record['id']}-c{candidate_index + 1}",
            **generated,
        })
        if config.get("clear_cache_each_candidate"):
            mx, _, _ = require_mlx()
            mx.clear_cache()
    judge_input = [{"candidate_id": row["candidate_id"], "assistant": row["assistant"]} for row in candidates]
    messages = build_judge_messages(system_prompt, prompt_record["user"], judge_input)
    ids = [row["candidate_id"] for row in candidates]
    judge, attempts = judge_group_with_retry(
        client, config["judge_model"], messages, ids, max_attempts=config["judge_max_attempts"]
    )
    scores = score_group(
        prompt_record["user"], judge_input, judge,
        persona_sources=sources[0], style_sources=sources[1],
    )
    for score in scores:
        score["judge_attempts"] = attempts
    return candidates, scores


def _trajectory_audit(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the minimal data needed to audit each GRPO policy ratio."""
    fields = (
        "candidate_id", "prompt_tokens", "completion_tokens",
        "old_logprobs", "finish_reason",
    )
    return [{field: candidate[field] for field in fields} for candidate in candidates]


def reward_preview(run_dir: Path, config_path: Path = DEFAULT_GRPO_CONFIG) -> dict[str, Any]:
    _require_sft_ready(run_dir)
    config = load_grpo_config(config_path)
    stage = run_dir / "grpo"
    atomic_write_json(stage / "config.json", config)
    prompts = read_jsonl(_relative_path(config["dataset"]))[: config["reward_preview_prompts"]]
    system_prompt = render_persona_prompt(load_persona(_relative_path(config["persona"])))
    sources = load_reward_sources(_relative_path(config["persona"]), _relative_path(config["style_examples"]))
    client = _judge_client(config)
    model, tokenizer = load_base(config, run_dir / "sft/adapter")
    audit = []
    for index, prompt in enumerate(prompts):
        candidates, scores = _collect_reward_group(
            model, tokenizer, config, system_prompt, prompt, index, client, sources
        )
        audit.append({
            "group_id": prompt["id"], "prompt": prompt, "candidates": scores,
            "trajectories": _trajectory_audit(candidates),
        })
    atomic_write_jsonl(stage / "reward_preview.jsonl", audit)
    review = {
        "schema_version": 1, "approved": False, "reviewer": "",
        "reviewed_group_ids": [], "expected_group_ids": [row["group_id"] for row in audit],
        "notes": "人工检查五组候选、Judge 分数、格式分与 penalty 后再将 approved 改为 true。",
    }
    atomic_write_json(stage / "reward_review_results.json", review)
    summary = {"stage": "reward_preview", "groups": len(audit), "awaiting_human_approval": True}
    atomic_write_json(stage / "reward_preview_summary.json", summary)
    return summary


def _require_reward_approval(run_dir: Path) -> None:
    result = read_json(run_dir / "grpo/reward_review_results.json")
    expected = result.get("expected_group_ids")
    reviewed = result.get("reviewed_group_ids")
    if (
        result.get("approved") is not True or not isinstance(result.get("reviewer"), str)
        or not result["reviewer"].strip() or reviewed != expected or len(expected or []) != 5
    ):
        raise PosttrainError("5 组奖励尚未完成显式人工批准，拒绝启动 GRPO")


def _cloud_manifest(config: dict[str, Any], run_dir: Path, errors: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1, "reason": "local_grpo_blocked", "automatic_cloud_spend": False,
        "must_restart_from_base": True, "do_not_convert_mlx_adapter": True,
        "base": {"model": config["model"], "revision": config["model_revision"]},
        "required_stages": ["SFT from frozen 50-row dataset", "SFT gates", "GRPO from cloud SFT", "unified evaluation"],
        "inputs": {relative: expected for relative, expected in FROZEN_HASHES.items()},
        "local_run": str(run_dir), "memory_attempts": errors,
    }


def _is_memory_failure(error: Exception) -> bool:
    message = str(error).lower()
    return any(term in message for term in ("oom", "out of memory", "metal", "memory", "内存"))


def run_grpo_stage(run_dir: Path, config_path: Path = DEFAULT_GRPO_CONFIG) -> dict[str, Any]:
    _require_sft_ready(run_dir)
    _require_reward_approval(run_dir)
    config = load_grpo_config(config_path)
    stage = run_dir / "grpo"
    atomic_write_json(stage / "config.json", config)
    prompts = read_jsonl(_relative_path(config["dataset"]))
    if len(prompts) != config["prompt_count"]:
        raise PosttrainError("冻结 GRPO Prompt 数量不正确")
    system_prompt = render_persona_prompt(load_persona(_relative_path(config["persona"])))
    sources = load_reward_sources(_relative_path(config["persona"]), _relative_path(config["style_examples"]))
    client = _judge_client(config)
    # Generate and score once. No optimizer exists yet, so partial rewards cannot update policy.
    generation_model, tokenizer = load_base(config, run_dir / "sft/adapter")
    preflight_groups = {}
    first_update_data = None
    generation_peak = 0.0
    for preflight_index, prompt in enumerate(prompts):
        candidates, scores = _collect_reward_group(
            generation_model, tokenizer, config, system_prompt, prompt,
            preflight_index, client, sources,
        )
        preflight_groups[preflight_index] = (candidates, scores)
        generation_peak = max(
            generation_peak,
            max(candidate["peak_memory_gb"] for candidate in candidates),
        )
        rewards = [row["reward"] for row in scores]
        if not group_should_skip(rewards):
            first_update_data = (candidates, rewards)
            break
    if first_update_data is None:
        raise PosttrainError("20 个 GRPO 组全部奖励相同，训练无效且未调用 optimizer")
    mx, _, optim = require_mlx()
    del generation_model
    mx.clear_cache()
    attempts = []
    selected = None
    for strategy in MEMORY_STRATEGIES:
        policy = None
        optimizer = None
        try:
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            policy, _ = prepare_trainable_model(config, run_dir / "sft/adapter")
            configure_grpo_memory_strategy(policy, strategy)
            optimizer = optim.Adam(learning_rate=config["learning_rate"])
            trial_config = {**config, **strategy}
            update = apply_grpo_group_update(
                policy, optimizer, *first_update_data, trial_config, update=True
            )
            peak = max(generation_peak, float(mx.get_peak_memory()) / 1e9)
            attempts.append({"strategy": strategy, "peak_memory_gb": peak, "result": update})
            if peak <= config["memory_limit_gb"]:
                selected = strategy
                break
        except Exception as exc:
            attempts.append({"strategy": strategy, "error": str(exc)})
        finally:
            if policy is not None:
                del policy
            if optimizer is not None:
                del optimizer
            mx.clear_cache()
    atomic_write_json(stage / "memory_preflight.json", {"attempts": attempts, "selected": selected})
    if selected is None:
        manifest = _cloud_manifest(config, run_dir, attempts)
        atomic_write_json(stage / "cloud_fallback_manifest.json", manifest)
        summary = {"stage": "grpo", "status": "local_grpo_blocked", "valid_training": False}
        atomic_write_json(stage / "summary.json", summary)
        return summary
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    policy, tokenizer = prepare_trainable_model(config, run_dir / "sft/adapter")
    configure_grpo_memory_strategy(policy, selected)
    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    active_config = {**config, **selected}
    rewards_artifact = []
    updated_groups = 0
    equal_groups = 0
    for group_index, prompt in enumerate(prompts):
        try:
            if group_index in preflight_groups:
                candidates, scores = preflight_groups[group_index]
            else:
                candidates, scores = _collect_reward_group(
                    policy, tokenizer, active_config, system_prompt, prompt,
                    group_index, client, sources,
                )
            rewards = [row["reward"] for row in scores]
            result = apply_grpo_group_update(
                policy, optimizer, candidates, rewards, active_config, update=True
            )
        except Exception as exc:
            if not _is_memory_failure(exc):
                raise
            failure = [{
                "strategy": selected, "group_index": group_index + 1,
                "error": str(exc),
            }]
            atomic_write_json(
                stage / "cloud_fallback_manifest.json",
                _cloud_manifest(config, run_dir, failure),
            )
            summary = {
                "stage": "grpo", "status": "local_grpo_blocked",
                "valid_training": False, "failed_group": group_index + 1,
            }
            atomic_write_json(stage / "summary.json", summary)
            return summary
        if result["updated"]:
            updated_groups += 1
        else:
            equal_groups += 1
        peak = float(mx.get_peak_memory()) / 1e9
        record = {
            "group_index": group_index + 1, "prompt": prompt, "candidates": scores,
            "trajectories": _trajectory_audit(candidates),
            "update": result, "peak_memory_gb": peak,
        }
        rewards_artifact.append(record)
        atomic_write_jsonl(stage / "rewards.jsonl", rewards_artifact)
        if peak > config["memory_limit_gb"]:
            failure = [{"strategy": selected, "group_index": group_index + 1, "peak_memory_gb": peak}]
            atomic_write_json(
                stage / "cloud_fallback_manifest.json",
                _cloud_manifest(config, run_dir, failure),
            )
            summary = {
                "stage": "grpo", "status": "local_grpo_blocked",
                "valid_training": False, "failed_group": group_index + 1,
            }
            atomic_write_json(stage / "summary.json", summary)
            return summary
    if updated_groups == 0:
        raise PosttrainError("全部 GRPO 组均无奖励方差，训练无效")
    adapter_check = save_adapter(policy, stage / "adapter", config)
    grpo_peak_memory = float(mx.get_peak_memory()) / 1e9
    del policy, optimizer, tokenizer
    mx.clear_cache()
    smoke_config = load_sft_config(DEFAULT_SFT_CONFIG)
    smoke = adapter_reload_smoke(smoke_config, stage / "adapter", system_prompt, prompts[0]["user"])
    summary = {
        "stage": "grpo", "status": "complete", "valid_training": True,
        "groups": len(prompts), "updated_groups": updated_groups, "equal_reward_groups": equal_groups,
        "strategy": selected, "peak_memory_gb": grpo_peak_memory,
        "adapter": adapter_check, "adapter_reload_smoke": smoke,
    }
    atomic_write_json(stage / "summary.json", summary)
    return summary


def _judge_evaluation_outputs(
    outputs: dict[str, list[dict[str, Any]]],
    eval_rows: list[dict[str, Any]],
    system_prompt: str,
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    client = _judge_client(config)
    sources = load_reward_sources(
        _relative_path(config["persona"]),
        _relative_path(config["style_examples"]),
    )
    indexed = {
        model_name: {
            (row["seed"], row["id"]): row for row in model_outputs
        }
        for model_name, model_outputs in outputs.items()
    }
    audit = []
    totals = {
        model_name: {
            "role_consistency": 0.0,
            "format_consistency": 0.0,
            "dialogue_quality": 0.0,
            "penalty": 0.0,
            "reward": 0.0,
            "records": 0,
        }
        for model_name in outputs
    }
    seeds = sorted({row["seed"] for row in outputs["base"]})
    for seed in seeds:
        for record in eval_rows:
            candidates = []
            id_to_model = {}
            for model_name in ("base", "sft", "grpo"):
                candidate_id = f"{seed}-{record['id']}-{model_name}"
                id_to_model[candidate_id] = model_name
                candidates.append({
                    "candidate_id": candidate_id,
                    "assistant": indexed[model_name][(seed, record["id"])]["assistant"],
                })
            messages = build_judge_messages(
                system_prompt, record["user"], candidates
            )
            candidate_ids = [item["candidate_id"] for item in candidates]
            judge, attempts = judge_group_with_retry(
                client,
                config["judge_model"],
                messages,
                candidate_ids,
                max_attempts=config["judge_max_attempts"],
            )
            scores = score_group(
                record["user"],
                candidates,
                judge,
                persona_sources=sources[0],
                style_sources=sources[1],
            )
            for score in scores:
                model_name = id_to_model[score["candidate_id"]]
                score["model"] = model_name
                summary = totals[model_name]
                for name in (
                    "role_consistency", "format_consistency",
                    "dialogue_quality", "reward",
                ):
                    summary[name] += score[name]
                summary["penalty"] += score["penalties"]["total"]
                summary["records"] += 1
            audit.append({
                "seed": seed,
                "id": record["id"],
                "user": record["user"],
                "judge_attempts": attempts,
                "candidates": scores,
            })
            atomic_write_jsonl(output_path, audit)
    averages = {}
    for model_name, values in totals.items():
        count = values["records"]
        averages[model_name] = {
            name: value if name == "records" else value / count
            for name, value in values.items()
        }
        averages[model_name]["macro_average"] = sum(
            averages[model_name][name]
            for name in (
                "role_consistency", "format_consistency", "dialogue_quality"
            )
        ) / 3
    return {"groups": len(audit), "models": averages}


def evaluate_stage(run_dir: Path, sft_config_path: Path = DEFAULT_SFT_CONFIG) -> dict[str, Any]:
    grpo = read_json(run_dir / "grpo/summary.json")
    if grpo.get("valid_training") is not True:
        raise PosttrainError("没有有效 GRPO checkpoint，拒绝统一评测")
    config = load_sft_config(sft_config_path)
    stage = run_dir / "evaluation"
    eval_rows = read_jsonl(ROOT / "data/runs/morgana-v1/eval.jsonl")
    persona = render_persona_prompt(load_persona(_relative_path(config["persona"])))
    outputs = {
        "base": generate_grid(config, eval_rows, persona, stage / "base_eval_outputs.jsonl"),
        "sft": generate_grid(config, eval_rows, persona, stage / "sft_eval_outputs.jsonl", adapter_path=run_dir / "sft/adapter"),
        "grpo": generate_grid(config, eval_rows, persona, stage / "grpo_eval_outputs.jsonl", adapter_path=run_dir / "grpo/adapter"),
    }
    ids = [row["id"] for row in eval_rows]
    seeds = tuple(config["generation"]["seeds"])
    sft_relative = evaluate_relative_behavior_gate(outputs["base"], outputs["sft"], seeds, ids)
    grpo_relative = evaluate_relative_behavior_gate(outputs["sft"], outputs["grpo"], seeds, ids)
    judge_summary = _judge_evaluation_outputs(
        outputs,
        eval_rows,
        persona,
        load_grpo_config(DEFAULT_GRPO_CONFIG),
        stage / "judge_scores.jsonl",
    )
    manual_ids = ids[:10]
    manual_id_set = set(manual_ids)
    manual_outputs = {
        name: [row for row in rows if row["id"] in manual_id_set]
        for name, rows in outputs.items()
    }
    packet_sft, key_sft = build_manual_review(
        manual_outputs["base"], manual_outputs["sft"], manual_ids
    )
    packet_grpo, key_grpo = build_manual_review(
        manual_outputs["sft"], manual_outputs["grpo"], manual_ids
    )
    automatic = {
        "base_to_sft": sft_relative,
        "sft_to_grpo": grpo_relative,
        "judge": judge_summary,
    }
    atomic_write_json(stage / "automatic_summary.json", automatic)
    atomic_write_json(stage / "base_sft_manual_packet.json", packet_sft)
    atomic_write_json(stage / "base_sft_manual_answer_key.json", key_sft)
    atomic_write_json(
        stage / "base_sft_manual_results.json",
        empty_manual_review_results(packet_sft),
    )
    atomic_write_json(stage / "sft_grpo_manual_packet.json", packet_grpo)
    atomic_write_json(stage / "sft_grpo_manual_answer_key.json", key_grpo)
    atomic_write_json(
        stage / "sft_grpo_manual_results.json",
        empty_manual_review_results(packet_grpo),
    )
    summary = {
        "stage": "evaluation", "status": "awaiting_manual_review",
        "models": ["base", "sft", "grpo"], "records_per_model": len(outputs["base"]),
        "automatic_summary": automatic,
    }
    atomic_write_json(stage / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="morgana-v1 Mac/MLX 后训练与统一评测")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "sft", "gate-sft", "reward-preview", "grpo", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-id", required=name not in {"doctor", "sft"})
        if name in {"doctor", "sft"}:
            command.add_argument("--sft-config", type=Path, default=DEFAULT_SFT_CONFIG)
        if name in {"reward-preview", "grpo"}:
            command.add_argument("--grpo-config", type=Path, default=DEFAULT_GRPO_CONFIG)
        if name == "gate-sft":
            command.add_argument("--results", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        create = args.command in {"doctor", "sft"}
        run_dir = resolve_run_dir(args.run_id, create=create)
        if args.command == "doctor":
            result = run_doctor(run_dir, args.sft_config)
        elif args.command == "sft":
            result = run_sft_stage(run_dir, args.sft_config)
        elif args.command == "gate-sft":
            result = gate_sft(run_dir, args.results)
        elif args.command == "reward-preview":
            result = reward_preview(run_dir, args.grpo_config)
        elif args.command == "grpo":
            result = run_grpo_stage(run_dir, args.grpo_config)
        else:
            result = evaluate_stage(run_dir)
        print(json.dumps({"run_dir": str(run_dir), **result}, ensure_ascii=False, indent=2))
        if args.command == "doctor" and not result["passed"]:
            raise SystemExit(2)
    except (ConfigurationError, PosttrainError, MLXRuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
