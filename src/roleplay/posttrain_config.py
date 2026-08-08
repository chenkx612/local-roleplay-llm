"""Strict configuration contracts for the Mac/MLX post-training path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL = "mlx-community/Qwen3.5-2B-4bit"
MODEL_REVISION = "674aaa7240b91e8012fcad5d791b7dfe5ba90207"
SFT_REQUIRED = {
    "schema_version", "model", "model_revision", "dataset", "dev_dataset",
    "persona", "num_layers", "lora_rank", "lora_scale", "lora_dropout",
    "max_seq_length", "batch_size", "grad_accumulation_steps", "epochs",
    "microbatches", "optimizer_steps", "learning_rate", "mask_prompt",
    "seed", "report_every", "memory_limit_gb", "generation",
}
GRPO_REQUIRED = {
    "schema_version", "model", "model_revision", "dataset", "persona",
    "style_examples", "prompt_count", "generations_per_prompt",
    "max_completion_tokens", "group_updates", "learning_rate", "temperature",
    "top_p", "top_k", "clip_epsilon", "beta", "num_layers", "lora_rank",
    "lora_scale", "lora_dropout", "seed", "memory_limit_gb", "judge_model",
    "judge_base_url", "judge_max_attempts", "reward_preview_prompts",
}
SFT_FIXED = {
    "dataset": "data/runs/morgana-v1/sft_train.jsonl",
    "dev_dataset": "data/runs/morgana-v1/dev.jsonl",
    "persona": "data/runs/morgana-v1/inputs/persona.json",
    "num_layers": 24,
    "lora_rank": 16,
    "lora_scale": 2.0,
    "lora_dropout": 0.05,
    "max_seq_length": 1024,
    "batch_size": 1,
    "grad_accumulation_steps": 10,
    "epochs": 3,
    "microbatches": 150,
    "optimizer_steps": 15,
    "learning_rate": 5e-5,
    "mask_prompt": True,
    "seed": 20260807,
    "report_every": 10,
    "memory_limit_gb": 12.0,
}
GRPO_FIXED = {
    "dataset": "data/runs/morgana-v1/rl_train.jsonl",
    "persona": "data/runs/morgana-v1/inputs/persona.json",
    "style_examples": "data/runs/morgana-v1/inputs/style_examples.jsonl",
    "prompt_count": 20,
    "generations_per_prompt": 4,
    "max_completion_tokens": 256,
    "group_updates": 20,
    "learning_rate": 1e-6,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 20,
    "clip_epsilon": 0.2,
    "beta": 0.0,
    "num_layers": 24,
    "lora_rank": 16,
    "lora_scale": 2.0,
    "lora_dropout": 0.05,
    "seed": 20260807,
    "memory_limit_gb": 12.0,
    "judge_model": "deepseek-v4-flash",
    "judge_base_url": "https://api.deepseek.com",
    "judge_max_attempts": 3,
    "reward_preview_prompts": 5,
}


class ConfigurationError(ValueError):
    """Raised when a post-training configuration is unsafe or inconsistent."""


def _positive_int(config: dict[str, Any], name: str) -> int:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} 必须是正整数")
    return value


def _positive_number(config: dict[str, Any], name: str) -> float:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{name} 必须是有限正数")
    value = float(value)
    if value == float("inf") or value != value:
        raise ConfigurationError(f"{name} 必须是有限正数")
    return value


def _load(path: Path, expected: set[str]) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"无法读取配置 {path}: {exc}") from exc
    if not isinstance(config, dict) or set(config) != expected:
        actual = set(config) if isinstance(config, dict) else set()
        raise ConfigurationError(
            f"配置字段不完整: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    if config["schema_version"] != 1:
        raise ConfigurationError("仅支持 schema_version=1")
    if config["model"] != MODEL or config["model_revision"] != MODEL_REVISION:
        raise ConfigurationError("模型名称或 revision 不等于冻结的 MLX Base")
    return config


def _require_fixed(config: dict[str, Any], expected: dict[str, Any]) -> None:
    changed = {
        name: {"expected": value, "actual": config.get(name)}
        for name, value in expected.items()
        if config.get(name) != value
    }
    if changed:
        raise ConfigurationError(f"主配置不允许超参数或输入漂移: {changed}")


def load_sft_config(path: Path) -> dict[str, Any]:
    """Load and validate the single supported SFT configuration."""
    config = _load(path, SFT_REQUIRED)
    _require_fixed(config, SFT_FIXED)
    for name in (
        "num_layers", "lora_rank", "max_seq_length", "batch_size",
        "grad_accumulation_steps", "epochs", "microbatches", "optimizer_steps",
        "seed", "report_every",
    ):
        _positive_int(config, name)
    for name in ("lora_scale", "learning_rate", "memory_limit_gb"):
        _positive_number(config, name)
    dropout = config["lora_dropout"]
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ConfigurationError("lora_dropout 必须在 [0, 1) 内")
    if config["microbatches"] != config["epochs"] * 50:
        raise ConfigurationError("SFT microbatches 必须恰好覆盖 50 条数据的 3 epochs")
    expected_steps = config["microbatches"] // config["grad_accumulation_steps"]
    if config["microbatches"] % config["grad_accumulation_steps"]:
        raise ConfigurationError("microbatches 必须可被梯度累积步数整除")
    if config["optimizer_steps"] != expected_steps:
        raise ConfigurationError("optimizer_steps 与 microbatch/累积步数不一致")
    if config["mask_prompt"] is not True:
        raise ConfigurationError("SFT 必须只计算最后一轮 assistant loss")
    generation = config["generation"]
    expected_generation = {
        "max_tokens", "temperature", "top_p", "top_k", "presence_penalty",
        "presence_context_size", "repetition_penalty",
        "repetition_context_size", "enable_thinking", "seeds",
    }
    if not isinstance(generation, dict) or set(generation) != expected_generation:
        raise ConfigurationError("generation 字段不完整")
    if generation["enable_thinking"] is not False:
        raise ConfigurationError("训练评测必须关闭 thinking")
    if generation["seeds"] != [20260807, 20260808, 20260809]:
        raise ConfigurationError("评测 seed 必须使用冻结的三组值")
    expected_generation_values = {
        "max_tokens": 256,
        "temperature": 0.6,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 0.4,
        "presence_context_size": 128,
        "repetition_penalty": 1.45,
        "repetition_context_size": 128,
        "enable_thinking": False,
        "seeds": [20260807, 20260808, 20260809],
    }
    _require_fixed(generation, expected_generation_values)
    return config


def load_grpo_config(path: Path) -> dict[str, Any]:
    """Load and validate the single supported GRPO configuration."""
    config = _load(path, GRPO_REQUIRED)
    _require_fixed(config, GRPO_FIXED)
    for name in (
        "prompt_count", "generations_per_prompt", "max_completion_tokens",
        "group_updates", "top_k", "num_layers", "lora_rank", "seed",
        "judge_max_attempts", "reward_preview_prompts",
    ):
        _positive_int(config, name)
    for name in (
        "learning_rate", "temperature", "top_p", "clip_epsilon", "lora_scale",
        "memory_limit_gb",
    ):
        _positive_number(config, name)
    if config["beta"] != 0:
        raise ConfigurationError("本机最小 GRPO 必须使用 beta=0")
    if config["prompt_count"] != config["group_updates"]:
        raise ConfigurationError("每个冻结 GRPO Prompt 必须恰好更新一组")
    if config["generations_per_prompt"] != 4:
        raise ConfigurationError("每组必须顺序生成 4 个候选")
    if not 0 < config["top_p"] <= 1:
        raise ConfigurationError("top_p 必须在 (0, 1] 内")
    if not 0 <= config["lora_dropout"] < 1:
        raise ConfigurationError("lora_dropout 必须在 [0, 1) 内")
    return config


def assistant_loss_bounds(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[int, int]:
    """Return the final assistant token interval used by mlx-lm mask_prompt."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ConfigurationError("SFT messages 最后一轮必须是 assistant")
    all_tokens = tokenizer.apply_chat_template(messages, return_dict=False)
    prompt_tokens = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, return_dict=False
    )
    if len(prompt_tokens) >= len(all_tokens):
        raise ConfigurationError("assistant-only mask 没有留下 completion token")
    return len(prompt_tokens), len(all_tokens)
