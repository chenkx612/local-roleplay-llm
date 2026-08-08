"""Lazy MLX backend for comparable SFT, GRPO and evaluation runs."""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from .artifacts import append_jsonl, atomic_write_json, atomic_write_jsonl, read_jsonl
from .grpo import accumulate_tree, mlx_policy_loss, standardized_advantages
from .sft_eval import normalize_empty_think_wrapper


class MLXRuntimeError(RuntimeError):
    """Raised when a required MLX numerical or runtime assertion fails."""


def require_finite_positive(value: float, label: str) -> float:
    """Validate a scalar numerical gate without depending on MLX imports."""
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise MLXRuntimeError(f"{label} 必须是有限正数")
    return result


def require_mlx() -> tuple[Any, Any, Any]:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError as exc:
        raise MLXRuntimeError(
            "缺少 Mac 训练依赖；请用 Python 3.13 安装 requirements-mac.txt"
        ) from exc
    return mx, nn, optim


def lora_parameters(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": config["lora_rank"],
        "scale": config["lora_scale"],
        "dropout": config["lora_dropout"],
    }


def load_base(config: dict[str, Any], adapter_path: Path | None = None) -> tuple[Any, Any]:
    require_mlx()
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise MLXRuntimeError("缺少 mlx-lm==0.31.3") from exc
    return load(
        config["model"],
        revision=config["model_revision"],
        adapter_path=str(adapter_path) if adapter_path else None,
    )


def prepare_trainable_model(config: dict[str, Any], adapter_path: Path | None = None) -> tuple[Any, Any]:
    model, tokenizer = load_base(config)
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], lora_parameters(config))
    if adapter_path is not None:
        model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)
    return model, tokenizer


def _flatten(tree: Any) -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    return list(tree_flatten(tree))


def _tree_finite_norm(tree: Any) -> float:
    mx, _, _ = require_mlx()
    leaves = _flatten(tree)
    if not leaves:
        raise MLXRuntimeError("没有可训练参数或梯度")
    squared = mx.array(0.0)
    for _, value in leaves:
        if not bool(mx.all(mx.isfinite(value)).item()):
            raise MLXRuntimeError("检测到非有限梯度")
        squared = squared + mx.sum(value.astype(mx.float32) ** 2)
    norm = math.sqrt(float(squared.item()))
    return require_finite_positive(norm, "梯度范数")


def _weight_delta(before: dict[str, Any], after: Any) -> float:
    mx, _, _ = require_mlx()
    total = mx.array(0.0)
    for name, value in _flatten(after):
        total = total + mx.sum((value.astype(mx.float32) - before[name].astype(mx.float32)) ** 2)
    delta = math.sqrt(float(total.item()))
    return require_finite_positive(delta, "optimizer update 权重增量")


def build_chat_dataset(rows: list[dict[str, Any]], tokenizer: Any, mask_prompt: bool = True) -> Any:
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset

    return CacheDataset(ChatDataset(rows, tokenizer, mask_prompt=mask_prompt))


def token_lengths(rows: list[dict[str, Any]], tokenizer: Any, mask_prompt: bool = True) -> list[int]:
    dataset = build_chat_dataset(rows, tokenizer, mask_prompt)
    return [len(dataset[index][0]) for index in range(len(dataset))]


def _first_batches(dataset: Any, config: dict[str, Any], count: int) -> list[Any]:
    from mlx_lm.tuner.trainer import iterate_batches

    iterator = iterate_batches(
        dataset, batch_size=1, max_seq_length=config.get("max_seq_length", 1024),
        loop=True, seed=config["seed"],
    )
    return [next(iterator) for _ in range(count)]


def gradient_precheck(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reload Base and make one throwaway accumulated optimizer update."""
    mx, nn, optim = require_mlx()
    from mlx_lm.tuner.trainer import default_loss
    from mlx.utils import tree_map

    mx.random.seed(config["seed"])
    model, tokenizer = prepare_trainable_model(config)
    model.train()
    dataset = build_chat_dataset(rows, tokenizer, mask_prompt=True)
    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    value_and_grad = nn.value_and_grad(model, default_loss)
    accumulated = None
    losses = []
    for batch in _first_batches(dataset, config, config["grad_accumulation_steps"]):
        (loss, _), gradient = value_and_grad(model, *batch)
        mx.eval(loss, gradient)
        loss_value = float(loss.item())
        require_finite_positive(loss_value, "梯度预检 loss")
        losses.append(loss_value)
        accumulated = accumulate_tree(accumulated, gradient, tree_map)
    accumulated = tree_map(
        lambda value: value / config["grad_accumulation_steps"], accumulated
    )
    grad_norm = _tree_finite_norm(accumulated)
    before = {name: value for name, value in _flatten(model.trainable_parameters())}
    optimizer.update(model, accumulated)
    mx.eval(model.trainable_parameters(), optimizer.state)
    delta = _weight_delta(before, model.trainable_parameters())
    peak = float(mx.get_peak_memory()) / 1e9
    result = {
        "passed": peak <= config["memory_limit_gb"],
        "loss": sum(losses) / len(losses),
        "gradient_norm": grad_norm,
        "weight_delta_norm": delta,
        "microbatches": len(losses),
        "peak_memory_gb": peak,
        "memory_limit_gb": config["memory_limit_gb"],
    }
    del model, optimizer, dataset
    mx.clear_cache()
    return result


def _save_adapter(model: Any, adapter_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    mx, _, _ = require_mlx()
    # Save every LoRA tensor, including frozen SFT layers when the 8-layer GRPO
    # fallback is active. Saving trainable_parameters() there would silently
    # discard the untouched first 16 layers of the accepted SFT adapter.
    weights = {
        name: value for name, value in _flatten(model.parameters())
        if "lora_a" in name.lower() or "lora_b" in name.lower()
    }
    if not weights:
        raise MLXRuntimeError("adapter 没有可保存张量")
    b_names = [name for name in weights if "lora_b" in name.lower()]
    if not b_names:
        raise MLXRuntimeError("adapter 中没有 LoRA-B 张量")
    for name, value in weights.items():
        if not bool(mx.all(mx.isfinite(value)).item()):
            raise MLXRuntimeError(f"adapter 张量非有限: {name}")
    zero_b = [name for name in b_names if not bool(mx.any(weights[name] != 0).item())]
    if zero_b:
        raise MLXRuntimeError(f"LoRA-B 未全部更新: {zero_b[:5]}")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    temporary = adapter_dir / ".adapters.safetensors.tmp"
    final = adapter_dir / "adapters.safetensors"
    mx.save_safetensors(str(temporary), weights)
    temporary.replace(final)
    atomic_write_json(adapter_dir / "adapter_config.json", {
        "fine_tune_type": "lora",
        "num_layers": config["num_layers"],
        "lora_parameters": lora_parameters(config),
        "model": config["model"],
        "model_revision": config["model_revision"],
    })
    return {"tensor_count": len(weights), "lora_b_count": len(b_names), "zero_lora_b": []}


def save_adapter(model: Any, adapter_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate and atomically save a final adapter checkpoint."""
    return _save_adapter(model, adapter_dir, config)


def configure_grpo_memory_strategy(model: Any, strategy: dict[str, Any]) -> None:
    """Apply one monotonic memory mitigation to an already loaded policy."""
    if strategy.get("grad_checkpoint"):
        from mlx_lm.tuner.trainer import grad_checkpoint

        layer_type = type(model.layers[0])
        if not getattr(layer_type, "_roleplay_grad_checkpointed", False):
            grad_checkpoint(model.layers[0])
            layer_type._roleplay_grad_checkpointed = True
    train_layers = strategy.get("train_layers")
    if train_layers is not None:
        model.freeze()
        for layer in model.layers[-train_layers:]:
            for _, module in layer.named_modules():
                if type(module).__name__.startswith("LoRA"):
                    module.unfreeze(
                        recurse=False, keys=["lora_a", "lora_b"], strict=True
                    )


def train_sft(
    config: dict[str, Any], rows: list[dict[str, Any]], adapter_dir: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    """Run exactly 150 official-MLX QLoRA microbatches with strict checks."""
    mx, nn, optim = require_mlx()
    from mlx_lm.tuner.trainer import default_loss, iterate_batches
    from mlx.utils import tree_map

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    mx.random.seed(config["seed"])
    model, tokenizer = prepare_trainable_model(config)
    model.train()
    dataset = build_chat_dataset(rows, tokenizer, mask_prompt=True)
    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    value_and_grad = nn.value_and_grad(model, default_loss)
    iterator = iterate_batches(
        dataset, batch_size=1, max_seq_length=config["max_seq_length"],
        loop=True, seed=config["seed"],
    )
    accumulated = None
    report_losses = []
    report_tokens = 0
    report_started = time.perf_counter()
    optimizer_steps = 0
    grad_norms = []
    for microbatch in range(1, config["microbatches"] + 1):
        batch = next(iterator)
        (loss, tokens), gradient = value_and_grad(model, *batch)
        mx.eval(loss, tokens, gradient)
        loss_value = float(loss.item())
        require_finite_positive(loss_value, f"microbatch {microbatch} loss")
        report_losses.append(loss_value)
        report_tokens += int(tokens.item())
        accumulated = accumulate_tree(accumulated, gradient, tree_map)
        if microbatch % config["grad_accumulation_steps"] == 0:
            averaged = tree_map(
                lambda value: value / config["grad_accumulation_steps"], accumulated
            )
            grad_norm = _tree_finite_norm(averaged)
            optimizer.update(model, averaged)
            mx.eval(model.trainable_parameters(), optimizer.state)
            grad_norms.append(grad_norm)
            optimizer_steps += 1
            accumulated = None
        if microbatch % config["report_every"] == 0:
            elapsed = time.perf_counter() - report_started
            peak = float(mx.get_peak_memory()) / 1e9
            metric = {
                "microbatch": microbatch,
                "optimizer_step": optimizer_steps,
                "loss": sum(report_losses) / len(report_losses),
                "gradient_norm": grad_norms[-1],
                "tokens_per_second": report_tokens / elapsed,
                "peak_memory_gb": peak,
            }
            append_jsonl(metrics_path, metric)
            if peak > config["memory_limit_gb"]:
                raise MLXRuntimeError(
                    f"SFT 峰值 MLX 内存 {peak:.3f}GB 超过 {config['memory_limit_gb']}GB"
                )
            report_losses = []
            report_tokens = 0
            report_started = time.perf_counter()
    if optimizer_steps != config["optimizer_steps"]:
        raise MLXRuntimeError("正式 SFT optimizer step 数不符合配置")
    adapter_check = _save_adapter(model, adapter_dir, config)
    peak = float(mx.get_peak_memory()) / 1e9
    result = {
        "passed": True,
        "microbatches": config["microbatches"],
        "optimizer_steps": optimizer_steps,
        "all_gradient_norms_finite_positive": all(
            math.isfinite(value) and value > 0 for value in grad_norms
        ),
        "gradient_norms": grad_norms,
        "peak_memory_gb": peak,
        "adapter": adapter_check,
    }
    del model, optimizer, dataset
    mx.clear_cache()
    return result


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )


def generate_one(
    model: Any, tokenizer: Any, messages: list[dict[str, str]],
    generation: dict[str, Any], seed: int,
) -> dict[str, Any]:
    mx, _, _ = require_mlx()
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    mx.random.seed(seed)
    model.eval()
    prompt = render_prompt(tokenizer, messages)
    sampler = make_sampler(
        temp=generation["temperature"], top_p=generation["top_p"],
        top_k=generation["top_k"],
    )
    processors = make_logits_processors(
        repetition_penalty=generation.get("repetition_penalty"),
        repetition_context_size=generation.get("repetition_context_size"),
        presence_penalty=generation.get("presence_penalty"),
        presence_context_size=generation.get("presence_context_size"),
    )
    text = ""
    final = None
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=generation["max_tokens"],
        sampler=sampler, logits_processors=processors,
    ):
        text += response.text
        final = response
    if final is None:
        raise MLXRuntimeError("MLX 推理没有返回 generation response")
    normalized = normalize_empty_think_wrapper(text)
    if not normalized:
        raise MLXRuntimeError("MLX 推理返回空回答")
    return {
        "assistant": normalized,
        "raw_assistant": text,
        "finish_reason": final.finish_reason,
        "prompt_tokens": final.prompt_tokens,
        "completion_tokens": final.generation_tokens,
        "peak_memory_gb": final.peak_memory,
    }


def generate_grid(
    config: dict[str, Any], records: list[dict[str, Any]], system_prompt: str,
    output_path: Path, *, adapter_path: Path | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    model, tokenizer = load_base(config, adapter_path)
    generation = config["generation"]
    selected_seeds = list(seeds or generation["seeds"])
    outputs = []
    for seed in selected_seeds:
        for index, record in enumerate(records):
            answer = generate_one(
                model, tokenizer,
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": record["user"]}],
                generation, seed + index,
            )
            outputs.append({**record, "seed": seed, "attempts": 1, **answer})
    atomic_write_jsonl(output_path, outputs)
    del model, tokenizer
    mx, _, _ = require_mlx()
    mx.clear_cache()
    return outputs


def adapter_reload_smoke(
    config: dict[str, Any], adapter_path: Path, system_prompt: str, user: str
) -> dict[str, Any]:
    model, tokenizer = load_base(config, adapter_path)
    result = generate_one(
        model, tokenizer,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        config["generation"], config["seed"],
    )
    del model, tokenizer
    mx, _, _ = require_mlx()
    mx.clear_cache()
    return {"passed": bool(result["assistant"]), **result}


def make_temporary_mlx_dataset(rows: list[dict[str, Any]], parent: Path):
    """Create an ephemeral MLX ``train.jsonl`` without a second frozen dataset."""
    directory = tempfile.TemporaryDirectory(prefix="mlx-data-", dir=parent)
    path = Path(directory.name)
    atomic_write_jsonl(path / "train.jsonl", rows)
    return directory, path


def generate_candidate_with_logprobs(
    model: Any, tokenizer: Any, messages: list[dict[str, str]],
    config: dict[str, Any], seed: int,
) -> dict[str, Any]:
    mx, _, _ = require_mlx()
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    model.eval()
    prompt = render_prompt(tokenizer, messages)
    sampler = make_sampler(
        temp=config["temperature"], top_p=config["top_p"], top_k=config["top_k"]
    )
    token_ids = []
    old_logprobs = []
    text = ""
    final = None
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=config["max_completion_tokens"], sampler=sampler
    ):
        text += response.text
        final = response
        # mlx-lm 0.31.3 puts the final sampled action only on its terminal
        # response (EOS for stop, or the last visible token for length).
        token_ids.append(response.token)
        old_logprobs.append(float(response.logprobs[response.token].item()))
    if final is None or not text.strip() or not token_ids:
        raise MLXRuntimeError("GRPO 候选为空，整组失败")
    return {
        "assistant": normalize_empty_think_wrapper(text),
        "prompt_tokens": prompt.tolist() if hasattr(prompt, "tolist") else list(prompt),
        "completion_tokens": token_ids,
        "old_logprobs": old_logprobs,
        "finish_reason": final.finish_reason,
        "peak_memory_gb": final.peak_memory,
    }


def apply_grpo_group_update(
    model: Any, optimizer: Any, candidates: Sequence[dict[str, Any]],
    rewards: Sequence[float], config: dict[str, Any], *, update: bool = True,
) -> dict[str, Any]:
    mx, nn, _ = require_mlx()

    # LoRA dropout must remain disabled so old/new log-probability ratios are
    # comparable. MLX can still compute gradients while the model is in eval.
    model.eval()
    advantages = standardized_advantages(rewards)
    if all(value == 0 for value in advantages):
        return {"updated": False, "reason": "equal_rewards", "advantages": advantages}
    from mlx.utils import tree_map

    value_and_grad = nn.value_and_grad(model, mlx_policy_loss)
    accumulated = None
    losses = []
    for candidate, advantage in zip(candidates, advantages):
        prompt = candidate["prompt_tokens"]
        completion = candidate["completion_tokens"]
        full = mx.array(prompt + completion)
        old = mx.array(candidate["old_logprobs"], dtype=mx.float32)
        loss, gradient = value_and_grad(
            model, full, len(prompt), old, advantage, config["clip_epsilon"]
        )
        mx.eval(loss, gradient)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise MLXRuntimeError("GRPO policy loss 非有限")
        _tree_finite_norm(gradient)
        accumulated = accumulate_tree(accumulated, gradient, tree_map)
        losses.append(loss_value)
        if config.get("clear_cache_each_candidate"):
            mx.clear_cache()
    averaged = tree_map(lambda value: value / len(candidates), accumulated)
    grad_norm = _tree_finite_norm(averaged)
    before = {name: value for name, value in _flatten(model.trainable_parameters())}
    if update:
        optimizer.update(model, averaged)
        mx.eval(model.trainable_parameters(), optimizer.state)
        delta = _weight_delta(before, model.trainable_parameters())
    else:
        delta = None
    return {
        "updated": update,
        "loss": sum(losses) / len(losses),
        "gradient_norm": grad_norm,
        "weight_delta_norm": delta,
        "advantages": advantages,
    }
