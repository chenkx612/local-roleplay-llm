"""Minimal GRPO math and the MLX execution primitives used by posttrain."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def standardized_advantages(rewards: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    """Standardize rewards within exactly one candidate group."""
    if not rewards:
        raise ValueError("rewards 不能为空")
    values = [float(value) for value in rewards]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("rewards 必须全部有限")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= epsilon:
        return [0.0] * len(values)
    std = math.sqrt(variance)
    return [(value - mean) / std for value in values]


def clipped_surrogate_terms(
    new_logprobs: Sequence[float],
    old_logprobs: Sequence[float],
    advantage: float,
    clip_epsilon: float,
) -> list[float]:
    """Return the PPO/GRPO surrogate term for each completion token."""
    if len(new_logprobs) != len(old_logprobs) or not new_logprobs:
        raise ValueError("新旧 log-prob 必须非空且逐 token 对齐")
    terms = []
    for new, old in zip(new_logprobs, old_logprobs):
        ratio = math.exp(float(new) - float(old))
        clipped = min(1 + clip_epsilon, max(1 - clip_epsilon, ratio))
        terms.append(min(ratio * advantage, clipped * advantage))
    return terms


def clipped_policy_loss(
    new_logprobs: Sequence[float],
    old_logprobs: Sequence[float],
    advantage: float,
    clip_epsilon: float,
) -> float:
    terms = clipped_surrogate_terms(
        new_logprobs, old_logprobs, advantage, clip_epsilon
    )
    return -sum(terms) / len(terms)


def completion_target_slice(prompt_tokens: int, completion_tokens: int) -> slice:
    """Map completion tokens to next-token logits of prompt+completion input."""
    if prompt_tokens <= 0 or completion_tokens <= 0:
        raise ValueError("prompt 和 completion token 数必须为正")
    return slice(prompt_tokens - 1, prompt_tokens + completion_tokens - 1)


def group_should_skip(rewards: Sequence[float], epsilon: float = 1e-8) -> bool:
    advantages = standardized_advantages(rewards, epsilon=epsilon)
    return all(value == 0 for value in advantages)


def mlx_policy_loss(
    model: Any,
    full_tokens: Any,
    prompt_length: int,
    old_logprobs: Any,
    advantage: float,
    clip_epsilon: float,
) -> Any:
    """MLX completion-only clipped policy loss; imports MLX lazily."""
    import mlx.core as mx

    inputs = full_tokens[None, :-1]
    targets = full_tokens[1:]
    logits = model(inputs)[0]
    completion_count = int(old_logprobs.shape[0])
    target_slice = completion_target_slice(prompt_length, completion_count)
    completion_logits = logits[target_slice]
    completion_targets = targets[target_slice]
    log_probs = completion_logits - mx.logsumexp(completion_logits, axis=-1, keepdims=True)
    indices = mx.arange(completion_count)
    selected = log_probs[indices, completion_targets]
    ratio = mx.exp(selected - old_logprobs)
    clipped = mx.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = mx.minimum(ratio * advantage, clipped * advantage)
    return -mx.mean(surrogate)


def accumulate_tree(accumulator: Any, gradient: Any, tree_map: Any) -> Any:
    """Accumulate per-candidate gradient trees without performing an update."""
    if accumulator is None:
        return gradient
    return tree_map(lambda left, right: left + right, accumulator, gradient)
