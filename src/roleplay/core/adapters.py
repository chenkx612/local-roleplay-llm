"""Validation helpers shared by PEFT training stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def inspect_adapter_change(
    source_dir: Path,
    trained_dir: Path,
    *,
    error_type: type[Exception],
) -> dict[str, Any]:
    """Require finite trained LoRA tensors with at least one changed value."""
    try:
        import numpy as np
        from safetensors import safe_open
    except ImportError as exc:
        raise error_type("缺少 numpy 或 safetensors") from exc

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
            raise error_type("训练后 adapter tensor 集合发生变化")
        for key in sorted(source_keys):
            before = source.get_tensor(key)
            after = trained.get_tensor(key)
            if before.shape != after.shape or not np.isfinite(after).all():
                raise error_type(f"训练后 adapter tensor 无效: {key}")
            delta = np.abs(after - before)
            changed = int(np.count_nonzero(delta))
            changed_elements += changed
            changed_tensors += int(changed > 0)
            if delta.size:
                max_abs_delta = max(max_abs_delta, float(delta.max()))
    if not changed_tensors:
        raise error_type("训练后 adapter 没有发生变化")
    return {
        "changed_tensors": changed_tensors,
        "changed_elements": changed_elements,
        "max_abs_delta": max_abs_delta,
    }
