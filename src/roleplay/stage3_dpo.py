"""Run, publish, download, and review morgana-v2 Stage 3 DPO."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from roleplay.sft_eval import (
    EVALUATION_SEEDS,
    build_manual_review,
    empty_manual_review_results,
    evaluate_core_behavior_gate,
    evaluate_manual_review,
    normalize_empty_think_wrapper,
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
    read_metric,
    run_logged,
    sha256_file,
    validate_effective_training_args,
    validate_pinned_packages,
    write_json_atomic,
    write_jsonl_exclusive,
)


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "965dcc54bc9c0591873df0e9869c056a54d323d1"
CONFIG_RELATIVE_PATH = Path("configs/morgana_v2_dpo.yaml")
TRAIN_RELATIVE_PATH = Path("data/runs/morgana-v2/dpo_train_run3.jsonl")
DEV_RELATIVE_PATH = Path("data/runs/morgana-v2/dev.jsonl")
SYSTEM_PROMPT_RELATIVE_PATH = Path("data/runs/morgana-v2/system_prompt.txt")
SFT_ADAPTER_RELATIVE_PATH = Path(
    "output/morgana-v2/stage2-sft/final/adapter"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path("output/morgana-v2/stage3-dpo")

TRAIN_SHA256 = "55d6b0efd31dce48fca2009323dd2b7a9f841e74b206187b081a9c6e8e3c47c3"
TRAIN_PAIR_COUNT = 31
DEV_SHA256 = "74cf6d05921155cec5c070ca8a611c7a8e6751b00ca0b77a6f4e9085aeeecb22"
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

DPO_SAMPLING_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.45,
}

EXPECTED_CONFIG = {
    "rlhf_type": "dpo",
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "use_hf": True,
    "split_dataset_ratio": 0.0,
    "tuner_type": "lora",
    "adapters": [str(SFT_ADAPTER_RELATIVE_PATH)],
    "ref_adapters": [str(SFT_ADAPTER_RELATIVE_PATH)],
    "target_modules": ["all-linear"],
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "torch_dtype": "float32",
    "fp16": False,
    "bf16": False,
    "quant_method": "bnb",
    "quant_bits": 4,
    "bnb_4bit_compute_dtype": "float32",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "lora_dtype": "float32",
    "max_length": 1024,
    "loss_scale": "last_round",
    "beta": 0.1,
    "loss_type": "sigmoid",
    "rpo_alpha": 1.0,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1.0e-6,
    "gradient_checkpointing": True,
    "enable_thinking": False,
    "add_non_thinking_prefix": True,
    "packing": False,
    "padding_free": False,
}

EXPECTED_ARCHIVE_FILES = frozenset(
    {
        "run_summary.json",
        "training_config.yaml",
        "train.jsonl",
        "train.log",
        "sft_dev_outputs.jsonl",
        "dpo_dev_outputs.jsonl",
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
        *(f"adapter/{name}" for name in ADAPTER_FILES),
    }
)


class Stage3DPOError(RuntimeError):
    """Raised when the frozen Stage 3 DPO contract is violated."""


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[2]


def validate_frozen_file(path: Path, expected_sha256: str) -> str:
    """Require one frozen file and return its verified digest."""
    if not path.is_file():
        raise Stage3DPOError(f"缺少冻结文件: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise Stage3DPOError(f"冻结文件哈希不匹配: {path}")
    return digest


def validate_sft_adapter(adapter_dir: Path) -> dict[str, str]:
    """Require the exact accepted Stage 2 adapter."""
    return {
        name: validate_frozen_file(adapter_dir / name, expected)
        for name, expected in SFT_ADAPTER_HASHES.items()
    }


def validate_training_rows(path: Path) -> list[dict[str, Any]]:
    """Require the frozen high-quality dataset from the current DPO run."""
    validate_frozen_file(path, TRAIN_SHA256)
    rows = read_jsonl(path)
    if len(rows) != TRAIN_PAIR_COUNT:
        raise Stage3DPOError(
            f"DPO 偏好对必须为 {TRAIN_PAIR_COUNT}，实际 {len(rows)}"
        )
    for index, row in enumerate(rows, 1):
        if set(row) != {"messages", "rejected_response"}:
            raise Stage3DPOError(f"DPO 第 {index} 条字段不正确")
        messages = row["messages"]
        rejected = row["rejected_response"]
        if not isinstance(messages, list) or len(messages) != 3:
            raise Stage3DPOError(f"DPO 第 {index} 条 messages 不正确")
        if [message.get("role") for message in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise Stage3DPOError(f"DPO 第 {index} 条角色顺序不正确")
        if any(
            set(message) != {"role", "content"}
            or not isinstance(message["content"], str)
            or not message["content"].strip()
            for message in messages
        ):
            raise Stage3DPOError(f"DPO 第 {index} 条消息内容不正确")
        chosen = messages[-1]["content"]
        if (
            not isinstance(rejected, str)
            or not rejected.strip()
            or rejected == chosen
        ):
            raise Stage3DPOError(f"DPO 第 {index} 条偏好回答不正确")
    return rows


def validate_training_config(config: dict[str, Any]) -> int:
    """Require the single frozen DPO configuration and return its steps."""
    mismatches = [
        f"{name}={config.get(name)!r} (预期 {expected!r})"
        for name, expected in EXPECTED_CONFIG.items()
        if config.get(name) != expected
    ]
    if config.get("dataset") != "__DPO_DATA__":
        mismatches.append("dataset 占位符不正确")
    if config.get("output_dir") != "__DPO_OUTPUT__":
        mismatches.append("output_dir 占位符不正确")
    if mismatches:
        raise Stage3DPOError("DPO 冻结配置不正确: " + ", ".join(mismatches))
    planned_steps = (
        math.ceil(
            math.ceil(TRAIN_PAIR_COUNT / config["per_device_train_batch_size"])
            / config["gradient_accumulation_steps"]
        )
        * config["num_train_epochs"]
    )
    if planned_steps != 8:
        raise Stage3DPOError(
            f"DPO 预期 optimizer steps 不再是 8: {planned_steps}"
        )
    return planned_steps


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise Stage3DPOError("缺少 PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage3DPOError(f"YAML 不是对象: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage3DPOError(f"{label} 不是有效 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Stage3DPOError(f"{label} 必须是 JSON 对象: {path}")
    return value


def _records_from_responses(
    responses: Any, seed: int, dev_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(responses) != len(dev_rows):
        raise Stage3DPOError("Dev 推理输出数量不正确")
    records = []
    for source, response in zip(dev_rows, responses, strict=True):
        choice = response.choices[0]
        raw_assistant = choice.message.content
        if not isinstance(raw_assistant, str) or not raw_assistant.strip():
            raise Stage3DPOError(f"Dev 推理输出为空: {source['id']}")
        assistant = normalize_empty_think_wrapper(raw_assistant)
        if not assistant:
            raise Stage3DPOError(f"规范化后输出为空: {source['id']}")
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


def _dpo_answer_key(answer_key: dict[str, Any]) -> dict[str, Any]:
    return {
        **answer_key,
        "answers": [
            {
                "review_id": row["review_id"],
                "id": row["id"],
                "sft_label": row["base_label"],
                "dpo_label": row["sft_label"],
            }
            for row in answer_key["answers"]
        ],
    }


def _evaluate_dpo_manual_review(
    packet: dict[str, Any],
    answer_key: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    generic_key = {
        **answer_key,
        "answers": [
            {
                "review_id": row["review_id"],
                "id": row["id"],
                "base_label": row["sft_label"],
                "sft_label": row["dpo_label"],
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
            key.replace("sft_", "dpo_", 1): value
            for key, value in result["checks"].items()
        },
        "dpo_wins": result["sft_wins"],
        "dpo_clear_losses": result["sft_clear_losses"],
        "dpo_severe_issue_ids": result["sft_severe_issue_ids"],
        "mean_scores": {
            "sft": result["mean_scores"]["base"],
            "dpo": result["mean_scores"]["sft"],
        },
    }


def generate_dev_review_artifacts(
    repo_dir: Path,
    sft_adapter: Path,
    dpo_adapter: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate aligned SFT/DPO outputs and an anonymous review packet."""
    try:
        import torch
        from peft import PeftModel
        from swift import (
            InferRequest,
            RequestConfig,
            TransformersEngine,
            get_model_processor,
            get_template,
        )
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise Stage3DPOError(f"缺少 Dev 评估依赖: {exc.name}") from exc

    dev_path = repo_dir / DEV_RELATIVE_PATH
    validate_frozen_file(dev_path, DEV_SHA256)
    dev_rows = read_jsonl(dev_path)
    if len(dev_rows) != 10:
        raise Stage3DPOError(f"Dev 数量必须为 10，实际 {len(dev_rows)}")
    system_prompt = (repo_dir / SYSTEM_PROMPT_RELATIVE_PATH).read_text(
        encoding="utf-8"
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
    request_config = RequestConfig(max_tokens=512, **DPO_SAMPLING_CONFIG)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    def infer(adapter: Path) -> list[dict[str, Any]]:
        model, processor = get_model_processor(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.float32,
            quantization_config=quantization,
            use_hf=True,
        )
        model = PeftModel.from_pretrained(model, adapter)
        engine = TransformersEngine(
            model, template=get_template(processor, enable_thinking=False)
        )
        records: list[dict[str, Any]] = []
        for seed in EVALUATION_SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            records.extend(
                _records_from_responses(
                    engine.infer(requests, request_config=request_config),
                    seed,
                    dev_rows,
                )
            )
        del engine, model, processor
        gc.collect()
        torch.cuda.empty_cache()
        return records

    sft_rows = infer(sft_adapter)
    dpo_rows = infer(dpo_adapter)
    expected_ids = [row["id"] for row in dev_rows]
    automatic_gate = evaluate_core_behavior_gate(
        sft_rows, dpo_rows, EVALUATION_SEEDS, expected_ids
    )
    packet, generic_key = build_manual_review(
        sft_rows, dpo_rows, expected_ids
    )
    answer_key = _dpo_answer_key(generic_key)
    paths = {
        "sft_dev_outputs.jsonl": output_dir / "sft_dev_outputs.jsonl",
        "dpo_dev_outputs.jsonl": output_dir / "dpo_dev_outputs.jsonl",
        "manual_review_packet.json": output_dir / "manual_review_packet.json",
        "manual_review_answer_key.json": output_dir
        / "manual_review_answer_key.json",
        "manual_review_results.json": output_dir / "manual_review_results.json",
    }
    write_jsonl_exclusive(paths["sft_dev_outputs.jsonl"], sft_rows)
    write_jsonl_exclusive(paths["dpo_dev_outputs.jsonl"], dpo_rows)
    write_json_atomic(paths["manual_review_packet.json"], packet)
    write_json_atomic(paths["manual_review_answer_key.json"], answer_key)
    write_json_atomic(
        paths["manual_review_results.json"], empty_manual_review_results(packet)
    )
    return {"paths": paths, "automatic_gate": automatic_gate}


def inspect_adapter_change(source_dir: Path, trained_dir: Path) -> dict[str, Any]:
    """Require finite trained LoRA tensors with at least one changed value."""
    try:
        import numpy as np
        from safetensors import safe_open
    except ImportError as exc:
        raise Stage3DPOError("缺少 numpy 或 safetensors") from exc

    source_path = source_dir / "adapter_model.safetensors"
    trained_path = trained_dir / "adapter_model.safetensors"
    changed_tensors = 0
    changed_elements = 0
    max_abs_delta = 0.0
    with safe_open(source_path, framework="np") as source, safe_open(
        trained_path, framework="np"
    ) as trained:
        source_keys = set(source.keys())
        if source_keys != set(trained.keys()):
            raise Stage3DPOError("训练后 adapter tensor 集合发生变化")
        for key in sorted(source_keys):
            before = source.get_tensor(key)
            after = trained.get_tensor(key)
            if before.shape != after.shape or not np.isfinite(after).all():
                raise Stage3DPOError(f"训练后 adapter tensor 无效: {key}")
            delta = np.abs(after - before)
            changed = int(np.count_nonzero(delta))
            changed_elements += changed
            changed_tensors += int(changed > 0)
            if delta.size:
                max_abs_delta = max(max_abs_delta, float(delta.max()))
    if not changed_tensors:
        raise Stage3DPOError("训练后 adapter 没有发生变化")
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
    """Require exactly the DPO publication artifact set."""
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != EXPECTED_ARCHIVE_FILES:
        raise Stage3DPOError(
            "DPO 归档目录不匹配: "
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
        raise Stage3DPOError("DPO run 未完成自动复核，不能发布")


def _token_length(processor: Any, messages: list[dict[str, str]]) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        token_ids = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    return (
        len(token_ids[0])
        if token_ids and isinstance(token_ids[0], list)
        else len(token_ids)
    )


def run_stage3_dpo(output_root: Path | None = None) -> Path:
    """Execute the frozen DPO run and return its publication directory."""
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
        raise Stage3DPOError(f"运行目录已存在: {run_id}")
    create_exclusive_directory(work_dir)
    summary_path = work_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage3_dpo",
        "status": "starting",
        "run": {
            "id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json_atomic(summary_path, summary)

    try:
        huggingface = configure_huggingface_environment()
        environment, torch_module = capture_environment()
        packages = validate_pinned_packages(PINNED_PACKAGES)
        summary["run"].update(git_context(repo_dir))
        summary["environment"] = environment
        summary["huggingface"] = huggingface
        summary["packages"] = packages
        del torch_module

        train_path = repo_dir / TRAIN_RELATIVE_PATH
        dev_path = repo_dir / DEV_RELATIVE_PATH
        system_path = repo_dir / SYSTEM_PROMPT_RELATIVE_PATH
        adapter_dir = repo_dir / SFT_ADAPTER_RELATIVE_PATH
        rows = validate_training_rows(train_path)
        inputs = {
            str(TRAIN_RELATIVE_PATH): {
                "sha256": TRAIN_SHA256,
                "records": len(rows),
            },
            str(DEV_RELATIVE_PATH): validate_frozen_file(dev_path, DEV_SHA256),
            str(SYSTEM_PROMPT_RELATIVE_PATH): validate_frozen_file(
                system_path, SYSTEM_PROMPT_SHA256
            ),
            str(SFT_ADAPTER_RELATIVE_PATH): validate_sft_adapter(adapter_dir),
        }

        source_config_path = repo_dir / CONFIG_RELATIVE_PATH
        config = _load_yaml(source_config_path)
        planned_steps = validate_training_config(config)

        try:
            from huggingface_hub import HfApi
            from transformers import AutoProcessor
        except ImportError as exc:
            raise Stage3DPOError(f"缺少输入校验依赖: {exc.name}") from exc
        model_info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
        if model_info.sha != MODEL_REVISION:
            raise Stage3DPOError(f"模型 revision 不匹配: {model_info.sha}")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        token_lengths: list[int] = []
        for index, row in enumerate(rows, 1):
            chosen_messages = row["messages"]
            rejected_messages = [
                *chosen_messages[:-1],
                {"role": "assistant", "content": row["rejected_response"]},
            ]
            for label, messages in (
                ("chosen", chosen_messages),
                ("rejected", rejected_messages),
            ):
                length = _token_length(processor, messages)
                if length > config["max_length"]:
                    raise Stage3DPOError(
                        f"DPO 第 {index} 条 {label} 超过 max_length: {length}"
                    )
                token_lengths.append(length)
        del processor

        train_data_path = work_dir / "train.jsonl"
        shutil.copy2(train_path, train_data_path)
        train_output = work_dir / "swift-output"
        config["dataset"] = str(train_data_path)
        config["output_dir"] = str(train_output)
        effective_config_path = work_dir / "training_config.yaml"
        _write_yaml(effective_config_path, config)

        summary["model"] = {"name": MODEL_ID, "revision": MODEL_REVISION}
        summary["inputs"] = inputs
        summary["training"] = {
            "config_source": str(CONFIG_RELATIVE_PATH),
            "config_sha256": sha256_file(source_config_path),
            "planned_optimizer_steps": planned_steps,
            "max_sequence_tokens": max(token_lengths),
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
            expected_steps=planned_steps,
        )

        effective_precision = validate_effective_training_args(train_output)
        losses = read_metric(train_output, "loss")
        grad_norms = read_metric(train_output, "grad_norm")
        if not losses or not all(
            math.isfinite(value) and value > 0 for value in losses
        ):
            raise Stage3DPOError(f"loss 无效: {losses}")
        if not grad_norms or not all(
            math.isfinite(value) and value > 0 for value in grad_norms
        ):
            raise Stage3DPOError(f"grad_norm 无效: {grad_norms}")
        if len(grad_norms) != planned_steps:
            raise Stage3DPOError(
                f"optimizer step 数不正确: {len(grad_norms)} != {planned_steps}"
            )
        final_adapter = find_final_adapter(train_output)
        adapter_update = inspect_adapter_change(adapter_dir, final_adapter)

        review = generate_dev_review_artifacts(
            repo_dir, adapter_dir, final_adapter, work_dir
        )
        automatic = review["automatic_gate"]
        archive_stage = work_dir / "archive"
        archive_stage.mkdir()
        shutil.copy2(effective_config_path, archive_stage / "training_config.yaml")
        shutil.copy2(train_data_path, archive_stage / "train.jsonl")
        shutil.copy2(train_log, archive_stage / "train.log")
        for name, source in review["paths"].items():
            shutil.copy2(source, archive_stage / name)
        published_adapter = archive_stage / "adapter"
        published_adapter.mkdir()
        for name in ADAPTER_FILES:
            source = final_adapter / name
            if not source.is_file():
                raise Stage3DPOError(f"训练 adapter 缺少 {name}")
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
                "command": command,
                "duration_seconds": round(duration, 3),
                "effective_precision": effective_precision,
                "optimizer_steps": len(grad_norms),
                "losses": losses,
                "grad_norms": grad_norms,
                "adapter_update": adapter_update,
            }
        )
        summary["automatic_review"] = {
            "passed": automatic["passed"],
            "checks": automatic["checks"],
            "sft": automatic["base"],
            "dpo": automatic["sft"],
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
        summary["ready_for_grpo"] = False
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
    except BaseException as exc:
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
    tag = f"morgana-v2-stage3-dpo-{run_id}"
    return output_dir / f"{tag}.tar.gz", output_dir / f"{tag}.manifest.json", tag


def create_release_bundle(run_dir: Path, output_dir: Path) -> tuple[Path, Path, str]:
    """Create or safely reuse one verified DPO release bundle."""
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
        raise Stage3DPOError("run_summary.json 缺少有效 run.id")
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise Stage3DPOError("发布包目录不能位于 run 目录内")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path, manifest_path, tag = _bundle_paths(output_dir, run_id)
    contents = {
        name: _artifact_metadata(run_dir / name, run_dir)
        for name in sorted(EXPECTED_ARCHIVE_FILES)
    }
    if bundle_path.exists() or manifest_path.exists():
        if not bundle_path.is_file() or not manifest_path.is_file():
            raise Stage3DPOError("发布包或 manifest 不完整")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("contents") != contents
            or manifest.get("bundle", {}).get("sha256")
            != sha256_file(bundle_path)
        ):
            raise Stage3DPOError("现有发布包与 run 不一致")
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
    """Bundle and upload one technically valid DPO run."""
    bundle_path, manifest_path, tag = create_release_bundle(run_dir, output_dir)
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
        f"morgana-v2 Stage 3 DPO {manifest['run_id']}",
        "--notes",
        "Stage 3 DPO 训练产物、Dev 复核材料和 adapter。",
    ]
    source_commit = manifest.get("source_commit")
    if isinstance(source_commit, str) and source_commit:
        command.extend(["--target", source_commit])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise Stage3DPOError("未安装 GitHub CLI：gh") from exc
    except subprocess.CalledProcessError as exc:
        raise Stage3DPOError(
            "GitHub Release 上传失败；本地发布包已保留"
        ) from exc
    return bundle_path, manifest_path, tag


def format_download_command(
    tag: str, repository: str = DEFAULT_GITHUB_REPOSITORY
) -> str:
    command = ["roleplay-stage3-dpo", "download", "--tag", tag]
    if repository != DEFAULT_GITHUB_REPOSITORY:
        command.extend(["--repo", repository])
    return shlex.join(command)


def extract_release_bundle(
    bundle_path: Path, manifest_path: Path, output_root: Path
) -> Path:
    """Verify and atomically extract one DPO release bundle."""
    bundle_path = bundle_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle", {}).get("sha256") != sha256_file(bundle_path):
        raise Stage3DPOError("下载包 SHA-256 与 manifest 不匹配")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise Stage3DPOError("manifest 缺少有效 run_id")
    contents = manifest.get("contents")
    if (
        not isinstance(contents, dict)
        or set(contents) != EXPECTED_ARCHIVE_FILES
        or any(not isinstance(metadata, dict) for metadata in contents.values())
    ):
        raise Stage3DPOError("manifest 文件清单不满足 DPO 归档契约")
    expected_names = {f"{run_id}/{name}" for name in contents}
    output_root = output_root.resolve()
    destination = output_root / run_id
    if destination.exists():
        raise Stage3DPOError(
            f"本地 run 目录已存在，拒绝覆盖: {destination}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / f".download-{uuid4().hex}"
    staging_root.mkdir()
    try:
        with tarfile.open(bundle_path, "r:gz") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != expected_names or not all(
                member.isfile() for member in members
            ):
                raise Stage3DPOError("发布包内容与 manifest 不一致")
            for member in members:
                target = (staging_root / member.name).resolve()
                try:
                    target.relative_to(staging_root)
                except ValueError as exc:
                    raise Stage3DPOError("发布包包含不安全路径") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise Stage3DPOError(f"无法读取发布包文件: {member.name}")
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
                raise Stage3DPOError(f"解包文件校验失败: {relative}")
        staged_run.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return destination


def download_release(
    tag: str,
    output_root: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> Path:
    """Download, verify, and extract one DPO GitHub Release."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise Stage3DPOError("release tag 格式无效")
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
            raise Stage3DPOError("未安装 GitHub CLI：gh") from exc
        except subprocess.CalledProcessError as exc:
            raise Stage3DPOError(f"GitHub Release 下载失败: {tag}") from exc
        bundle = download_dir / f"{tag}.tar.gz"
        manifest = download_dir / f"{tag}.manifest.json"
        if not bundle.is_file() or not manifest.is_file():
            raise Stage3DPOError("Release 缺少发布包或 manifest")
        return extract_release_bundle(bundle, manifest, output_root)


def review_run(run_dir: Path) -> dict[str, Any]:
    """Validate and record one completed local DPO manual review."""
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise Stage3DPOError(f"缺少 run_summary.json: {run_dir}")
    summary = _read_json_object(summary_path, "run_summary.json")
    if "manual_reviewed_at_utc" in summary:
        raise Stage3DPOError("该 run 已完成人工复核，拒绝重复提交")
    required = (
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            raise Stage3DPOError(f"缺少人工复核文件: {name}")
    submitted = _read_json_object(
        run_dir / "manual_review_results.json",
        "manual_review_results.json",
    )
    if not submitted.get("results"):
        print("尚未提交人工复核；run_summary 保持不变。")
        return summary
    packet = _read_json_object(
        run_dir / "manual_review_packet.json",
        "manual_review_packet.json",
    )
    answer_key = _read_json_object(
        run_dir / "manual_review_answer_key.json",
        "manual_review_answer_key.json",
    )
    try:
        gate = _evaluate_dpo_manual_review(packet, answer_key, submitted)
    except ValueError as exc:
        raise Stage3DPOError(str(exc)) from exc
    automatic_passed = bool(summary.get("automatic_review", {}).get("passed"))
    technically_valid = bool(summary.get("technically_valid"))
    summary["manual_review"] = {
        **summary.get("manual_review", {}),
        "status": "passed" if gate["passed"] else "failed",
        "gate": gate,
    }
    summary["ready_for_grpo"] = (
        technically_valid and automatic_passed and gate["passed"]
    )
    summary["status"] = (
        "ready_for_grpo" if summary["ready_for_grpo"] else "dpo_failed"
    )
    result_path = run_dir / "manual_review_results.json"
    summary.setdefault("artifacts", {})["manual_review_results.json"] = (
        _artifact_metadata(result_path, run_dir)
    )
    summary["manual_reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary["manual_review"], ensure_ascii=False, indent=2))
    print(f"ready_for_grpo={summary['ready_for_grpo']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 3 DPO command-line parser."""
    parser = argparse.ArgumentParser(
        description="在 AutoDL 上运行 morgana-v2 Stage 3 DPO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行完整 DPO 训练")
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
    """Run the requested AutoDL Stage 3 DPO command."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_dir = run_stage3_dpo(args.output_root)
            print(f"DPO run: {run_dir}")
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
    except (Stage2SFTError, Stage3DPOError, subprocess.CalledProcessError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
