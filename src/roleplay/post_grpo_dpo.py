"""Run, publish, download, and review the targeted post-GRPO DPO stage."""

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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from roleplay.post_grpo_dpo_data import _reward_v2
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
from roleplay.stage3_dpo import Stage3DPOError, inspect_adapter_change


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "965dcc54bc9c0591873df0e9869c056a54d323d1"
CONFIG_RELATIVE_PATH = Path("configs/morgana_v2_post_grpo_dpo.yaml")
ORIGINAL_TRAIN_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_train.jsonl"
)
ORIGINAL_TRAIN_AUDIT_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_train_audit.json"
)
EXPANSION_TRAIN_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_train_expansion.jsonl"
)
EXPANSION_TRAIN_AUDIT_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_train_expansion_audit.json"
)
HOLDOUT_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/post_grpo_dpo_holdout.jsonl"
)
SYSTEM_PROMPT_RELATIVE_PATH = Path(
    "data/runs/morgana-v2/system_prompt.txt"
)
GRPO_ADAPTER_RELATIVE_PATH = Path(
    "output/morgana-v2/stage4-grpo/20260812-2144/adapter"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "output/morgana-v2/post-grpo-dpo/train"
)

ORIGINAL_TRAIN_SHA256 = (
    "0dc429a33d67fd79d2e22012927af538037da19f5e4bd748c929f792f4490abc"
)
ORIGINAL_TRAIN_AUDIT_SHA256 = (
    "a7395ea3ac461843c5e8b1946b1e4cb83d4da46b3a660fe283fe54e1e1de0b2e"
)
EXPANSION_TRAIN_SHA256 = (
    "b305148a7ba69b96fcab2cf480611f754779abcdd4ace0525b311ce36732023b"
)
EXPANSION_TRAIN_AUDIT_SHA256 = (
    "33c7ca2ba96897c5163dfa626eead680d221fe0e8a7c04419d5fa1249fe9c192"
)
HOLDOUT_SHA256 = "59e043347e1b1436f75ed357e5edd10adef1ad2e749003dea5e195b3ec2bedde"
SYSTEM_PROMPT_SHA256 = (
    "d88993aaa1178ced740f6b54530a27e5fcdb2486a66d8b460367e842b53ee112"
)
GRPO_ADAPTER_HASHES = {
    "adapter_model.safetensors": (
        "89ca4fa213ea16eeee002b088ea46b3012b6311b02f09b2b27b0937ab6dcd30f"
    ),
    "adapter_config.json": (
        "67f3ab10168164cc014c7f9c8720984760b1d94be33fde9abf89fb3004a886dc"
    ),
    "additional_config.json": (
        "2b7ed6cc0ca6c21dc39bf80fd3351f5f87c462df23a237dd6ad20473eb9a33a2"
    ),
}
ORIGINAL_TRAIN_PAIR_COUNT = 20
EXPANSION_TRAIN_PAIR_COUNT = 41
TRAIN_PAIR_COUNT = ORIGINAL_TRAIN_PAIR_COUNT + EXPANSION_TRAIN_PAIR_COUNT
HOLDOUT_COUNT = 9
TARGET_ISSUES = (
    "fabricated_background",
    "perspective_shift",
    "emotion_response",
)
SAMPLING_CONFIG = {
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
    "adapters": [str(GRPO_ADAPTER_RELATIVE_PATH)],
    "ref_adapters": [str(GRPO_ADAPTER_RELATIVE_PATH)],
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
    "rpo_alpha": 0.3,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2.0e-7,
    "gradient_checkpointing": True,
    "enable_thinking": False,
    "add_non_thinking_prefix": True,
    "packing": False,
    "padding_free": False,
    "seed": 20260812,
    "data_seed": 20260812,
}
EXPECTED_ARCHIVE_FILES = frozenset(
    {
        "run_summary.json",
        "training_config.yaml",
        "train.jsonl",
        "train.log",
        "grpo_holdout_outputs.jsonl",
        "dpo_holdout_outputs.jsonl",
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
        *(f"adapter/{name}" for name in ADAPTER_FILES),
    }
)


class PostGRPODPOError(RuntimeError):
    """Raised when the frozen post-GRPO DPO contract is violated."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_frozen_file(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise PostGRPODPOError(f"缺少冻结文件: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise PostGRPODPOError(f"冻结文件哈希不匹配: {path}")
    return digest


def validate_grpo_adapter(adapter_dir: Path) -> dict[str, str]:
    return {
        name: validate_frozen_file(adapter_dir / name, expected)
        for name, expected in GRPO_ADAPTER_HASHES.items()
    }


def validate_training_rows(
    path: Path, expected_sha256: str, expected_count: int
) -> list[dict[str, Any]]:
    validate_frozen_file(path, expected_sha256)
    rows = read_jsonl(path)
    if len(rows) != expected_count:
        raise PostGRPODPOError(
            f"DPO 偏好对必须为 {expected_count}，实际 {len(rows)}"
        )
    for index, row in enumerate(rows, 1):
        if set(row) != {"messages", "rejected_response"}:
            raise PostGRPODPOError(f"DPO 第 {index} 条字段不正确")
        messages = row["messages"]
        rejected = row["rejected_response"]
        if not isinstance(messages, list) or len(messages) != 3:
            raise PostGRPODPOError(f"DPO 第 {index} 条 messages 不正确")
        if [message.get("role") for message in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise PostGRPODPOError(f"DPO 第 {index} 条角色顺序不正确")
        if any(
            set(message) != {"role", "content"}
            or not isinstance(message["content"], str)
            or not message["content"].strip()
            for message in messages
        ):
            raise PostGRPODPOError(f"DPO 第 {index} 条消息内容不正确")
        if (
            not isinstance(rejected, str)
            or not rejected.strip()
            or rejected == messages[-1]["content"]
        ):
            raise PostGRPODPOError(f"DPO 第 {index} 条偏好回答不正确")
    return rows


def validate_holdout_rows(path: Path) -> list[dict[str, Any]]:
    validate_frozen_file(path, HOLDOUT_SHA256)
    rows = read_jsonl(path)
    if len(rows) != HOLDOUT_COUNT:
        raise PostGRPODPOError(
            f"holdout 必须为 {HOLDOUT_COUNT} 条，实际 {len(rows)}"
        )
    counts = {issue: 0 for issue in TARGET_ISSUES}
    for index, row in enumerate(rows, 1):
        if set(row) != {
            "id",
            "target_issue",
            "scenario",
            "user",
            "preference_criteria",
        }:
            raise PostGRPODPOError(f"holdout 第 {index} 条字段不正确")
        if row["id"] != f"post_dpo_dev_{index:04d}":
            raise PostGRPODPOError(f"holdout 第 {index} 条 id 不正确")
        issue = row["target_issue"]
        if issue not in counts:
            raise PostGRPODPOError(f"holdout 第 {index} 条目标不正确")
        if any(
            not isinstance(row[name], str) or not row[name].strip()
            for name in ("scenario", "user", "preference_criteria")
        ):
            raise PostGRPODPOError(f"holdout 第 {index} 条文本为空")
        counts[issue] += 1
    if set(counts.values()) != {3}:
        raise PostGRPODPOError(f"holdout 目标分布不正确: {counts}")
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise PostGRPODPOError("缺少 PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostGRPODPOError(f"YAML 不是对象: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def validate_training_config(config: dict[str, Any]) -> int:
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
        raise PostGRPODPOError(
            "post-GRPO DPO 冻结配置不正确: " + ", ".join(mismatches)
        )
    steps = (
        math.ceil(
            math.ceil(TRAIN_PAIR_COUNT / config["per_device_train_batch_size"])
            / config["gradient_accumulation_steps"]
        )
        * config["num_train_epochs"]
    )
    if steps != 31:
        raise PostGRPODPOError(f"预期 optimizer steps 不再是 31: {steps}")
    return steps


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


def _records_from_responses(
    responses: Any, seed: int, holdout_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(responses) != len(holdout_rows):
        raise PostGRPODPOError("holdout 推理输出数量不正确")
    records = []
    for source, response in zip(holdout_rows, responses, strict=True):
        choice = response.choices[0]
        raw = choice.message.content
        if not isinstance(raw, str) or not raw.strip():
            raise PostGRPODPOError(f"holdout 推理输出为空: {source['id']}")
        assistant = normalize_empty_think_wrapper(raw)
        reward, _ = _reward_v2(assistant, choice.finish_reason)
        records.append(
            {
                "seed": seed,
                "id": source["id"],
                "scenario": source["scenario"],
                "target_issue": source["target_issue"],
                "target_goals": [source["target_issue"]],
                "preference_criteria": source["preference_criteria"],
                "user": source["user"],
                "assistant": assistant,
                "raw_assistant": raw,
                "finish_reason": choice.finish_reason,
                "attempts": 1,
                "reward_v2": reward,
            }
        )
    return records


def _reward_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row["reward_v2"] for row in rows]
    return {
        "records": len(rows),
        "mean_total_reward": sum(
            reward["total_reward"] for reward in rewards
        )
        / len(rewards),
        "hard_invalid_count": sum(
            bool(reward["hard_invalid_reasons"]) for reward in rewards
        ),
        "wrong_self_count": sum(
            reward["wrong_self_count"] for reward in rewards
        ),
    }


def _dpo_answer_key(generic: dict[str, Any]) -> dict[str, Any]:
    return {
        **generic,
        "answers": [
            {
                "review_id": row["review_id"],
                "id": row["id"],
                "grpo_label": row["base_label"],
                "dpo_label": row["sft_label"],
            }
            for row in generic["answers"]
        ],
    }


def evaluate_post_dpo_review(
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
                "base_label": row["grpo_label"],
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
            "grpo": result["mean_scores"]["base"],
            "dpo": result["mean_scores"]["sft"],
        },
    }


def generate_holdout_review_artifacts(
    repo_dir: Path,
    grpo_adapter: Path,
    dpo_adapter: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate aligned GRPO/DPO holdout outputs and an anonymous packet."""
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
        raise PostGRPODPOError(f"缺少 holdout 评估依赖: {exc.name}") from exc

    holdout_rows = validate_holdout_rows(repo_dir / HOLDOUT_RELATIVE_PATH)
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
        for row in holdout_rows
    ]
    request_config = RequestConfig(max_tokens=512, **SAMPLING_CONFIG)
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
        records = []
        for seed in EVALUATION_SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            records.extend(
                _records_from_responses(
                    engine.infer(requests, request_config=request_config),
                    seed,
                    holdout_rows,
                )
            )
        del engine, model, processor
        gc.collect()
        torch.cuda.empty_cache()
        return records

    grpo_rows = infer(grpo_adapter)
    dpo_rows = infer(dpo_adapter)
    ids = [row["id"] for row in holdout_rows]
    core = evaluate_core_behavior_gate(
        grpo_rows, dpo_rows, EVALUATION_SEEDS, ids
    )
    grpo_reward = _reward_summary(grpo_rows)
    dpo_reward = _reward_summary(dpo_rows)
    reward_checks = {
        "mean_reward_not_lower": (
            dpo_reward["mean_total_reward"]
            >= grpo_reward["mean_total_reward"]
        ),
        "hard_invalid_count_not_higher": (
            dpo_reward["hard_invalid_count"]
            <= grpo_reward["hard_invalid_count"]
        ),
        "wrong_self_count_not_higher": (
            dpo_reward["wrong_self_count"]
            <= grpo_reward["wrong_self_count"]
        ),
    }
    automatic_gate = {
        "passed": core["passed"] and all(reward_checks.values()),
        "checks": {
            "generation_stability": core["passed"],
            **reward_checks,
        },
        "core": core,
        "reward_v2": {"grpo": grpo_reward, "dpo": dpo_reward},
    }
    packet, generic_key = build_manual_review(grpo_rows, dpo_rows, ids)
    source_by_id = {row["id"]: row for row in holdout_rows}
    for item in packet["items"]:
        source = source_by_id[item["id"]]
        item["target_issue"] = source["target_issue"]
        item["preference_criteria"] = source["preference_criteria"]
    packet["comparison"] = "GRPO vs post-GRPO DPO"
    packet["severe_issue_examples"] = [
        "unreadable",
        "role_break",
        *TARGET_ISSUES,
    ]
    answer_key = _dpo_answer_key(generic_key)
    paths = {
        "grpo_holdout_outputs.jsonl": output_dir
        / "grpo_holdout_outputs.jsonl",
        "dpo_holdout_outputs.jsonl": output_dir
        / "dpo_holdout_outputs.jsonl",
        "manual_review_packet.json": output_dir / "manual_review_packet.json",
        "manual_review_answer_key.json": output_dir
        / "manual_review_answer_key.json",
        "manual_review_results.json": output_dir
        / "manual_review_results.json",
    }
    write_jsonl_exclusive(paths["grpo_holdout_outputs.jsonl"], grpo_rows)
    write_jsonl_exclusive(paths["dpo_holdout_outputs.jsonl"], dpo_rows)
    write_json_atomic(paths["manual_review_packet.json"], packet)
    write_json_atomic(paths["manual_review_answer_key.json"], answer_key)
    write_json_atomic(
        paths["manual_review_results.json"],
        empty_manual_review_results(packet),
    )
    return {"paths": paths, "automatic_gate": automatic_gate}


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_archive_contract(run_dir: Path) -> None:
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != EXPECTED_ARCHIVE_FILES:
        raise PostGRPODPOError(
            "post-GRPO DPO 归档目录不匹配: "
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
        raise PostGRPODPOError("run 未完成自动复核，不能发布")


def run_post_grpo_dpo(output_root: Path | None = None) -> Path:
    """Execute the frozen lightweight DPO run."""
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
        raise PostGRPODPOError(f"运行目录已存在: {run_id}")
    create_exclusive_directory(work_dir)
    summary_path = work_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "post_grpo_dpo",
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

        original_train_path = repo_dir / ORIGINAL_TRAIN_RELATIVE_PATH
        expansion_train_path = repo_dir / EXPANSION_TRAIN_RELATIVE_PATH
        holdout_path = repo_dir / HOLDOUT_RELATIVE_PATH
        system_path = repo_dir / SYSTEM_PROMPT_RELATIVE_PATH
        adapter_dir = repo_dir / GRPO_ADAPTER_RELATIVE_PATH
        original_rows = validate_training_rows(
            original_train_path,
            ORIGINAL_TRAIN_SHA256,
            ORIGINAL_TRAIN_PAIR_COUNT,
        )
        expansion_rows = validate_training_rows(
            expansion_train_path,
            EXPANSION_TRAIN_SHA256,
            EXPANSION_TRAIN_PAIR_COUNT,
        )
        rows = [*original_rows, *expansion_rows]
        users = [row["messages"][1]["content"] for row in rows]
        if len(users) != len(set(users)):
            raise PostGRPODPOError("原始与扩充 DPO 数据存在重复 Prompt")
        holdout = validate_holdout_rows(holdout_path)
        inputs = {
            str(ORIGINAL_TRAIN_RELATIVE_PATH): {
                "sha256": ORIGINAL_TRAIN_SHA256,
                "records": len(original_rows),
            },
            str(ORIGINAL_TRAIN_AUDIT_RELATIVE_PATH): validate_frozen_file(
                repo_dir / ORIGINAL_TRAIN_AUDIT_RELATIVE_PATH,
                ORIGINAL_TRAIN_AUDIT_SHA256,
            ),
            str(EXPANSION_TRAIN_RELATIVE_PATH): {
                "sha256": EXPANSION_TRAIN_SHA256,
                "records": len(expansion_rows),
            },
            str(EXPANSION_TRAIN_AUDIT_RELATIVE_PATH): validate_frozen_file(
                repo_dir / EXPANSION_TRAIN_AUDIT_RELATIVE_PATH,
                EXPANSION_TRAIN_AUDIT_SHA256,
            ),
            str(HOLDOUT_RELATIVE_PATH): {
                "sha256": HOLDOUT_SHA256,
                "records": len(holdout),
            },
            str(SYSTEM_PROMPT_RELATIVE_PATH): validate_frozen_file(
                system_path, SYSTEM_PROMPT_SHA256
            ),
            str(GRPO_ADAPTER_RELATIVE_PATH): validate_grpo_adapter(adapter_dir),
        }
        source_config = repo_dir / CONFIG_RELATIVE_PATH
        config = _load_yaml(source_config)
        planned_steps = validate_training_config(config)

        try:
            from huggingface_hub import HfApi
            from transformers import AutoProcessor
        except ImportError as exc:
            raise PostGRPODPOError(f"缺少输入校验依赖: {exc.name}") from exc
        model_info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
        if model_info.sha != MODEL_REVISION:
            raise PostGRPODPOError(f"模型 revision 不匹配: {model_info.sha}")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        token_lengths = []
        for index, row in enumerate(rows, 1):
            chosen = row["messages"]
            rejected = [
                *chosen[:-1],
                {"role": "assistant", "content": row["rejected_response"]},
            ]
            for label, messages in (("chosen", chosen), ("rejected", rejected)):
                length = _token_length(processor, messages)
                if length > config["max_length"]:
                    raise PostGRPODPOError(
                        f"DPO 第 {index} 条 {label} 超过 max_length: {length}"
                    )
                token_lengths.append(length)
        del processor

        train_data = work_dir / "train.jsonl"
        write_jsonl_exclusive(train_data, rows)
        train_output = work_dir / "swift-output"
        config["dataset"] = str(train_data)
        config["output_dir"] = str(train_output)
        effective_config = work_dir / "training_config.yaml"
        _write_yaml(effective_config, config)
        summary["model"] = {"name": MODEL_ID, "revision": MODEL_REVISION}
        summary["inputs"] = inputs
        summary["training"] = {
            "config_source": str(CONFIG_RELATIVE_PATH),
            "config_sha256": sha256_file(source_config),
            "training_pairs": len(rows),
            "source_pair_counts": {
                "original": len(original_rows),
                "expansion": len(expansion_rows),
            },
            "planned_optimizer_steps": planned_steps,
            "max_sequence_tokens": max(token_lengths),
        }
        summary["status"] = "training"
        write_json_atomic(summary_path, summary)

        train_log = work_dir / "train.log"
        command = ["swift", "rlhf", str(effective_config)]
        duration = run_logged(
            command,
            train_log,
            repo_dir,
            {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TOKENIZERS_PARALLELISM": "false",
            },
            expected_steps=planned_steps,
        )
        precision = validate_effective_training_args(train_output)
        losses = read_metric(train_output, "loss")
        grad_norms = read_metric(train_output, "grad_norm")
        if not losses or not all(
            math.isfinite(value) and value > 0 for value in losses
        ):
            raise PostGRPODPOError(f"loss 无效: {losses}")
        if not grad_norms or not all(
            math.isfinite(value) and value > 0 for value in grad_norms
        ):
            raise PostGRPODPOError(f"grad_norm 无效: {grad_norms}")
        if len(grad_norms) != planned_steps:
            raise PostGRPODPOError(
                f"optimizer step 数不正确: {len(grad_norms)} != {planned_steps}"
            )
        final_adapter = find_final_adapter(train_output)
        adapter_update = inspect_adapter_change(adapter_dir, final_adapter)
        review = generate_holdout_review_artifacts(
            repo_dir, adapter_dir, final_adapter, work_dir
        )
        automatic = review["automatic_gate"]

        archive_stage = work_dir / "archive"
        archive_stage.mkdir()
        shutil.copy2(effective_config, archive_stage / "training_config.yaml")
        shutil.copy2(train_data, archive_stage / "train.jsonl")
        shutil.copy2(train_log, archive_stage / "train.log")
        for name, source in review["paths"].items():
            shutil.copy2(source, archive_stage / name)
        published_adapter = archive_stage / "adapter"
        published_adapter.mkdir()
        for name in ADAPTER_FILES:
            source = final_adapter / name
            if not source.is_file():
                raise PostGRPODPOError(f"训练 adapter 缺少 {name}")
            shutil.copy2(source, published_adapter / name)

        summary["status"] = (
            "awaiting_manual_review"
            if automatic["passed"]
            else "automatic_review_failed"
        )
        summary["technically_valid"] = True
        summary["ready_to_publish"] = True
        summary["ready_for_final_eval"] = False
        summary["training"].update(
            {
                "command": command,
                "duration_seconds": round(duration, 3),
                "effective_precision": precision,
                "optimizer_steps": len(grad_norms),
                "losses": losses,
                "grad_norms": grad_norms,
                "adapter_update": adapter_update,
            }
        )
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
        summary["artifacts"] = {
            relative: _artifact(archive_stage / relative, archive_stage)
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
        summary["ready_for_final_eval"] = False
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        summary["retained_work_dir"] = str(work_dir)
        write_json_atomic(summary_path, summary)
        if not run_dir.exists():
            create_exclusive_directory(run_dir)
        shutil.copy2(summary_path, run_dir / "run_summary.json")
        raise


def _bundle_paths(output_dir: Path, run_id: str) -> tuple[Path, Path, str]:
    tag = f"morgana-v2-post-grpo-dpo-{run_id}"
    return (
        output_dir / f"{tag}.tar.gz",
        output_dir / f"{tag}.manifest.json",
        tag,
    )


def create_release_bundle(
    run_dir: Path, output_dir: Path
) -> tuple[Path, Path, str]:
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
        raise PostGRPODPOError("run_summary.json 缺少有效 run.id")
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise PostGRPODPOError("发布包目录不能位于 run 目录内")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, manifest_path, tag = _bundle_paths(output_dir, run_id)
    contents = {
        name: _artifact(run_dir / name, run_dir)
        for name in sorted(EXPECTED_ARCHIVE_FILES)
    }
    if bundle.exists() or manifest_path.exists():
        if not bundle.is_file() or not manifest_path.is_file():
            raise PostGRPODPOError("发布包或 manifest 不完整")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("contents") != contents
            or manifest.get("bundle", {}).get("sha256")
            != sha256_file(bundle)
        ):
            raise PostGRPODPOError("现有发布包与 run 不一致")
        return bundle, manifest_path, tag
    temporary = output_dir / f".{bundle.name}.{uuid4().hex}.tmp"
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name in sorted(EXPECTED_ARCHIVE_FILES):
                archive.add(
                    run_dir / name,
                    arcname=f"{run_id}/{name}",
                    recursive=False,
                )
        temporary.replace(bundle)
        manifest = {
            "schema_version": 1,
            "stage": "post_grpo_dpo",
            "run_id": run_id,
            "run_status": summary["status"],
            "source_commit": summary.get("run", {}).get("commit"),
            "github_release_tag": tag,
            "bundle": {
                "file": bundle.name,
                "bytes": bundle.stat().st_size,
                "sha256": sha256_file(bundle),
            },
            "contents": contents,
        }
        write_json_atomic(manifest_path, manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        bundle.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return bundle, manifest_path, tag


def publish_run(
    run_dir: Path,
    output_dir: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> tuple[Path, Path, str]:
    bundle, manifest_path, tag = create_release_bundle(run_dir, output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command = [
        "gh",
        "release",
        "create",
        tag,
        str(bundle),
        str(manifest_path),
        "--repo",
        repository,
        "--title",
        f"morgana-v2 post-GRPO DPO {manifest['run_id']}",
        "--notes",
        "post-GRPO DPO 训练产物、holdout 复核材料和 adapter。",
    ]
    source_commit = manifest.get("source_commit")
    if isinstance(source_commit, str) and source_commit:
        command.extend(["--target", source_commit])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise PostGRPODPOError("未安装 GitHub CLI：gh") from exc
    except subprocess.CalledProcessError as exc:
        raise PostGRPODPOError(
            "GitHub Release 上传失败；本地发布包已保留"
        ) from exc
    return bundle, manifest_path, tag


def format_download_command(
    tag: str, repository: str = DEFAULT_GITHUB_REPOSITORY
) -> str:
    command = ["roleplay-post-grpo-dpo", "download", "--tag", tag]
    if repository != DEFAULT_GITHUB_REPOSITORY:
        command.extend(["--repo", repository])
    return shlex.join(command)


def extract_release_bundle(
    bundle: Path, manifest_path: Path, output_root: Path
) -> Path:
    bundle = bundle.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "post_grpo_dpo":
        raise PostGRPODPOError("manifest stage 不正确")
    if manifest.get("bundle", {}).get("sha256") != sha256_file(bundle):
        raise PostGRPODPOError("下载包 SHA-256 与 manifest 不匹配")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise PostGRPODPOError("manifest 缺少有效 run_id")
    contents = manifest.get("contents")
    if not isinstance(contents, dict) or set(contents) != EXPECTED_ARCHIVE_FILES:
        raise PostGRPODPOError("manifest 文件清单不满足归档契约")
    expected_names = {f"{run_id}/{name}" for name in contents}
    output_root = output_root.resolve()
    destination = output_root / run_id
    if destination.exists():
        raise PostGRPODPOError(
            f"本地 run 目录已存在，拒绝覆盖: {destination}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".download-{uuid4().hex}"
    staging.mkdir()
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != expected_names or not all(
                member.isfile() for member in members
            ):
                raise PostGRPODPOError("发布包内容与 manifest 不一致")
            for member in members:
                target = (staging / member.name).resolve()
                try:
                    target.relative_to(staging)
                except ValueError as exc:
                    raise PostGRPODPOError("发布包包含不安全路径") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise PostGRPODPOError(
                        f"无法读取发布包文件: {member.name}"
                    )
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        staged_run = staging / run_id
        for relative, metadata in contents.items():
            path = staged_run / relative
            if (
                not path.is_file()
                or path.stat().st_size != metadata.get("bytes")
                or sha256_file(path) != metadata.get("sha256")
            ):
                raise PostGRPODPOError(f"解包文件校验失败: {relative}")
        staged_run.replace(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def download_release(
    tag: str,
    output_root: Path,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise PostGRPODPOError("release tag 格式无效")
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
            raise PostGRPODPOError("未安装 GitHub CLI：gh") from exc
        except subprocess.CalledProcessError as exc:
            raise PostGRPODPOError(f"GitHub Release 下载失败: {tag}") from exc
        bundle = download_dir / f"{tag}.tar.gz"
        manifest = download_dir / f"{tag}.manifest.json"
        if not bundle.is_file() or not manifest.is_file():
            raise PostGRPODPOError("Release 缺少发布包或 manifest")
        return extract_release_bundle(bundle, manifest, output_root)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostGRPODPOError(f"{label} 不是有效 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PostGRPODPOError(f"{label} 必须是 JSON 对象: {path}")
    return value


def review_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise PostGRPODPOError(f"缺少 run_summary.json: {run_dir}")
    summary = _read_json_object(summary_path, "run_summary.json")
    if "manual_reviewed_at_utc" in summary:
        raise PostGRPODPOError("该 run 已完成人工复核，拒绝重复提交")
    for name in (
        "manual_review_packet.json",
        "manual_review_answer_key.json",
        "manual_review_results.json",
    ):
        if not (run_dir / name).is_file():
            raise PostGRPODPOError(f"缺少人工复核文件: {name}")
    submitted = _read_json_object(
        run_dir / "manual_review_results.json", "manual_review_results.json"
    )
    if not submitted.get("results"):
        print("尚未提交人工复核；run_summary 保持不变。")
        return summary
    packet = _read_json_object(
        run_dir / "manual_review_packet.json", "manual_review_packet.json"
    )
    answer_key = _read_json_object(
        run_dir / "manual_review_answer_key.json",
        "manual_review_answer_key.json",
    )
    try:
        gate = evaluate_post_dpo_review(packet, answer_key, submitted)
    except ValueError as exc:
        raise PostGRPODPOError(str(exc)) from exc
    ready = (
        bool(summary.get("technically_valid"))
        and bool(summary.get("automatic_review", {}).get("passed"))
        and gate["passed"]
    )
    summary["manual_review"] = {
        **summary.get("manual_review", {}),
        "status": "passed" if gate["passed"] else "failed",
        "gate": gate,
    }
    summary["ready_for_final_eval"] = ready
    summary["status"] = "ready_for_final_eval" if ready else "dpo_failed"
    summary.setdefault("artifacts", {})["manual_review_results.json"] = (
        _artifact(run_dir / "manual_review_results.json", run_dir)
    )
    summary["manual_reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary["manual_review"], ensure_ascii=False, indent=2))
    print(f"ready_for_final_eval={ready}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 morgana-v2 post-GRPO DPO"
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
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_dir = run_post_grpo_dpo(args.output_root)
            print(f"post-GRPO DPO run: {run_dir}")
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
        else:
            review_run(args.run_dir)
    except (
        Stage2SFTError,
        Stage3DPOError,
        PostGRPODPOError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
