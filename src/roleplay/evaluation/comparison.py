"""Aligned adapter comparison and anonymous review artifact generation."""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any

from roleplay.core.artifacts import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_exclusive,
)
from roleplay.sft_eval import (
    EVALUATION_SEEDS,
    build_manual_review,
    empty_manual_review_results,
    evaluate_core_behavior_gate,
    normalize_empty_think_wrapper,
)


def _records_from_responses(
    responses: Any,
    seed: int,
    dev_rows: list[dict[str, Any]],
    *,
    error_type: type[Exception],
) -> list[dict[str, Any]]:
    if len(responses) != len(dev_rows):
        raise error_type("Dev 推理输出数量不正确")
    records = []
    for source, response in zip(dev_rows, responses, strict=True):
        choice = response.choices[0]
        raw_assistant = choice.message.content
        if not isinstance(raw_assistant, str) or not raw_assistant.strip():
            raise error_type(f"Dev 推理输出为空: {source['id']}")
        assistant = normalize_empty_think_wrapper(raw_assistant)
        if not assistant:
            raise error_type(f"规范化后输出为空: {source['id']}")
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


def _grpo_answer_key(answer_key: dict[str, Any]) -> dict[str, Any]:
    return {
        **answer_key,
        "answers": [
            {
                "review_id": row["review_id"],
                "id": row["id"],
                "sft_label": row["base_label"],
                "grpo_label": row["sft_label"],
            }
            for row in answer_key["answers"]
        ],
    }


def generate_adapter_review_artifacts(
    repo_dir: Path,
    sft_adapter: Path,
    grpo_adapter: Path,
    output_dir: Path,
    *,
    model_id: str,
    model_revision: str,
    dev_relative_path: Path,
    dev_sha256: str,
    system_prompt_relative_path: Path,
    sampling_config: dict[str, Any],
    error_type: type[Exception],
) -> dict[str, Any]:
    """Generate aligned SFT/GRPO outputs and an anonymous A/B packet."""
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
        raise error_type(f"缺少 Dev 评估依赖: {exc.name}") from exc

    dev_path = repo_dir / dev_relative_path
    if not dev_path.is_file() or sha256_file(dev_path) != dev_sha256:
        raise error_type(f"冻结文件哈希不匹配: {dev_path}")
    try:
        dev_rows = read_jsonl(dev_path)
    except ValueError as exc:
        raise error_type(str(exc)) from exc
    if len(dev_rows) != 10:
        raise error_type(f"Dev 数量必须为 10，实际 {len(dev_rows)}")
    system_prompt = (repo_dir / system_prompt_relative_path).read_text(
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
    request_config = RequestConfig(max_tokens=512, **sampling_config)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    def infer(adapter: Path) -> list[dict[str, Any]]:
        model, processor = get_model_processor(
            model_id,
            revision=model_revision,
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
                    dev_rows,
                    error_type=error_type,
                )
            )
        del engine, model, processor
        gc.collect()
        torch.cuda.empty_cache()
        return records

    sft_rows = infer(sft_adapter)
    grpo_rows = infer(grpo_adapter)
    expected_ids = [row["id"] for row in dev_rows]
    automatic_gate = evaluate_core_behavior_gate(
        sft_rows, grpo_rows, EVALUATION_SEEDS, expected_ids
    )
    packet, generic_key = build_manual_review(
        sft_rows, grpo_rows, expected_ids
    )
    answer_key = _grpo_answer_key(generic_key)
    paths = {
        "sft_dev_outputs.jsonl": output_dir / "sft_dev_outputs.jsonl",
        "grpo_dev_outputs.jsonl": output_dir / "grpo_dev_outputs.jsonl",
        "manual_review_packet.json": output_dir / "manual_review_packet.json",
        "manual_review_answer_key.json": output_dir
        / "manual_review_answer_key.json",
        "manual_review_results.json": output_dir / "manual_review_results.json",
    }
    write_jsonl_exclusive(paths["sft_dev_outputs.jsonl"], sft_rows)
    write_jsonl_exclusive(paths["grpo_dev_outputs.jsonl"], grpo_rows)
    write_json_atomic(paths["manual_review_packet.json"], packet)
    write_json_atomic(paths["manual_review_answer_key.json"], answer_key)
    write_json_atomic(
        paths["manual_review_results.json"],
        empty_manual_review_results(packet),
    )
    return {"paths": paths, "automatic_gate": automatic_gate}
