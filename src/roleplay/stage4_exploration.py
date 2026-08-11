"""Probe Stage 4 GRPO sampling support without updating model weights."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from roleplay.grpo_rule_reward import PROMPTS_SHA256, score_completion
from roleplay.sft_eval import normalize_empty_think_wrapper
from roleplay.stage2_sft import (
    Stage2SFTError,
    capture_environment,
    configure_huggingface_environment,
    create_exclusive_directory,
    generate_run_id,
    git_context,
    read_jsonl,
    validate_pinned_packages,
    write_json_atomic,
    write_jsonl_exclusive,
)
from roleplay.stage3_grpo import (
    MODEL_ID,
    MODEL_REVISION,
    SFT_ADAPTER_RELATIVE_PATH,
    SYSTEM_PROMPT_RELATIVE_PATH,
    SYSTEM_PROMPT_SHA256,
)
from roleplay.stage4_grpo import (
    PROMPTS_RELATIVE_PATH,
    STAGE4_DISABLED_ACCELERATION_PACKAGES,
    STAGE4_PINNED_PACKAGES,
    Stage4GRPOError,
    validate_frozen_file,
    validate_prompt_isolation,
    validate_sft_adapter,
)


DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "output/morgana-v2/stage4-exploration"
)
BASE_SEED = 20260807
MAX_TOKENS = 256
TOP_K = 20
REPETITION_PENALTY = 1.45


@dataclass(frozen=True)
class SamplingConfig:
    """One ordered candidate-generation configuration."""

    name: str
    num_generations: int
    temperature: float
    top_p: float
    probe_rounds: int = 1

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.num_generations <= 0
            or self.temperature <= 0
            or not 0 < self.top_p <= 1
            or self.probe_rounds <= 0
        ):
            raise ValueError("采样配置参数无效")


# Ordered first by generation cost, then by distance from the frozen Stage 4
# sampler. The first passing entry is the minimum accepted configuration.
SAMPLING_LADDER = (
    SamplingConfig("g4-t06-p08", 4, 0.6, 0.8, 2),
    SamplingConfig("g4-t08-p09", 4, 0.8, 0.9, 2),
    SamplingConfig("g8-t06-p08", 8, 0.6, 0.8, 2),
    SamplingConfig("g8-t08-p09", 8, 0.8, 0.9, 2),
    SamplingConfig("g16-t08-p09", 16, 0.8, 0.9, 2),
)

# Frozen before observing new exploration outputs. Rates are prompt-group
# support rates, not candidate-level success rates.
SUPPORT_THRESHOLDS = {
    "hard_valid": 1.0,
    "reward_variance": 1.0,
    "brevity": 0.8,
    "signature": 0.7,
    "no_wrong_self": 0.8,
    "action": 0.8,
    "format": 1.0,
    "fully_compliant": 0.7,
    "forbidden_action": 1.0,
}


class Stage4ExplorationError(RuntimeError):
    """Raised when the sampling exploration contract is invalid."""


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[2]


def _prompt_policies(
    prompts: Sequence[dict[str, Any]],
) -> dict[str, str]:
    policies: dict[str, str] = {}
    for row in prompts:
        try:
            record_id = row["id"]
            policy = row["reward_policy"]["action"]
        except (KeyError, TypeError) as exc:
            raise Stage4ExplorationError("规则 Prompt 缺少动作策略") from exc
        if not isinstance(record_id, str) or not isinstance(policy, str):
            raise Stage4ExplorationError("规则 Prompt id/动作策略无效")
        policies[record_id] = policy
    if len(policies) != len(prompts):
        raise Stage4ExplorationError("规则 Prompt id 重复")
    return policies


def _candidate_compliance(components: dict[str, Any]) -> dict[str, bool]:
    action = components.get("action")
    if not isinstance(action, dict):
        raise Stage4ExplorationError("候选缺少动作分析")
    checks = {
        "hard_valid": not components.get("hard_invalid_reasons"),
        "brevity": 30 <= components.get("normalized_length", -1) <= 90,
        "signature": components.get("signature_count") == 1,
        "no_wrong_self": components.get("wrong_self_penalty") == 0.0,
        "action": components.get("action_score") == 1.0,
        "format": not components.get("format_reasons"),
    }
    checks["fully_compliant"] = all(checks.values())
    return checks


def normalize_candidate_rows(
    rows: Sequence[dict[str, Any]],
    policies: dict[str, str],
    config: SamplingConfig,
) -> list[dict[str, Any]]:
    """Normalize generated rows or Stage 4 reward logs for analysis."""
    normalized = []
    for index, row in enumerate(rows):
        record_id = row.get("record_id", row.get("prompt_id"))
        if record_id not in policies:
            raise Stage4ExplorationError(
                f"候选 {index} 缺少有效 record_id: {record_id!r}"
            )
        completion = row.get("completion", row.get("assistant"))
        if not isinstance(completion, str) or not completion.strip():
            raise Stage4ExplorationError(f"候选 {index} 回复为空")
        finish_reason = row.get("finish_reason")
        components = row.get("components")
        if not isinstance(components, dict):
            components = score_completion(
                completion,
                policies[record_id],
                finish_reason=finish_reason,
                is_truncated=bool(row.get("is_truncated", False)),
            ).as_log_dict()
        total_reward = components.get("total_reward")
        if isinstance(total_reward, bool) or not isinstance(
            total_reward, (int, float)
        ):
            raise Stage4ExplorationError(f"候选 {index} 奖励无效")
        normalized.append(
            {
                "schema_version": 1,
                "config": config.name,
                "record_id": record_id,
                "probe_round": row.get("probe_round", 1),
                "candidate_index": row.get("candidate_index", index + 1),
                "seed": row.get("seed"),
                "completion": completion,
                "finish_reason": finish_reason,
                "components": components,
                "compliance": _candidate_compliance(components),
                "total_reward": float(total_reward),
            }
        )
    return normalized


def summarize_candidate_support(
    rows: Sequence[dict[str, Any]],
    prompts: Sequence[dict[str, Any]],
    config: SamplingConfig,
) -> dict[str, Any]:
    """Summarize whether each prompt group contains learnable candidates."""
    policies = _prompt_policies(prompts)
    normalized = normalize_candidate_rows(rows, policies, config)
    grouped = {
        (probe_round, record_id): []
        for probe_round in range(1, config.probe_rounds + 1)
        for record_id in policies
    }
    for row in normalized:
        key = (row["probe_round"], row["record_id"])
        if key not in grouped:
            raise Stage4ExplorationError(
                f"候选包含意外采样轮次: {key[0]}"
            )
        grouped[key].append(row)
    invalid_counts = {
        f"round-{key[0]}:{key[1]}": len(group)
        for key, group in grouped.items()
        if len(group) != config.num_generations
    }
    if invalid_counts:
        raise Stage4ExplorationError(
            f"候选组大小不正确: {invalid_counts}"
        )

    prompt_results = []
    support_counts = {
        name: 0
        for name in (
            "hard_valid",
            "reward_variance",
            "brevity",
            "signature",
            "no_wrong_self",
            "action",
            "format",
            "fully_compliant",
        )
    }
    forbidden_count = 0
    forbidden_support = 0
    round_support_counts = {
        probe_round: {name: 0 for name in support_counts}
        for probe_round in range(1, config.probe_rounds + 1)
    }
    round_forbidden_support = {
        probe_round: 0
        for probe_round in range(1, config.probe_rounds + 1)
    }
    for probe_round in range(1, config.probe_rounds + 1):
        for prompt in prompts:
            record_id = prompt["id"]
            group = grouped[(probe_round, record_id)]
            support = {
                name: any(row["compliance"][name] for row in group)
                for name in (
                    "hard_valid",
                    "brevity",
                    "signature",
                    "no_wrong_self",
                    "action",
                    "format",
                    "fully_compliant",
                )
            }
            support["reward_variance"] = len(
                {row["total_reward"] for row in group}
            ) > 1
            for name, value in support.items():
                support_counts[name] += int(value)
                round_support_counts[probe_round][name] += int(value)
            if policies[record_id] == "forbidden":
                forbidden_count += 1
                forbidden_support += int(support["action"])
                round_forbidden_support[probe_round] += int(
                    support["action"]
                )
            prompt_results.append(
                {
                    "probe_round": probe_round,
                    "record_id": record_id,
                    "action_policy": policies[record_id],
                    "support": support,
                    "mean_reward": mean(
                        row["total_reward"] for row in group
                    ),
                    "minimum_reward": min(
                        row["total_reward"] for row in group
                    ),
                    "maximum_reward": max(
                        row["total_reward"] for row in group
                    ),
                }
            )

    prompt_count = len(prompts)
    probe_group_count = prompt_count * config.probe_rounds
    support_rates = {
        name: count / probe_group_count
        for name, count in support_counts.items()
    }
    support_rates["forbidden_action"] = (
        forbidden_support / forbidden_count if forbidden_count else 1.0
    )
    round_support_rates = {
        str(probe_round): {
            **{
                name: count / prompt_count
                for name, count in counts.items()
            },
            "forbidden_action": round_forbidden_support[probe_round]
            / sum(policy == "forbidden" for policy in policies.values()),
        }
        for probe_round, counts in round_support_counts.items()
    }
    checks = {
        name: support_rates[name] >= threshold
        and all(
            rates[name] >= threshold
            for rates in round_support_rates.values()
        )
        for name, threshold in SUPPORT_THRESHOLDS.items()
    }
    rewards = [row["total_reward"] for row in normalized]
    candidate_compliance_rates = {
        name: sum(row["compliance"][name] for row in normalized)
        / len(normalized)
        for name in normalized[0]["compliance"]
    }
    return {
        "schema_version": 1,
        "config": asdict(config),
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": SUPPORT_THRESHOLDS,
        "prompts": prompt_count,
        "probe_rounds": config.probe_rounds,
        "probe_groups": probe_group_count,
        "candidate_rows": len(normalized),
        "reward": {
            "minimum": min(rewards),
            "mean": mean(rewards),
            "maximum": max(rewards),
        },
        "support_counts": {
            **support_counts,
            "forbidden_action": forbidden_support,
        },
        "support_rates": support_rates,
        "round_support_rates": round_support_rates,
        "candidate_compliance_rates": candidate_compliance_rates,
        "groups": prompt_results,
    }


def select_minimum_config(
    summaries: Sequence[dict[str, Any]],
) -> str | None:
    """Return the first passing config from the frozen ladder order."""
    by_name = {summary["config"]["name"]: summary for summary in summaries}
    for config in SAMPLING_LADDER:
        summary = by_name.get(config.name)
        if summary is not None and summary.get("passed") is True:
            return config.name
    return None


def _load_exact_sft_engine(adapter_dir: Path) -> tuple[Any, Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from swift import TransformersEngine, get_model_processor, get_template
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise Stage4ExplorationError(
            f"缺少候选生成依赖: {exc.name}"
        ) from exc

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model, processor = get_model_processor(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        quantization_config=quantization,
        use_hf=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    engine = TransformersEngine(
        model, template=get_template(processor, enable_thinking=False)
    )
    return engine, model, processor


def generate_config_candidates(
    engine: Any,
    prompts: Sequence[dict[str, Any]],
    system_prompt: str,
    config: SamplingConfig,
    config_index: int,
) -> list[dict[str, Any]]:
    """Generate one candidate group per prompt with the exact GRPO backend."""
    try:
        import torch
        from swift import InferRequest, RequestConfig
    except ImportError as exc:
        raise Stage4ExplorationError(
            f"缺少候选生成依赖: {exc.name}"
        ) from exc

    request_config = RequestConfig(
        max_tokens=MAX_TOKENS,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
    )
    rows = []
    for probe_round in range(1, config.probe_rounds + 1):
        for prompt_index, prompt in enumerate(prompts):
            seed = (
                BASE_SEED
                + config_index * 10000
                + (probe_round - 1) * 2000
                + prompt_index * 100
            )
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt["user"]},
            ]
            requests = [
                InferRequest(messages=messages)
                for _ in range(config.num_generations)
            ]
            responses = engine.infer(
                requests, request_config=request_config
            )
            if len(responses) != config.num_generations:
                raise Stage4ExplorationError(
                    f"{prompt['id']} 候选数量不正确: {len(responses)}"
                )
            for candidate_index, response in enumerate(responses, 1):
                choice = response.choices[0]
                raw_assistant = choice.message.content
                if (
                    not isinstance(raw_assistant, str)
                    or not raw_assistant.strip()
                ):
                    raise Stage4ExplorationError(
                        f"{prompt['id']} 候选 {candidate_index} 为空"
                    )
                assistant = normalize_empty_think_wrapper(raw_assistant)
                components = score_completion(
                    assistant,
                    prompt["reward_policy"]["action"],
                    finish_reason=choice.finish_reason,
                ).as_log_dict()
                rows.append(
                    {
                        "schema_version": 1,
                        "config": config.name,
                        "probe_round": probe_round,
                        "record_id": prompt["id"],
                        "candidate_index": candidate_index,
                        "seed": seed,
                        "completion": assistant,
                        "raw_completion": raw_assistant,
                        "finish_reason": choice.finish_reason,
                        "components": components,
                        "compliance": _candidate_compliance(components),
                        "total_reward": components["total_reward"],
                    }
                )
            print(
                f"[{config.name}] round {probe_round}/"
                f"{config.probe_rounds} {prompt_index + 1}/"
                f"{len(prompts)} {prompt['id']} 完成"
            )
    return rows


def run_exploration(output_root: Path | None = None) -> Path:
    """Run the adaptive sampling ladder without training model weights."""
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
        raise Stage4ExplorationError(f"运行目录已存在: {run_id}")
    create_exclusive_directory(work_dir)
    summary_path = work_dir / "exploration_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage4_sampling_exploration",
        "status": "starting",
        "run": {"id": run_id},
        "sampling_ladder": [asdict(config) for config in SAMPLING_LADDER],
        "thresholds": SUPPORT_THRESHOLDS,
        "results": [],
        "selected_config": None,
    }
    write_json_atomic(summary_path, summary)
    engine = model = processor = torch_module = None
    try:
        configure_huggingface_environment()
        environment, torch_module = capture_environment()
        packages = validate_pinned_packages(
            STAGE4_PINNED_PACKAGES,
            STAGE4_DISABLED_ACCELERATION_PACKAGES,
        )
        summary["run"].update(git_context(repo_dir))
        summary["environment"] = environment
        summary["packages"] = packages

        prompts_path = repo_dir / PROMPTS_RELATIVE_PATH
        system_path = repo_dir / SYSTEM_PROMPT_RELATIVE_PATH
        adapter_dir = repo_dir / SFT_ADAPTER_RELATIVE_PATH
        validate_frozen_file(prompts_path, PROMPTS_SHA256)
        validate_frozen_file(system_path, SYSTEM_PROMPT_SHA256)
        validate_sft_adapter(adapter_dir)
        prompts = read_jsonl(prompts_path)
        validate_prompt_isolation(repo_dir, prompts)
        policies = _prompt_policies(prompts)
        system_prompt = system_path.read_text(encoding="utf-8")
        summary["status"] = "generating"
        write_json_atomic(summary_path, summary)

        engine, model, processor = _load_exact_sft_engine(adapter_dir)
        for config_index, config in enumerate(SAMPLING_LADDER):
            rows = generate_config_candidates(
                engine,
                prompts,
                system_prompt,
                config,
                config_index,
            )
            normalized = normalize_candidate_rows(rows, policies, config)
            candidate_path = work_dir / f"candidates-{config.name}.jsonl"
            write_jsonl_exclusive(candidate_path, normalized)
            result = summarize_candidate_support(rows, prompts, config)
            result["candidate_file"] = candidate_path.name
            summary["results"].append(result)
            summary["selected_config"] = select_minimum_config(
                summary["results"]
            )
            if summary["selected_config"]:
                break
            write_json_atomic(summary_path, summary)

        summary["status"] = (
            "minimum_config_found"
            if summary["selected_config"]
            else "no_viable_config"
        )
        write_json_atomic(summary_path, summary)
        work_dir.replace(run_dir)
        return run_dir
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        write_json_atomic(summary_path, summary)
        raise
    finally:
        del engine, model, processor, torch_module
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def analyze_file(
    input_path: Path,
    output_path: Path | None,
    config: SamplingConfig,
) -> dict[str, Any]:
    """Analyze a candidate JSONL or an existing Stage 4 reward log."""
    repo_dir = repository_root()
    prompts_path = repo_dir / PROMPTS_RELATIVE_PATH
    validate_frozen_file(prompts_path, PROMPTS_SHA256)
    prompts = read_jsonl(prompts_path)
    summary = summarize_candidate_support(
        read_jsonl(input_path), prompts, config
    )
    if output_path is not None:
        write_json_atomic(output_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 4 sampling-exploration CLI."""
    parser = argparse.ArgumentParser(
        description="探索 Stage 4 GRPO 的最小合规候选采样配置"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="在冻结 SFT 上运行采样阶梯")
    run_parser.add_argument("--output-root", type=Path)
    analyze_parser = subparsers.add_parser(
        "analyze", help="分析已有候选或奖励日志"
    )
    analyze_parser.add_argument("--input", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path)
    analyze_parser.add_argument("--name", default=SAMPLING_LADDER[0].name)
    analyze_parser.add_argument("--num-generations", type=int, default=4)
    analyze_parser.add_argument("--temperature", type=float, default=0.6)
    analyze_parser.add_argument("--top-p", type=float, default=0.8)
    analyze_parser.add_argument("--probe-rounds", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run or analyze a Stage 4 sampling exploration."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_dir = run_exploration(args.output_root)
            summary = json.loads(
                (run_dir / "exploration_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            print(f"探索结果: {run_dir}")
            print(f"最小配置: {summary['selected_config']}")
        else:
            config = SamplingConfig(
                args.name,
                args.num_generations,
                args.temperature,
                args.top_p,
                args.probe_rounds,
            )
            summary = analyze_file(args.input, args.output, config)
            print(
                json.dumps(
                    {
                        "config": summary["config"],
                        "passed": summary["passed"],
                        "checks": summary["checks"],
                        "support_rates": summary["support_rates"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except (
        OSError,
        Stage2SFTError,
        Stage4GRPOError,
        Stage4ExplorationError,
        ValueError,
    ) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
