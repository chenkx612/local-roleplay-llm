"""Generate local SFT candidates for GRPO reward calibration on Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from roleplay.sft_eval import normalize_empty_think_wrapper


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "mlx-community/Qwen3.5-2B-4bit"
BASE_REVISION = "674aaa7240b91e8012fcad5d791b7dfe5ba90207"
BASE_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--mlx-community--Qwen3.5-2B-4bit"
    / "snapshots"
    / BASE_REVISION
)
PEFT_ADAPTER = ROOT / "output/morgana-v2/stage2-sft/4/adapter"
PEFT_ADAPTER_SHA256 = (
    "617e6e00535fa356272d32fb16d8fe8d451a9c3cfd2f766f56af02cdf2f9b78d"
)
MLX_ADAPTER = (
    ROOT / "output/morgana-v2/stage3-grpo/reward-validation/mlx-sft-adapter"
)
PROMPTS_PATH = ROOT / "data/runs/morgana-v2/rl_train.jsonl"
PROMPTS_SHA256 = "b36b4f01f232901ab0b5f6011fa64b66f48e02c75b6b0050035e4caf703e7231"
SYSTEM_PROMPT_PATH = ROOT / "data/runs/morgana-v2/system_prompt.txt"
SYSTEM_PROMPT_SHA256 = (
    "d88993aaa1178ced740f6b54530a27e5fcdb2486a66d8b460367e842b53ee112"
)
OUTPUT_PATH = (
    ROOT / "output/morgana-v2/stage3-grpo/reward-validation/sft_candidates.jsonl"
)

# One prompt per important calibration behavior: natural dialogue, fabricated
# history, identity boundary, signature voice and emotional response.
DEFAULT_PROMPT_IDS = (
    "grpo_0001",
    "grpo_0005",
    "grpo_0010",
    "grpo_0011",
    "grpo_0018",
)
PEFT_KEY_PREFIX = "base_model.model.model.language_model.layers."
MLX_KEY_PREFIX = "language_model.model.layers."
EXPECTED_LAYER_COUNT = 24
EXPECTED_TENSOR_COUNT = 372
EXPECTED_LORA_B_COUNT = 186

MAX_TOKENS = 512
TEMPERATURE = 0.6
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.45
REPETITION_CONTEXT_SIZE = 128
BASE_SEED = 20260807


class CandidateGenerationError(RuntimeError):
    """Raised when frozen inputs or the local adapter are invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise CandidateGenerationError(f"缺少冻结文件: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise CandidateGenerationError(
            f"冻结文件哈希不匹配: {path}，期望 {expected}，实际 {actual}"
        )


def mlx_weight_key(peft_key: str) -> tuple[str, bool]:
    """Map a PEFT LoRA tensor key to MLX-LM and report whether to transpose."""
    if not peft_key.startswith(PEFT_KEY_PREFIX):
        raise CandidateGenerationError(f"未知 PEFT tensor 前缀: {peft_key}")
    suffix = peft_key.removeprefix(PEFT_KEY_PREFIX)
    if suffix.endswith(".lora_A.weight"):
        return MLX_KEY_PREFIX + suffix.removesuffix(".lora_A.weight") + ".lora_a", True
    if suffix.endswith(".lora_B.weight"):
        return MLX_KEY_PREFIX + suffix.removesuffix(".lora_B.weight") + ".lora_b", True
    raise CandidateGenerationError(f"未知 PEFT LoRA tensor: {peft_key}")


def _adapter_modules(peft_keys: Iterable[str]) -> tuple[list[str], set[int]]:
    modules: set[str] = set()
    layers: set[int] = set()
    for key in peft_keys:
        if not key.startswith(PEFT_KEY_PREFIX) or not key.endswith(".lora_A.weight"):
            continue
        suffix = key.removeprefix(PEFT_KEY_PREFIX)
        layer_text, module = suffix.split(".", 1)
        layers.add(int(layer_text))
        modules.add(module.removesuffix(".lora_A.weight"))
    return sorted(modules), layers


def convert_peft_adapter_to_mlx(source_dir: Path, output_dir: Path) -> Path:
    """Convert the frozen PEFT adapter into MLX-LM's inference format."""
    weights_path = source_dir / "adapter_model.safetensors"
    config_path = source_dir / "adapter_config.json"
    require_hash(weights_path, PEFT_ADAPTER_SHA256)
    if not config_path.is_file():
        raise CandidateGenerationError(f"缺少 PEFT adapter 配置: {config_path}")

    try:
        import numpy as np
        from safetensors import safe_open
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise CandidateGenerationError(
            "本地转换需要 mlx-lm（包含 numpy 与 safetensors）"
        ) from exc

    peft_config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = peft_config.get("r")
    alpha = peft_config.get("lora_alpha")
    if type(rank) is not int or rank <= 0 or type(alpha) not in (int, float):
        raise CandidateGenerationError("PEFT adapter 的 rank/alpha 无效")

    converted: dict[str, np.ndarray[Any, Any]] = {}
    with safe_open(weights_path, framework="np") as source:
        peft_keys = list(source.keys())
        if len(peft_keys) != EXPECTED_TENSOR_COUNT:
            raise CandidateGenerationError(
                f"PEFT tensor 数量异常: {len(peft_keys)}，期望 {EXPECTED_TENSOR_COUNT}"
            )
        modules, layers = _adapter_modules(peft_keys)
        if layers != set(range(EXPECTED_LAYER_COUNT)):
            raise CandidateGenerationError(f"PEFT adapter 层集合异常: {sorted(layers)}")
        for key in peft_keys:
            output_key, transpose = mlx_weight_key(key)
            value = source.get_tensor(key)
            converted[output_key] = value.T.copy() if transpose else value.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"
    adapter_config = output_dir / "adapter_config.json"
    temporary_adapter = adapter_file.with_suffix(".safetensors.tmp")
    temporary_config = adapter_config.with_suffix(".json.tmp")
    save_file(converted, temporary_adapter)
    temporary_config.write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": EXPECTED_LAYER_COUNT,
                "lora_parameters": {
                    "rank": rank,
                    "dropout": 0.0,
                    "scale": alpha / rank,
                    "keys": modules,
                },
                "source_format": "peft",
                "source_sha256": PEFT_ADAPTER_SHA256,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_adapter.replace(adapter_file)
    temporary_config.replace(adapter_config)
    return output_dir


def load_prompts(path: Path, prompt_ids: Iterable[str]) -> list[dict[str, Any]]:
    require_hash(path, PROMPTS_SHA256)
    wanted = list(prompt_ids)
    if len(wanted) != len(set(wanted)):
        raise CandidateGenerationError("prompt id 不得重复")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt_id = row["id"]
                user = row["user"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise CandidateGenerationError(
                    f"Prompt 第 {line_no} 行格式无效: {exc}"
                ) from exc
            if not isinstance(prompt_id, str) or not isinstance(user, str):
                raise CandidateGenerationError(f"Prompt 第 {line_no} 行字段类型无效")
            rows[prompt_id] = row
    missing = [prompt_id for prompt_id in wanted if prompt_id not in rows]
    if missing:
        raise CandidateGenerationError(f"找不到 prompt: {missing}")
    return [rows[prompt_id] for prompt_id in wanted]


def _generate_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    seed: int,
) -> tuple[str, str]:
    try:
        import mlx.core as mx
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
    except ImportError as exc:
        raise CandidateGenerationError("本地生成需要 mlx-lm") from exc

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
    processors = make_logits_processors(
        repetition_penalty=REPETITION_PENALTY,
        repetition_context_size=REPETITION_CONTEXT_SIZE,
    )
    mx.random.seed(seed)
    text_parts: list[str] = []
    finish_reason = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        logits_processors=processors,
    ):
        text_parts.append(response.text)
        if response.finish_reason is not None:
            finish_reason = response.finish_reason
    return "".join(text_parts), finish_reason


def _verify_loaded_adapter(model: Any, adapter_path: Path) -> None:
    """Reject MLX's permissive load if any converted LoRA tensor was skipped."""
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
    except ImportError as exc:
        raise CandidateGenerationError("本地校验需要 mlx-lm") from exc

    expected = mx.load(str(adapter_path / "adapters.safetensors"))
    actual = dict(tree_flatten(model.parameters()))
    expected_keys = set(expected)
    actual_keys = {key for key in actual if key.endswith((".lora_a", ".lora_b"))}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise CandidateGenerationError(
            f"MLX adapter 未完整加载: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    shape_mismatches = [
        key for key in expected_keys if actual[key].shape != expected[key].shape
    ]
    if shape_mismatches:
        raise CandidateGenerationError(
            f"MLX adapter tensor shape 不匹配: {shape_mismatches[:3]}"
        )
    nonzero_lora_b = sum(
        bool(mx.any(actual[key] != 0).item())
        for key in expected_keys
        if key.endswith(".lora_b")
    )
    if nonzero_lora_b != EXPECTED_LORA_B_COUNT:
        raise CandidateGenerationError(
            f"MLX LoRA-B 非零数量异常: {nonzero_lora_b}，"
            f"期望 {EXPECTED_LORA_B_COUNT}"
        )


def generate_candidates(
    *,
    base_model_path: Path,
    adapter_path: Path,
    prompts: list[dict[str, Any]],
    system_prompt: str,
    output_path: Path,
    candidates_per_prompt: int = 4,
) -> list[dict[str, Any]]:
    if candidates_per_prompt <= 0:
        raise CandidateGenerationError("candidates_per_prompt 必须大于 0")
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise CandidateGenerationError("本地生成需要 mlx-lm") from exc

    model, tokenizer = load(str(base_model_path), adapter_path=str(adapter_path))
    _verify_loaded_adapter(model, adapter_path)
    records: list[dict[str, Any]] = []
    total = len(prompts) * candidates_per_prompt
    for prompt_index, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt["user"]},
        ]
        for candidate_index in range(candidates_per_prompt):
            seed = BASE_SEED + prompt_index * 100 + candidate_index
            raw_assistant, finish_reason = _generate_one(
                model, tokenizer, messages, seed
            )
            assistant = normalize_empty_think_wrapper(raw_assistant)
            record = {
                "prompt_id": prompt["id"],
                "scenario": prompt.get("scenario"),
                "user": prompt["user"],
                "candidate_index": candidate_index + 1,
                "seed": seed,
                "messages": messages,
                "raw_assistant": raw_assistant,
                "assistant": assistant,
                "finish_reason": finish_reason,
                "is_truncated": finish_reason == "length",
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "adapter_sha256": PEFT_ADAPTER_SHA256,
                "generation": {
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "repetition_penalty": REPETITION_PENALTY,
                    "repetition_context_size": REPETITION_CONTEXT_SIZE,
                    "enable_thinking": False,
                },
            }
            records.append(record)
            print(
                f"[{len(records)}/{total}] {prompt['id']} "
                f"candidate {candidate_index + 1}: {finish_reason}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用冻结的第 4 次 SFT adapter 生成 GRPO 奖励校准候选"
    )
    parser.add_argument("--base-model", type=Path, default=BASE_SNAPSHOT)
    parser.add_argument("--peft-adapter", type=Path, default=PEFT_ADAPTER)
    parser.add_argument("--mlx-adapter", type=Path, default=MLX_ADAPTER)
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--system-prompt", type=Path, default=SYSTEM_PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids")
    parser.add_argument("--candidates-per-prompt", type=int, default=4)
    args = parser.parse_args()

    try:
        if not args.base_model.is_dir():
            raise CandidateGenerationError(f"缺少本地 MLX 基座: {args.base_model}")
        require_hash(args.system_prompt, SYSTEM_PROMPT_SHA256)
        prompts = load_prompts(args.prompts, args.prompt_ids or DEFAULT_PROMPT_IDS)
        adapter_path = convert_peft_adapter_to_mlx(
            args.peft_adapter, args.mlx_adapter
        )
        records = generate_candidates(
            base_model_path=args.base_model,
            adapter_path=adapter_path,
            prompts=prompts,
            system_prompt=args.system_prompt.read_text(encoding="utf-8"),
            output_path=args.output,
            candidates_per_prompt=args.candidates_per_prompt,
        )
        print(f"完成，共 {len(records)} 个候选，输出: {args.output}")
    except (CandidateGenerationError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
