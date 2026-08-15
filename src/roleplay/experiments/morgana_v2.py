"""Canonical cross-stage contract for the archived morgana-v2 experiment."""

from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "965dcc54bc9c0591873df0e9869c056a54d323d1"

DEV_RELATIVE_PATH = Path("data/runs/morgana-v2/dev.jsonl")
DEV_SHA256 = "74cf6d05921155cec5c070ca8a611c7a8e6751b00ca0b77a6f4e9085aeeecb22"
SYSTEM_PROMPT_RELATIVE_PATH = Path("data/runs/morgana-v2/system_prompt.txt")
SYSTEM_PROMPT_SHA256 = (
    "d88993aaa1178ced740f6b54530a27e5fcdb2486a66d8b460367e842b53ee112"
)

SFT_ADAPTER_RELATIVE_PATH = Path("output/morgana-v2/stage2-sft/final/adapter")
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

SAMPLING_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.45,
}
