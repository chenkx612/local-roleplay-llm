"""One discoverable command tree for the complete learning pipeline."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


Command = tuple[str, str]

DIRECT_COMMANDS: Mapping[str, Command] = {
    "chat": ("roleplay.chat", "main"),
    "infer": ("roleplay.inference", "main"),
}

COMMAND_GROUPS: Mapping[str, Mapping[str, Command]] = {
    "data": {
        "prompts": ("roleplay.datagen", "main"),
        "sft": ("roleplay.sft_data", "main"),
        "dpo": ("roleplay.dpo_data", "main"),
        "dpo-editability": ("roleplay.dpo_editability", "main"),
        "grpo-candidates": ("roleplay.grpo_candidates", "main"),
        "post-dpo": ("roleplay.post_grpo_dpo_data", "main"),
        "post-dpo-sampling": ("roleplay.post_grpo_dpo_sampling", "main"),
    },
    "train": {
        "sft": ("roleplay.stage2_sft", "main"),
        "dpo": ("roleplay.stage3_dpo", "main"),
        "grpo": ("roleplay.stage4_grpo", "main"),
        "grpo-judge": ("roleplay.stage3_grpo", "main"),
        "post-dpo": ("roleplay.post_grpo_dpo", "main"),
    },
    "eval": {
        "inspect": ("roleplay.evaluation.cli", "main"),
        "compare": ("roleplay.evaluation.cli", "main"),
    },
}


def _resolve(command: Command) -> Callable[[list[str] | None], Any]:
    module_name, attribute = command
    return getattr(importlib.import_module(module_name), attribute)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roleplay",
        description="角色扮演模型后训练学习流水线",
    )
    subparsers = parser.add_subparsers(dest="area", required=True)
    for name in DIRECT_COMMANDS:
        subparsers.add_parser(name, add_help=False)
    for group_name, commands in COMMAND_GROUPS.items():
        group = subparsers.add_parser(group_name)
        group_subparsers = group.add_subparsers(dest="command", required=True)
        for command_name in commands:
            group_subparsers.add_parser(command_name, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, forwarded = parser.parse_known_args(arguments)
    if args.area in DIRECT_COMMANDS:
        target = DIRECT_COMMANDS[args.area]
    else:
        target = COMMAND_GROUPS[args.area][args.command]
        if args.area == "eval":
            forwarded.insert(0, args.command)
    legacy_program = sys.argv[0]
    command_path = ["roleplay", args.area]
    if args.area not in DIRECT_COMMANDS and args.area != "eval":
        command_path.append(args.command)
    try:
        sys.argv[0] = " ".join(command_path)
        result = _resolve(target)(forwarded)
    finally:
        sys.argv[0] = legacy_program
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
