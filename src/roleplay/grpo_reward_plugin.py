"""ms-swift 4.4.1 AsyncORM adapter for the morgana-v2 reward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swift.rewards import AsyncORM, orms

from roleplay.grpo_reward import GRPORewardError, MorganaRewardEngine


class MorganaRewardORM(AsyncORM):
    """Expose the frozen composite reward through ms-swift's registry."""

    def __init__(self, args: Any = None, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            raise GRPORewardError("ms-swift args.output_dir 不能为空")
        self.engine = MorganaRewardEngine(
            log_path=Path(output_dir) / "reward_samples.jsonl"
        )

    async def __call__(
        self,
        completions: list[str],
        messages: list[list[dict[str, Any]]],
        finish_reason: list[str | None] | None = None,
        is_truncated: list[bool] | None = None,
        prompt_id: list[Any] | None = None,
        request_id: list[Any] | None = None,
        trainer_state: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        global_step = getattr(trainer_state, "global_step", None)
        return await self.engine.score_batch(
            completions,
            messages,
            finish_reasons=finish_reason,
            is_truncated=is_truncated,
            prompt_ids=prompt_id,
            request_ids=request_id,
            record_ids=kwargs.get("id"),
            global_step=global_step,
        )


orms["morgana_reward"] = MorganaRewardORM
