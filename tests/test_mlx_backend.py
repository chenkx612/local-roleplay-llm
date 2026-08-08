"""Numerical blockers that can be tested without importing Metal."""

import math
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from roleplay.mlx_backend import (
    MLXRuntimeError,
    apply_grpo_group_update,
    generate_candidate_with_logprobs,
    gradient_precheck,
    require_finite_positive,
    train_sft,
)


class NumericalGateTests(unittest.TestCase):
    def test_accepts_positive_loss_gradient_and_weight_delta(self):
        for label in ("loss", "gradient_norm", "weight_delta"):
            self.assertEqual(require_finite_positive(0.25, label), 0.25)

    def test_nonfinite_gradient_is_a_hard_failure(self):
        for value in (math.nan, math.inf, -math.inf, 0.0, -1.0):
            with self.subTest(value=value), self.assertRaises(MLXRuntimeError):
                require_finite_positive(value, "gradient_norm")


class GrpoTrajectoryTests(unittest.TestCase):
    def test_terminal_response_token_and_logprob_are_preserved(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        responses = [
            SimpleNamespace(
                text="甲", token=7, logprobs={7: Scalar(-0.7)},
                finish_reason=None, peak_memory=1.0,
            ),
            SimpleNamespace(
                text="乙", token=8, logprobs={8: Scalar(-0.8)},
                finish_reason="length", peak_memory=1.0,
            ),
        ]
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.stream_generate = lambda *args, **kwargs: iter(responses)
        sample_utils = types.ModuleType("mlx_lm.sample_utils")
        sample_utils.make_sampler = lambda **kwargs: object()
        fake_mx = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None))
        model = SimpleNamespace(eval=lambda: None)
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: [1, 2]
        )
        config = {
            "temperature": 0.8, "top_p": 0.9, "top_k": 20,
            "max_completion_tokens": 2,
        }
        with patch.dict(sys.modules, {
            "mlx_lm": mlx_lm, "mlx_lm.sample_utils": sample_utils,
        }), patch(
            "roleplay.mlx_backend.require_mlx",
            return_value=(fake_mx, None, None),
        ):
            candidate = generate_candidate_with_logprobs(
                model, tokenizer, [{"role": "user", "content": "问"}], config, 3
            )
        self.assertEqual(candidate["completion_tokens"], [7, 8])
        self.assertEqual(candidate["old_logprobs"], [-0.7, -0.8])

    def test_policy_ratio_computation_keeps_model_in_eval_mode(self):
        calls = []
        model = SimpleNamespace(
            eval=lambda: calls.append("eval"),
            train=lambda: calls.append("train"),
        )
        with patch(
            "roleplay.mlx_backend.require_mlx",
            return_value=(None, None, None),
        ):
            result = apply_grpo_group_update(
                model, None, [], [1.0, 1.0], {}, update=False
            )
        self.assertEqual(calls, ["eval"])
        self.assertEqual(result["reason"], "equal_rewards")


class SftSeedTests(unittest.TestCase):
    def test_mlx_is_seeded_before_precheck_and_formal_adapter_construction(self):
        trainer = types.ModuleType("mlx_lm.tuner.trainer")
        trainer.default_loss = object()
        trainer.iterate_batches = object()
        mlx_utils = types.ModuleType("mlx.utils")
        mlx_utils.tree_map = object()
        fake_modules = {
            "mlx": types.ModuleType("mlx"),
            "mlx.utils": mlx_utils,
            "mlx_lm": types.ModuleType("mlx_lm"),
            "mlx_lm.tuner": types.ModuleType("mlx_lm.tuner"),
            "mlx_lm.tuner.trainer": trainer,
        }
        for function, args in (
            (gradient_precheck, ({"seed": 17}, [])),
            (train_sft, ({"seed": 17}, [], None, None)),
        ):
            with self.subTest(function=function.__name__):
                calls = []
                fake_mx = SimpleNamespace(
                    random=SimpleNamespace(seed=lambda seed: calls.append(("seed", seed)))
                )

                def stop_after_seed(config):
                    calls.append(("construct", config["seed"]))
                    raise RuntimeError("stop after construction check")

                with patch.dict(sys.modules, fake_modules), patch(
                    "roleplay.mlx_backend.require_mlx",
                    return_value=(fake_mx, None, None),
                ), patch(
                    "roleplay.mlx_backend.prepare_trainable_model",
                    side_effect=stop_after_seed,
                ), self.assertRaisesRegex(RuntimeError, "construction check"):
                    function(*args)
                self.assertEqual(calls, [("seed", 17), ("construct", 17)])
