# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.main_model import TSDiffuser_base


def _observed_window() -> torch.Tensor:
    return torch.tensor(
        [[[0.0, 5.0, 10.0, 15.0], [20.0, 25.0, 30.0, 35.0]]],
        dtype=torch.float32,
    )


def _expected_training_scale(observed_data: torch.Tensor) -> torch.Tensor:
    mean = observed_data.mean(dim=(1, 2), keepdim=True)
    std = observed_data.std(dim=(1, 2), keepdim=True) + 1e-5
    return torch.clamp((observed_data - mean) / std, min=-10.0, max=10.0)


def _bare_model() -> TSDiffuser_base:
    model = TSDiffuser_base.__new__(TSDiffuser_base)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.num_steps = 1
    model.is_unconditional = False
    model.use_aux_loss = False
    model.alpha_torch = torch.tensor([[[0.9]]], dtype=torch.float32)
    model.alpha_hat = torch.tensor([0.9], dtype=torch.float32)
    model.alpha = torch.tensor([0.9], dtype=torch.float32)
    model.beta = torch.tensor([0.1], dtype=torch.float32)
    return model


class RecordingDiffModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, total_input, _side_info, _t, _strategy_type):
        self.inputs.append(total_input.detach().clone())
        return torch.zeros_like(total_input[:, 0])


def test_training_and_evaluation_present_the_same_conditioning_scale_to_denoiser() -> None:
    observed_data = _observed_window()
    observed_mask = torch.ones_like(observed_data)
    cond_mask = torch.tensor(
        [[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    strategy_type = torch.zeros(1, dtype=torch.long)
    side_info = torch.empty(1)
    recorder = RecordingDiffModel()
    model = _bare_model()
    model.add_module("diffmodel", recorder)

    model.calc_loss(
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        is_train=1,
        strategy_type=strategy_type,
    )

    def process_data(_self, _batch):
        return (
            observed_data,
            observed_mask,
            torch.zeros(1, observed_data.shape[-1]),
            cond_mask,
            observed_mask,
            torch.zeros(1, dtype=torch.long),
            strategy_type,
        )

    model.process_data = MethodType(process_data, model)
    model.get_side_info = MethodType(lambda _self, *_args: side_info, model)
    model.evaluate({}, n_samples=1)

    training_conditioning = recorder.inputs[0][:, 0]
    evaluation_conditioning = recorder.inputs[1][:, 0]

    assert torch.allclose(evaluation_conditioning, training_conditioning)


@pytest.mark.parametrize(
    ("evaluate_name", "impute_name"),
    [
        ("evaluate", "impute"),
        ("evaluate_with_dpm", "dpm_solver_impute"),
    ],
)
def test_evaluation_normalizes_model_input_and_denormalizes_returned_samples(
    evaluate_name: str,
    impute_name: str,
) -> None:
    observed_data = _observed_window()
    observed_mask = torch.ones_like(observed_data)
    cond_mask = torch.ones_like(observed_data)
    strategy_type = torch.zeros(1, dtype=torch.long)
    expected_model_input = _expected_training_scale(observed_data)
    captured: dict[str, torch.Tensor] = {}
    model = _bare_model()

    def process_data(_self, _batch):
        return (
            observed_data,
            observed_mask,
            torch.zeros(1, observed_data.shape[-1]),
            cond_mask,
            observed_mask,
            torch.zeros(1, dtype=torch.long),
            strategy_type,
        )

    def fake_impute(
        _self: TSDiffuser_base,
        model_input: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> torch.Tensor:
        captured["model_input"] = model_input.detach().clone()
        return model_input.unsqueeze(1)

    model.process_data = MethodType(process_data, model)
    model.get_side_info = MethodType(lambda _self, *_args: torch.empty(1), model)
    setattr(model, impute_name, MethodType(fake_impute, model))

    samples, returned_target, *_ = getattr(model, evaluate_name)({}, n_samples=1)

    assert torch.allclose(captured["model_input"], expected_model_input)
    assert torch.allclose(samples[:, 0], observed_data)
    assert torch.equal(returned_target, observed_data)
