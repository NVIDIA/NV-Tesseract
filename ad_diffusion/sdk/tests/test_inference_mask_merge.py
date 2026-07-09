# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.utils import evaluate
from sdk.inference_ad import evaluate_ad_tesseract2


class ComplementaryMaskModel:
    """Deterministic model whose samples are valid only on their target mask."""

    target = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    strategy_0_target_mask = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
    strategy_1_target_mask = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])

    def eval(self) -> ComplementaryMaskModel:
        return self

    def evaluate(self, batch: dict[str, int], nsample: int):
        return self._evaluate(batch["strategy"], nsample)

    def evaluate_with_dpm(self, batch: dict[str, int], nsample: int, dpm_steps: int):
        return self._evaluate(batch["strategy"], nsample)

    def _evaluate(self, strategy: int, nsample: int):
        target_mask = self.strategy_0_target_mask if strategy == 0 else self.strategy_1_target_mask

        # A diffusion reconstruction is meaningful only at the positions masked
        # for that strategy. Make those predictions perfect and conditioned
        # positions deliberately bad so selecting the wrong strategy is visible.
        samples = self.target.unsqueeze(1).repeat(1, nsample, 1, 1) + 100.0
        samples = torch.where(target_mask.unsqueeze(1).bool(), self.target.unsqueeze(1), samples)

        observed_mask = torch.ones_like(self.target)
        timepoints = torch.arange(self.target.shape[-1]).reshape(1, -1)
        return samples, self.target, target_mask, observed_mask, timepoints


def complementary_loaders():
    return [{"strategy": 0}], [{"strategy": 1}]


@pytest.mark.parametrize("use_dpm_solver", [False, True])
def test_evaluate_merges_each_strategy_on_its_own_target_mask(use_dpm_solver: bool):
    loader_0, loader_1 = complementary_loaders()

    result = evaluate(
        ComplementaryMaskModel(),
        loader_0,
        loader_1,
        nsample=1,
        save_results=False,
        use_dpm_solver=use_dpm_solver,
        dpm_steps=2,
    )

    expected = ComplementaryMaskModel.target.permute(0, 2, 1).unsqueeze(1)
    torch.testing.assert_close(result["generated_samples"], expected)


def test_evaluate_ad_tesseract2_residuals_use_target_mask_reconstructions():
    loader_0, loader_1 = complementary_loaders()

    result = evaluate_ad_tesseract2(
        ComplementaryMaskModel(),
        loader_0,
        loader_1,
        nsample=1,
        use_dpm_solver=False,
    )

    np.testing.assert_allclose(result["residual"], np.zeros(4))
    np.testing.assert_allclose(result["residual_l2"], np.zeros(4))
