# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from backbone import BackboneModel, RevIN


@pytest.mark.parametrize("affine", [False, True])
def test_revin_round_trip_uses_explicit_statistics(affine: bool):
    revin = RevIN(num_features=1, affine=affine)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    normalized, statistics = revin(x, mode="norm", return_statistics=True)
    restored = revin(normalized, mode="denorm", statistics=statistics)

    assert not hasattr(revin, "mean")
    assert not hasattr(revin, "stdev")
    torch.testing.assert_close(restored, x)


def test_revin_denormalization_requires_matching_statistics():
    revin = RevIN(num_features=1)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    normalized = revin(x, mode="norm")

    with pytest.raises(ValueError, match="matching normalization call"):
        revin(normalized, mode="denorm")


class _CoordinatedTokenizer:
    """Pause one forecast after RevIN norm until a second forecast has normalized."""

    def __init__(self):
        self._label = threading.local()
        self.request_one_normalized = threading.Event()
        self.request_two_normalized = threading.Event()

    def set_request_label(self, label: str):
        self._label.value = label

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._label.value == "request_one":
            self.request_one_normalized.set()
            assert self.request_two_normalized.wait(timeout=5)
        else:
            self.request_two_normalized.set()
        return x.unfold(dimension=-1, size=1, step=1)


class _IdentityPatchEmbedding(torch.nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x


class _IdentityEncoder(torch.nn.Module):
    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _LastPatchHead(torch.nn.Module):
    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        return enc_out[:, :, -1, :]


class _ForecastHarness:
    """Minimal object that executes the production BackboneModel.forecast path."""

    forecast = BackboneModel.forecast
    _apply_cross_channel = BackboneModel._apply_cross_channel

    def __init__(self):
        self.normalizer = RevIN(num_features=1)
        self.tokenizer = _CoordinatedTokenizer()
        self.patch_embedding = _IdentityPatchEmbedding()
        self.encoder = _IdentityEncoder()
        self.head = _LastPatchHead()
        self.patch_len = 1
        self.config = SimpleNamespace(d_model=1)
        self.use_cross_channel = False


def test_concurrent_forecasts_keep_revin_statistics_request_local():
    model = _ForecastHarness()
    input_mask = torch.ones((1, 4))
    request_one = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    request_two = torch.tensor([[[100.0, 200.0, 300.0, 400.0]]])

    def invoke(label: str, x: torch.Tensor):
        if label == "request_two":
            assert model.tokenizer.request_one_normalized.wait(timeout=5)
        model.tokenizer.set_request_label(label)
        return model.forecast(x_enc=x, input_mask=input_mask).forecast

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoke, "request_one", request_one)
        second = pool.submit(invoke, "request_two", request_two)
        output_one = first.result(timeout=10)
        output_two = second.result(timeout=10)

    torch.testing.assert_close(output_one, torch.tensor([[[4.0]]]))
    torch.testing.assert_close(output_two, torch.tensor([[[400.0]]]))
