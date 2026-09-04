# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
from types import SimpleNamespace

import pytest
import torch
from backbone import _REVIN_LEGACY_STATISTICS, BackboneModel, RevIN
from sdk import forecasting


@pytest.mark.parametrize("affine", [False, True])
def test_revin_round_trip_uses_explicit_statistics(affine: bool):
    revin = RevIN(num_features=1, affine=affine)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    normalized, statistics = revin(x, mode="norm", return_statistics=True)
    restored = revin(normalized, mode="denorm", statistics=statistics)

    assert not statistics[0].requires_grad
    assert not statistics[1].requires_grad
    assert not hasattr(revin, "mean")
    assert not hasattr(revin, "stdev")
    torch.testing.assert_close(restored, x)


def test_revin_legacy_sequential_round_trip_uses_context_local_statistics():
    revin = RevIN(num_features=1)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    normalized = revin(x, mode="norm")
    restored = revin(normalized, mode="denorm")

    torch.testing.assert_close(restored, x)


def test_revin_denormalization_without_context_statistics_fails():
    def invoke():
        revin = RevIN(num_features=1)
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

        with pytest.raises(ValueError, match="current thread/task context"):
            revin(x, mode="denorm")

    Context().run(invoke)


def test_revin_legacy_statistics_release_dead_modules():
    def invoke():
        revin = RevIN(num_features=1)
        reference = weakref.ref(revin)
        revin(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]), mode="norm")

        statistics_by_module = _REVIN_LEGACY_STATISTICS.get()
        assert statistics_by_module is not None
        assert len(statistics_by_module) == 1

        del revin
        gc.collect()

        assert reference() is None
        assert len(statistics_by_module) == 0

    Context().run(invoke)


def test_revin_legacy_statistics_are_copy_on_write_across_async_contexts():
    revin = RevIN(num_features=1)
    parent_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    child_input = torch.tensor([[[100.0, 200.0, 300.0, 400.0]]])
    parent_normalized = revin(parent_input, mode="norm")

    async def child_round_trip() -> torch.Tensor:
        child_normalized = revin(child_input, mode="norm")
        await asyncio.sleep(0)
        return revin(child_normalized, mode="denorm")

    child_output = asyncio.run(child_round_trip())
    parent_output = revin(parent_normalized, mode="denorm")

    torch.testing.assert_close(child_output, child_input)
    torch.testing.assert_close(parent_output, parent_input)


def test_revin_legacy_concurrent_round_trips_are_context_local():
    revin = RevIN(num_features=1)
    request_one_normalized = threading.Event()
    request_two_normalized = threading.Event()
    request_one = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    request_two = torch.tensor([[[100.0, 200.0, 300.0, 400.0]]])

    def round_trip(label: str, x: torch.Tensor) -> torch.Tensor:
        if label == "request_two":
            assert request_one_normalized.wait(timeout=5)

        normalized = revin(x, mode="norm")

        if label == "request_one":
            request_one_normalized.set()
            assert request_two_normalized.wait(timeout=5)
        else:
            request_two_normalized.set()

        return revin(normalized, mode="denorm")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(round_trip, "request_one", request_one)
        second = pool.submit(round_trip, "request_two", request_two)
        output_one = first.result(timeout=10)
        output_two = second.result(timeout=10)

    torch.testing.assert_close(output_one, request_one)
    torch.testing.assert_close(output_two, request_two)


def _statistics(revin: RevIN, x: torch.Tensor):
    _, statistics = revin(x, mode="norm", return_statistics=True)
    return statistics


def test_revin_gradient_statistics_preserve_forward_math():
    revin = RevIN(num_features=1)
    model = torch.nn.Sequential(revin)
    x = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]], requires_grad=True)

    detached_mean, detached_stdev = _statistics(revin, x)
    with forecasting._normalization_statistics_gradient(model, enabled=True) as normalizers:
        normalized, (gradient_mean, gradient_stdev) = revin(x, mode="norm", return_statistics=True)

    assert normalizers == 1
    assert gradient_mean.requires_grad
    assert gradient_stdev.requires_grad
    torch.testing.assert_close(gradient_mean, detached_mean, rtol=0, atol=0)
    torch.testing.assert_close(gradient_stdev, detached_stdev, rtol=0, atol=0)
    torch.testing.assert_close(normalized, (x - detached_mean) / detached_stdev)

    gradient = torch.autograd.grad(gradient_mean.sum() + gradient_stdev.sum(), x)[0]
    assert torch.isfinite(gradient).all()


def test_revin_gradient_statistics_use_finite_clamp_surrogate():
    revin = RevIN(num_features=1)
    model = torch.nn.Sequential(revin)
    x = torch.ones((1, 1, 4), requires_grad=True)

    with forecasting._normalization_statistics_gradient(model, enabled=True):
        _, (_, stdev) = revin(x, mode="norm", return_statistics=True)

    gradient = torch.autograd.grad(stdev.sum(), x)[0]
    assert torch.isfinite(gradient).all()


def test_normalization_statistics_gradient_handles_out_of_order_exits():
    revin = RevIN(num_features=1)
    model = torch.nn.Sequential(revin)
    x = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6).requires_grad_(True)
    first = forecasting._normalization_statistics_gradient(model, enabled=True)
    second = forecasting._normalization_statistics_gradient(model, enabled=True)

    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    try:
        mean, stdev = _statistics(revin, x)
        assert mean.requires_grad
        assert stdev.requires_grad
        assert "_get_statistics" not in vars(revin)
    finally:
        second.__exit__(None, None, None)

    mean, stdev = _statistics(revin, x)
    assert not mean.requires_grad
    assert not stdev.requires_grad
    assert "_get_statistics" not in vars(revin)


def test_normalization_statistics_gradient_disabled_flow_is_scoped():
    revin = RevIN(num_features=1)
    model = torch.nn.Sequential(revin)
    x = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6).requires_grad_(True)

    with forecasting._normalization_statistics_gradient(model, enabled=True):
        mean, stdev = _statistics(revin, x)
        assert mean.requires_grad
        assert stdev.requires_grad

        with forecasting._normalization_statistics_gradient(model, enabled=False) as normalizers:
            mean, stdev = _statistics(revin, x)
            assert normalizers == 0
            assert not mean.requires_grad
            assert not stdev.requires_grad

        mean, stdev = _statistics(revin, x)
        assert mean.requires_grad
        assert stdev.requires_grad

    mean, stdev = _statistics(revin, x)
    assert not mean.requires_grad
    assert not stdev.requires_grad


def test_normalization_statistics_gradient_is_thread_local():
    revin = RevIN(num_features=1)
    model = torch.nn.Sequential(revin)
    barrier = threading.Barrier(2)

    def invoke(enabled: bool) -> tuple[bool, bool]:
        x = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6).requires_grad_(True)
        with forecasting._normalization_statistics_gradient(model, enabled=enabled):
            barrier.wait(timeout=5)
            mean, stdev = _statistics(revin, x)
            barrier.wait(timeout=5)
            return mean.requires_grad, stdev.requires_grad

    with ThreadPoolExecutor(max_workers=2) as pool:
        enabled = pool.submit(invoke, True)
        disabled = pool.submit(invoke, False)
        enabled_result = enabled.result(timeout=10)
        disabled_result = disabled.result(timeout=10)

    assert enabled_result == (True, True)
    assert disabled_result == (False, False)
    assert "_get_statistics" not in vars(revin)


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
