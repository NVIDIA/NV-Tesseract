# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for GPU-count-independent inference randomness."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.main_model import TSDiffuser_base
from sdk.inference_ad import INFERENCE_BATCH_SIZE, _split_indexes


def _generators(seeds: list[int]) -> list[torch.Generator]:
    return [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]


def test_per_window_noise_is_independent_of_batch_partition() -> None:
    reference = torch.empty(5, 3, 4)
    seeds = [42 + index for index in range(len(reference))]

    noise_as_one_batch = TSDiffuser_base._randn_like_per_window(reference, _generators(seeds))
    noise_as_two_batches = torch.cat(
        [
            TSDiffuser_base._randn_like_per_window(reference[:2], _generators(seeds[:2])),
            TSDiffuser_base._randn_like_per_window(reference[2:], _generators(seeds[2:])),
        ]
    )

    torch.testing.assert_close(noise_as_one_batch, noise_as_two_batches, rtol=0, atol=0)


def test_worker_split_preserves_canonical_batch_boundaries() -> None:
    indexes = list(range(INFERENCE_BATCH_SIZE * 2 + 6))

    chunks = _split_indexes(indexes, num_parts=2)

    assert chunks == [indexes[: INFERENCE_BATCH_SIZE * 2], indexes[INFERENCE_BATCH_SIZE * 2 :]]
    assert [len(chunk) for chunk in chunks] == [INFERENCE_BATCH_SIZE * 2, 6]
