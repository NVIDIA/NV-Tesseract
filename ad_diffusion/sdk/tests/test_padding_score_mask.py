# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk import inference_ad


def test_raw_right_padding_records_only_caller_features_as_valid() -> None:
    frame = pd.DataFrame(
        {
            "signal_a": [0.0, 0.0, 0.0],
            "signal_b": [1.0, 2.0, 3.0],
            "signal_c": [3.0, 2.0, 1.0],
        }
    )

    dataset = inference_ad.InferenceData(frame, target_dim=4)

    assert dataset.valid_feature_mask.tolist() == [True, True, True, False]
    np.testing.assert_allclose(dataset.data[:, 3].numpy(), 0.0)


def test_transformed_feature_space_treats_all_model_dimensions_as_valid() -> None:
    mask = inference_ad._build_score_feature_mask(3, 40, transformed=True)
    assert mask.tolist() == [True] * 40


def test_residuals_exclude_nonzero_reconstruction_in_padding() -> None:
    recon = np.array([[1.0, 3.0, 9.0], [2.0, 6.0, 8.0]])
    target = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]])

    mae, l2, mask = inference_ad._compute_reconstruction_residuals(
        recon,
        target,
        [True, True, False],
    )

    np.testing.assert_allclose(mae, [1.5, 2.5])
    np.testing.assert_allclose(l2, [np.sqrt(5), np.sqrt(17)])
    assert mask.tolist() == [True, True, False]


def test_all_valid_mask_preserves_legacy_formula() -> None:
    recon = np.array([[1.0, 3.0, 9.0]])
    target = np.array([[0.0, 1.0, 0.0]])

    mae, l2, mask = inference_ad._compute_reconstruction_residuals(recon, target)

    np.testing.assert_allclose(mae, np.mean(np.abs(recon - target), axis=1))
    np.testing.assert_allclose(l2, np.linalg.norm(recon - target, axis=1))
    assert mask.tolist() == [True, True, True]


@pytest.mark.parametrize("mask", [[True, False], [False, False, False]])
def test_invalid_score_feature_mask_fails_explicitly(mask: list[bool]) -> None:
    with pytest.raises(ValueError, match="valid_feature_mask"):
        inference_ad._compute_reconstruction_residuals(
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            mask,
        )


def test_evaluate_uses_dataset_mask_and_scores_only_valid_features(monkeypatch) -> None:
    monkeypatch.setattr(
        inference_ad,
        "evaluate",
        lambda *_args, **_kwargs: {
            "generated_samples": np.array([[[[1.0, 3.0, 9.0]]]]),
            "target": np.array([[[0.0, 1.0, 0.0]]]),
        },
    )

    loader = SimpleNamespace(dataset=SimpleNamespace(valid_feature_mask=np.array([True, True, False])))
    result = inference_ad.evaluate_ad_tesseract2(
        object(),
        loader,
        object(),
        nsample=1,
    )

    np.testing.assert_allclose(result["residual"], [1.5])
    np.testing.assert_allclose(result["residual_l2"], [np.sqrt(5)])
    assert result["valid_feature_mask"].tolist() == [True, True, False]
    assert result["score_feature_count"] == 2


def test_chunk_merge_preserves_parent_score_mask() -> None:
    chunks = [
        {
            "residual": np.array([1.5]),
            "residual_l2": np.array([2.0]),
            "target": np.zeros((1, 3)),
            "recon": np.ones((1, 3)),
        },
        {
            "residual": np.array([2.5]),
            "residual_l2": np.array([3.0]),
            "target": np.zeros((1, 3)),
            "recon": np.ones((1, 3)),
        },
    ]
    mask = np.array([True, True, False])

    result = inference_ad._merge_chunked_results(chunks, target_dim=3, valid_feature_mask=mask)

    np.testing.assert_array_equal(result["valid_feature_mask"], mask)
    assert result["score_feature_count"] == 2
    np.testing.assert_allclose(result["residual"], [1.5, 2.5])
