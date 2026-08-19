# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reconstruction-based anomaly explanations."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk.explainability import explain_reconstruction_anomalies


def test_ranks_exact_reconstruction_error_contributions() -> None:
    explanations = explain_reconstruction_anomalies(
        target=np.array([[3.0, 4.0, 5.0]]),
        reconstruction=np.array([[1.0, 3.0, 5.0]]),
        feature_names=["pressure", "flow", "power"],
        top_k=2,
    )

    assert json.loads(explanations.loc[0, "TopContributors"]) == ["pressure", "flow"]
    assert json.loads(explanations.loc[0, "ContributionShares"]) == pytest.approx([2 / 3, 1 / 3], abs=1e-6)
    assert explanations.loc[0, "ExplanationCoverage"] == pytest.approx(1.0)
    assert explanations.loc[0, "ExplanationMethod"] == "reconstruction_error"


def test_explains_only_rows_selected_by_anomaly_mask() -> None:
    explanations = explain_reconstruction_anomalies(
        target=np.array([[1.0, 2.0], [4.0, 8.0]]),
        reconstruction=np.zeros((2, 2)),
        anomaly_mask=np.array([False, True]),
        top_k=1,
    )

    assert explanations.loc[0, "TopContributors"] == "[]"
    assert explanations.loc[0, "ExplanationMethod"] == ""
    assert json.loads(explanations.loc[1, "TopContributors"]) == ["component_1"]
    assert explanations.loc[1, "ExplanationCoverage"] == pytest.approx(2 / 3)


def test_zero_error_has_zero_contribution_shares() -> None:
    explanations = explain_reconstruction_anomalies(
        target=np.ones((1, 2)),
        reconstruction=np.ones((1, 2)),
        top_k=2,
    )

    assert json.loads(explanations.loc[0, "ContributionShares"]) == [0.0, 0.0]
    assert explanations.loc[0, "ExplanationCoverage"] == 0.0


def test_maps_input_features_without_renormalizing_padded_model_error() -> None:
    """Mapped inputs should keep their shares of total model-space MAE."""
    explanations = explain_reconstruction_anomalies(
        target=np.zeros((1, 4)),
        reconstruction=np.array([[4.0, 3.0, 2.0, 1.0]]),
        feature_names=["temperature", "pressure"],
        feature_indices=[0, 1],
        top_k=2,
    )

    assert json.loads(explanations.loc[0, "TopContributors"]) == ["temperature", "pressure"]
    assert json.loads(explanations.loc[0, "ContributionShares"]) == pytest.approx([0.4, 0.3])
    assert explanations.loc[0, "ExplanationCoverage"] == pytest.approx(0.7)


@pytest.mark.parametrize("feature_indices", [[], [0, 0], [0, 2]])
def test_rejects_invalid_feature_indices(feature_indices: list[int]) -> None:
    with pytest.raises(ValueError, match="feature_indices"):
        explain_reconstruction_anomalies(
            target=np.zeros((1, 2)),
            reconstruction=np.ones((1, 2)),
            feature_names=["a"] * len(feature_indices),
            feature_indices=feature_indices,
        )


def test_rejects_mismatched_feature_names() -> None:
    with pytest.raises(ValueError, match="feature_names has 1 entries"):
        explain_reconstruction_anomalies(
            target=np.ones((2, 2)),
            reconstruction=np.zeros((2, 2)),
            feature_names=["only_one"],
        )


@pytest.mark.parametrize("top_k", [1.5, "2", True])
def test_rejects_non_integer_top_k(top_k: object) -> None:
    with pytest.raises(TypeError, match="top_k must be an integer"):
        explain_reconstruction_anomalies(
            target=np.ones((2, 2)),
            reconstruction=np.zeros((2, 2)),
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("target", "reconstruction"),
    [
        (np.array([[np.nan, 1.0]]), np.zeros((1, 2))),
        (np.ones((1, 2)), np.array([[0.0, np.inf]])),
    ],
)
def test_rejects_non_finite_reconstruction_errors(target: np.ndarray, reconstruction: np.ndarray) -> None:
    with pytest.raises(ValueError, match="only finite reconstruction errors"):
        explain_reconstruction_anomalies(target=target, reconstruction=reconstruction)
