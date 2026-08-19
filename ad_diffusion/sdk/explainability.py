# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explanations for reconstruction-based anomaly scores."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


def _as_feature_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert one- or two-dimensional values to a feature matrix."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array, got shape {matrix.shape}.")
    return matrix


def _resolve_feature_names(feature_names: Sequence[str] | None, feature_count: int) -> list[str]:
    """Return validated feature names or conservative component labels."""
    if feature_names is None:
        return [f"component_{index}" for index in range(feature_count)]

    names = [str(feature_name) for feature_name in feature_names]
    if len(names) != feature_count:
        raise ValueError(
            f"feature_names has {len(names)} entries, but target and reconstruction have {feature_count} features."
        )
    return names


def explain_reconstruction_anomalies(
    target: ArrayLike,
    reconstruction: ArrayLike,
    *,
    anomaly_mask: ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    feature_indices: Sequence[int] | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Explain MAE scores with exact per-feature reconstruction-error shares.

    ``feature_indices`` limits the ranked contributors while retaining all model
    dimensions in the MAE denominator. This maps real input columns through
    right-padding without pretending that padding dimensions are input features.
    """
    target_matrix = _as_feature_matrix(target, name="target")
    reconstruction_matrix = _as_feature_matrix(reconstruction, name="reconstruction")
    if target_matrix.shape != reconstruction_matrix.shape:
        raise ValueError(
            "target and reconstruction must have the same shape, "
            f"got {target_matrix.shape} and {reconstruction_matrix.shape}."
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(f"top_k must be an integer >= 1, got {type(top_k).__name__}.")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    row_count, feature_count = target_matrix.shape
    if feature_indices is None:
        resolved_indices = np.arange(feature_count, dtype=int)
    else:
        raw_indices = list(feature_indices)
        if not raw_indices:
            raise ValueError("feature_indices must contain at least one model dimension.")
        if any(not isinstance(index, int | np.integer) or isinstance(index, bool) for index in raw_indices):
            raise TypeError("feature_indices must contain only integers.")
        resolved_indices = np.asarray(raw_indices, dtype=int)
        if len(np.unique(resolved_indices)) != len(resolved_indices):
            raise ValueError("feature_indices must not contain duplicates.")
        if ((resolved_indices < 0) | (resolved_indices >= feature_count)).any():
            raise ValueError(f"feature_indices must be between 0 and {feature_count - 1}.")
    names = _resolve_feature_names(feature_names, len(resolved_indices))
    selected_count = min(top_k, len(resolved_indices))

    if anomaly_mask is None:
        explain_rows = np.ones(row_count, dtype=bool)
    else:
        explain_rows = np.asarray(anomaly_mask, dtype=bool).reshape(-1)
        if len(explain_rows) != row_count:
            raise ValueError(f"anomaly_mask has length {len(explain_rows)}, expected {row_count}.")

    with np.errstate(invalid="ignore", over="ignore"):
        absolute_errors = np.abs(target_matrix - reconstruction_matrix)
    if not np.isfinite(absolute_errors).all():
        raise ValueError("target and reconstruction must produce only finite reconstruction errors.")
    error_totals = absolute_errors.sum(axis=1)
    contribution_shares = np.divide(
        absolute_errors,
        error_totals[:, np.newaxis],
        out=np.zeros_like(absolute_errors),
        where=error_totals[:, np.newaxis] > 0,
    )
    eligible_shares = contribution_shares[:, resolved_indices]
    ranked_indices = np.argsort(-eligible_shares, axis=1, kind="stable")[:, :selected_count]

    contributors: list[str] = []
    shares: list[str] = []
    coverage = np.zeros(row_count, dtype=np.float64)
    methods: list[str] = []
    for row_index, is_explained in enumerate(explain_rows):
        if not is_explained:
            contributors.append("[]")
            shares.append("[]")
            methods.append("")
            continue

        indices = ranked_indices[row_index]
        row_shares = eligible_shares[row_index, indices]
        contributors.append(json.dumps([names[index] for index in indices], separators=(",", ":")))
        shares.append(json.dumps([round(float(value), 6) for value in row_shares], separators=(",", ":")))
        coverage[row_index] = float(row_shares.sum())
        methods.append("reconstruction_error")

    return pd.DataFrame(
        {
            "TopContributors": contributors,
            "ContributionShares": shares,
            "ExplanationCoverage": coverage,
            "ExplanationMethod": methods,
        }
    )
