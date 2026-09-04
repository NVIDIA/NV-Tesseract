# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Counterfactual operators for reconstruction-failure experiments.

The functions in this module deliberately operate on input windows and a
normal-reference window.  They do not inspect model internals, which lets the
same experiment harness re-score the repaired windows with any anomaly
detector that accepts the same input representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

FailureMode = Literal["level_shift", "trend_change", "transient_spike", "variance_burst"]
FAILURE_MODES: tuple[FailureMode, ...] = (
    "level_shift",
    "trend_change",
    "transient_spike",
    "variance_burst",
)
FAILURE_MODE_LABELS: dict[FailureMode, str] = {
    "level_shift": "Level shift",
    "trend_change": "Trend change",
    "transient_spike": "Transient spike",
    "variance_burst": "Variance burst",
}
DIAGNOSIS_COLUMNS: tuple[str, ...] = (
    "LikelyReconstructionIssue",
    "RepairImpact",
    "DiagnosticConfidence",
    "ExplanationConsistency",
)


def _window(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (timestamps, features), got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _validate_inputs(
    values: ArrayLike,
    reference: ArrayLike,
    feature_index: int,
    segment: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], slice]:
    window = _window(values, name="values")
    normal = _window(reference, name="reference")
    if window.shape != normal.shape:
        raise ValueError(f"values and reference must have the same shape, got {window.shape} and {normal.shape}.")
    if not isinstance(feature_index, int) or isinstance(feature_index, bool):
        raise TypeError("feature_index must be an integer.")
    if not 0 <= feature_index < window.shape[1]:
        raise ValueError(f"feature_index must be between 0 and {window.shape[1] - 1}.")
    start, end = segment
    if not (0 <= start < end <= window.shape[0]):
        raise ValueError(f"segment must satisfy 0 <= start < end <= {window.shape[0]}, got {segment}.")
    return window, normal, slice(start, end)


def inject_failure(
    reference: ArrayLike,
    *,
    feature_index: int,
    segment: tuple[int, int],
    mode: FailureMode,
    magnitude: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Inject one controlled failure signature into a normal-reference window."""
    normal, _, affected = _validate_inputs(reference, reference, feature_index, segment)
    if mode not in FAILURE_MODES:
        raise ValueError(f"Unknown failure mode: {mode}.")
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("magnitude must be a positive finite number.")

    result = normal.copy()
    baseline = normal[affected, feature_index]
    scale = max(float(np.std(baseline)), 0.05)
    length = len(baseline)

    if mode == "level_shift":
        result[affected, feature_index] += magnitude * scale
    elif mode == "trend_change":
        ramp = np.linspace(-1.0, 1.0, length)
        result[affected, feature_index] += magnitude * scale * ramp
    elif mode == "transient_spike":
        width = max(1, length // 10)
        center = affected.start + length // 2
        lo = max(affected.start, center - width)
        hi = min(affected.stop, center + width + 1)
        result[lo:hi, feature_index] += magnitude * scale
    else:
        result[affected, feature_index] += rng.normal(0.0, magnitude * scale, size=length)
    return result


def repair_under_hypothesis(
    values: ArrayLike,
    reference: ArrayLike,
    *,
    feature_index: int,
    segment: tuple[int, int],
    hypothesis: FailureMode,
) -> NDArray[np.float64]:
    """Apply a hypothesis-specific repair using a normal-reference window.

    These operators are intentionally distinct.  A level repair removes only a
    constant offset; a trend repair removes only a centered linear slope; a
    transient repair replaces a compact peak; and a variance repair attenuates
    high-frequency deviations across the segment.
    """
    window, normal, affected = _validate_inputs(values, reference, feature_index, segment)
    if hypothesis not in FAILURE_MODES:
        raise ValueError(f"Unknown failure-mode hypothesis: {hypothesis}.")

    repaired = window.copy()
    delta = window[affected, feature_index] - normal[affected, feature_index]
    length = len(delta)

    if hypothesis == "level_shift":
        repaired[affected, feature_index] -= float(np.median(delta))
    elif hypothesis == "trend_change":
        axis = np.linspace(-1.0, 1.0, length)
        slope = float(np.dot(delta, axis) / max(np.dot(axis, axis), np.finfo(float).eps))
        repaired[affected, feature_index] -= slope * axis
    elif hypothesis == "transient_spike":
        width = max(1, length // 10)
        center = int(np.argmax(np.abs(delta))) + affected.start
        lo = max(affected.start, center - width)
        hi = min(affected.stop, center + width + 1)
        repaired[lo:hi, feature_index] = normal[lo:hi, feature_index]
    else:
        kernel_size = min(5, length if length % 2 else max(1, length - 1))
        kernel = np.ones(kernel_size, dtype=float) / kernel_size
        low_frequency = np.convolve(delta, kernel, mode="same")
        high_frequency = delta - low_frequency
        repaired[affected, feature_index] -= 0.8 * high_frequency
    return repaired


def normalized_score_reduction(original_score: float, repaired_score: float, *, epsilon: float = 1e-12) -> float:
    """Return the fraction of the original score removed by a repair."""
    if not np.isfinite(original_score) or not np.isfinite(repaired_score):
        raise ValueError("Scores must be finite.")
    return float((original_score - repaired_score) / max(abs(original_score), epsilon))


def build_repair_variants(
    values: ArrayLike,
    reference: ArrayLike,
    *,
    anomaly_mask: ArrayLike,
    candidate_feature_indices: Sequence[Sequence[int]],
    lookback: int = 20,
) -> dict[FailureMode, NDArray[np.float64]]:
    """Build one structure-preserving input variant per failure hypothesis.

    Consecutive anomalous rows are treated as one event.  Each repair covers
    the event and up to ``lookback`` rows of preceding context, and is applied
    only to features selected by the existing contribution explanation.
    """
    window = _window(values, name="values")
    normal = _window(reference, name="reference")
    if window.shape != normal.shape:
        raise ValueError(f"values and reference must have the same shape, got {window.shape} and {normal.shape}.")
    if not isinstance(lookback, int) or isinstance(lookback, bool):
        raise TypeError("lookback must be an integer.")
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    mask = np.asarray(anomaly_mask, dtype=bool).reshape(-1)
    if len(mask) != len(window):
        raise ValueError(f"anomaly_mask has length {len(mask)}, expected {len(window)}.")
    if len(candidate_feature_indices) != len(window):
        raise ValueError(
            f"candidate_feature_indices has length {len(candidate_feature_indices)}, expected {len(window)}."
        )

    variants = {hypothesis: window.copy() for hypothesis in FAILURE_MODES}
    anomaly_indexes = np.flatnonzero(mask)
    if not len(anomaly_indexes):
        return variants

    event_starts = np.r_[0, np.flatnonzero(np.diff(anomaly_indexes) > 1) + 1]
    event_ends = np.r_[event_starts[1:], len(anomaly_indexes)]
    for event_start, event_end in zip(event_starts, event_ends, strict=True):
        event_indexes = anomaly_indexes[event_start:event_end]
        segment = (max(0, int(event_indexes[0]) - lookback + 1), int(event_indexes[-1]) + 1)
        features = sorted(
            {
                int(feature_index)
                for row_index in event_indexes
                for feature_index in candidate_feature_indices[int(row_index)]
            }
        )
        invalid = [feature_index for feature_index in features if not 0 <= feature_index < window.shape[1]]
        if invalid:
            raise ValueError(f"Candidate feature indexes are outside the input width {window.shape[1]}: {invalid}.")
        for hypothesis in FAILURE_MODES:
            for feature_index in features:
                variants[hypothesis] = repair_under_hypothesis(
                    variants[hypothesis],
                    normal,
                    feature_index=feature_index,
                    segment=segment,
                    hypothesis=hypothesis,
                )
    return variants


def summarize_repair_diagnostics(
    original_scores: ArrayLike,
    repaired_scores: Mapping[FailureMode, ArrayLike],
    *,
    anomaly_mask: ArrayLike,
    consistency_predictions: Sequence[Sequence[FailureMode]] | None = None,
) -> pd.DataFrame:
    """Convert counterfactual score changes into four end-user metrics.

    The returned frame intentionally excludes per-hypothesis scores.  Confidence
    measures the margin between the best and second-best repair impacts; it is
    not a probability.  Consistency is reported only when repeated-reference or
    repeated-seed predictions are supplied.
    """
    original = np.asarray(original_scores, dtype=float).reshape(-1)
    mask = np.asarray(anomaly_mask, dtype=bool).reshape(-1)
    if len(mask) != len(original):
        raise ValueError(f"anomaly_mask has length {len(mask)}, expected {len(original)}.")
    if not np.isfinite(original).all():
        raise ValueError("original_scores must contain only finite values.")

    missing = [hypothesis for hypothesis in FAILURE_MODES if hypothesis not in repaired_scores]
    if missing:
        raise ValueError(f"repaired_scores is missing hypotheses: {missing}.")
    repaired = np.column_stack(
        [np.asarray(repaired_scores[hypothesis], dtype=float).reshape(-1) for hypothesis in FAILURE_MODES]
    )
    if repaired.shape != (len(original), len(FAILURE_MODES)):
        raise ValueError(
            "Each repaired score vector must have the same length as original_scores; "
            f"got combined shape {repaired.shape}."
        )
    if not np.isfinite(repaired).all():
        raise ValueError("repaired_scores must contain only finite values.")

    denominator = np.maximum(np.abs(original), np.finfo(float).eps)
    effects = (original[:, None] - repaired) / denominator[:, None]
    records = [
        {
            "LikelyReconstructionIssue": "",
            "RepairImpact": 0.0,
            "DiagnosticConfidence": "",
            "ExplanationConsistency": "",
        }
        for _ in original
    ]
    repeat_arrays = None
    if consistency_predictions is not None:
        repeat_arrays = [np.asarray(predictions, dtype=object).reshape(-1) for predictions in consistency_predictions]
        if any(len(predictions) != len(original) for predictions in repeat_arrays):
            raise ValueError("Each consistency prediction vector must have the same length as original_scores.")

    for row_index in np.flatnonzero(mask):
        order = np.argsort(-effects[row_index], kind="stable")
        best_index, second_index = int(order[0]), int(order[1])
        best_effect = float(effects[row_index, best_index])
        margin = best_effect - float(effects[row_index, second_index])
        if best_effect <= 0:
            issue = "No supported repair"
            impact = 0.0
            confidence = "Inconclusive"
        else:
            best_mode = FAILURE_MODES[best_index]
            issue = FAILURE_MODE_LABELS[best_mode]
            impact = min(best_effect, 1.0)
            if margin >= 0.20:
                confidence = "High"
            elif margin >= 0.10:
                confidence = "Moderate"
            elif margin > 0:
                confidence = "Low"
            else:
                confidence = "Inconclusive"

        consistency = "Not evaluated"
        if repeat_arrays:
            best_mode = FAILURE_MODES[best_index]
            agreement = np.mean([predictions[row_index] == best_mode for predictions in repeat_arrays])
            consistency = f"{agreement:.0%}"
        records[row_index] = {
            "LikelyReconstructionIssue": issue,
            "RepairImpact": impact,
            "DiagnosticConfidence": confidence,
            "ExplanationConsistency": consistency,
        }
    return pd.DataFrame.from_records(records, columns=DIAGNOSIS_COLUMNS)
