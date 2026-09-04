# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for counterfactual reconstruction-failure operators."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk.reconstruction_failure import (
    FAILURE_MODES,
    build_repair_variants,
    inject_failure,
    normalized_score_reduction,
    repair_under_hypothesis,
    summarize_repair_diagnostics,
)


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_matching_repair_is_best_for_controlled_failure(mode: str) -> None:
    rng = np.random.default_rng(7)
    time = np.linspace(0, 4 * np.pi, 60)
    reference = np.column_stack((np.sin(time), np.cos(time)))
    anomalous = inject_failure(
        reference,
        feature_index=0,
        segment=(15, 45),
        mode=mode,
        magnitude=4.0,
        rng=rng,
    )

    def score(candidate: np.ndarray) -> float:
        return float(np.mean(np.abs(candidate - reference)))

    scores = {
        hypothesis: score(
            repair_under_hypothesis(
                anomalous,
                reference,
                feature_index=0,
                segment=(15, 45),
                hypothesis=hypothesis,
            )
        )
        for hypothesis in FAILURE_MODES
    }
    assert min(scores, key=scores.get) == mode
    assert scores[mode] < score(anomalous)


def test_repair_changes_only_selected_feature_and_segment() -> None:
    reference = np.zeros((20, 3))
    anomalous = reference.copy()
    anomalous[5:15, 1] = 3.0
    repaired = repair_under_hypothesis(
        anomalous,
        reference,
        feature_index=1,
        segment=(5, 15),
        hypothesis="level_shift",
    )
    np.testing.assert_array_equal(repaired[:5], anomalous[:5])
    np.testing.assert_array_equal(repaired[15:], anomalous[15:])
    np.testing.assert_array_equal(repaired[:, 0], anomalous[:, 0])
    np.testing.assert_array_equal(repaired[:, 2], anomalous[:, 2])


def test_normalized_score_reduction() -> None:
    assert normalized_score_reduction(0.8, 0.2) == pytest.approx(0.75)


def test_build_repair_variants_changes_only_selected_event_feature() -> None:
    values = np.zeros((12, 3))
    values[5:7, 1] = 4.0
    candidates = [[] for _ in range(len(values))]
    candidates[5] = [1]
    candidates[6] = [1]

    variants = build_repair_variants(
        values,
        np.zeros_like(values),
        anomaly_mask=[False, False, False, False, False, True, True, False, False, False, False, False],
        candidate_feature_indices=candidates,
        lookback=1,
    )

    assert set(variants) == set(FAILURE_MODES)
    for repaired in variants.values():
        np.testing.assert_array_equal(repaired[:, 0], values[:, 0])
        np.testing.assert_array_equal(repaired[:, 2], values[:, 2])
        np.testing.assert_array_equal(repaired[:5], values[:5])
        np.testing.assert_array_equal(repaired[7:], values[7:])


def test_summarize_repair_diagnostics_returns_only_user_facing_metrics() -> None:
    diagnosis = summarize_repair_diagnostics(
        [0.2, 1.0, 0.5],
        {
            "level_shift": [0.2, 0.35, 0.6],
            "trend_change": [0.2, 0.75, 0.6],
            "transient_spike": [0.2, 0.85, 0.6],
            "variance_burst": [0.2, 0.95, 0.6],
        },
        anomaly_mask=[False, True, True],
    )

    assert list(diagnosis.columns) == [
        "LikelyReconstructionIssue",
        "RepairImpact",
        "DiagnosticConfidence",
        "ExplanationConsistency",
    ]
    assert diagnosis.loc[1].to_dict() == {
        "LikelyReconstructionIssue": "Level shift",
        "RepairImpact": pytest.approx(0.65),
        "DiagnosticConfidence": "High",
        "ExplanationConsistency": "Not evaluated",
    }
    assert diagnosis.loc[2].to_dict() == {
        "LikelyReconstructionIssue": "No supported repair",
        "RepairImpact": 0.0,
        "DiagnosticConfidence": "Inconclusive",
        "ExplanationConsistency": "Not evaluated",
    }
