# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for anomaly-detection PDF reporting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk.reporting import (
    _create_explanation_page,
    _resolve_explanation_rows,
    generate_anomaly_detection_report,
)


def _report_frame() -> pd.DataFrame:
    """Create a compact result frame with timestamps, predictions, and labels."""
    samples = 12
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=samples, freq="min"),
            "temperature": np.linspace(20.0, 24.0, samples),
            "pressure": np.sin(np.linspace(0.0, 2.0 * np.pi, samples)),
            "GT": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            "Anomaly": [False, False, False, True, False, False, False, True, False, False, False, False],
            "MAE": np.linspace(0.05, 0.7, samples),
        }
    )


def test_generate_report_with_detected_and_ground_truth_anomalies(tmp_path: Path) -> None:
    """A labeled result should produce a non-empty multi-page PDF."""
    destination = tmp_path / "anomaly_report.pdf"

    result = generate_anomaly_detection_report(_report_frame(), destination, max_features_per_page=1)

    assert result == destination
    assert destination.read_bytes().startswith(b"%PDF")
    assert destination.stat().st_size > 5_000


def test_generate_report_without_ground_truth(tmp_path: Path) -> None:
    """Ground truth should be optional."""
    destination = tmp_path / "unlabeled_report.pdf"
    frame = _report_frame().drop(columns="GT")

    generate_anomaly_detection_report(frame, destination)

    assert destination.exists()


def test_generate_report_rejects_missing_ad_columns(tmp_path: Path) -> None:
    """The report requires the standard anomaly flag and score columns."""
    with pytest.raises(ValueError, match="must contain"):
        generate_anomaly_detection_report(
            _report_frame().drop(columns="MAE"),
            tmp_path / "invalid.pdf",
        )


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("Anomaly", [-1, 1] * 6),
        ("GT", np.linspace(0.0, 1.0, 12)),
    ],
)
def test_generate_report_rejects_non_binary_masks(tmp_path: Path, column: str, values) -> None:
    """Detected and ground-truth masks must contain true binary values."""
    frame = _report_frame()
    frame[column] = values

    with pytest.raises(ValueError, match="must contain only binary values"):
        generate_anomaly_detection_report(frame, tmp_path / "non_binary.pdf")


def test_generate_report_preserves_non_string_feature_labels(tmp_path: Path) -> None:
    """Explicit integer feature labels should match the original DataFrame columns."""
    frame = _report_frame().rename(columns={"temperature": 10, "pressure": 20})
    destination = tmp_path / "integer_features.pdf"

    generate_anomaly_detection_report(frame, destination, feature_columns=[10, 20])

    assert destination.read_bytes().startswith(b"%PDF")


def test_generate_report_includes_explainability_metrics(tmp_path: Path) -> None:
    """Explainability output should add anomaly-detail metrics to the PDF."""
    frame = _report_frame()
    frame["TopContributors"] = ["[]"] * len(frame)
    frame["ContributionShares"] = ["[]"] * len(frame)
    frame["ExplanationCoverage"] = 0.0
    frame.loc[3, ["TopContributors", "ContributionShares", "ExplanationCoverage"]] = [
        '["temperature","pressure"]',
        "[0.75,0.25]",
        1.0,
    ]
    frame.loc[7, ["TopContributors", "ContributionShares", "ExplanationCoverage"]] = [
        '["pressure"]',
        "[0.6]",
        0.6,
    ]
    destination = tmp_path / "explainable_report.pdf"

    generate_anomaly_detection_report(frame, destination)

    rows = _resolve_explanation_rows(
        frame,
        frame["Anomaly"].to_numpy(dtype=bool),
        frame["timestamp"],
        is_datetime=True,
    )
    assert rows == [
        ("2026-01-01\n00:03:00", "temperature\npressure", "75.0%\n25.0%", "100.0%"),
        ("2026-01-01\n00:07:00", "pressure", "60.0%", "60.0%"),
    ]
    assert destination.read_bytes().startswith(b"%PDF")


def test_generate_report_without_explanations_keeps_metrics_optional() -> None:
    """Reports without explanation columns should preserve the existing path."""
    frame = _report_frame()

    rows = _resolve_explanation_rows(
        frame,
        frame["Anomaly"].to_numpy(dtype=bool),
        frame["timestamp"],
        is_datetime=True,
    )

    assert rows is None


def test_explanation_table_labels_anomalous_timestamp() -> None:
    """The first explanation column should identify anomalous timestamps."""
    figure = _create_explanation_page(
        [("2026-01-01\n00:03:00", "temperature", "75.0%", "75.0%")],
        page_number=3,
        explanation_page_number=1,
        explanation_page_count=1,
    )

    table = figure.axes[0].tables[0]
    assert table.get_celld()[(0, 0)].get_text().get_text() == "Anomalous timestamp"
