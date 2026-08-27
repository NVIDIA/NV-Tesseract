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
    _create_mae_note_page,
    _resolve_x_axis,
    generate_anomaly_detection_report,
    infer_ground_truth_column,
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


def test_infers_lowercase_anomaly_as_ground_truth() -> None:
    """A conventional lowercase anomaly label should be treated as report metadata."""
    frame = pd.DataFrame({"signal": [1.0, 2.0], "anomaly": [0, 1]})

    assert infer_ground_truth_column(frame) == "anomaly"


def test_resolve_x_axis_identifies_row_numbers_without_timestamp() -> None:
    """Reports without timestamps should identify their zero-based row-number axis."""
    values, label, is_datetime = _resolve_x_axis(pd.DataFrame({"signal": [1.0, 2.0]}), None)

    assert values.tolist() == [0, 1]
    assert label == "Row number (no timestamp column)"
    assert is_datetime is False


def test_resolve_x_axis_preserves_timestamp_values() -> None:
    """A valid timestamp column should remain the report's datetime axis."""
    frame = _report_frame()

    values, label, is_datetime = _resolve_x_axis(frame, "timestamp")

    assert values.equals(frame["timestamp"])
    assert label == "timestamp"
    assert is_datetime is True


def test_mae_note_page_documents_row_number_time_steps() -> None:
    """The PDF should document row-number time steps when timestamps are unavailable."""
    figure = _create_mae_note_page(page_number=3, uses_row_numbers=True)

    assert any(
        text.get_text()
        == "No usable timestamp column was provided, so time steps in this report are zero-based DataFrame row numbers."
        for text in figure.axes[0].texts
    )


def test_mae_note_page_omits_row_number_note_for_timestamped_data() -> None:
    """Timestamped reports should not show the row-number fallback note."""
    figure = _create_mae_note_page(page_number=3, uses_row_numbers=False)

    assert not any("zero-based DataFrame row numbers" in text.get_text() for text in figure.axes[0].texts)
