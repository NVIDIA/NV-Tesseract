# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for anomaly-detection PDF reporting."""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk import reporting
from sdk.reporting import (
    _aggregate_explanation_contributions,
    _create_contribution_graph_page,
    _create_mae_note_page,
    _resolve_diagnosis_rows,
    _resolve_explanation_rows,
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


def test_mae_note_page_explains_row_number_time_steps() -> None:
    """The PDF should explain row-number time steps when timestamps are unavailable."""
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


def test_overview_page_explains_anomaly_score_direction() -> None:
    """The first report page should state what lower and higher MAE mean."""
    figure = reporting._create_overview_page(
        np.arange(3),
        np.array([0.1, 0.2, 0.3]),
        np.array([False, False, True]),
        None,
        title="Anomaly report",
        x_label="Row number",
        is_datetime=False,
    )

    page_text = [text.get_text() for axis in figure.axes for text in axis.texts]
    assert "How to read the anomaly score" in page_text
    assert reporting.SCORE_DIRECTION_NOTE in page_text


def test_diagnosis_note_page_explains_metrics_and_likely_issue_values() -> None:
    """Question 2 reports should explain every end-user metric and issue label."""
    figure = _create_mae_note_page(page_number=3, include_diagnosis=True)

    page_text = "\n".join(text.get_text() for text in figure.axes[0].texts)
    assert "Likely issue" in page_text
    assert "Repair impact" in page_text
    assert "Diagnostic confidence" in page_text
    assert "Explanation consistency" in page_text
    for issue in ["Level shift", "Trend change", "Transient spike", "Variance burst", "No supported repair"]:
        assert issue in page_text


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
    csv_destination = tmp_path / "explainable_report_explanations.csv"
    exported = pd.read_csv(csv_destination)
    assert list(exported.columns) == [
        "Anomalous timestamp",
        "Top contributors",
        "Contribution shares",
        "Explanation coverage",
    ]
    assert len(exported) == 2
    assert exported.loc[0, "Anomalous timestamp"] == "2026-01-01 00:03:00"
    assert exported.loc[0, "Top contributors"] == '["temperature","pressure"]'
    assert exported.loc[0, "Contribution shares"] == "[0.75,0.25]"
    assert exported.loc[0, "Explanation coverage"] == 1.0


def test_report_adds_simplified_reconstruction_diagnosis_without_raw_scores(tmp_path: Path) -> None:
    frame = _report_frame()
    frame["TopContributors"] = ["[]"] * len(frame)
    frame["ContributionShares"] = ["[]"] * len(frame)
    frame["ExplanationCoverage"] = 0.0
    frame["LikelyReconstructionIssue"] = ""
    frame["RepairImpact"] = 0.0
    frame["DiagnosticConfidence"] = ""
    frame["ExplanationConsistency"] = ""
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
    frame.loc[3, ["LikelyReconstructionIssue", "RepairImpact", "DiagnosticConfidence", "ExplanationConsistency"]] = [
        "Level shift",
        0.63,
        "High",
        "Not evaluated",
    ]
    frame.loc[7, ["LikelyReconstructionIssue", "RepairImpact", "DiagnosticConfidence", "ExplanationConsistency"]] = [
        "Transient spike",
        0.31,
        "Moderate",
        "75%",
    ]

    destination = tmp_path / "diagnosis_report.pdf"
    generate_anomaly_detection_report(frame, destination)

    assert _resolve_diagnosis_rows(frame, frame["Anomaly"].to_numpy(dtype=bool)) == [
        ("Level shift", "63.0%", "High", "Not evaluated"),
        ("Transient spike", "31.0%", "Moderate", "75%"),
    ]
    exported = pd.read_csv(tmp_path / "diagnosis_report_explanations.csv")
    assert list(exported.columns[-4:]) == [
        "Likely reconstruction issue",
        "Repair impact",
        "Diagnostic confidence",
        "Explanation consistency",
    ]
    assert not any(column.startswith(("effect_", "score_after_")) for column in exported.columns)
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


def test_contribution_graph_visualizes_shares_and_coverage() -> None:
    """The graph should label feature shares and retain timestamp-only row labels."""
    figure = _create_contribution_graph_page(
        [("2026-01-01\n00:03:00", "temperature\npressure", "75.0%\n15.0%", "90.0%")],
        page_number=3,
        graph_page_number=1,
        graph_page_count=1,
    )

    axis = figure.axes[0]
    assert len(axis.patches) == 3  # Two contributor segments plus uncovered error.
    assert axis.get_title() == ""
    assert axis.get_yticklabels()[0].get_text() == "2026-01-01 00:03:00"
    assert any(text.get_text() == "temperature\n75%" for text in axis.texts)
    assert any(text.get_text() == "pressure\n15%" for text in axis.texts)
    assert any(text.get_text() == "90.0% covered" for text in axis.texts)


def test_aggregate_explanations_weights_features_by_anomaly_score() -> None:
    """Overall contributors should reflect their contribution to total anomaly MAE."""
    frame = pd.DataFrame(
        {
            "Anomaly": [True, True],
            "MAE": [1.0, 3.0],
            "TopContributors": ['["temperature","pressure"]', '["pressure","temperature"]'],
            "ContributionShares": ["[0.5,0.5]", "[0.75,0.25]"],
            "ExplanationCoverage": [1.0, 1.0],
        }
    )

    contributions, coverage = _aggregate_explanation_contributions(
        frame,
        frame["Anomaly"].to_numpy(dtype=bool),
        score_column="MAE",
        top_k=2,
    )

    assert [feature for feature, _ in contributions] == ["pressure", "temperature"]
    assert [share for _, share in contributions] == pytest.approx([0.6875, 0.3125])
    assert coverage == pytest.approx(1.0)


def test_large_explanation_report_uses_one_consolidated_page(monkeypatch, tmp_path: Path) -> None:
    """Large anomaly sets should switch from per-anomaly pages to one summary page."""
    sample_count = 80
    frame = pd.DataFrame(
        {
            "signal": np.linspace(0.0, 1.0, sample_count),
            "Anomaly": [True] * sample_count,
            "MAE": np.linspace(0.1, 1.0, sample_count),
            "TopContributors": ['["signal"]'] * sample_count,
            "ContributionShares": ["[1.0]"] * sample_count,
            "ExplanationCoverage": [1.0] * sample_count,
        }
    )
    consolidated = Mock(wraps=reporting._create_consolidated_contribution_page)
    detailed = Mock(wraps=reporting._create_contribution_graph_page)
    monkeypatch.setattr(reporting, "_create_consolidated_contribution_page", consolidated)
    monkeypatch.setattr(reporting, "_create_contribution_graph_page", detailed)

    generate_anomaly_detection_report(frame, tmp_path / "consolidated.pdf", max_report_pages=10)

    consolidated.assert_called_once()
    detailed.assert_not_called()
    assert consolidated.call_args.kwargs["max_report_pages"] == 10
    assert consolidated.call_args.kwargs["anomaly_count"] == sample_count
