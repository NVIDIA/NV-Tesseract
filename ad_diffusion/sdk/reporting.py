# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDF reporting for time-series anomaly detection results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

GROUND_TRUTH_CANDIDATES = ("GT", "ground_truth", "is_anomaly", "label", "Label")
TIMESTAMP_CANDIDATES = ("timestamp", "Timestamp", "time", "Time", "ts", "datetime", "date")
EXPLANATION_COLUMNS = {
    "TopContributors",
    "ContributionShares",
    "ExplanationCoverage",
    "ExplanationMethod",
}


def infer_ground_truth_column(df: pd.DataFrame) -> str | None:
    """Return the first conventional ground-truth column present in a frame."""
    return next((column for column in GROUND_TRUTH_CANDIDATES if column in df.columns), None)


def infer_timestamp_column(df: pd.DataFrame) -> str | None:
    """Return the first conventional timestamp column present in a frame."""
    return next((column for column in TIMESTAMP_CANDIDATES if column in df.columns), None)


def _resolve_column(df: pd.DataFrame, requested: str | None, *, role: str) -> str | None:
    if requested is not None and requested not in df.columns:
        raise ValueError(f"{role} column {requested!r} was not found in the report data.")
    return requested


def _as_boolean_mask(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name} must contain only numeric or boolean values.")
    if not set(numeric.to_numpy(dtype=float)) <= {0.0, 1.0}:
        raise ValueError(f"{name} must contain only binary values (0/1 or False/True).")
    return numeric.to_numpy(dtype=bool)


def _resolve_x_axis(df: pd.DataFrame, timestamp_column: str | None) -> tuple[np.ndarray | pd.Series, str, bool]:
    if timestamp_column is None:
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(df.index, index=df.index), "Time", True
        return np.arange(len(df)), "Sample", False

    values = df[timestamp_column]
    if pd.api.types.is_numeric_dtype(values):
        return values, timestamp_column, False

    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().all():
        return parsed, timestamp_column, True
    return np.arange(len(df)), f"Sample ({timestamp_column} unavailable as time)", False


def _resolve_feature_columns(
    result_df: pd.DataFrame,
    feature_columns: Sequence[Hashable] | None,
    *,
    timestamp_column: str | None,
    ground_truth_column: str | None,
    anomaly_column: str,
    score_column: str,
) -> list[Hashable]:
    excluded = {
        anomaly_column,
        score_column,
        timestamp_column,
        ground_truth_column,
        *EXPLANATION_COLUMNS,
    }
    if feature_columns is None:
        resolved = [column for column in result_df.select_dtypes(include=[np.number]).columns if column not in excluded]
    else:
        resolved = list(feature_columns)
        missing = [column for column in resolved if column not in result_df.columns]
        if missing:
            raise ValueError(f"Report feature columns were not found: {missing}.")

    if not resolved:
        raise ValueError("No numeric feature columns are available for the anomaly report.")
    return resolved


def _classification_summary(detected: np.ndarray, ground_truth: np.ndarray | None) -> list[tuple[str, str]]:
    rows = [
        ("Samples", f"{len(detected):,}"),
        ("Detected anomalies", f"{int(detected.sum()):,}"),
        ("Detection rate", f"{detected.mean():.2%}"),
    ]
    if ground_truth is None:
        rows.append(("Ground truth", "Not provided"))
        return rows

    true_positive = int(np.sum(detected & ground_truth))
    false_positive = int(np.sum(detected & ~ground_truth))
    false_negative = int(np.sum(~detected & ground_truth))
    true_negative = int(np.sum(~detected & ~ground_truth))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rows.extend(
        [
            ("Ground-truth anomalies", f"{int(ground_truth.sum()):,}"),
            ("TP / FP / FN / TN", f"{true_positive} / {false_positive} / {false_negative} / {true_negative}"),
            ("Precision / Recall / F1", f"{precision:.3f} / {recall:.3f} / {f1:.3f}"),
        ]
    )
    return rows


def _style_axis(axis, *, is_datetime: bool) -> None:
    axis.grid(True, color="#D8DEE9", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, colors="#344054")
    if is_datetime:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _add_footer(figure: Figure, page_number: int) -> None:
    figure.text(0.07, 0.018, "NV-Tesseract anomaly detection", fontsize=7, color="#667085")
    figure.text(0.93, 0.018, f"Page {page_number}", fontsize=7, color="#667085", ha="right")


def _create_overview_page(
    x_values,
    scores: np.ndarray,
    detected: np.ndarray,
    ground_truth: np.ndarray | None,
    *,
    title: str,
    x_label: str,
    is_datetime: bool,
) -> Figure:
    figure = Figure(figsize=(11.0, 8.5), facecolor="white")
    grid = figure.add_gridspec(2, 1, height_ratios=[0.42, 0.58], left=0.08, right=0.95, top=0.89, bottom=0.10)
    summary_axis = figure.add_subplot(grid[0])
    score_axis = figure.add_subplot(grid[1])

    figure.suptitle(title, x=0.08, y=0.955, ha="left", fontsize=20, fontweight="bold", color="#17365D")

    summary_axis.axis("off")
    summary_axis.text(0.0, 0.98, "Run summary", fontsize=12, fontweight="bold", color="#175CD3", va="top")
    for row_index, (label, value) in enumerate(_classification_summary(detected, ground_truth)):
        y_position = 0.78 - row_index * 0.13
        summary_axis.text(0.0, y_position, label, fontsize=9, color="#667085", va="center")
        summary_axis.text(0.32, y_position, value, fontsize=10, color="#101828", fontweight="bold", va="center")

    score_axis.plot(x_values, scores, color="#475467", linewidth=1.2, label="MAE anomaly score")
    if ground_truth is not None and ground_truth.any():
        score_axis.scatter(
            np.asarray(x_values)[ground_truth],
            scores[ground_truth],
            color="#175CD3",
            marker="o",
            s=46,
            linewidth=0,
            label="Ground truth",
            zorder=3,
        )
    if detected.any():
        score_axis.scatter(
            np.asarray(x_values)[detected],
            scores[detected],
            color="#D92D20",
            marker="o",
            s=24,
            linewidth=0,
            label="Detected anomaly",
            zorder=4,
        )
    score_axis.set_title("Anomaly score", loc="left", fontsize=11, fontweight="bold", color="#101828")
    score_axis.set_xlabel(x_label, fontsize=9)
    score_axis.set_ylabel("MAE", fontsize=9)
    _style_axis(score_axis, is_datetime=is_datetime)
    score_axis.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0,
        frameon=False,
        fontsize=8,
        ncols=3,
    )
    _add_footer(figure, 1)
    return figure


def _create_feature_page(
    result_df: pd.DataFrame,
    feature_columns: Sequence[Hashable],
    x_values,
    detected: np.ndarray,
    ground_truth: np.ndarray | None,
    *,
    x_label: str,
    is_datetime: bool,
    page_number: int,
    feature_page_number: int,
    feature_page_count: int,
) -> Figure:
    figure = Figure(figsize=(11.0, 8.5), facecolor="white")
    axes = figure.subplots(len(feature_columns), 1, sharex=True, squeeze=False)
    figure.subplots_adjust(left=0.09, right=0.96, top=0.90, bottom=0.10, hspace=0.42)
    figure.suptitle(
        f"Original signals with anomaly overlays ({feature_page_number}/{feature_page_count})",
        x=0.09,
        y=0.955,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#17365D",
    )

    for axis, feature in zip(axes[:, 0], feature_columns, strict=True):
        values = pd.to_numeric(result_df[feature], errors="coerce").to_numpy(dtype=float)
        axis.plot(x_values, values, color="#344054", linewidth=1.0, label="Original signal")
        if ground_truth is not None and ground_truth.any():
            axis.scatter(
                np.asarray(x_values)[ground_truth],
                values[ground_truth],
                color="#175CD3",
                marker="o",
                s=42,
                linewidth=0,
                label="Ground truth",
                zorder=3,
            )
        if detected.any():
            axis.scatter(
                np.asarray(x_values)[detected],
                values[detected],
                color="#D92D20",
                marker="o",
                s=22,
                linewidth=0,
                label="Detected anomaly",
                zorder=4,
            )
        axis.set_title(str(feature), loc="left", fontsize=10, fontweight="bold", color="#101828")
        axis.set_ylabel("Value", fontsize=8)
        _style_axis(axis, is_datetime=is_datetime)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.94),
        frameon=False,
        fontsize=8,
        ncols=3,
    )
    axes[-1, 0].set_xlabel(x_label, fontsize=9)
    _add_footer(figure, page_number)
    return figure


def _create_mae_note_page(*, page_number: int) -> Figure:
    figure = Figure(figsize=(11.0, 8.5), facecolor="white")
    axis = figure.add_axes((0.09, 0.12, 0.82, 0.76))
    axis.axis("off")

    axis.text(0.0, 0.96, "About the MAE anomaly score", fontsize=18, fontweight="bold", color="#17365D", va="top")
    axis.text(
        0.0,
        0.83,
        "MAE means mean absolute error. In this report, it measures the average absolute difference "
        "between the observed values and NV-Tesseract's reconstructed values at each sample.",
        fontsize=11,
        color="#344054",
        va="top",
        wrap=True,
    )
    axis.text(
        0.0,
        0.65,
        r"$\mathrm{MAE}(t)=\frac{1}{F}\sum_{f=1}^{F}|x_{t,f}-\hat{x}_{t,f}|$",
        fontsize=16,
        color="#101828",
        va="top",
    )
    notes = [
        "A larger MAE means the detector reconstructed that sample less accurately.",
        "The anomaly flag is produced by applying the selected threshold strategy to the MAE sequence.",
        "MAE scale depends on preprocessing and feature scaling, so values should be interpreted in the context of the same model and data pipeline.",
        "A high MAE identifies unusual reconstruction behavior; by itself, it does not establish physical causality or root cause.",
    ]
    axis.text(0.0, 0.48, "Interpretation notes", fontsize=12, fontweight="bold", color="#175CD3", va="top")
    for note_index, note in enumerate(notes):
        y_position = 0.39 - note_index * 0.10
        axis.text(0.0, y_position, "-", fontsize=12, color="#175CD3", va="top")
        axis.text(0.035, y_position, note, fontsize=10, color="#344054", va="top", wrap=True)

    _add_footer(figure, page_number)
    return figure


def generate_anomaly_detection_report(
    result_df: pd.DataFrame,
    output_path: str | Path,
    *,
    feature_columns: Sequence[Hashable] | None = None,
    timestamp_column: str | None = None,
    ground_truth_column: str | None = None,
    anomaly_column: str = "Anomaly",
    score_column: str = "MAE",
    title: str = "Anomaly Detection Report",
    max_features_per_page: int = 4,
) -> Path:
    """Create a PDF report from an anomaly-analysis result DataFrame.

    The report contains a score overview followed by original-signal plots with
    detected anomalies and, when available, ground-truth anomalies overlaid.
    Timestamp and ground-truth columns are inferred from conventional names when
    they are not supplied explicitly.
    """
    if result_df.empty:
        raise ValueError("Cannot generate an anomaly report from an empty DataFrame.")
    if anomaly_column not in result_df.columns or score_column not in result_df.columns:
        raise ValueError(f"Report data must contain {anomaly_column!r} and {score_column!r} columns.")
    if not isinstance(max_features_per_page, int) or isinstance(max_features_per_page, bool):
        raise TypeError("max_features_per_page must be an integer.")
    if max_features_per_page < 1:
        raise ValueError("max_features_per_page must be at least 1.")

    resolved_timestamp = _resolve_column(
        result_df,
        timestamp_column or infer_timestamp_column(result_df),
        role="Timestamp",
    )
    resolved_ground_truth = _resolve_column(
        result_df,
        ground_truth_column or infer_ground_truth_column(result_df),
        role="Ground-truth",
    )
    resolved_features = _resolve_feature_columns(
        result_df,
        feature_columns,
        timestamp_column=resolved_timestamp,
        ground_truth_column=resolved_ground_truth,
        anomaly_column=anomaly_column,
        score_column=score_column,
    )

    detected = _as_boolean_mask(result_df[anomaly_column], name=anomaly_column)
    ground_truth = (
        _as_boolean_mask(result_df[resolved_ground_truth], name=resolved_ground_truth)
        if resolved_ground_truth
        else None
    )
    scores = pd.to_numeric(result_df[score_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_column} must contain only finite numeric values.")
    x_values, x_label, is_datetime = _resolve_x_axis(result_df, resolved_timestamp)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    feature_groups = [
        resolved_features[index : index + max_features_per_page]
        for index in range(0, len(resolved_features), max_features_per_page)
    ]
    page_count = 2 + len(feature_groups)

    with PdfPages(destination, metadata={"Title": title, "Subject": "Time-series anomaly detection report"}) as pdf:
        pdf.savefig(
            _create_overview_page(
                x_values,
                scores,
                detected,
                ground_truth,
                title=title,
                x_label=x_label,
                is_datetime=is_datetime,
            )
        )
        for feature_page_number, group in enumerate(feature_groups, start=1):
            page_number = feature_page_number + 1
            pdf.savefig(
                _create_feature_page(
                    result_df,
                    group,
                    x_values,
                    detected,
                    ground_truth,
                    x_label=x_label,
                    is_datetime=is_datetime,
                    page_number=page_number,
                    feature_page_number=feature_page_number,
                    feature_page_count=len(feature_groups),
                )
            )
        pdf.savefig(_create_mae_note_page(page_number=page_count))
    return destination
