# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDF reporting for time-series anomaly detection results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

GROUND_TRUTH_CANDIDATES = ("GT", "ground_truth", "is_anomaly", "anomaly", "label", "Label")
TIMESTAMP_CANDIDATES = ("timestamp", "Timestamp", "time", "Time", "ts", "datetime", "date")
EXPLANATION_COLUMNS = {
    "TopContributors",
    "ContributionShares",
    "ExplanationCoverage",
    "ExplanationMethod",
    "LikelyReconstructionIssue",
    "RepairImpact",
    "DiagnosticConfidence",
    "ExplanationConsistency",
}
REQUIRED_EXPLANATION_COLUMNS = ("TopContributors", "ContributionShares", "ExplanationCoverage")
REQUIRED_DIAGNOSIS_COLUMNS = (
    "LikelyReconstructionIssue",
    "RepairImpact",
    "DiagnosticConfidence",
    "ExplanationConsistency",
)
MAX_EXPLANATIONS_PER_PAGE = 7
DEFAULT_MAX_REPORT_PAGES = 10
DEFAULT_CONSOLIDATED_TOP_K = 5
FEATURE_COLORS = ("#175CD3", "#039855", "#F79009", "#7F56D9", "#0E9384", "#D92D20", "#444CE7", "#C11574")
SCORE_DIRECTION_NOTE = (
    "Lower MAE means a closer reconstruction\nand more typical behavior.\n"
    "Higher MAE means a larger mismatch\nand more anomalous behavior."
)


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
        return np.arange(len(df)), "Row number (no timestamp column)", False

    values = df[timestamp_column]
    if pd.api.types.is_numeric_dtype(values):
        return values, timestamp_column, False

    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().all():
        return parsed, timestamp_column, True
    return np.arange(len(df)), f"Row number ({timestamp_column} unavailable as time)", False


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


def _parse_explanation_list(value, *, name: str) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} must contain a valid JSON list.") from error
    if not isinstance(value, list | tuple | np.ndarray):
        raise TypeError(f"{name} must contain a list for each detected anomaly.")
    return list(value)


def _format_explanation_sample(value, *, is_datetime: bool) -> str:
    if is_datetime:
        return pd.Timestamp(value).strftime("%Y-%m-%d\n%H:%M:%S")
    return str(value)


def _resolve_explanation_rows(
    result_df: pd.DataFrame,
    detected: np.ndarray,
    x_values,
    *,
    is_datetime: bool,
) -> list[tuple[str, str, str, str]] | None:
    present = [column for column in REQUIRED_EXPLANATION_COLUMNS if column in result_df.columns]
    if not present:
        return None
    missing = [column for column in REQUIRED_EXPLANATION_COLUMNS if column not in result_df.columns]
    if missing:
        raise ValueError(f"Explainability report data is missing columns: {missing}.")

    rows = []
    x_array = np.asarray(x_values)
    for row_index in np.flatnonzero(detected):
        contributors = _parse_explanation_list(result_df.iloc[row_index]["TopContributors"], name="TopContributors")
        shares = _parse_explanation_list(result_df.iloc[row_index]["ContributionShares"], name="ContributionShares")
        if len(contributors) != len(shares):
            raise ValueError("TopContributors and ContributionShares must have matching lengths.")

        numeric_shares = np.asarray(shares, dtype=float)
        if not np.isfinite(numeric_shares).all() or ((numeric_shares < 0) | (numeric_shares > 1)).any():
            raise ValueError("ContributionShares must contain only finite values between 0 and 1.")
        coverage = pd.to_numeric(result_df.iloc[row_index]["ExplanationCoverage"], errors="coerce")
        if not np.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError("ExplanationCoverage must contain only finite values between 0 and 1.")

        rows.append(
            (
                _format_explanation_sample(x_array[row_index], is_datetime=is_datetime),
                "\n".join(map(str, contributors)) or "Not available",
                "\n".join(f"{share:.1%}" for share in numeric_shares) or "Not available",
                f"{coverage:.1%}",
            )
        )
    return rows


def _resolve_diagnosis_rows(
    result_df: pd.DataFrame,
    detected: np.ndarray,
) -> list[tuple[str, str, str, str]] | None:
    """Validate and format the four user-facing reconstruction-diagnosis fields."""
    present = [column for column in REQUIRED_DIAGNOSIS_COLUMNS if column in result_df.columns]
    if not present:
        return None
    missing = [column for column in REQUIRED_DIAGNOSIS_COLUMNS if column not in result_df.columns]
    if missing:
        raise ValueError(f"Reconstruction-diagnosis report data is missing columns: {missing}.")

    rows = []
    for row_index in np.flatnonzero(detected):
        impact = pd.to_numeric(result_df.iloc[row_index]["RepairImpact"], errors="coerce")
        if not np.isfinite(impact) or not 0 <= impact <= 1:
            raise ValueError("RepairImpact must contain only finite values between 0 and 1.")
        issue = str(result_df.iloc[row_index]["LikelyReconstructionIssue"]).strip() or "Not available"
        confidence = str(result_df.iloc[row_index]["DiagnosticConfidence"]).strip() or "Not available"
        consistency = str(result_df.iloc[row_index]["ExplanationConsistency"]).strip() or "Not available"
        rows.append((issue, f"{impact:.1%}", confidence, consistency))
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

    summary_axis.text(
        0.56,
        0.82,
        "How to read the anomaly score",
        fontsize=10,
        fontweight="bold",
        color="#175CD3",
        va="top",
    )
    summary_axis.text(
        0.56,
        0.66,
        SCORE_DIRECTION_NOTE,
        fontsize=8.5,
        color="#344054",
        va="top",
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#F5F8FF", "edgecolor": "#B2CCFF"},
    )

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
    figure.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.10, hspace=0.42)
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


def _create_contribution_graph_page(
    rows: Sequence[tuple[str, str, str, str]],
    *,
    page_number: int,
    graph_page_number: int,
    graph_page_count: int,
    feature_colors: dict[str, str] | None = None,
    diagnosis_rows: Sequence[tuple[str, str, str, str]] | None = None,
) -> Figure:
    figure = Figure(figsize=(11.0, 8.5), facecolor="white")
    axis = figure.add_axes((0.20, 0.14, 0.44 if diagnosis_rows is not None else 0.72, 0.66))
    figure.suptitle(
        f"Feature contributions and diagnosis by anomaly ({graph_page_number}/{graph_page_count})",
        x=0.07,
        y=0.955,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#17365D",
    )
    figure.text(
        0.07,
        0.895,
        "Each segment shows the contributing feature and its share; gray shows error outside the displayed features",
        fontsize=10,
        color="#667085",
    )
    if diagnosis_rows is not None:
        figure.text(0.71, 0.825, "Reconstruction diagnosis", fontsize=8, fontweight="bold", color="#175CD3")
    if not rows:
        axis.axis("off")
        axis.text(
            0.0,
            0.88,
            "No detected anomalies were available to graph.",
            fontsize=11,
            color="#475467",
            va="top",
        )
        _add_footer(figure, page_number)
        return figure

    y_positions = np.arange(
        MAX_EXPLANATIONS_PER_PAGE - 1,
        MAX_EXPLANATIONS_PER_PAGE - len(rows) - 1,
        -1,
    )
    y_labels = []
    if feature_colors is None:
        ordered_features = list(
            dict.fromkeys(
                feature
                for _, contributor_text, _, _ in rows
                for feature in contributor_text.splitlines()
                if contributor_text != "Not available"
            )
        )
        feature_colors = {
            feature: FEATURE_COLORS[index % len(FEATURE_COLORS)] for index, feature in enumerate(ordered_features)
        }
    for y_position, (timestamp, contributor_text, share_text, coverage_text) in zip(y_positions, rows, strict=True):
        contributors = contributor_text.splitlines() if contributor_text != "Not available" else []
        shares = (
            [float(value.removesuffix("%")) / 100 for value in share_text.splitlines()]
            if share_text != "Not available"
            else []
        )
        left = 0.0
        for contributor, share in zip(contributors, shares, strict=True):
            color = feature_colors[contributor]
            axis.barh(y_position, share, left=left, height=0.58, color=color, edgecolor="white", linewidth=0.8)
            if share >= 0.075:
                display_name = contributor if len(contributor) <= 14 else f"{contributor[:12]}..."
                axis.text(
                    left + share / 2,
                    y_position,
                    f"{display_name}\n{share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    fontweight="bold",
                    color="#101828" if color == "#F79009" else "white",
                    linespacing=0.9,
                )
            left += share

        uncovered = max(0.0, 1.0 - left)
        axis.barh(
            y_position,
            uncovered,
            left=left,
            height=0.58,
            color="#D0D5DD",
            edgecolor="white",
            linewidth=0.8,
        )
        axis.text(1.015, y_position, f"{coverage_text} covered", va="center", fontsize=7.5, color="#475467")
        y_labels.append(timestamp.replace("\n", " "))

    if diagnosis_rows is not None:
        for y_position, (issue, impact, confidence, consistency) in zip(y_positions, diagnosis_rows, strict=True):
            normalized_y = 0.14 + 0.66 * ((y_position + 0.75) / MAX_EXPLANATIONS_PER_PAGE)
            figure.text(0.71, normalized_y + 0.018, issue, fontsize=7, fontweight="bold", color="#101828")
            figure.text(
                0.71,
                normalized_y - 0.002,
                f"Impact {impact}  |  Confidence {confidence}",
                fontsize=6.5,
                color="#344054",
            )
            figure.text(
                0.71,
                normalized_y - 0.022,
                f"Consistency {consistency}",
                fontsize=6.5,
                color="#667085",
            )

    axis.set_yticks(y_positions, labels=y_labels)
    axis.set_ylim(-0.75, MAX_EXPLANATIONS_PER_PAGE - 0.25)
    axis.set_xlim(0.0, 1.15)
    axis.set_xticks(np.linspace(0.0, 1.0, 6), labels=[f"{value:.0%}" for value in np.linspace(0.0, 1.0, 6)])
    axis.set_xlabel("Contribution share of total reconstruction error", fontsize=9, color="#344054")
    axis.grid(axis="x", color="#D8DEE9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=7.5, colors="#344054", pad=8)
    axis.tick_params(axis="x", labelsize=8, colors="#667085")
    _add_footer(figure, page_number)
    return figure


def _aggregate_explanation_contributions(
    result_df: pd.DataFrame,
    detected: np.ndarray,
    *,
    score_column: str,
    top_k: int,
) -> tuple[list[tuple[str, float]], float]:
    """Aggregate contributor shares across anomalies, weighted by anomaly MAE."""
    scores = pd.to_numeric(result_df[score_column], errors="coerce").to_numpy(dtype=float)
    total_anomaly_score = float(scores[detected].sum())
    weighted_contributions: dict[str, float] = {}
    for row_index in np.flatnonzero(detected):
        contributors = _parse_explanation_list(result_df.iloc[row_index]["TopContributors"], name="TopContributors")
        shares = _parse_explanation_list(result_df.iloc[row_index]["ContributionShares"], name="ContributionShares")
        for contributor, share in zip(contributors, shares, strict=True):
            feature = str(contributor)
            weighted_contributions[feature] = weighted_contributions.get(feature, 0.0) + scores[row_index] * float(
                share
            )

    ranked = sorted(weighted_contributions.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    if total_anomaly_score <= 0:
        contributions = [(feature, 0.0) for feature, _ in ranked]
    else:
        contributions = [(feature, value / total_anomaly_score) for feature, value in ranked]
    return contributions, sum(share for _, share in contributions)


def _summarize_diagnoses(
    result_df: pd.DataFrame,
    detected: np.ndarray,
) -> list[tuple[str, str]] | None:
    if not all(column in result_df.columns for column in REQUIRED_DIAGNOSIS_COLUMNS):
        return None
    anomalous = result_df.loc[detected]
    if anomalous.empty:
        return [
            ("Most common issue", "Not available"),
            ("Median impact", "0.0%"),
            ("High confidence", "0.0%"),
            ("Consistency", "Not evaluated"),
        ]
    issues = anomalous["LikelyReconstructionIssue"].astype(str)
    most_common_issue = issues.value_counts().index[0]
    median_impact = float(pd.to_numeric(anomalous["RepairImpact"], errors="coerce").median())
    high_confidence = float(anomalous["DiagnosticConfidence"].astype(str).eq("High").mean())
    consistency_values = anomalous["ExplanationConsistency"].astype(str)
    numeric_consistency = pd.to_numeric(consistency_values.str.removesuffix("%"), errors="coerce")
    consistency = (
        f"{float(numeric_consistency.median()):.0f}%" if numeric_consistency.notna().any() else "Not evaluated"
    )
    return [
        ("Most common issue", most_common_issue),
        ("Median impact", f"{median_impact:.1%}"),
        ("High confidence", f"{high_confidence:.1%}"),
        ("Consistency", consistency),
    ]


def _create_consolidated_contribution_page(
    contributions: Sequence[tuple[str, float]],
    *,
    coverage: float,
    anomaly_count: int,
    page_number: int,
    max_report_pages: int,
    diagnosis_summary: Sequence[tuple[str, str]] | None = None,
) -> Figure:
    """Create one page summarizing the strongest contributors across all anomalies."""
    figure = Figure(figsize=(11.0, 8.5), facecolor="white")
    axis = figure.add_axes((0.25, 0.20, 0.66, 0.55))
    figure.suptitle(
        "Overall anomaly feature contributions",
        x=0.07,
        y=0.955,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#17365D",
    )
    figure.text(
        0.07,
        0.895,
        f"Top {len(contributions)} contributors across {anomaly_count:,} detected anomalies, weighted by anomaly MAE.",
        fontsize=10,
        color="#667085",
    )
    figure.text(
        0.07,
        0.855,
        f"Per-anomaly charts were consolidated to keep this report within {max_report_pages} pages; full details are in the companion CSV.",
        fontsize=9,
        color="#667085",
    )
    if not contributions:
        axis.axis("off")
        axis.text(0.0, 0.8, "No contributor data was available to summarize.", fontsize=11, color="#475467")
        _add_footer(figure, page_number)
        return figure

    features = [feature for feature, _ in contributions][::-1]
    shares = np.asarray([share for _, share in contributions][::-1], dtype=float)
    colors = [FEATURE_COLORS[index % len(FEATURE_COLORS)] for index in range(len(features))]
    bars = axis.barh(np.arange(len(features)), shares, color=colors, height=0.62)
    axis.set_yticks(np.arange(len(features)), labels=features)
    axis.set_xlim(0.0, max(0.05, min(1.0, float(shares.max()) * 1.22)))
    axis.xaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    axis.set_xlabel("Share of total detected-anomaly MAE", fontsize=9, color="#344054")
    axis.grid(axis="x", color="#D8DEE9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=8, colors="#344054", pad=8)
    axis.tick_params(axis="x", labelsize=8, colors="#667085")
    for bar, share in zip(bars, shares, strict=True):
        axis.text(
            bar.get_width() + axis.get_xlim()[1] * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{share:.1%}",
            va="center",
            fontsize=8,
            color="#344054",
        )
    figure.text(
        0.25,
        0.125,
        f"Displayed features explain {coverage:.1%} of total detected-anomaly MAE; the remainder is outside the displayed contributors.",
        fontsize=9,
        color="#475467",
    )
    if diagnosis_summary:
        figure.text(0.07, 0.075, "Reconstruction diagnosis", fontsize=9, fontweight="bold", color="#175CD3")
        for index, (label, value) in enumerate(diagnosis_summary):
            x_position = 0.27 + index * 0.17
            figure.text(x_position, 0.077, label, fontsize=7, color="#667085")
            figure.text(x_position, 0.052, value, fontsize=8, fontweight="bold", color="#101828")
    _add_footer(figure, page_number)
    return figure


def _write_explanation_csv(
    result_df: pd.DataFrame,
    detected: np.ndarray,
    x_values,
    output_path: Path,
    *,
    is_datetime: bool,
) -> Path:
    columns = [
        "Anomalous timestamp",
        "Top contributors",
        "Contribution shares",
        "Explanation coverage",
    ]
    include_diagnosis = all(column in result_df.columns for column in REQUIRED_DIAGNOSIS_COLUMNS)
    if include_diagnosis:
        columns.extend(
            ["Likely reconstruction issue", "Repair impact", "Diagnostic confidence", "Explanation consistency"]
        )
    records = []
    x_array = np.asarray(x_values)
    for row_index in np.flatnonzero(detected):
        contributors = _parse_explanation_list(result_df.iloc[row_index]["TopContributors"], name="TopContributors")
        shares = _parse_explanation_list(result_df.iloc[row_index]["ContributionShares"], name="ContributionShares")
        coverage = float(result_df.iloc[row_index]["ExplanationCoverage"])
        record = {
            "Anomalous timestamp": _format_explanation_sample(x_array[row_index], is_datetime=is_datetime).replace(
                "\n", " "
            ),
            "Top contributors": json.dumps([str(value) for value in contributors], separators=(",", ":")),
            "Contribution shares": json.dumps([float(value) for value in shares], separators=(",", ":")),
            "Explanation coverage": coverage,
        }
        if include_diagnosis:
            record.update(
                {
                    "Likely reconstruction issue": result_df.iloc[row_index]["LikelyReconstructionIssue"],
                    "Repair impact": float(result_df.iloc[row_index]["RepairImpact"]),
                    "Diagnostic confidence": result_df.iloc[row_index]["DiagnosticConfidence"],
                    "Explanation consistency": result_df.iloc[row_index]["ExplanationConsistency"],
                }
            )
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records, columns=columns).to_csv(output_path, index=False)
    return output_path


def _create_mae_note_page(
    *,
    page_number: int,
    uses_row_numbers: bool = False,
    include_diagnosis: bool = False,
) -> Figure:
    figure = Figure(figsize=(11.0, 8.5), facecolor="white")
    axis = figure.add_axes((0.07, 0.10, 0.86, 0.80))
    axis.axis("off")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_autoscale_on(False)

    title = "How to interpret this report" if include_diagnosis else "About the MAE anomaly score"
    axis.text(0.0, 0.98, title, fontsize=18, fontweight="bold", color="#17365D", va="top")
    axis.text(
        0.0,
        0.86,
        "MAE means mean absolute error. It is the average absolute\n"
        "difference between observed and reconstructed values at each sample.",
        fontsize=9.5,
        color="#344054",
        va="top",
        wrap=True,
    )
    axis.text(
        0.0,
        0.70,
        r"$\mathrm{MAE}(t)=\frac{1}{F}\sum_{f=1}^{F}|x_{t,f}-\hat{x}_{t,f}|$",
        fontsize=14,
        color="#101828",
        va="top",
    )
    notes = [
        (
            "Lower MAE means a closer reconstruction and more typical\n"
            "behavior. Higher MAE means a larger mismatch and more\n"
            "anomalous behavior."
        ),
        "The anomaly flag is produced by applying the selected\nthreshold strategy to the MAE sequence.",
        "MAE scale depends on preprocessing and feature scaling.\nCompare values within the same model and data pipeline.",
        "High MAE identifies unusual reconstruction behavior. It does\nnot establish physical causality or root cause.",
    ]
    if uses_row_numbers:
        notes.append(
            "No usable timestamp column was provided, so time steps in this report are zero-based DataFrame row numbers."
        )
    axis.text(0.0, 0.57, "MAE interpretation", fontsize=11, fontweight="bold", color="#175CD3", va="top")
    note_positions = [0.49, 0.34, 0.23, 0.12, 0.02]
    for y_position, note in zip(note_positions[: len(notes)], notes, strict=True):
        axis.text(0.0, y_position, "-", fontsize=10, color="#175CD3", va="top")
        axis.text(0.025, y_position, note, fontsize=8.5, color="#344054", va="top", wrap=True)

    if include_diagnosis:
        x_position = 0.53
        axis.plot([0.49, 0.49], [0.05, 0.89], color="#D8DEE9", linewidth=0.8, transform=axis.transAxes)
        axis.text(
            x_position,
            0.86,
            "Reconstruction diagnosis",
            fontsize=11,
            fontweight="bold",
            color="#175CD3",
            va="top",
        )
        axis.text(
            x_position,
            0.79,
            "These metrics describe how the detector responded to four\n"
            "tested repairs. They do not prove physical causality or root cause.",
            fontsize=8.5,
            color="#344054",
            va="top",
            wrap=True,
        )
        axis.text(x_position, 0.69, "Likely issue", fontsize=9, fontweight="bold", color="#101828", va="top")
        axis.text(
            x_position,
            0.645,
            "The pattern whose repair reduced the anomaly score the most.\nPossible values:",
            fontsize=8,
            color="#475467",
            va="top",
        )
        issue_values = [
            "Level shift - sustained upward or downward change.",
            "Trend change - unexpected change in direction or slope.",
            "Transient spike - brief, isolated jump or drop.",
            "Variance burst - temporary increase in fluctuation or noise.",
            "No supported repair - none of the tested repairs reduced the score.",
        ]
        for index, value in enumerate(issue_values):
            y_position = 0.56 - index * 0.045
            axis.text(x_position, y_position, "-", fontsize=8, color="#175CD3", va="top")
            axis.text(x_position + 0.018, y_position, value, fontsize=7.6, color="#344054", va="top")

        axis.text(x_position, 0.31, "Repair impact", fontsize=9, fontweight="bold", color="#101828", va="top")
        axis.text(
            x_position,
            0.265,
            "Percentage decrease in anomaly score after the best repair.\n"
            "Larger values mean a stronger effect on the detector.",
            fontsize=8,
            color="#475467",
            va="top",
        )
        axis.text(x_position, 0.18, "Diagnostic confidence", fontsize=9, fontweight="bold", color="#101828", va="top")
        axis.text(
            x_position,
            0.135,
            "Best-versus-next-best margin: High >=20 points; Moderate\n"
            "10-20; Low <10; Inconclusive means no clear improvement.",
            fontsize=8,
            color="#475467",
            va="top",
        )
        axis.text(x_position, 0.07, "Explanation consistency", fontsize=9, fontweight="bold", color="#101828", va="top")
        axis.text(
            x_position,
            0.025,
            "Agreement across repeated evaluations; Not evaluated means one evaluation was used.",
            fontsize=7.4,
            color="#475467",
            va="top",
        )

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
    explanation_csv_path: str | Path | None = None,
    max_report_pages: int = DEFAULT_MAX_REPORT_PAGES,
    consolidated_top_k: int = DEFAULT_CONSOLIDATED_TOP_K,
) -> Path:
    """Create a PDF report from an anomaly-analysis result DataFrame.

    The report contains a score overview followed by original-signal plots with
    detected anomalies and, when available, ground-truth anomalies overlaid.
    Explainability metrics are plotted in the PDF and exported in full to a
    companion CSV when explanation columns are available.
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
    if not isinstance(max_report_pages, int) or isinstance(max_report_pages, bool):
        raise TypeError("max_report_pages must be an integer.")
    if max_report_pages < 4:
        raise ValueError("max_report_pages must be at least 4.")
    if not isinstance(consolidated_top_k, int) or isinstance(consolidated_top_k, bool):
        raise TypeError("consolidated_top_k must be an integer.")
    if consolidated_top_k < 1:
        raise ValueError("consolidated_top_k must be at least 1.")

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
    explanation_rows = _resolve_explanation_rows(
        result_df,
        detected,
        x_values,
        is_datetime=is_datetime,
    )
    diagnosis_rows = _resolve_diagnosis_rows(result_df, detected)
    if diagnosis_rows is not None and explanation_rows is None:
        raise ValueError("Reconstruction-diagnosis report data requires feature-contribution explanation columns.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if explanation_rows is not None:
        csv_destination = (
            Path(explanation_csv_path)
            if explanation_csv_path is not None
            else destination.with_name(f"{destination.stem}_explanations.csv")
        )
        _write_explanation_csv(
            result_df,
            detected,
            x_values,
            csv_destination,
            is_datetime=is_datetime,
        )
    reserved_pages = 2 + (1 if explanation_rows is not None else 0)
    available_feature_pages = max(1, max_report_pages - reserved_pages)
    effective_features_per_page = max(
        max_features_per_page,
        int(np.ceil(len(resolved_features) / available_feature_pages)),
    )
    feature_groups = [
        resolved_features[index : index + effective_features_per_page]
        for index in range(0, len(resolved_features), effective_features_per_page)
    ]
    explanation_groups = (
        [
            explanation_rows[index : index + MAX_EXPLANATIONS_PER_PAGE]
            for index in range(0, len(explanation_rows), MAX_EXPLANATIONS_PER_PAGE)
        ]
        if explanation_rows
        else ([[]] if explanation_rows is not None else [])
    )
    diagnosis_groups = (
        [
            diagnosis_rows[index : index + MAX_EXPLANATIONS_PER_PAGE]
            for index in range(0, len(diagnosis_rows), MAX_EXPLANATIONS_PER_PAGE)
        ]
        if diagnosis_rows
        else ([[]] if diagnosis_rows is not None else [])
    )
    explanation_features = list(
        dict.fromkeys(
            feature
            for _, contributor_text, _, _ in (explanation_rows or [])
            for feature in contributor_text.splitlines()
            if contributor_text != "Not available"
        )
    )
    explanation_feature_colors = {
        feature: FEATURE_COLORS[index % len(FEATURE_COLORS)] for index, feature in enumerate(explanation_features)
    }
    detailed_page_count = 2 + len(feature_groups) + len(explanation_groups)
    use_consolidated_explanations = explanation_rows is not None and detailed_page_count > max_report_pages
    consolidated_contributions: list[tuple[str, float]] = []
    consolidated_coverage = 0.0
    if use_consolidated_explanations:
        consolidated_contributions, consolidated_coverage = _aggregate_explanation_contributions(
            result_df,
            detected,
            score_column=score_column,
            top_k=consolidated_top_k,
        )
        page_count = 3 + len(feature_groups)
    else:
        page_count = detailed_page_count

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
        if use_consolidated_explanations:
            pdf.savefig(
                _create_consolidated_contribution_page(
                    consolidated_contributions,
                    coverage=consolidated_coverage,
                    anomaly_count=int(detected.sum()),
                    page_number=2 + len(feature_groups),
                    max_report_pages=max_report_pages,
                    diagnosis_summary=_summarize_diagnoses(result_df, detected),
                )
            )
        for graph_page_number, group in enumerate(
            [] if use_consolidated_explanations else explanation_groups,
            start=1,
        ):
            page_number = 1 + len(feature_groups) + graph_page_number
            diagnosis_group = diagnosis_groups[graph_page_number - 1] if diagnosis_groups else None
            pdf.savefig(
                _create_contribution_graph_page(
                    group,
                    page_number=page_number,
                    graph_page_number=graph_page_number,
                    graph_page_count=len(explanation_groups),
                    feature_colors=explanation_feature_colors,
                    diagnosis_rows=diagnosis_group,
                )
            )
        pdf.savefig(
            _create_mae_note_page(
                page_number=page_count,
                uses_row_numbers=x_label.startswith("Row number"),
                include_diagnosis=diagnosis_rows is not None,
            )
        )
    return destination
