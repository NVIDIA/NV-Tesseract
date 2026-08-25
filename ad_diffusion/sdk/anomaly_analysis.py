# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdk.explainability import explain_reconstruction_anomalies
from sdk.inference_ad import (
    _resolve_model_paths,
    get_model_target_dim,
    inference_ad_tesseract2_mp,
)
from sdk.reporting import (
    generate_anomaly_detection_report,
    infer_ground_truth_column,
    infer_timestamp_column,
)
from sdk.thresholds import MACSThresholdStrategy, SCSThresholdStrategy

# Set up logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ADDiffusionConfig:
    """Explainability and reporting configuration for :func:`perform_anomaly_analysis_with_diffusion`.

    Scoped to the explain toggle and explainability/report-related parameters —
    inference-behavior knobs like ``threshold_strategy``, ``nsample``, and the
    model/config paths stay as direct function arguments.
    """

    explain: bool = False
    explanation_top_k: int = 3
    report_path: str | Path | None = None
    timestamp_column: str | None = None
    ground_truth_column: str | None = None
    report_title: str = "Anomaly Detection Report"
    report_explanation_csv_path: str | Path | None = None
    report_max_pages: int = 10
    report_consolidated_top_k: int = 5


_SDK_CONFIG_FIELDS = frozenset(field.name for field in fields(ADDiffusionConfig))


def load_sdk_config(config_path: str | Path) -> ADDiffusionConfig:
    """Load and validate an :class:`ADDiffusionConfig` from YAML.

    The YAML may be a flat mapping or contain a top-level ``sdk`` mapping,
    which allows the file to carry other sections (e.g. model config) alongside it.
    """
    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    if not isinstance(raw_config, dict):
        raise TypeError(f"SDK config must be a mapping, got {type(raw_config).__name__}.")

    config = raw_config.get("sdk", raw_config)
    if config is None:
        return ADDiffusionConfig()
    if not isinstance(config, dict):
        raise TypeError(f"SDK config 'sdk' section must be a mapping, got {type(config).__name__}.")

    unknown_keys = sorted(set(config) - _SDK_CONFIG_FIELDS)
    if unknown_keys:
        allowed = ", ".join(sorted(_SDK_CONFIG_FIELDS))
        raise ValueError(f"Unknown SDK config keys: {unknown_keys}. Allowed keys: {allowed}")

    return ADDiffusionConfig(**config)


def _resolve_sdk_config(config: ADDiffusionConfig | str | Path | None) -> ADDiffusionConfig:
    if config is None:
        return ADDiffusionConfig()
    if isinstance(config, ADDiffusionConfig):
        return config
    if isinstance(config, str | Path):
        return load_sdk_config(config)
    raise TypeError("sdk_config must be an ADDiffusionConfig, YAML path, or None")


def perform_anomaly_analysis_with_diffusion(
    df: pd.DataFrame,
    *,
    threshold_strategy: str,
    model_path: str | Path | None = None,
    model_config_path: str | Path = "",
    nsample: int = 15,
    preprocess_model_dir: str | Path | None = None,
    sdk_config: ADDiffusionConfig | str | Path | None = None,
) -> pd.DataFrame:
    """
    Perform anomaly analysis using Tesseract AD Diffusion Model.

    If ``model_path``/``model_config_path`` do not exist locally, the default weights
    (``final_model.pth`` + ``curriculum_medium.yaml``) are automatically
    downloaded from the Hugging Face repository
    ``nvidia/nv-tesseract-ad-diffusion``.

    Args:
        df: DataFrame containing numeric data
        threshold_strategy: Strategy to use for threshold calculation ('scs' or 'macs')
        model_path: Path to the diffusion model checkpoint. If ``None`` or missing
            locally, the default checkpoint is downloaded from Hugging Face.
        model_config_path: Path to the model architecture config file (optional if config is in checkpoint)
        nsample: Number of samples for diffusion model inference
        preprocess_model_dir: Directory containing preprocessing model (optional)
        sdk_config: Explainability/reporting `ADDiffusionConfig`, YAML path, or `None` for defaults.
            See `ADDiffusionConfig` for the field reference.

    Returns:
        DataFrame with original data and anomaly detection results
    """
    cfg = _resolve_sdk_config(sdk_config)

    # Prepare data for diffusion model
    # The diffusion model expects all numeric columns
    resolved_timestamp_column = None
    resolved_ground_truth_column = None
    metadata_columns = []
    if cfg.report_path is not None:
        resolved_timestamp_column = cfg.timestamp_column or infer_timestamp_column(df)
        resolved_ground_truth_column = cfg.ground_truth_column or infer_ground_truth_column(df)
        metadata_columns = [
            column for column in (resolved_timestamp_column, resolved_ground_truth_column) if column is not None
        ]
        missing_metadata = [column for column in metadata_columns if column not in df.columns]
        if missing_metadata:
            raise ValueError(f"Report metadata columns were not found: {missing_metadata}.")

    input_df = df.drop(columns=metadata_columns).copy()
    if input_df.empty and len(input_df.columns) == 0:
        raise ValueError("No feature columns remain after excluding report metadata.")

    # Validate all columns are numeric by attempting to convert the entire DataFrame
    original_columns = input_df.columns.tolist()

    # Try to convert all columns to numeric at once
    numeric_df = input_df.apply(pd.to_numeric, errors="coerce")

    # Check which columns introduced NaNs (indicating non-numeric values)
    non_numeric_cols = []
    for col in original_columns:
        original_na_count = input_df[col].isna().sum()
        converted_na_count = numeric_df[col].isna().sum()

        if converted_na_count > original_na_count:
            non_numeric_cols.append(col)
        else:
            # Update the original dataframe with successfully converted column
            input_df[col] = numeric_df[col]

    if non_numeric_cols:
        raise ValueError(
            f"The following columns contain non-numeric values: {non_numeric_cols}. "
            f"All input values must be numeric for anomaly detection."
        )

    # Resolve / auto-download weights once up front so downstream calls share them.
    resolved_model, resolved_config = _resolve_model_paths(
        str(model_path) if model_path else None,
        str(model_config_path) if model_config_path else "",
    )

    # Get target_dim from model and validate data size BEFORE running inference
    target_dim = get_model_target_dim(resolved_model, resolved_config)
    n_samples = len(input_df)

    if n_samples < target_dim:
        raise ValueError(
            f"Insufficient data samples for PCA: got {n_samples} samples but model requires "
            f"at least {target_dim} samples (target_dim={target_dim}). "
            f"Please provide more data."
        )

    # Run inference with diffusion model (auto-uses multi-GPU when available)
    results = inference_ad_tesseract2_mp(
        data=input_df,
        model_path=resolved_model,
        config_path=resolved_config,
        nsample=nsample,
        preprocess_model_dir=str(preprocess_model_dir) if preprocess_model_dir else None,
    )

    # Extract residual scores (MAE) from results
    residual_scores = results["residual"]

    # Get target data for advanced thresholding methods
    target_data = results["target"]

    # Model evaluation works on fixed-size windows and may append padded rows to
    # the final window. Align outputs before threshold calibration so synthetic
    # padding cannot change the thresholds used for real input rows.
    original_length = len(df)
    if len(residual_scores) != original_length:
        logger.info(f"Aligning lengths: residual_scores={len(residual_scores)}, original_data={original_length}")
        residual_scores = residual_scores[:original_length]
    if len(target_data) != original_length:
        target_data = target_data[:original_length]

    # Apply thresholding strategy
    if threshold_strategy == "scs":
        # Use actual target data from the model results
        anomalies = SCSThresholdStrategy().scs_thresholder.detect_anomalies(residual_scores, target_data)
    elif threshold_strategy == "macs":
        # Use actual target data from the model results
        anomalies = MACSThresholdStrategy().macs_thresholder.detect_anomalies(residual_scores, target_data)
    else:
        raise ValueError(f"Unknown threshold strategy: {threshold_strategy}")

    # Create result dataframe. A thresholder may return a longer mask than the
    # score array, so keep assignment aligned with the input rows as well.
    result_df = df.copy()
    anomalies = anomalies[:original_length]

    result_df["Anomaly"] = anomalies
    result_df["MAE"] = residual_scores  # Using residual (MAE) as anomaly score

    if cfg.explain:
        reconstruction = results.get("recon")
        if reconstruction is None:
            raise ValueError("Inference results must include 'recon' when explain=True.")

        explanation_length = min(original_length, len(target_data), len(reconstruction), len(anomalies))
        target_for_explanation = target_data[:explanation_length]
        reconstruction_for_explanation = reconstruction[:explanation_length]
        target_feature_count = target_for_explanation.shape[1] if target_for_explanation.ndim > 1 else 1
        input_feature_count = len(input_df.columns)
        directly_mapped = preprocess_model_dir is None and input_feature_count <= target_feature_count
        feature_names = list(input_df.columns) if directly_mapped else None
        feature_indices = list(range(input_feature_count)) if directly_mapped else None
        explanations = explain_reconstruction_anomalies(
            target_for_explanation,
            reconstruction_for_explanation,
            anomaly_mask=anomalies[:explanation_length],
            feature_names=feature_names,
            feature_indices=feature_indices,
            top_k=cfg.explanation_top_k,
        )
        explanations = explanations.reindex(range(original_length)).fillna(
            {
                "TopContributors": "[]",
                "ContributionShares": "[]",
                "ExplanationCoverage": 0.0,
                "ExplanationMethod": "",
            }
        )
        for column in explanations.columns:
            result_df[column] = explanations[column].to_numpy()

    if cfg.report_path is not None:
        generate_anomaly_detection_report(
            result_df,
            cfg.report_path,
            feature_columns=list(input_df.columns),
            timestamp_column=resolved_timestamp_column,
            ground_truth_column=resolved_ground_truth_column,
            title=cfg.report_title,
            explanation_csv_path=cfg.report_explanation_csv_path,
            max_report_pages=cfg.report_max_pages,
            consolidated_top_k=cfg.report_consolidated_top_k,
        )

    return result_df
