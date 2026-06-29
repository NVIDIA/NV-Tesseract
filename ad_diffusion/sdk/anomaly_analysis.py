# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path  # noqa: TC003

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdk.inference_ad import (
    _resolve_model_paths,
    get_model_target_dim,
    inference_ad_tesseract2_mp,
)
from sdk.thresholds import MACSThresholdStrategy, SCSThresholdStrategy

# Set up logging
logger = logging.getLogger(__name__)


def perform_anomaly_analysis_with_diffusion(
    df: pd.DataFrame,
    *,
    threshold_strategy: str,
    model_path: str | Path | None = None,
    config_path: str | Path = "",
    nsample: int = 15,
    preprocess_model_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Perform anomaly analysis using Tesseract AD Diffusion Model.

    If ``model_path``/``config_path`` do not exist locally, the default weights
    (``final_model.pth`` + ``curriculum_medium.yaml``) are automatically
    downloaded from the Hugging Face repository
    ``nvidia/nv-tesseract-ad-diffusion``.

    Args:
        df: DataFrame containing numeric data
        threshold_strategy: Strategy to use for threshold calculation ('scs' or 'macs')
        model_path: Path to the diffusion model checkpoint. If ``None`` or missing
            locally, the default checkpoint is downloaded from Hugging Face.
        config_path: Path to the model config file (optional if config is in checkpoint)
        nsample: Number of samples for diffusion model inference
        preprocess_model_dir: Directory containing preprocessing model (optional)

    Returns:
        DataFrame with original data and anomaly detection results
    """
    # Prepare data for diffusion model
    # The diffusion model expects all numeric columns
    input_df = df.copy()

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
        str(config_path) if config_path else "",
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

    return result_df
