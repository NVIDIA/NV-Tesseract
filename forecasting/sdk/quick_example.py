#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick example showing how to load a CSV file and perform forecasting.

This example demonstrates:
1. Auto-downloading model weights from Hugging Face (on first run)
2. Standard forecasting mode
3. DARR mode (context-enhanced forecasting)
4. Interpretability mode (lag and feature attributions, JSON + PDF report)

Model weights download automatically from the public Hugging Face repo on first run.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from sdk.forecasting import perform_forecasting

logger = logging.getLogger(__name__)

# Load your CSV file
csv_path = (
    Path(__file__).resolve().parent / "tests" / "datasets" / "ETTh_single_feature.csv"
)  # Replace with your file path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(csv_path):
        logger.error("CSV file not found at %s", csv_path)
        logger.error("Please supply your own data with 'timestamp' and target columns.")
        logger.error("Example CSV format:")
        logger.error("timestamp,target")
        logger.error("2024-01-01 00:00:00,1.5")
        logger.error("2024-01-01 01:00:00,1.7")
        logger.error("...")
        raise SystemExit("Data file required")

    # Model weights will be auto-downloaded from Hugging Face on first run
    # You can also specify custom paths if you have the files locally
    df = pd.read_csv(csv_path)
    timestamp_col = "timestamp"
    target_col = "LULL"
    seq_len = 512
    forecast_horizon = 100

    # Standard forecasting (no external memory)
    # Model weights will be auto-downloaded if not present
    forecast_df = perform_forecasting(
        df=df,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        timestamp_column=timestamp_col,
        target_column=target_col,
        save_preds="forecast_ETTh_seq_len_100.csv",
    )
    logger.info(
        "\nStandard forecast (only predicted rows with '%s_forecast' column):\n%s", target_col, forecast_df.to_csv()
    )

    # DARR mode requires a user-provided context dataset that mirrors the input schema
    # but contains different data (e.g., historical slices, previously predicted values, etc.).
    # This example loads one such CSV to demonstrate how to run DARR.
    context_csv_path = Path(__file__).resolve().parent / "tests" / "datasets" / "ETTh_single_feature_darr_context.csv"
    if not context_csv_path.exists():
        raise SystemExit(f"Context CSV not found at {context_csv_path}, please provide your own for DARR mode.")

    context_df = pd.read_csv(context_csv_path)
    darr_df = perform_forecasting(
        df=df,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        context_df=context_df,  # This enables DARR mode
        timestamp_column=timestamp_col,
        target_column=target_col,
        save_preds="forecast_ETTh_darr_100.csv",
        # Model weights auto-downloaded if needed
    )
    logger.info("\nDARR forecast (hybrid prediction in '%s_forecast' column):\n%s", target_col, darr_df.to_csv())

    # Interpretability mode: produces a lag x horizon attribution heatmap and a
    # multi-page PDF report alongside the forecast. Set ``interpretability=True``
    # and choose the output format via ``interpretability_output``:
    #   - "json" -> writes only forecast.csv + explanation.json
    #   - "pdf"  -> writes forecast.csv + lag_horizon_*.csv/png + explanation_report.pdf
    #   - None   -> writes both (full bundle)
    interp_out_dir = Path(__file__).resolve().parent / "interpretability_output"
    # Add two derived channels so the feature-attribution and feature-axis
    # embedding-stability pages are included. The run also writes the complete
    # feature_axis_embedding_stability.csv report. Replace these with the actual
    # feature columns used by your model in production.
    df_interp = df.copy()
    df_interp[f"{target_col}_diff"] = df_interp[target_col].diff().fillna(0.0)
    df_interp[f"{target_col}_roll12"] = df_interp[target_col].rolling(window=12, min_periods=1).mean()

    interp_df = perform_forecasting(
        df=df_interp,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        timestamp_column=timestamp_col,
        target_column=target_col,
        # Interpretability controls
        interpretability=True,
        interpretability_output=None,  # write both JSON and PDF
        interpretability_out_dir=interp_out_dir,
        interpretability_dataset_name=csv_path.name,
        n_lags=128,
        softmax_tau=1.0,
        integrated_gradients=True,
        save_preds="forecast_ETTh_with_explanations.csv",
    )
    logger.info(
        "\nInterpretability forecast (single-window baseline in '%s_forecast' column):\n%s",
        target_col,
        interp_df.head().to_string(index=False),
    )
    logger.info("\nReport bundle written under: %s", interp_out_dir)
    logger.info("Multivariate input includes feature-axis attribution and stability pages.")
