#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick example showing how to load a CSV file and perform forecasting.

This example demonstrates:
1. Auto-downloading model weights from Hugging Face (on first run)
2. Standard forecasting mode
3. DARR mode (context-enhanced forecasting)
4. Interpretability mode (temporal lag x horizon attributions plus, for
   multivariate inputs, feature-axis channel x horizon attributions; JSON + PDF)

Make sure you're authenticated with Hugging Face for the private repo:
    huggingface-cli login
"""

import os
from pathlib import Path

import pandas as pd

from sdk.forecasting import perform_forecasting

# Load your CSV file
csv_path = (
    Path(__file__).resolve().parent / "tests" / "datasets" / "ETTh_single_feature.csv"
)  # Replace with your file path

if __name__ == "__main__":
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        print("Please supply your own data with 'timestamp' and target columns.")
        print("Example CSV format:")
        print("timestamp,target")
        print("2024-01-01 00:00:00,1.5")
        print("2024-01-01 01:00:00,1.7")
        print("...")
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
    print(f"\nStandard forecast (only predicted rows with '{target_col}_forecast' column):")
    print(forecast_df.to_csv())

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
    print(f"\nDARR forecast (hybrid prediction in '{target_col}_forecast' column):")
    print(darr_df.to_csv())

    # Interpretability mode: produces a lag x horizon attribution heatmap and a
    # multi-page PDF report alongside the forecast. Set ``interpretability=True``
    # and choose the output format via ``interpretability_output``:
    #   - "json" -> writes only forecast.csv + explanation.json
    #   - "pdf"  -> writes forecast.csv + lag_horizon_*.csv/png + explanation_report.pdf
    #   - None   -> writes both (full bundle)
    #
    # Feature axis: when the input has more than one numeric column (a
    # multivariate frame), the SDK also decomposes the forecast along the
    # *feature axis*. The report gains a channel x horizon attribution page
    # (which input channel drives each forecast step) and writes
    # channel_horizon_attributions.csv / channel_horizon_heatmap.png. Below we
    # derive two extra channels from the target so the feature-axis page is
    # produced; swap in your own feature columns for a real multivariate signal.
    # (By default the feature-axis attribution is measured in latent space; pass
    # channel_output_aware=True to attribute the target channel's forecast
    # directly instead, at the cost of extra forward passes.)
    multivariate_df = df.copy()
    multivariate_df[f"{target_col}_diff"] = multivariate_df[target_col].diff().fillna(0.0)
    multivariate_df[f"{target_col}_roll12"] = (
        multivariate_df[target_col].rolling(window=12, min_periods=1).mean()
    )

    interp_out_dir = Path(__file__).resolve().parent / "interpretability_output"
    interp_df = perform_forecasting(
        df=multivariate_df,
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
        save_preds="forecast_ETTh_with_explanations.csv",
    )
    print(f"\nInterpretability forecast (single-window baseline in '{target_col}_forecast' column):")
    print(interp_df.head().to_string(index=False))
    print(f"\nReport bundle written under: {interp_out_dir}")
    print("Multivariate input -> the report includes a feature-axis (channel x horizon) page.")
