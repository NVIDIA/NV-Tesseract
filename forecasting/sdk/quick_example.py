#!/usr/bin/env python
"""
Quick example showing how to load a CSV file and perform forecasting
"""

import os
from pathlib import Path

import pandas as pd

from forecasting.sdk.forecasting import perform_forecasting

# Load your CSV file
csv_path = (
    Path(__file__).resolve().parent / "tests" / "datasets" / "ETTh_single_feature.csv"
)  # Replace with your file path
if __name__ == "__main__":
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV file not found at {csv_path}, please supply your own data.")
        
    # Replace with your own paths for the weights and standardizer
    ckpt = "moment_head_512_6hr.pt"
    standardizer_pkl = "standardizer.pkl"
    df = pd.read_csv(csv_path)
    timestamp_col = "timestamp"
    target_col = "LULL"
    seq_len = 512
    forecast_horizon = 100

    # Standard forecasting (no external memory)
    forecast_df = perform_forecasting(
        ckpt=ckpt,
        standardizer_pkl=standardizer_pkl,
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
        ckpt=ckpt,
        standardizer_pkl=standardizer_pkl,
        df=df,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        context_df=context_df,
        timestamp_column=timestamp_col,
        target_column=target_col,
        save_preds="forecast_ETTh_darr_100.csv",
    )
    print(f"\nDARR forecast (hybrid prediction in '{target_col}_forecast' column):")
    print(darr_df.to_csv())
