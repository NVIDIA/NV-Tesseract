import json
import os

# Import required modules from parent directory
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from forecasting.dataset_longhorizon import (
    CSVLongHorizonSimpleDataset,
    Standardizer,
)
from forecasting.model import build_model
from momentfm.utils.utils import control_randomness

from forecasting import DEVICE


class InferenceOnlyDataset(Dataset):
    """
    Dataset for pure inference when no ground truth is available.
    Only provides input windows, no future values needed.
    """

    def __init__(self, csv_path, seq_len, standardizer):
        # Load data
        df = pd.read_csv(csv_path)

        # Get timestamp and value columns
        self.times = pd.to_datetime(df["timestamp"]).values
        value_cols = [col for col in df.columns if col != "timestamp"]
        self.values = df[value_cols].values.astype(np.float32)

        # Standardize
        self.standardizer = standardizer
        if self.standardizer is not None:
            self.series = self.standardizer.transform(self.values)
        else:
            self.series = self.values.copy()

        self.seq_len = seq_len
        self.n_channels = self.series.shape[1]

        # For inference, we only need one window from the end
        if len(self.series) < seq_len:
            raise ValueError(f"Series has {len(self.series)} points but seq_len requires {seq_len}")

        # Single window from the end
        self._start = len(self.series) - seq_len

    def __len__(self):
        return 1  # Only one window

    def __getitem__(self, idx):
        # Get the last seq_len points
        start = self._start
        end = start + self.seq_len

        # Input window [C, seq_len]
        x = self.series[start:end].T

        # Input mask (all ones - no missing values)
        input_mask = np.ones(self.seq_len, dtype=np.int64)

        # No ground truth available - return dummy
        y_dummy = np.zeros((self.n_channels, 1), dtype=np.float32)

        return (torch.from_numpy(x).float(), torch.from_numpy(y_dummy).float(), torch.from_numpy(input_mask).long())

    def inverse_transform(self, data):
        """Transform predictions back to original scale"""
        if self.standardizer is not None:
            return self.standardizer.inverse(data)
        return data


def json_to_csv(json_data: str | dict | list, csv_path: str) -> str:
    """
    Convert JSON data to CSV format.

    Args:
        json_data: Either a path to JSON file, or the JSON data itself (dict/list)
        csv_path: Path where CSV will be saved

    Returns:
        Path to the created CSV file
    """
    # Load JSON if it's a file path
    if isinstance(json_data, str):
        with open(json_data) as f:
            data = json.load(f)
    else:
        data = json_data

    # Convert to DataFrame
    if isinstance(data, list) or isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        raise ValueError("JSON must be either a list of objects or an object with arrays")

    # Save as CSV
    df.to_csv(csv_path, index=False)
    return csv_path


def l2_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize vectors"""
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, eps)


@torch.no_grad()
def embed_batch(model, x_enc, input_mask):
    """Generate embeddings for a batch"""
    out = model.embed(x_enc=x_enc, input_mask=input_mask)
    if not hasattr(out, "embeddings") or out.embeddings is None:
        raise RuntimeError("model.embed(...) did not return .embeddings")
    return out.embeddings


def nan_safe_weighted_average(weights: np.ndarray, y_neighbors: np.ndarray) -> np.ndarray:
    """
    Compute weighted average handling NaN values

    Args:
        weights: [k] weight array
        y_neighbors: [k, C, H] neighbor predictions

    Returns:
        [C, H] weighted average
    """
    w = weights.astype(np.float32)
    Y = y_neighbors.astype(np.float32)
    valid = np.isfinite(Y)

    # expand weights: [k] -> [k, 1, 1]
    w_exp = w[:, None, None]
    w_mask = np.where(valid, w_exp, 0.0)
    num = np.nansum(w_mask * np.nan_to_num(Y), axis=0)
    den = np.sum(w_mask, axis=0)
    return num / np.maximum(den, 1e-12)


@torch.no_grad()
def build_context_memory(model, context_loader, device, cosine=True):
    """
    Build memory from context data for kNN retrieval

    Returns:
        DB_E: [N, d] embeddings (L2-normalized if cosine=True)
        DB_Y: [N, C, H] future values
    """
    model.eval()
    E, Y = [], []
    for batch in tqdm(context_loader, desc="Building context memory"):
        x_enc, y_future, input_mask = batch[:3]
        x_enc = x_enc.to(device, dtype=torch.float32)
        input_mask = input_mask.to(device)
        emb = embed_batch(model, x_enc, input_mask)
        E.append(emb.detach().cpu().numpy().astype(np.float32))
        Y.append(y_future.detach().cpu().numpy().astype(np.float32))

    DB_E = np.concatenate(E, axis=0)
    DB_E = np.nan_to_num(DB_E, nan=0.0, posinf=0.0, neginf=0.0)
    if cosine:
        DB_E = l2_normalize(DB_E)
    DB_Y = np.concatenate(Y, axis=0).astype(np.float32)
    return DB_E, DB_Y


def knn_forecast(DB_E, DB_Y, Q_E, k=64, temperature=0.05):
    """
    kNN retrieval-based forecasting

    Args:
        DB_E: [N, d] database embeddings
        DB_Y: [N, C, H] database future values
        Q_E: [M, d] query embeddings
        k: number of nearest neighbors
        temperature: softmax temperature (None for uniform weights)

    Returns:
        Yhat_knn: [M, C, H] kNN predictions
    """
    M, d = Q_E.shape
    N = DB_E.shape[0]
    k = min(k, N)

    # similarity matrix (cosine = dot on unit vectors)
    S = Q_E @ DB_E.T  # [M, N]

    # top-k indices per row
    idxs = np.argpartition(-S, k - 1, axis=1)[:, :k]
    row = np.arange(M)[:, None]
    sims_k = S[row, idxs]
    order = np.argsort(-sims_k, axis=1)
    idxs = idxs[row, order]  # [M, k]
    sims = S[row, idxs]  # [M, k]

    # weights
    if temperature is None or temperature < 0:
        w = np.ones_like(sims, dtype=np.float32) / k
    else:
        T = max(temperature, 1e-12)
        sm = sims / T
        sm = sm - np.max(sm, axis=1, keepdims=True)
        ex = np.exp(sm)
        w = ex / np.maximum(np.sum(ex, axis=1, keepdims=True), 1e-12)

    # combine neighbor futures
    Yhat = []
    for m in range(M):
        neigh = DB_Y[idxs[m]]  # [k, C, H]
        yhat_m = nan_safe_weighted_average(w[m], neigh)  # [C, H]
        Yhat.append(yhat_m.astype(np.float32))

    return np.stack(Yhat, axis=0)  # [M, C, H]


@torch.no_grad()
def autoregressive_forecast(model, x_enc, input_mask, model_horizon, target_horizon, standardizer, device):
    """
    Universal autoregressive forecasting that works with any model_horizon and target_horizon.

    Strategy:
    - If model predicts K steps at once, use all K predictions efficiently
    - Slide window by K steps each iteration (not 1 step)
    - This minimizes the number of forward passes

    Args:
        model: The forecasting model
        x_enc: Input tensor [B, C, seq_len]
        input_mask: Input mask tensor [B, seq_len]
        model_horizon: Native horizon the model was trained on (e.g., 1, 24, 72, 96)
        target_horizon: Desired forecast horizon (can be any value)
        standardizer: Standardizer object for inverse transform
        device: Device to use for computation

    Returns:
        predictions: [B, C, target_horizon] array of predictions
    """
    B, C, seq_len = x_enc.shape

    # If target horizon <= model horizon, just do single forward pass
    if target_horizon <= model_horizon:
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(x_enc=x_enc, input_mask=input_mask)
        # Truncate to target_horizon if needed
        return output.forecast[:, :, :target_horizon].detach().cpu().numpy()

    # Calculate number of iterations needed
    # We predict model_horizon steps at a time
    num_iterations = int(np.ceil(target_horizon / model_horizon))
    all_predictions = []

    # Current input window
    current_input = x_enc.clone()
    current_mask = input_mask.clone()

    remaining_steps = target_horizon

    for i in range(num_iterations):
        # Forward pass with current window
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(x_enc=current_input, input_mask=current_mask)

        # Get predictions for this iteration [B, C, model_horizon]
        iteration_preds = output.forecast.detach()

        # Determine how many predictions to use from this iteration
        steps_to_use = min(model_horizon, remaining_steps)

        # Only keep the predictions we need
        preds_to_keep = iteration_preds[:, :, :steps_to_use]
        all_predictions.append(preds_to_keep)

        remaining_steps -= steps_to_use

        # If this is not the last iteration, prepare next input window
        if remaining_steps > 0:
            # Slide the window by model_horizon steps (or remaining steps)
            # current_input: [B, C, seq_len]
            # iteration_preds: [B, C, model_horizon]

            # Determine how many steps to slide
            slide_amount = min(model_horizon, seq_len)

            # Concatenate current input with new predictions
            # [B, C, seq_len + model_horizon]
            extended = torch.cat([current_input, iteration_preds], dim=2)

            # Take the last seq_len values as new input
            # This effectively slides the window by model_horizon steps
            current_input = extended[:, :, -seq_len:]

            # Update mask (all ones since we're using predictions)
            current_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)

    # Concatenate all predictions
    # List of [B, C, steps] -> [B, C, target_horizon]
    final_preds = torch.cat(all_predictions, dim=2)

    return final_preds.cpu().numpy()


def perform_forecasting(
    # Input data
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    target_column: str = "target",
    context_df: pd.DataFrame | None = None,  # Optional context DataFrame for DARR mode
    # Model configuration - Replace with your own paths for the weights and standardizer
    standardizer_pkl: str = "artifacts_512_72/standardizer.pkl",
    ckpt: str = "artifacts_512_72/moment_head_512_6hr.pt",
    seq_len: int = 512,
    forecast_horizon: int = 72,
    model_horizon: int = 72,  # Override with other weights' values if needed
    # Output configuration
    save_preds: str | None = None,
    # DARR mode configuration
    alpha: float = 0.01,
    # Additional parameters (with sensible defaults)
    model_name: str = "AutonLab/MOMENT-1-large",
    batch_size: int = 8,
    num_workers: int = 2,
    stride: int | None = None,
    context_stride: int | None = None,
    seed: int = 13,
    k: int = 64,
    temperature: float = 0.05,
    device: str | None = None,
    local_files_only: bool = False,
) -> pd.DataFrame:
    """
    Perform time series forecasting using Tesseract v2 with optional context-enhanced mode (DARR).
    Supports autoregressive forecasting for horizons beyond the model's native capability.

    ALWAYS uses InferenceOnlyDataset - only requires seq_len rows for inference.
    """
    # Set model_horizon to forecast_horizon if not specified
    if model_horizon is None:
        model_horizon = forecast_horizon

    # Validate that model_horizon is reasonable
    if model_horizon <= 0:
        raise ValueError(f"model_horizon must be positive, got {model_horizon}")

    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")

    # Maximum forecast horizon limit
    MAX_FORECAST_HORIZON = 512
    if forecast_horizon > MAX_FORECAST_HORIZON:
        raise ValueError(f"forecast_horizon must be <= {MAX_FORECAST_HORIZON}, got {forecast_horizon}")


    # Input validation
    if df is None or df.empty:
        raise ValueError("Input DataFrame is required and cannot be empty")

    # Validate minimum rows
    if len(df) < seq_len:
        raise ValueError(f"DataFrame has {len(df)} rows but seq_len requires at least {seq_len} rows")

    # Validate timestamp column
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    if df[timestamp_column].isnull().any():
        raise ValueError(f"Timestamp column '{timestamp_column}' contains NULL values")

    # Try to convert timestamp column to datetime if it's not already
    working_df = df.copy()
    try:
        if not pd.api.types.is_datetime64_any_dtype(working_df[timestamp_column]):
            working_df[timestamp_column] = pd.to_datetime(working_df[timestamp_column])
    except Exception as e:
        raise ValueError(f"Cannot parse timestamp column '{timestamp_column}' as datetime: {e}")

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # Check if target column is numeric
    if not pd.api.types.is_numeric_dtype(working_df[target_column]):
        raise ValueError(f"Target column '{target_column}' must contain numeric values")

    # Handle NULL values in target column - fill with zeros
    if working_df[target_column].isnull().any():
        print(f"Warning: Found NULL values in '{target_column}', filling with zeros")
        working_df[target_column] = working_df[target_column].fillna(0)

    # Automatically detect all numeric columns to use as features
    numeric_columns = working_df.select_dtypes(include=[np.number]).columns.tolist()

    # Make sure target column is first in the list
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    columns_to_process = [target_column] + numeric_columns

    # Fill NaN values with zeros for all numeric columns
    for col in columns_to_process:
        if working_df[col].isnull().any():
            print(f"Warning: Found NULL values in '{col}', filling with zeros")
            working_df[col] = working_df[col].fillna(0)

    # Set default values
    # Use model_horizon for strides to ensure consistent context memory regardless of forecast_horizon
    if stride is None:
        stride = model_horizon
    if context_stride is None:
        context_stride = model_horizon

    # Set device
    if device is None:
        device = DEVICE
    else:
        device = torch.device(device)

    # Set random seed
    control_randomness(seed=seed)

    # Determine mode
    if context_df is not None:
        mode = "darr"
        print(f"Using DARR mode (Context-Enhanced Forecasting) with alpha={alpha}")
    else:
        mode = "standard"
        print("Using standard forecasting mode (inference only)")

    # Temporary files for DataFrame conversion
    temp_test_csv = None
    temp_context_csv = None

    try:
        # Create temporary CSV from DataFrame for processing
        temp_dir = tempfile.gettempdir()
        temp_test_csv = os.path.join(temp_dir, f"temp_test_{os.getpid()}.csv")

        # Save only the necessary columns (timestamp + value columns)
        csv_df = working_df[[timestamp_column] + columns_to_process].copy()
        csv_df.rename(columns={timestamp_column: "timestamp"}, inplace=True)
        csv_df.to_csv(temp_test_csv, index=False)
        test_csv_path = temp_test_csv

        # Handle context DataFrame if provided for DARR mode
        if context_df is not None:
            temp_context_csv = os.path.join(temp_dir, f"temp_context_{os.getpid()}.csv")

            # Validate context DataFrame columns
            if timestamp_column not in context_df.columns:
                raise ValueError(f"Context DataFrame missing timestamp column '{timestamp_column}'")

            if target_column not in context_df.columns:
                raise ValueError(f"Context DataFrame missing target column '{target_column}'")

            # Validate context DataFrame has enough rows for at least one window
            min_context_rows = seq_len + model_horizon
            if len(context_df) < min_context_rows:
                raise ValueError(
                    f"Context DataFrame has {len(context_df)} rows but requires at least "
                    f"{min_context_rows} rows (seq_len={seq_len} + model_horizon={model_horizon})"
                )

            # Process context DataFrame similarly to main DataFrame
            context_working = context_df.copy()

            # Convert timestamp if needed
            try:
                if not pd.api.types.is_datetime64_any_dtype(context_working[timestamp_column]):
                    context_working[timestamp_column] = pd.to_datetime(context_working[timestamp_column])
            except Exception as e:
                raise ValueError(f"Cannot parse context timestamp column '{timestamp_column}' as datetime: {e}")

            # Get numeric columns from context DataFrame
            context_numeric = context_working.select_dtypes(include=[np.number]).columns.tolist()

            # Ensure target column is included
            if target_column in context_numeric:
                context_numeric.remove(target_column)
            
            # COLUMN COMPATIBILITY CHECK AND ALIGNMENT
            # Find intersection of numeric columns between main and context datasets
            main_numeric_set = set(numeric_columns)
            context_numeric_set = set(context_numeric)
            
            # Common columns (excluding target which is always included)
            common_columns = main_numeric_set.intersection(context_numeric_set)
            
            # Check if we have column compatibility issues
            if len(main_numeric_set) != len(context_numeric_set) or main_numeric_set != context_numeric_set:
                print(f"Warning: Column mismatch detected between input and context datasets")
                print(f"  Input dataset columns: {sorted(main_numeric_set)}")
                print(f"  Context dataset columns: {sorted(context_numeric_set)}")
                print(f"  Common columns: {sorted(common_columns)}")
                
                if len(common_columns) == 0:
                    raise ValueError(
                        f"No common numeric columns found between input and context datasets.\n"
                        f"Input dataset has: {sorted(main_numeric_set)}\n"
                        f"Context dataset has: {sorted(context_numeric_set)}\n"
                        f"For DARR mode to work, both datasets must share at least some numeric columns."
                    )
                
                print(f"  Using only common columns for consistent predictions: {sorted(common_columns)}")
                
                # Update both datasets to use only common columns
                columns_to_process = [target_column] + sorted(common_columns)
                context_columns_to_use = [timestamp_column, target_column] + sorted(common_columns)
                
                # Re-save the main CSV with aligned columns
                csv_df = working_df[[timestamp_column] + columns_to_process].copy()
                csv_df.rename(columns={timestamp_column: "timestamp"}, inplace=True)
                csv_df.to_csv(temp_test_csv, index=False)
            else:
                context_columns_to_use = [timestamp_column, target_column] + context_numeric

            # Create CSV with selected columns
            context_csv_df = context_working[context_columns_to_use].copy()
            context_csv_df.rename(columns={timestamp_column: "timestamp"}, inplace=True)

            # Fill NULLs in context data
            for col in context_csv_df.columns:
                if col != "timestamp" and context_csv_df[col].isnull().any():
                    print(f"Warning: Found NULL values in context DataFrame column '{col}', filling with zeros")
                    context_csv_df[col] = context_csv_df[col].fillna(0)

            context_csv_df.to_csv(temp_context_csv, index=False)
            context_csv_path = temp_context_csv

        # Load standardizer
        stds = joblib.load(standardizer_pkl)
        mean = np.array(stds["mean"], dtype=np.float32)
        std = np.array(stds["std"], dtype=np.float32)
        std[std < 1e-8] = 1.0
        standardizer = Standardizer(mean=mean, std=std)

        # ALWAYS use InferenceOnlyDataset for the main test data
        test_dataset = InferenceOnlyDataset(csv_path=test_csv_path, seq_len=seq_len, standardizer=standardizer)

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Always 1 for inference-only
            shuffle=False,
            num_workers=0,  # Set to 0 for single sample
            pin_memory=True,
        )

        # Build model with model_horizon
        model = build_model(
            model_name=model_name,
            seq_len=seq_len,
            forecast_horizon=model_horizon,  # Use model's native horizon
            freeze_encoder=False,
            freeze_embedder=False,
            freeze_head=False,
            local_files_only=local_files_only,
            device=str(device),
        )
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        # Perform inference based on mode
        if mode == "darr":
            # Context-Enhanced Forecasting (DARR Mode)

            # Create context dataset with model_horizon (needs ground truth)
            context_dataset = CSVLongHorizonSimpleDataset(
                csv_path=context_csv_path,
                data_split="train",
                seq_len=seq_len,
                forecast_horizon=model_horizon,  # Use model's native horizon
                standardizer=None,
                standardize=False,
                stride=context_stride,
            )
            context_dataset.standardizer = standardizer
            context_dataset.series = context_dataset.standardizer.transform(context_dataset.values)

            context_loader = DataLoader(
                context_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )

            # Build context memory (using model_horizon)
            DB_E, DB_Y = build_context_memory(model, context_loader, device, cosine=True)
            # print(f"Context memory built: {DB_E.shape} embeddings, {DB_Y.shape} futures")

            # Embed test data and get direct predictions (with autoregressive if needed)
            Q_E_list = []
            preds_direct = []

            for timeseries, forecast, input_mask in tqdm(test_loader, desc="Direct prediction + embedding"):
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)

                # Embed for kNN
                emb = embed_batch(model, timeseries, input_mask)
                Q_E_list.append(emb.detach().cpu().numpy().astype(np.float32))

                # Direct prediction with autoregressive extension if needed
                batch_preds = autoregressive_forecast(
                    model, timeseries, input_mask, model_horizon, forecast_horizon, standardizer, device
                )
                preds_direct.append(batch_preds)

            Q_E = np.concatenate(Q_E_list, axis=0)
            Q_E = np.nan_to_num(Q_E, nan=0.0, posinf=0.0, neginf=0.0)
            Q_E = l2_normalize(Q_E)

            preds_direct = np.concatenate(preds_direct, axis=0)

            # kNN retrieval forecast
            preds_knn_base = knn_forecast(DB_E, DB_Y, Q_E, k=k, temperature=temperature)

            # If forecast_horizon > model_horizon, extend kNN predictions
            if forecast_horizon > model_horizon:
                B, C, H = preds_knn_base.shape
                preds_knn = np.zeros((B, C, forecast_horizon), dtype=np.float32)
                preds_knn[:, :, :H] = preds_knn_base
                # Extend with last value
                for h in range(H, forecast_horizon):
                    preds_knn[:, :, h] = preds_knn_base[:, :, -1]
            else:
                preds_knn = preds_knn_base[:, :, :forecast_horizon]

            # Validate shapes before combining predictions
            if preds_direct.shape != preds_knn.shape:
                raise ValueError(
                    f"Shape mismatch between direct and kNN predictions.\n"
                    f"Direct prediction shape: {preds_direct.shape}\n"
                    f"kNN prediction shape: {preds_knn.shape}\n"
                    f"This typically occurs when input and context datasets have different numbers of columns.\n"
                    f"Ensure both datasets have the same numeric columns, or the column alignment failed."
                )
            
            # Hybrid predictions
            preds_hybrid = alpha * preds_direct + (1 - alpha) * preds_knn

            # Use hybrid as main predictions
            preds = preds_hybrid

        else:
            # Direct prediction mode with autoregressive extension if needed
            preds = []

            for timeseries, forecast, input_mask in tqdm(test_loader, desc="Inference"):
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)

                # Autoregressive forecast
                batch_preds = autoregressive_forecast(
                    model, timeseries, input_mask, model_horizon, forecast_horizon, standardizer, device
                )
                preds.append(batch_preds)

            preds = np.concatenate(preds, axis=0)

        # Convert to original scale
        B, C, H = preds.shape
        P_flat = preds.transpose(0, 2, 1).reshape(-1, C)

        P_orig = test_dataset.inverse_transform(P_flat)

        # Reshape back to [B, C, H]
        P_orig_reshaped = P_orig.reshape(B * H, C).reshape(B, H, C).transpose(0, 2, 1)

        # Build prediction timestamps and values for future rows only
        all_timestamps = []
        all_predictions = []

        # Infer frequency from timestamp column
        time_diffs = working_df[timestamp_column].diff().dropna()
        if len(time_diffs) > 0:
            inferred_freq = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        else:
            inferred_freq = pd.Timedelta(hours=1)

        last_input_time = working_df[timestamp_column].iloc[-1]
        forecast_timestamps = pd.date_range(
            start=last_input_time + inferred_freq, periods=forecast_horizon, freq=inferred_freq
        )

        forecast_values = P_orig[:forecast_horizon, 0]
        all_timestamps.extend(forecast_timestamps.tolist())
        all_predictions.extend(forecast_values.tolist())

        if all_timestamps:
            result_df = pd.DataFrame(
                {timestamp_column: pd.to_datetime(all_timestamps), f"{target_column}_forecast": all_predictions}
            )
        else:
            result_df = pd.DataFrame(columns=[timestamp_column, f"{target_column}_forecast"])

        # Save predictions if requested
        if save_preds:
            result_df.to_csv(save_preds, index=False)
            print(f"Saved predictions to {save_preds}")

        if mode == "darr":
            print(f"\n{'=' * 60}")
            print("DARR Mode Results")
            print(f"{'=' * 60}")
            print(f"Added column: {target_column}_forecast")
        else:
            print(f"\n{'=' * 60}")
            print("Results")
            print(f"{'=' * 60}")
            print(f"Added column: {target_column}_forecast")

        return result_df

    finally:
        # Clean up temporary files
        if temp_test_csv and os.path.exists(temp_test_csv):
            Path(temp_test_csv).unlink()
        if temp_context_csv and os.path.exists(temp_context_csv):
            Path(temp_context_csv).unlink()
