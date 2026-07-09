# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
utils.py

-----------------------------------------------------------------------
This script provides utility functions for training, validating, and evaluating the model.
The utilities include methods for  computing various evaluation metrics, handling model inference,
and leveraging different sampling techniques.

Key Functions:
-----------------------------------------------------------------------
- evaluate(): Runs inference on test datasets, generates samples, and evaluates performance
  using RMSE, MAE, R^2, and CRPS metrics. **NEW: Supports DPM-Solver for fast inference**
- sliding_window_evaluate(): Applies a window-based technique to merge multiple time
  series predictions, improving robustness in anomaly detection.
- ensemble(): Runs multiple evaluation iterations to create an ensemble of generated
  samples for better probabilistic forecasting.
- estimate_training_time(): Estimates total training time based on partial epoch duration.
- calc_quantile_CRPS(): Calculates Continuous Ranked Probability Score for evaluation.

Evaluation Metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Continuous Ranked Probability Score (CRPS)
- Coefficient of Determination (R^2)

Usage:
-----------------------------------------------------------------------
- Import the required function(s) from `utils.py` and call them as needed.

**NEW: Fast Inference with DPM-Solver**
-----------------------------------------------------------------------
Set use_dpm_solver=True for 50-100x speedup:

    results = evaluate(
        model, test_loader1, test_loader2,
        nsample=30,
        use_dpm_solver=True,
        dpm_steps=20
    )

"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Standard library imports
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_utils import load_data, save_data

# Optional imports
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EPSILON = 1e-8


def estimate_training_time(start_time, end_time, epochs_run, total_epochs, save_dir=None):
    """
    Estimates total training time based on partial epoch duration.

    Args:
        start_time (datetime): Start time of the training window.
        end_time (datetime): End time after a few epochs.
        epochs_run (int): Number of epochs run during timing.
        total_epochs (int): Total number of epochs intended.
        save_dir (str or Path, optional): Directory to save the timing estimation.
    """
    elapsed = (end_time - start_time).total_seconds()
    avg_epoch_time = elapsed / epochs_run
    estimated_total_time = avg_epoch_time * total_epochs
    est_minutes = estimated_total_time / 60
    est_hours = est_minutes / 60

    results = {
        "avg_epoch_time_sec": round(avg_epoch_time, 2),
        "estimated_total_minutes": round(est_minutes, 2),
        "estimated_total_hours": round(est_hours, 2),
    }

    logger.info("===== Training Time Estimation =====")
    logger.info("Average epoch time: %s seconds", results["avg_epoch_time_sec"])
    logger.info(
        "Estimated time for %s epochs: %s minutes (~%.2f hours)",
        total_epochs,
        results["estimated_total_minutes"],
        results["estimated_total_hours"],
    )

    if save_dir is not None:
        save_path = Path(save_dir) / "training_time_estimate.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Timing estimation saved to %s", save_path)

    return results


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(torch.abs((forecast - target) * eval_points * ((target <= forecast).float() - q))).item()


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i].item(), dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i].item(), eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def convert_csv_to_json_if_needed(base_path: Path, dataset: str) -> None:
    """Convert CSV files to JSON format if JSON files don't exist"""

    # Define file mappings
    files_to_convert = [
        ("train.csv", "train.json"),
        ("test.csv", "test.json"),
    ]

    converted_any = False
    for csv_file, json_file in files_to_convert:
        csv_path = base_path / csv_file
        json_path = base_path / json_file

        # Only convert if CSV exists and JSON doesn't
        if csv_path.exists() and not json_path.exists():
            logger.info(f"Converting {csv_file} to {json_file}...")
            try:
                # Load CSV
                df = pd.read_csv(csv_path)
                logger.info(f"Loaded {csv_file}: shape={df.shape}")

                # Validate data
                if df.empty:
                    logger.error(f"{csv_file} is empty!")
                    raise ValueError(f"{csv_file} is empty")

                # Save as JSON with numpy/NaN handling
                save_data(df, json_path)

                # Verify the JSON file was written correctly
                file_size = json_path.stat().st_size
                logger.info(f"Successfully converted {csv_file} to {json_file} (size: {file_size} bytes)")

                # Try to reload to verify integrity
                load_data(json_path)
                logger.info(f"Verified {json_file} can be loaded successfully")

                converted_any = True
            except Exception as e:
                logger.error(f"Error converting {csv_file}: {e}")
                # Clean up partial file if it exists
                if json_path.exists():
                    json_path.unlink()
                    logger.info(f"Cleaned up corrupted {json_file}")
                raise

    # Handle test_labels.json - create from test.json if it doesn't exist
    label_json_path = base_path / "test_labels.json"
    if not label_json_path.exists():
        test_json_path = base_path / "test.json"
        if test_json_path.exists():
            logger.info("test_labels.json not found, creating from test.json...")
            try:
                test_df = load_data(test_json_path)

                # Check if there's a 'label' or 'anomaly' column
                if "label" in test_df.columns:
                    labels = test_df["label"].values
                    logger.info("Found 'label' column in test data")
                elif "anomaly" in test_df.columns:
                    labels = test_df["anomaly"].values
                    logger.info("Found 'anomaly' column in test data")
                else:
                    # Create dummy labels (all zeros)
                    labels = np.array([0] * len(test_df))
                    logger.warning(
                        f"No label column found, creating dummy labels (all zeros) for {len(test_df)} samples"
                    )

                save_data(labels, label_json_path)
                logger.info(f"Created test_labels.json with {len(labels)} labels")
                converted_any = True
            except Exception as e:
                logger.error(f"Error creating test_labels.json: {e}")
                raise

    if converted_any:
        logger.info("CSV to JSON conversion complete!")
    else:
        logger.info("JSON files already exist or no CSV files found")


def evaluate(
    model,
    test_loader1,
    test_loader2,
    nsample=20,
    scaler=1,
    mean_scaler=0,
    foldername="",
    epoch_number="",
    name="",
    save_results=True,
    use_dpm_solver=False,
    dpm_steps=20,
):
    """
    Evaluate the model on test data with optional DPM-Solver for fast inference.

    Args:
        model: The diffusion model
        test_loader1: First test dataloader (strategy 0)
        test_loader2: Second test dataloader (strategy 1)
        nsample: Number of samples to generate
        scaler: Scaling factor for metrics
        mean_scaler: Mean scaling factor
        foldername: Folder to save results
        epoch_number: Current epoch number
        name: Additional name for saved files
        save_results: Whether to save results to disk
        use_dpm_solver: If True, use DPM-Solver for 50-100x faster inference
        dpm_steps: Number of DPM-Solver steps (10-50, default: 20)

    Returns:
        eval_outputs: Dictionary containing evaluation results

    Note:
        Setting use_dpm_solver=True with dpm_steps=20 provides ~50x speedup
        with minimal quality loss compared to standard 1000-step diffusion.

    Example:
        # Standard inference (slow)
        results = evaluate(model, loader1, loader2, nsample=30)

        # Fast inference with DPM-Solver
        results = evaluate(model, loader1, loader2, nsample=30,
                          use_dpm_solver=True, dpm_steps=20)
    """
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        # Log sampling method
        if use_dpm_solver:
            logger.info(f"Using DPM-Solver with {dpm_steps} steps for fast inference")
        else:
            logger.info("Using standard diffusion sampling")

        with tqdm(zip(test_loader1, test_loader2, strict=False), mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, (test_batch1, test_batch2) in enumerate(it, start=1):
                # Process first batch with selected sampling method
                if use_dpm_solver:
                    samples, c_target, eval_points, observed_points, observed_time = model.evaluate_with_dpm(
                        test_batch1, nsample, dpm_steps
                    )
                else:
                    output = model.evaluate(test_batch1, nsample)
                    samples, c_target, eval_points, observed_points, observed_time = output

                samples = samples.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                # Process second batch with selected sampling method
                if use_dpm_solver:
                    samples2, _, eval_points2, _, _ = model.evaluate_with_dpm(test_batch2, nsample, dpm_steps)
                else:
                    output2 = model.evaluate(test_batch2, nsample)
                    samples2, _, eval_points2, _, _ = output2

                samples2 = samples2.permute(0, 1, 3, 2)
                eval_points2 = eval_points2.permute(0, 2, 1)

                # Combine samples based on target masks
                samples1_old = samples
                samples2_old = samples2
                target_mask1_expanded = eval_points.unsqueeze(1)
                target_mask2_expanded = eval_points2.unsqueeze(1)
                # Combine samples based on expanded target masks
                samples = target_mask1_expanded * samples1_old + target_mask2_expanded * samples2_old

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = ((samples_median.values - c_target) ** 2) * (scaler**2)
                mae_current = (torch.abs(samples_median.values - c_target)) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += torch.ones_like(mse_current).sum().item()

                # Use NumPy for sqrt for RMSE
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

        # Save generated outputs
        all_target = torch.cat(all_target, dim=0).to("cpu")
        all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
        all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
        all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
        all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")

        if save_results:
            if not os.path.exists(foldername):
                os.makedirs(foldername)

            # Add suffix to indicate DPM-Solver was used
            suffix = f"_dpm{dpm_steps}" if use_dpm_solver else ""

            save_data(
                [
                    all_generated_samples.numpy() if hasattr(all_generated_samples, "numpy") else all_generated_samples,
                    all_target.numpy() if hasattr(all_target, "numpy") else all_target,
                    all_evalpoint.numpy() if hasattr(all_evalpoint, "numpy") else all_evalpoint,
                    all_observed_point.numpy() if hasattr(all_observed_point, "numpy") else all_observed_point,
                    all_observed_time.numpy() if hasattr(all_observed_time, "numpy") else all_observed_time,
                    scaler,
                    mean_scaler,
                ],
                foldername + f"/{epoch_number}-generated_outputs_nsample{nsample}{name}{suffix}.json",
            )

        # Calculate CRPS
        CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

        # Calculate R² using NumPy
        y_true = all_target.reshape(-1).numpy()
        y_pred = all_generated_samples.median(dim=1).values.reshape(-1).numpy()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        results = {
            "RMSE": np.sqrt(mse_total / evalpoints_total),
            "MAE": mae_total / evalpoints_total,
            "CRPS": CRPS,
            "R2": r2.item() if hasattr(r2, "item") else r2,
        }

        eval_outputs = {
            "generated_samples": all_generated_samples,
            "target": all_target,
            "eval_points": all_evalpoint,
            "observed_points": all_observed_point,
            "observed_time": all_observed_time,
            **results,
        }

        if save_results:
            suffix = f"_dpm{dpm_steps}" if use_dpm_solver else ""
            with open(
                foldername + f"/{epoch_number}-result_nsample{nsample}{suffix}.json",
                "w",
            ) as f:
                json.dump(results, f, indent=4)

        return eval_outputs


def sliding_window_evaluate(
    model,
    test_loader1,
    test_loader2,
    nsample: int = 20,
    scaler: float = 1,
    mean_scaler: float = 0,
    foldername: str = "",
    epoch_number: str = "",
    name: str = "",
    split: int = 4,
    save_results: bool = True,
    use_dpm_solver: bool = False,
    dpm_steps: int = 20,
) -> Dict:
    """
    Sliding window evaluation with optional DPM-Solver support.

    Args:
        model: The diffusion model
        test_loader1: First test dataloader
        test_loader2: Second test dataloader
        nsample: Number of samples to generate
        scaler: Scaling factor
        mean_scaler: Mean scaling factor
        foldername: Folder to save results
        epoch_number: Current epoch number
        name: Additional name for saved files
        split: Number of splits for sliding window
        save_results: Whether to save results
        use_dpm_solver: If True, use DPM-Solver for fast inference
        dpm_steps: Number of DPM-Solver steps

    Returns:
        eval_outputs: Dictionary containing evaluation results
    """
    model.eval()

    # Initialize metrics using NumPy
    mse_total = np.zeros(1, dtype=np.float32)
    mae_total = np.zeros(1, dtype=np.float32)
    evalpoints_total = np.zeros(1, dtype=np.float32)

    # NumPy storage arrays
    all_target = np.array([], dtype=np.float32)
    all_generated = np.array([], dtype=np.float32)

    test_loader2 = iter(test_loader2)

    with torch.no_grad():
        for batch_no, test_batch in enumerate(tqdm(test_loader1), start=1):
            # GPU-accelerated model evaluation with selected sampling method
            if use_dpm_solver:
                samples, c_target, *_ = model.evaluate_with_dpm(test_batch, nsample, dpm_steps)
                samples2, *_ = model.evaluate_with_dpm(next(test_loader2), nsample, dpm_steps)
            else:
                samples, c_target, *_ = model.evaluate(test_batch, nsample)
                samples2, *_ = model.evaluate(next(test_loader2), nsample)

            # Convert to NumPy arrays
            def convert_to_numpy(tensor):
                return tensor.detach().cpu().numpy()

            samples = convert_to_numpy(samples.permute(0, 1, 3, 2).squeeze())
            c_target = convert_to_numpy(c_target.permute(0, 2, 1))
            samples2 = convert_to_numpy(samples2.permute(0, 1, 3, 2).squeeze())

            # Merge samples using NumPy operations
            samples = samples + samples2
            samples_length = samples.shape[1]

            # Slice processing with NumPy
            slice_idx = slice(samples_length // split, -samples_length // split)
            samples = samples[:, slice_idx, :]
            c_target = c_target[:, slice_idx, :]

            # NumPy metric calculations
            residuals = samples - c_target
            scaled_residuals = residuals * scaler

            # Update metrics
            mse_total[0] += np.sum(scaled_residuals**2)
            mae_total[0] += np.sum(np.abs(scaled_residuals))
            evalpoints_total[0] += samples.size

            # Concatenate using NumPy
            all_target = np.concatenate([all_target, c_target.ravel()])
            all_generated = np.concatenate([all_generated, samples.ravel()])

    # Final metric calculations using NumPy
    rmse = np.sqrt(mse_total[0] / evalpoints_total[0]).item()
    mae = (mae_total[0] / evalpoints_total[0]).item()

    # R² calculation using NumPy
    ss_res = np.sum((all_target - all_generated) ** 2)
    ss_tot = np.sum((all_target - np.mean(all_target)) ** 2)
    r2 = (1 - (ss_res / ss_tot)).item()

    # CRPS calculation using NumPy quantiles
    quantiles = np.arange(0.05, 1.0, 0.05)
    q_pred = np.quantile(all_generated, quantiles, axis=0)
    crps = np.mean(quantiles * (all_target - q_pred) * (all_target < q_pred)).item()

    results = {"rmse": rmse, "mae": mae, "r2": r2, "crps": crps}

    # Save results using h5py (if available)
    if save_results and HAS_H5PY:
        suffix = f"_dpm{dpm_steps}" if use_dpm_solver else ""
        output_path = Path(foldername) / f"{epoch_number}-sw_results_nsample{nsample}{name}{suffix}.h5"
        with h5py.File(output_path, "w") as h5:
            h5.create_dataset("target", data=all_target)
            h5.create_dataset("generated", data=all_generated)
    elif save_results and not HAS_H5PY:
        logger.warning("h5py not available, skipping HDF5 file save for sliding window results")

    eval_outputs = {
        "generated_samples": all_generated,
        "target": all_target,
        **results,
    }

    return eval_outputs


def ensemble(
    model,
    test_loader,
    nsample=10,
    scaler=1,
    name="",
    save_results=True,
    use_dpm_solver=False,
    dpm_steps=20,
):
    """
    Ensemble evaluation with optional DPM-Solver support.

    Args:
        model: The diffusion model
        test_loader: Test dataloader
        nsample: Number of ensemble iterations
        scaler: Scaling factor
        name: Name for saved files
        save_results: Whether to save results
        use_dpm_solver: If True, use DPM-Solver for fast inference
        dpm_steps: Number of DPM-Solver steps

    Returns:
        eval_outputs: Dictionary containing evaluation results
    """
    model.eval()

    # Initialize metrics using NumPy
    mse_total = np.zeros(1, dtype=np.float32)
    mae_total = np.zeros(1, dtype=np.float32)
    evalpoints_total = np.zeros(1, dtype=np.float32)

    # NumPy storage arrays
    all_target = np.array([], dtype=np.float32)
    all_generated = np.array([], dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(nsample), desc="Ensemble iterations"):
            batch_targets = np.array([], dtype=np.float32)
            batch_generated = np.array([], dtype=np.float32)

            for test_batch in test_loader:
                # GPU-accelerated model evaluation with selected sampling method
                if use_dpm_solver:
                    output = model.evaluate_with_dpm(test_batch, 1, dpm_steps)
                else:
                    output = model.evaluate(test_batch, 1)

                samples, c_target, eval_points, observed_points, _ = output

                # Convert to NumPy arrays
                samples = samples.permute(0, 1, 3, 2).squeeze().detach().cpu().numpy()
                c_target = c_target.permute(0, 2, 1).detach().cpu().numpy()
                eval_points = eval_points.permute(0, 2, 1).detach().cpu().numpy()

                # Calculate metrics using NumPy arrays
                residuals = (samples - c_target) * eval_points
                scaled_residuals = residuals * scaler

                # Update metrics
                mse_total[0] += np.sum(scaled_residuals**2)
                mae_total[0] += np.sum(np.abs(scaled_residuals))
                evalpoints_total[0] += np.sum(eval_points.astype(np.float32))

                # Accumulate results
                batch_targets = np.concatenate([batch_targets, c_target.ravel()])
                batch_generated = np.concatenate([batch_generated, samples.ravel()])

            # Store ensemble results
            all_target = np.concatenate([all_target, batch_targets])
            all_generated = np.concatenate([all_generated, batch_generated])

    # Final metric calculations
    rmse = np.sqrt(mse_total[0] / evalpoints_total[0]).item()
    mae = (mae_total[0] / evalpoints_total[0]).item()

    # Save results using h5py
    results = {"RMSE": rmse, "MAE": mae, "evalpoints_total": evalpoints_total[0].item(), "nsample": nsample}

    eval_outputs = {
        "generated_samples": all_generated,
        "target": all_target,
        **results,
    }

    if save_results and HAS_H5PY:
        base_path = Path(name).with_suffix("")
        suffix = f"_dpm{dpm_steps}" if use_dpm_solver else ""
        data_path = f"{base_path}_data{suffix}.h5"
        with h5py.File(data_path, "w") as h5:
            h5.create_dataset("target", data=all_target)
            h5.create_dataset("generated", data=all_generated)

        logger.info("Ensemble results saved to %s", data_path)
    elif save_results and not HAS_H5PY:
        logger.warning("h5py not available, skipping HDF5 file save for ensemble results")

    logger.info("RMSE: %.4f", rmse)
    logger.info("MAE: %.4f", mae)

    return eval_outputs
