#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick Example: Anomaly Detection with AD Diffusion

This example demonstrates how to use the perform_anomaly_analysis_with_diffusion
function for time series anomaly detection using a synthetic dataset.

Requirements:
- Python 3.12+
- All dependencies installed (run: uv sync)
- Optional: Pre-trained NV-Tesseract AD diffusion model

Usage:
    # From the ad_diffusion directory (weights auto-download from HF on first run):
    uv run python examples/quick_example.py

    # Use your own dataset:
    uv run python examples/quick_example.py --dataset-path /path/to/your/data.csv

    # Or pre-download the weights explicitly:
    uv run python examples/quick_example.py --download-weights

    # Or use a local checkpoint you already have:
    uv run python examples/quick_example.py --model-path /path/to/final_model.pth

    # Or run directly (after setting up dependencies):
    cd examples && python quick_example.py

Hugging Face authentication (only needed if the repo is gated/private):
    1. Install the CLI:  uv add "huggingface_hub[cli]"
    2. Login:            huggingface-cli login
    3. Or export a token: export HUGGINGFACE_HUB_TOKEN="hf_..."
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion
from sdk.inference_ad import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_MODEL_FILENAME,
    HF_REPO_ID,
    download_model_weights,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_dataset(n_samples: int = 1000, n_features: int = 5, anomaly_rate: float = 0.05) -> pd.DataFrame:
    """
    Create a synthetic time series dataset with anomalies.

    Args:
        n_samples: Number of time series samples
        n_features: Number of features/sensors
        anomaly_rate: Proportion of samples that are anomalies

    Returns:
        DataFrame with time series data
    """
    np.random.seed(42)  # For reproducibility

    # Create normal time series patterns
    time = np.linspace(0, 4 * np.pi, n_samples)

    # Generate base patterns for each feature
    data = np.zeros((n_samples, n_features))

    for i in range(n_features):
        # Mix of sine waves with different frequencies and phases
        frequency = 0.5 + i * 0.2
        phase = i * np.pi / 4

        # Base signal: sine wave + trend + noise
        signal = (
            np.sin(frequency * time + phase)  # Primary pattern
            + 0.3 * np.sin(2 * frequency * time)  # Harmonic
            + 0.1 * time / n_samples  # Slight trend
            + 0.1 * np.random.randn(n_samples)  # Gaussian noise
        )

        data[:, i] = signal

    # Add anomalies
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    ground_truth = np.zeros(n_samples)

    # Create different types of anomalies
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(["spike", "dip", "shift", "noise"])

        if anomaly_type == "spike":
            # Sharp spike anomaly
            data[idx] += np.random.uniform(3, 5, n_features) * np.sign(np.random.randn(n_features))
            ground_truth[idx] = 1
        elif anomaly_type == "dip":
            # Sharp dip anomaly
            data[idx] -= np.random.uniform(2, 4, n_features)
            ground_truth[idx] = 1
        elif anomaly_type == "shift":
            # Level shift anomaly
            shift_duration = min(10, n_samples - idx)
            shift_magnitude = np.random.uniform(1.5, 3, n_features)
            data[idx : idx + shift_duration] += shift_magnitude
            ground_truth[idx : idx + shift_duration] = 1
        elif anomaly_type == "noise":
            # High noise anomaly
            noise_duration = min(5, n_samples - idx)
            data[idx : idx + noise_duration] += np.random.randn(noise_duration, n_features) * 2
            ground_truth[idx : idx + noise_duration] = 1

    # Create DataFrame with meaningful column names
    columns = [f"sensor_{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)

    # Add timestamp column
    df["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="1H")

    # Add ground truth anomaly labels for evaluation
    df["is_anomaly"] = ground_truth

    return df


def save_sample_dataset():
    """Create and save a sample dataset for users."""
    logger.info("Creating sample dataset...")

    # Create datasets directory if it doesn't exist
    datasets_dir = Path(__file__).parent / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    # Generate sample data
    df = create_synthetic_dataset(n_samples=500, n_features=3, anomaly_rate=0.08)

    # Save dataset
    dataset_path = datasets_dir / "sample_timeseries.csv"

    # Save without ground truth for real-world simulation
    df_clean = df.drop(["is_anomaly"], axis=1)
    df_clean.to_csv(dataset_path, index=False)

    # Save ground truth separately for evaluation
    ground_truth_path = datasets_dir / "sample_timeseries_labels.csv"
    df[["timestamp", "is_anomaly"]].to_csv(ground_truth_path, index=False)

    logger.info(f"Sample dataset saved to: {dataset_path}")
    logger.info(f"Ground truth labels saved to: {ground_truth_path}")

    return dataset_path, ground_truth_path


def run_anomaly_detection_example(
    model_path: str | None = None,
    config_path: str | None = None,
    skip_download: bool = False,
    dataset_path: str | None = None,
):
    """
    Run the complete anomaly detection example.

    Args:
        model_path: Path to pre-trained model. If ``None`` or missing, the default
            weights are auto-downloaded from ``nvidia/nv-tesseract-ad-diffusion``.
        config_path: Path to the model config. If ``None`` or missing, the default
            ``curriculum_medium.yaml`` is fetched alongside the checkpoint.
        skip_download: If True, do not attempt to auto-download weights. The example
            will only print the synthetic dataset preview when no local model is found.
        dataset_path: Path to your own CSV dataset. If ``None``, a synthetic dataset
            will be created and used for the example.
    """
    try:
        logger.info("AD Diffusion - Quick Example")
        logger.info("=" * 50)

        # Step 0: Ensure model weights are available (auto-download from HF if missing).
        resolved_model_path = model_path or DEFAULT_MODEL_FILENAME
        resolved_config_path = config_path or DEFAULT_CONFIG_FILENAME

        if not skip_download and (not Path(resolved_model_path).exists() or not Path(resolved_config_path).exists()):
            logger.info(f"Model weights not found locally — downloading from Hugging Face ({HF_REPO_ID})...")
            try:
                resolved_model_path, resolved_config_path = download_model_weights(
                    model_path=resolved_model_path,
                    config_path=resolved_config_path,
                )
                logger.info(f"Using model:  {resolved_model_path}")
                logger.info(f"Using config: {resolved_config_path}")
            except ImportError:
                logger.error(
                    "huggingface_hub is not installed. Install it with "
                    "`uv add huggingface_hub` or run with --model-path pointing to local weights."
                )
                raise
            except Exception as e:
                logger.error(f"Could not download weights from Hugging Face: {e}")
                logger.info(
                    "If the repository is gated, run `huggingface-cli login` or "
                    "`export HUGGINGFACE_HUB_TOKEN='hf_...'` and try again."
                )
                resolved_model_path = model_path  # fall back to whatever the user passed

        # Step 1: Create or load dataset
        if dataset_path and Path(dataset_path).exists():
            logger.info(f"Using provided dataset: {dataset_path}")
            df = pd.read_csv(dataset_path)
            labels_path = None  # No ground truth for custom datasets
        else:
            if dataset_path:
                logger.warning(f"Dataset not found at {dataset_path}, creating synthetic dataset instead")
            else:
                logger.info("No dataset provided, creating synthetic dataset...")

            sample_dataset_path, labels_path = save_sample_dataset()
            df = pd.read_csv(sample_dataset_path)

        # Step 2: Load the dataset
        logger.info("Dataset loaded successfully")

        # Display basic info about the dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(
            f"Data range: {df.select_dtypes(include=[np.number]).min().min():.2f} to {df.select_dtypes(include=[np.number]).max().max():.2f}"
        )

        # Step 3: Prepare data for anomaly detection
        # Remove timestamp column for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        analysis_df = df[numeric_columns].copy()

        logger.info(f"Using {len(numeric_columns)} numeric columns for analysis: {numeric_columns}")

        # Step 4: Run anomaly detection
        if resolved_model_path and Path(resolved_model_path).exists():
            logger.info(f"Using model: {resolved_model_path}")

            try:
                results = perform_anomaly_analysis_with_diffusion(
                    df=analysis_df,
                    threshold_strategy="scs",  # Try "macs" as alternative
                    model_path=resolved_model_path,
                    config_path=resolved_config_path if Path(resolved_config_path).exists() else "",
                    nsample=15,
                    # preprocess_model_dir="/path/to/preprocessing/models"  # Optional
                )

                # Step 5: Display results
                logger.info("Anomaly Detection Results:")
                logger.info("-" * 30)

                n_anomalies = results["Anomaly"].sum()
                anomaly_rate = n_anomalies / len(results) * 100

                logger.info(f"Total samples analyzed: {len(results)}")
                logger.info(f"Anomalies detected: {n_anomalies}")
                logger.info(f"Anomaly rate: {anomaly_rate:.2f}%")

                # Show anomaly score statistics
                mae_stats = results["MAE"].describe()
                logger.info("MAE Score Statistics:")
                logger.info(f"  Mean: {mae_stats['mean']:.4f}")
                logger.info(f"  Std:  {mae_stats['std']:.4f}")
                logger.info(f"  Max:  {mae_stats['max']:.4f}")

                # Show top anomalies
                top_anomalies = results.nlargest(5, "MAE")[["MAE", "Anomaly"] + numeric_columns]
                logger.info("Top 5 Anomaly Scores:")
                logger.info(top_anomalies.to_string(index=False))

                # Step 6: Evaluate against ground truth (if available)
                if labels_path:
                    try:
                        labels_df = pd.read_csv(labels_path)
                        if len(labels_df) == len(results):
                            ground_truth = labels_df["is_anomaly"].values
                            predicted = results["Anomaly"].values

                            # Calculate basic metrics
                            true_positives = np.sum((ground_truth == 1) & (predicted == 1))
                            false_positives = np.sum((ground_truth == 0) & (predicted == 1))
                            false_negatives = np.sum((ground_truth == 1) & (predicted == 0))

                            precision = (
                                true_positives / (true_positives + false_positives)
                                if (true_positives + false_positives) > 0
                                else 0
                            )
                            recall = (
                                true_positives / (true_positives + false_negatives)
                                if (true_positives + false_negatives) > 0
                                else 0
                            )
                            f1_score = (
                                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            )

                            logger.info("Evaluation Metrics (vs Ground Truth):")
                            logger.info(f"  Precision: {precision:.3f}")
                            logger.info(f"  Recall:    {recall:.3f}")
                            logger.info(f"  F1-Score:  {f1_score:.3f}")
                        else:
                            logger.warning(
                                f"Ground truth length ({len(labels_df)}) doesn't match results ({len(results)})"
                            )

                    except Exception as e:
                        logger.warning(f"Could not evaluate against ground truth: {e}")
                else:
                    logger.info("No ground truth available for custom dataset - skipping evaluation metrics")

                # Step 7: Save results
                output_path = Path(__file__).parent / "datasets" / "anomaly_results.csv"
                results_with_timestamp = results.copy()
                if "timestamp" in df.columns:
                    results_with_timestamp["timestamp"] = df["timestamp"]
                results_with_timestamp.to_csv(output_path, index=False)
                logger.info(f"Results saved to: {output_path}")

            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                logger.info("This might be due to missing model file or incompatible data size.")
                logger.info(
                    "Please ensure you have a trained NV-Tesseract AD diffusion model and sufficient data samples."
                )

        else:
            logger.warning("No model weights found locally and auto-download was skipped/failed.")
            logger.info("To run with the pretrained Tesseract AD Diffusion model:")
            logger.info(f"  python quick_example.py                       # auto-download from {HF_REPO_ID}")
            logger.info("  python quick_example.py --download-weights     # pre-download only")
            logger.info("  python quick_example.py --model-path /path/to/final_model.pth")
            logger.info("")
            logger.info("For now, showing the prepared dataset:")
            logger.info("Dataset preview:")
            logger.info(analysis_df.head().to_string())

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install the package first: uv sync")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="AD Diffusion Quick Example")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to the NV-Tesseract diffusion model checkpoint (.pth file). "
            f"Defaults to '{DEFAULT_MODEL_FILENAME}' (auto-downloaded from {HF_REPO_ID})."
        ),
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Path to the model config YAML. "
            f"Defaults to '{DEFAULT_CONFIG_FILENAME}' (auto-downloaded from {HF_REPO_ID})."
        ),
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help=(
            "Pre-download the default weights from Hugging Face and exit. "
            "Useful to warm up the cache before running inference."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not auto-download weights. Requires --model-path to point to local weights.",
    )
    parser.add_argument(
        "--create-dataset-only",
        action="store_true",
        help="Only create the sample dataset without running anomaly detection",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Path to your own CSV dataset for anomaly detection. "
            "The dataset should contain numeric columns for analysis. "
            "If not provided, a synthetic dataset will be generated."
        ),
    )

    args = parser.parse_args()

    if args.create_dataset_only:
        save_sample_dataset()
        logger.info("Sample dataset created successfully!")
    elif args.download_weights:
        model_file = args.model_path or DEFAULT_MODEL_FILENAME
        config_file = args.config_path or DEFAULT_CONFIG_FILENAME
        logger.info(f"Downloading default AD Diffusion weights from {HF_REPO_ID}...")
        model_file, config_file = download_model_weights(
            model_path=model_file,
            config_path=config_file,
        )
        logger.info(f"✓ Model:  {model_file}")
        logger.info(f"✓ Config: {config_file}")
    else:
        run_anomaly_detection_example(
            args.model_path,
            args.config_path,
            skip_download=args.skip_download,
            dataset_path=args.dataset_path,
        )


if __name__ == "__main__":
    main()
