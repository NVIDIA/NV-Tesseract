# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference function for anomaly detection using NV-Tesseract AD diffusion model

This script defines functions that perform anomaly detection on datasets using NV-Tesseract diffusion models.
Note: This package uses a flat import structure - ensure the ad_diffusion directory is in your Python path.

Usage:
import sys, os
sys.path.append('/path/to/ad_diffusion')  # Adjust path as needed
from sdk.inference_ad import inference_ad_tesseract2

# Explicit paths
results = inference_ad_tesseract2(data, model_path, config_path, nsample=30)

# Or let the SDK auto-download weights from Hugging Face
# (nvidia/nv-tesseract-ad-diffusion -> final_model.pth + curriculum_medium.yaml)
results = inference_ad_tesseract2(data, nsample=30)

# Pre-fetch weights manually (e.g. to pick a custom cache directory)
from sdk.inference_ad import download_model_weights
model_path, config_path = download_model_weights(
    model_path="weights/final_model.pth",
    config_path="weights/curriculum_medium.yaml",
)

# The results is a dictionary containing the following keys:
# "residual": residual_mae, # The mean absolute error between the reconstructed and target data
# "residual_l2": residual_l2, # The L2 norm of the difference between the reconstructed and target data
# "target": target_flat, # The target/original data after preprocessing
# "recon": recon_flat, # The reconstructed data after imputation
# "target_dim": target_dim, # The target dimension used by the model

# For reproducible results, call set_seed() before inference:
from sdk.inference_ad import set_seed
set_seed(42)  # Sets Python, NumPy, PyTorch, CUDA seeds and cuDNN determinism
results = inference_ad_tesseract2(data, model_path, config_path, nsample=30)

# For fast inference with DPM-Solver (50-100x speedup):
results = inference_ad_tesseract2(data, model_path, config_path, nsample=30,
                                  use_dpm_solver=True, dpm_steps=20)

# For multi-GPU inference (spawns workers per call):
from sdk.inference_ad import inference_ad_tesseract2_mp
results = inference_ad_tesseract2_mp(data, "model.pth", nsample=30)

# For multi-GPU inference with DPM-Solver (massive speedup):
results = inference_ad_tesseract2_mp(data, "model.pth", nsample=30,
                                     use_dpm_solver=True, dpm_steps=20)
"""

import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.main_model import TSDiffuser_Generic
from models.utils import evaluate
from utils.tsb_ad_preprocessor import preprocess_for_inference

# DEVICE configuration - simplified for standalone use
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default Hugging Face repository and asset names for the AD Diffusion model
HF_REPO_ID = "nvidia/nv-tesseract-ad-diffusion"
DEFAULT_MODEL_FILENAME = "final_model.pth"
DEFAULT_CONFIG_FILENAME = "curriculum_medium.yaml"

# Path to worker script for multi-GPU inference
_WORKER_SCRIPT = Path(__file__).parent / "inference_worker.py"

# Logger for profiling residual calculations
logger = logging.getLogger(__name__)

# Enable profiling via environment variable
PROFILE_RESIDUALS = os.environ.get("TESSERACT_PROFILE_RESIDUALS", "0") == "1"

# Default seed for reproducibility
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducibility across all random number generators.

    This function ensures deterministic behavior by setting seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's CPU random number generator
    - PyTorch's CUDA random number generators (all GPUs)
    - cuDNN backend determinism flags

    Args:
        seed: Integer seed value for reproducibility. Default is 42.

    Note:
        Setting torch.backends.cudnn.deterministic = True may impact performance
        but ensures reproducible results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Note: set_seed() is NOT called at module load to avoid affecting other code.
# Call set_seed() explicitly when reproducibility is needed.


def min_max_normalize(data):
    """
    Normalize train and test arrays to [0, 1] per feature using min and max from train.
    Does NOT touch labels. Updates train/test in-place.
    """
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    denom = np.where(max_val - min_val == 0, 1, max_val - min_val)
    data_norm = (data - min_val) / denom
    return data_norm


def pca_features(data, target_dim, random_state=DEFAULT_SEED):
    """
    PCA features for the given data.

    Args:
        data: Input data array
        target_dim: Number of components to keep
        random_state: Random state for reproducibility (used in randomized SVD)

    Returns:
        Transformed data with reduced dimensions
    """
    pca = PCA(n_components=target_dim, random_state=random_state)
    pca.fit(data)
    return pca.transform(data)


def pad_features(data, target_dim):
    """Pad arr to target_cols columns with zeros (on the right)."""
    n_samples, n_cols = data.shape
    if n_cols == target_dim:
        return data
    if n_cols < target_dim:
        pad_width = target_dim - n_cols
        return np.pad(data, ((0, 0), (0, pad_width)), "constant", constant_values=0)
    # Shouldn't happen, but just in case
    return data[:, :target_dim]


def match_target_dim(data, target_dim, random_state=DEFAULT_SEED):
    """
    Match data dimensions to target dimension using padding or PCA.

    Args:
        data (np.ndarray): Input data array
        target_dim (int): Target number of dimensions
        random_state (int): Random state for PCA reproducibility

    Returns:
        np.ndarray: Data with matched dimensions

    Note:
        - If data has fewer columns than target_dim: pads with zeros
        - If data has more columns than target_dim: applies PCA reduction
        - If data has exactly target_dim columns: returns unchanged
    """
    _, n_cols = data.shape
    if n_cols == target_dim:
        return data
    if n_cols < target_dim:
        return pad_features(data, target_dim)
    return pca_features(data, target_dim, random_state=random_state)


class InferenceData(Dataset):
    """
    Dataset class for inferencing on time series data with consistent masking strategies.

    This dataset implements a sliding window approach for time series analysis,
    with support for multiple masking strategies and automatic padding for
    consistent tensor sizes across batches.

    Note:
        The dataset expects pre-cleaned data. Users should remove any unwanted columns
        before passing the DataFrame. All numeric columns in the DataFrame will be used
        as features. String/object columns are automatically excluded.

    Attributes:
        data (torch.Tensor): Preprocessed time series data
        window_length (int): Length of each time series window
        strategy (int): Masking strategy (0 or 1)
        split (int): Number of splits for masking
        begin_indexes (list): Starting indices for each window

    Methods:
        __init__: Initialize dataset with data loading and preprocessing
        _create_mask: Generate masks based on fixed strategy
        __len__: Return number of windows
        __getitem__: Get a single window with proper tensor formatting
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_dim: int,
        window_length=100,
        window_split=1,
        strategy=1,
        split=4,
        scale_factor=20,
        domain=None,
        model_dir=None,
        requires_preprocessing=True,
    ):
        """
        Initialize the InferenceData dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            target_dim (int): Target dimensionality for data
            window_length (int): Length of each time series window
            window_split (int): Split factor for window calculation
            strategy (int): Masking strategy (0: even splits, 1: odd splits)
            split (int): Number of splits within a window for masking
            scale_factor (int): Scale factor for data
            domain (str, optional): Domain for preprocessing
            model_dir (str, optional): Model directory for preprocessing
            requires_preprocessing (bool): Whether to apply preprocessing

        Note:
            Automatically handles dimensionality matching and data preprocessing.
        """
        # Drop the columns with string values
        df = df.select_dtypes(exclude=["object"])
        # Use all numeric columns as features
        self.data = df.values

        # Apply TSB-AD preprocessing
        if requires_preprocessing:
            if model_dir is not None:
                # Let the preprocessor handle the metadata columns based on target_dim
                self.data = preprocess_for_inference(
                    self.data,
                    model_dir=model_dir,
                    domain=domain,
                    target_dim=target_dim,
                    add_metadata=True,  # Always add metadata for consistent dimensions
                )
                # Check for extreme values before scaling
                if np.any(np.isnan(self.data)):
                    self.data = np.nan_to_num(self.data, nan=0.0)
                if np.any(np.isinf(self.data)):
                    self.data = np.nan_to_num(self.data, posinf=100.0, neginf=-100.0)

                # Additional clipping to ensure reasonable range
                data_max = np.max(np.abs(self.data))
                if data_max > 1000:
                    self.data = np.clip(self.data, -1000, 1000)

                # Scale the data
                self.data = torch.Tensor(self.data) * scale_factor
            else:
                # Fallback to original preprocessing if model_dir not provided
                self.data = match_target_dim(self.data, target_dim)
                self.data = min_max_normalize(self.data)
                self.data = torch.Tensor(self.data) * scale_factor
        else:
            logger.info("No preprocessing applied")

        self.window_length = window_length
        self.strategy = strategy
        self.split = split

        # Calculate window indices
        step = self.window_length // window_split
        self.begin_indexes = list(range(0, len(self.data) - window_length + step, step))

    def _create_mask(self, observed_mask):
        """
        Generate mask based on fixed strategy.

        Args:
            observed_mask (torch.Tensor): Base mask tensor

        Returns:
            torch.Tensor: Generated mask based on strategy

        Note:
            Strategy 0: Masks even-numbered splits
            Strategy 1: Masks odd-numbered splits
            Creates alternating masked/unmasked regions for robust evaluation.
        """
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        skip = length // self.split

        for split_idx in range(self.split):
            start = split_idx * skip
            end = min(start + skip, length)

            if (split_idx % 2 == 0 and self.strategy == 0) or (split_idx % 2 != 0 and self.strategy == 1):
                mask[start:end] = 1

        return mask

    def __len__(self):
        """Return the number of windows in the dataset."""
        return len(self.begin_indexes)

    def __getitem__(self, idx):
        start_idx = self.begin_indexes[idx]
        # Ensure both tensors have the correct window length

        observed_data = self.data[start_idx : start_idx + self.window_length]

        # Ensure observed_data is a tensor
        if not isinstance(observed_data, torch.Tensor):
            observed_data = torch.tensor(observed_data, dtype=torch.float32)

        if observed_data.shape[0] < self.window_length:
            # Pad with the last value
            pad_size = self.window_length - observed_data.shape[0]
            observed_data = torch.cat([observed_data, observed_data[-1:].repeat(pad_size, 1)], dim=0)

        item = {
            "observed_data": observed_data,
            "observed_mask": torch.ones_like(observed_data),
            "gt_mask": self._create_mask(observed_data),
            "timepoints": torch.arange(self.window_length),
            "strategy_type": self.strategy,
        }

        return item


class PreprocessedInferenceData(Dataset):
    """Dataset for inference when preprocessing is already done."""

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        window_length: int = 100,
        window_split: int = 1,
        strategy: int = 1,
        split: int = 4,
        begin_indexes: list[int] | None = None,
    ) -> None:
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = data.float()

        self.window_length = window_length
        self.strategy = strategy
        self.split = split

        if begin_indexes is None:
            step = self.window_length // window_split
            self.begin_indexes = list(range(0, len(self.data) - window_length + step, step))
        else:
            self.begin_indexes = begin_indexes

    def _create_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        skip = length // self.split

        for split_idx in range(self.split):
            start = split_idx * skip
            end = min(start + skip, length)

            if (split_idx % 2 == 0 and self.strategy == 0) or (split_idx % 2 != 0 and self.strategy == 1):
                mask[start:end] = 1

        return mask

    def __len__(self) -> int:
        return len(self.begin_indexes)

    def __getitem__(self, idx: int) -> dict:
        start_idx = self.begin_indexes[idx]
        observed_data = self.data[start_idx : start_idx + self.window_length]

        if observed_data.shape[0] < self.window_length:
            pad_size = self.window_length - observed_data.shape[0]
            observed_data = torch.cat([observed_data, observed_data[-1:].repeat(pad_size, 1)], dim=0)

        return {
            "observed_data": observed_data,
            "observed_mask": torch.ones_like(observed_data),
            "gt_mask": self._create_mask(observed_data),
            "timepoints": torch.arange(self.window_length),
            "strategy_type": self.strategy,
        }


class WindowTensorDataset(Dataset):
    """Dataset for inference when window tensors are already prepared."""

    def __init__(
        self,
        windows: np.ndarray | torch.Tensor,
        *,
        strategy: int = 1,
        split: int = 4,
        window_indices: list[int] | None = None,
    ) -> None:
        if isinstance(windows, np.ndarray):
            self.windows = torch.tensor(windows, dtype=torch.float32)
        else:
            self.windows = windows.float()

        self.strategy = strategy
        self.split = split
        self.window_indices = window_indices or list(range(len(self.windows)))

    def _create_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        skip = length // self.split

        for split_idx in range(self.split):
            start = split_idx * skip
            end = min(start + skip, length)

            if (split_idx % 2 == 0 and self.strategy == 0) or (split_idx % 2 != 0 and self.strategy == 1):
                mask[start:end] = 1

        return mask

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> dict:
        win_idx = self.window_indices[idx]
        observed_data = self.windows[win_idx]
        window_length = observed_data.shape[0]

        return {
            "observed_data": observed_data,
            "observed_mask": torch.ones_like(observed_data),
            "gt_mask": self._create_mask(observed_data),
            "timepoints": torch.arange(window_length),
            "strategy_type": self.strategy,
        }


def get_dataloader(
    data,
    target_dim,
    batch_size=32,
    window_split=1,
    split=4,
    scale_factor=20,
    domain=None,
    model_dir=None,
    requires_preprocessing=True,
):
    """
    Create data loaders for inference.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        target_dim (int): Target dimensionality
        batch_size (int): Batch size for data loaders
        window_split (int): Split factor for windows
        split (int): Number of splits for masking

    Returns:
        tuple: (test_loader1, test_loader2)
            - test_loader1: DataLoader with strategy 0
            - test_loader2: DataLoader with strategy 1

    Note:
        Creates two data loaders with different masking strategies
        for robust evaluation of the model's performance.
    """
    # Create test datasets
    test_args = {
        "window_split": window_split,
        "split": split,
        "scale_factor": scale_factor,
        "requires_preprocessing": requires_preprocessing,
    }
    test_data1 = InferenceData(data, target_dim, strategy=0, domain=domain, model_dir=model_dir, **test_args)
    test_data2 = InferenceData(data, target_dim, strategy=1, domain=domain, model_dir=model_dir, **test_args)
    test_loader1 = DataLoader(test_data1, batch_size=batch_size)
    test_loader2 = DataLoader(test_data2, batch_size=batch_size)

    return (test_loader1, test_loader2)


def get_dataloader_from_array(
    data: np.ndarray | torch.Tensor,
    batch_size: int = 32,
    window_split: int = 1,
    split: int = 4,
    window_length: int = 100,
    begin_indexes: list[int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create data loaders for inference from preprocessed arrays."""
    test_args = {
        "window_split": window_split,
        "split": split,
        "window_length": window_length,
        "begin_indexes": begin_indexes,
    }
    test_data1 = PreprocessedInferenceData(data, strategy=0, **test_args)
    test_data2 = PreprocessedInferenceData(data, strategy=1, **test_args)
    test_loader1 = DataLoader(test_data1, batch_size=batch_size)
    test_loader2 = DataLoader(test_data2, batch_size=batch_size)

    return (test_loader1, test_loader2)


def get_dataloader_from_windows(
    windows: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 32,
    split: int = 4,
    window_indices: list[int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create data loaders from precomputed window tensors."""
    test_data1 = WindowTensorDataset(windows, strategy=0, split=split, window_indices=window_indices)
    test_data2 = WindowTensorDataset(windows, strategy=1, split=split, window_indices=window_indices)
    test_loader1 = DataLoader(test_data1, batch_size=batch_size)
    test_loader2 = DataLoader(test_data2, batch_size=batch_size)
    return (test_loader1, test_loader2)


def preprocess_dataframe(
    df: pd.DataFrame,
    target_dim: int,
    *,
    scale_factor: float = 20,
    domain: str | None = None,
    model_dir: str | None = None,
) -> torch.Tensor:
    """Preprocess a dataframe once for multi-GPU inference."""
    df = df.select_dtypes(exclude=["object"])
    data = df.values

    if model_dir is not None:
        data = preprocess_for_inference(
            data,
            model_dir=model_dir,
            domain=domain,
            target_dim=target_dim,
            add_metadata=True,
        )
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
        if np.any(np.isinf(data)):
            data = np.nan_to_num(data, posinf=100.0, neginf=-100.0)
        data_max = np.max(np.abs(data))
        if data_max > 1000:
            data = np.clip(data, -1000, 1000)
        return torch.tensor(data, dtype=torch.float32) * scale_factor

    if data.shape[0] < target_dim and data.shape[1] > target_dim:
        raise ValueError(f"Insufficient samples for PCA: got {data.shape[0]} samples but target_dim={target_dim}.")
    data = match_target_dim(data, target_dim)
    data = min_max_normalize(data)
    return torch.tensor(data, dtype=torch.float32) * scale_factor


def _build_window_indexes(total_rows: int, window_length: int, window_split: int) -> list[int]:
    step = max(1, window_length // max(1, window_split))
    if total_rows < window_length:
        return [0]
    return list(range(0, total_rows - window_length + step, step))


def _split_indexes(indexes: list[int], num_parts: int) -> list[list[int]]:
    if num_parts <= 1:
        return [indexes]
    chunk_size = len(indexes) // num_parts
    remainder = len(indexes) % num_parts
    chunks: list[list[int]] = []
    start = 0
    for i in range(num_parts):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(indexes[start:end])
        start = end
    return chunks


def _build_window_tensor(
    data: np.ndarray,
    begin_indexes: list[int],
    window_length: int,
) -> np.ndarray:
    windows: list[np.ndarray] = []
    for start_idx in begin_indexes:
        window = data[start_idx : start_idx + window_length]
        if window.shape[0] < window_length:
            pad_size = window_length - window.shape[0]
            pad = np.repeat(window[-1:], pad_size, axis=0)
            window = np.concatenate([window, pad], axis=0)
        windows.append(window)
    if not windows:
        return np.empty((0, window_length, data.shape[1]), dtype=data.dtype)
    return np.stack(windows, axis=0)


def _create_shared_memory(array: np.ndarray) -> dict:
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shm_array[:] = array
    return {"name": shm.name, "shape": array.shape, "dtype": array.dtype}


def _cleanup_shared_memory(shm_info: dict) -> None:
    try:
        shm = shared_memory.SharedMemory(name=shm_info["name"])
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


def _merge_chunked_results(results_per_chunk: list[dict | None], target_dim: int) -> dict:
    all_residuals = []
    all_residuals_l2 = []
    all_targets = []
    all_recons = []

    for chunk_result in results_per_chunk:
        if chunk_result is None:
            continue
        all_residuals.append(chunk_result["residual"])
        all_residuals_l2.append(chunk_result["residual_l2"])
        all_targets.append(chunk_result["target"])
        all_recons.append(chunk_result["recon"])

    return {
        "residual": np.concatenate(all_residuals),
        "residual_l2": np.concatenate(all_residuals_l2),
        "target": np.concatenate(all_targets),
        "recon": np.concatenate(all_recons),
        "target_dim": target_dim,
    }


def prepare_for_metrics(recon, target):
    """
    Aligns and flattens the model outputs for metric calculation.
    """

    recon_flat = recon.reshape(-1, recon.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    n_points = min(recon_flat.shape[0], target_flat.shape[0])
    recon_flat = recon_flat[:n_points]
    target_flat = target_flat[:n_points]

    return recon_flat, target_flat


def evaluate_ad_raw_samples(model, test_loader1, test_loader2, nsample=30, use_dpm_solver=False, dpm_steps=20):
    """
    Evaluate the model and return raw samples before median.

    Used for multi-GPU parallelization where each GPU generates partial samples,
    then samples are combined before taking the final median.

    Args:
        model: The diffusion model
        test_loader1: First test dataloader
        test_loader2: Second test dataloader
        nsample: Number of samples to generate
        use_dpm_solver: If True, use DPM-Solver for fast inference
        dpm_steps: Number of DPM-Solver steps (10-50)

    Returns:
        dict with "generated_samples" (tensor) and "target" (tensor)
    """
    eval_outputs = evaluate(
        model,
        test_loader1,
        test_loader2,
        nsample=nsample,
        save_results=False,
        use_dpm_solver=use_dpm_solver,
        dpm_steps=dpm_steps,
    )
    return {
        "generated_samples": eval_outputs["generated_samples"],  # (batch, nsample, window, dim)
        "target": eval_outputs["target"],  # (batch, window, dim)
    }


def combine_samples_and_compute_residuals(samples_list: list, target) -> dict:
    """
    Combine samples from multiple GPUs and compute residuals.

    Args:
        samples_list: List of sample tensors from different GPUs, each (batch, partial_nsample, window, dim)
        target: Target tensor (batch, window, dim)

    Returns:
        dict with residual, residual_l2, target, recon keys
    """
    # Concatenate samples along nsample dimension (dim=1)
    if isinstance(samples_list[0], torch.Tensor):
        all_samples = torch.cat(samples_list, dim=1)  # (batch, total_nsample, window, dim)
        recon = all_samples.median(dim=1).values.cpu().numpy()
    else:
        all_samples = np.concatenate(samples_list, axis=1)
        recon = np.median(all_samples, axis=1)

    if hasattr(target, "cpu"):
        target_np = target.cpu().numpy()
    else:
        target_np = target

    # Align and flatten
    recon_flat, target_flat = prepare_for_metrics(recon, target_np)

    # Compute residuals with clipping
    MAX_VALUE = 1e8
    recon_clipped = np.clip(recon_flat, -MAX_VALUE, MAX_VALUE)
    target_clipped = np.clip(target_flat, -MAX_VALUE, MAX_VALUE)
    recon_clipped = np.nan_to_num(recon_clipped, nan=0.0, posinf=MAX_VALUE, neginf=-MAX_VALUE)
    target_clipped = np.nan_to_num(target_clipped, nan=0.0, posinf=MAX_VALUE, neginf=-MAX_VALUE)

    residual_l2 = np.linalg.norm(recon_clipped - target_clipped, axis=1)
    residual_mae = np.mean(np.abs(recon_clipped - target_clipped), axis=1)
    residual_l2 = np.clip(residual_l2, 0, MAX_VALUE)
    residual_mae = np.clip(residual_mae, 0, MAX_VALUE)

    return {
        "residual": residual_mae,
        "residual_l2": residual_l2,
        "target": target_flat,
        "recon": recon_flat,
    }


def evaluate_ad_tesseract2(model, test_loader1, test_loader2, nsample=30, use_dpm_solver=False, dpm_steps=20):
    """
    Evaluate the model on the test data with optional DPM-Solver support.

    Args:
        model: The diffusion model
        test_loader1: First test dataloader
        test_loader2: Second test dataloader
        nsample: Number of samples to generate
        use_dpm_solver: If True, use DPM-Solver for 50-100x faster inference
        dpm_steps: Number of DPM-Solver steps (10-50, default: 20)

    If TESSERACT_PROFILE_RESIDUALS=1 environment variable is set, logs detailed
    timing information for each step of residual calculation.
    """
    timings = {}

    # Step 1: Diffusion sampling (the heavy work)
    t0 = time.perf_counter()
    eval_outputs = evaluate(
        model,
        test_loader1,
        test_loader2,
        nsample=nsample,
        save_results=False,
        use_dpm_solver=use_dpm_solver,
        dpm_steps=dpm_steps,
    )
    timings["diffusion_sampling"] = time.perf_counter() - t0

    all_generated_samples = eval_outputs["generated_samples"]  # This is the reconstructed data meaning the imputed data
    all_target = eval_outputs["target"]  # This is the target data means the original data

    # Step 2: Median calculation over samples
    t0 = time.perf_counter()
    if isinstance(all_generated_samples, torch.Tensor):
        recon = all_generated_samples.median(dim=1).values.cpu().numpy()
    else:
        recon = np.median(all_generated_samples, axis=1)
    timings["median_calculation"] = time.perf_counter() - t0

    # Step 3: CPU/numpy conversion
    t0 = time.perf_counter()
    if hasattr(all_target, "cpu"):
        target = all_target.cpu().numpy()
    else:
        target = all_target
    timings["cpu_conversion"] = time.perf_counter() - t0

    # Step 4: Align and flatten
    # The output of the evaluate function for all_generated_samples and all_target are in the shape of (nsample, window_length, target_dim)
    # We need to align and flatten them to the shape of (nsample * window_length, target_dim)
    t0 = time.perf_counter()
    recon_flat, target_flat = prepare_for_metrics(recon, target)
    timings["prepare_for_metrics"] = time.perf_counter() - t0

    # Step 5: Clipping operations
    t0 = time.perf_counter()
    MAX_VALUE = 1e8  # Maximum reasonable value to prevent overflow
    recon_clipped = np.clip(recon_flat, -MAX_VALUE, MAX_VALUE)
    target_clipped = np.clip(target_flat, -MAX_VALUE, MAX_VALUE)
    timings["clipping"] = time.perf_counter() - t0

    # Step 6: NaN/Inf replacement
    t0 = time.perf_counter()
    recon_clipped = np.nan_to_num(recon_clipped, nan=0.0, posinf=MAX_VALUE, neginf=-MAX_VALUE)
    target_clipped = np.nan_to_num(target_clipped, nan=0.0, posinf=MAX_VALUE, neginf=-MAX_VALUE)
    timings["nan_to_num"] = time.perf_counter() - t0

    # Step 7: Compute residuals
    t0 = time.perf_counter()
    residual_l2 = np.linalg.norm(recon_clipped - target_clipped, axis=1)
    residual_mae = np.mean(np.abs(recon_clipped - target_clipped), axis=1)

    # Additional clipping for residuals
    residual_l2 = np.clip(residual_l2, 0, MAX_VALUE)
    residual_mae = np.clip(residual_mae, 0, MAX_VALUE)
    timings["residual_math"] = time.perf_counter() - t0

    # Log profiling information if enabled
    if PROFILE_RESIDUALS:
        total_time = sum(timings.values())
        logger.info("=" * 60)
        logger.info("RESIDUAL CALCULATION PROFILING (TESSERACT_PROFILE_RESIDUALS=1)")
        logger.info("=" * 60)
        for step, duration in timings.items():
            pct = (duration / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  {step:25s}: {duration:8.3f}s ({pct:5.1f}%)")
        logger.info("-" * 60)
        logger.info(f"  {'TOTAL':25s}: {total_time:8.3f}s")
        logger.info(f"  Data shape: recon={recon_flat.shape}, target={target_flat.shape}")
        logger.info("=" * 60)

    return_results = {
        "residual": residual_mae,  # The mean absolute error of the difference between the reconstructed and target data (MAE)
        "residual_l2": residual_l2,  # The L2 norm of the difference between the reconstructed and target data
        "target": target_flat,  # The target/original data after preprocessing
        "recon": recon_flat,  # The reconstructed data after imputation
    }

    return return_results


def download_model_weights(
    model_path: str = DEFAULT_MODEL_FILENAME,
    config_path: str = DEFAULT_CONFIG_FILENAME,
    repo_id: str = HF_REPO_ID,
    force_download: bool = False,
) -> tuple[str, str]:
    """
    Auto-download AD Diffusion model weights from Hugging Face if they don't exist locally.

    Args:
        model_path: Local path for the model checkpoint (default: final_model.pth)
        config_path: Local path for the model config YAML (default: curriculum_medium.yaml)
        repo_id: Hugging Face repository ID (default: nvidia/nv-tesseract-ad-diffusion)
        force_download: Force re-download even if files exist

    Returns:
        Tuple of (model_path, config_path) as strings pointing to the local files.

    Raises:
        ImportError: If huggingface_hub is not installed.
        Exception: If the download fails (e.g., due to missing authentication).

    Note:
        Weights are published on the Hugging Face repository and download
        without authentication. If a download fails with 401/403, accept the
        model license on the repo page or authenticate first:
            1. Install the CLI:  `uv add huggingface_hub[cli]`  (or `pip install huggingface_hub[cli]`)
            2. Login:            `huggingface-cli login`
            3. Or set a token:   `export HUGGINGFACE_HUB_TOKEN='your_token'`
    """
    model_file = Path(model_path)
    config_file = Path(config_path)

    # Fast path: both files already exist and we are not forcing a redownload.
    if not force_download and model_file.exists() and config_file.exists():
        return str(model_file), str(config_file)

    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to download model weights. "
            "Install it with: `uv add huggingface_hub` or `pip install huggingface_hub`."
        )

    logger.info("Downloading AD Diffusion weights from Hugging Face (%s)...", repo_id)

    # Create parent directories if the user specified a subdirectory.
    if model_file.parent != Path():
        model_file.parent.mkdir(parents=True, exist_ok=True)
    if config_file.parent != Path():
        config_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        if force_download or not model_file.exists():
            logger.info("Downloading %s...", model_file.name)
            hf_hub_download(
                repo_id=repo_id,
                filename=model_file.name,
                local_dir=str(model_file.parent) if model_file.parent != Path() else ".",
                local_dir_use_symlinks=False,
                force_download=force_download,
            )
            logger.info("Downloaded %s", model_file)

        if force_download or not config_file.exists():
            logger.info("Downloading %s...", config_file.name)
            hf_hub_download(
                repo_id=repo_id,
                filename=config_file.name,
                local_dir=str(config_file.parent) if config_file.parent != Path() else ".",
                local_dir_use_symlinks=False,
                force_download=force_download,
            )
            logger.info("Downloaded %s", config_file)

    except Exception as e:
        error_msg = f"Failed to download model weights from {repo_id}: {e}"
        if "401" in str(e) or "403" in str(e):
            error_msg += (
                "\n\nDownload failed. Accept the model license on Hugging Face or authenticate:"
                "\n  1. Install huggingface-cli: `uv add huggingface_hub[cli]`"
                "\n  2. Login:                  `huggingface-cli login`"
                "\n  3. Or set a token:         `export HUGGINGFACE_HUB_TOKEN='your_token'`"
            )
        raise Exception(error_msg) from e

    return str(model_file), str(config_file)


def _resolve_model_paths(
    model_path: str | None,
    config_path: str | None = "",
    repo_id: str = HF_REPO_ID,
) -> tuple[str, str]:
    """
    Resolve model and config paths, auto-downloading from Hugging Face if needed.

    Behavior:
        - If ``model_path`` is ``None`` or empty, defaults to ``final_model.pth`` in the CWD.
        - If ``config_path`` is ``None`` or empty, defaults to ``curriculum_medium.yaml`` in the CWD.
        - If either file does not exist locally, the missing asset is fetched from
          ``repo_id`` on Hugging Face Hub.

    Args:
        model_path: Local checkpoint path (or ``None`` to use the default).
        config_path: Local config path (or ``None``/empty to use the default).
        repo_id: Hugging Face repository ID.

    Returns:
        Tuple of (resolved_model_path, resolved_config_path) as strings.
    """
    resolved_model = str(model_path) if model_path else DEFAULT_MODEL_FILENAME
    resolved_config = str(config_path) if config_path else DEFAULT_CONFIG_FILENAME

    if not Path(resolved_model).exists() or not Path(resolved_config).exists():
        resolved_model, resolved_config = download_model_weights(
            model_path=resolved_model,
            config_path=resolved_config,
            repo_id=repo_id,
        )

    return resolved_model, resolved_config


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_target_dim(model_path: str | None = None, config_path: str = "") -> int:
    """
    Extract target_dim from model checkpoint or config without loading the full model.

    If ``model_path``/``config_path`` do not exist locally, they are automatically
    downloaded from the Hugging Face repository ``nvidia/nv-tesseract-ad-diffusion``.

    Args:
        model_path: Path to the model checkpoint. If ``None`` or missing locally,
            the default ``final_model.pth`` is downloaded from Hugging Face.
        config_path: Path to config file. Optional if the config is embedded in the
            checkpoint; otherwise defaults to ``curriculum_medium.yaml`` from HF.

    Returns:
        int: Target dimension used by the model
    """
    resolved_model, resolved_config = _resolve_model_paths(model_path, config_path)
    checkpoint = torch.load(resolved_model, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", None)
    if config is None:
        assert resolved_config.strip() != "", "Must pass config path when the model path is a checkpoint"
        config = load_config(resolved_config)

    return config["model"].get("target_dim", 40)


def inference_ad_tesseract2(
    data: pd.DataFrame,
    model_path: str | None = None,
    config_path: str = "",
    nsample=30,
    preprocess_model_dir=None,
    use_dpm_solver=True,
    dpm_steps=20,
):
    """
    Perform anomaly detection inference using NV-Tesseract AD diffusion model.

    If ``model_path``/``config_path`` do not exist locally, they are automatically
    downloaded from ``nvidia/nv-tesseract-ad-diffusion`` on Hugging Face Hub.

    Args:
        data: DataFrame with pre-cleaned numeric data. Users should remove any unwanted
              columns beforehand. All numeric columns will be used as features.
        model_path: Path to the model checkpoint. If ``None`` or missing locally,
            ``final_model.pth`` is downloaded from Hugging Face.
        config_path: Path to config file. Optional if the config is embedded in the
            checkpoint; otherwise ``curriculum_medium.yaml`` is downloaded from HF.
        nsample: Number of samples for diffusion model inference
        preprocess_model_dir: Directory containing preprocessing model (optional)
        use_dpm_solver: If True, use DPM-Solver for 50-100x faster inference
        dpm_steps: Number of DPM-Solver steps (10-50, default: 20)

    Returns:
        dict: Dictionary containing:
            - residual: MAE between reconstructed and target data
            - residual_l2: L2 norm of reconstruction error
            - target: Original data after preprocessing
            - recon: Reconstructed data after imputation
            - target_dim: Target dimension used by the model

    Note:
        For reproducible results, call set_seed() before this function.
        Using DPM-Solver can provide 50-100x speedup over standard diffusion.
    """
    device = DEVICE

    resolved_model, resolved_config = _resolve_model_paths(model_path, config_path)
    model_path = Path(resolved_model)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", None)
    if config is None:
        assert resolved_config.strip() != "", "Must pass config path when the model path is a checkpoint"
        config = load_config(resolved_config)

    target_dim = config["model"].get("target_dim", 40)

    model = TSDiffuser_Generic(config, device=device, target_dim=target_dim, ratio=0.7)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            # Load model state
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    scale_factor = config.get("dataset", {}).get("scale_factor", 1.0)

    test_loader1, test_loader2 = get_dataloader(
        data, target_dim, scale_factor=scale_factor, model_dir=preprocess_model_dir
    )

    # Run inference with DPM support
    results = evaluate_ad_tesseract2(
        model,
        test_loader1,
        test_loader2,
        nsample=nsample,
        use_dpm_solver=use_dpm_solver,
        dpm_steps=dpm_steps,
    )

    # Add target_dim to results
    results["target_dim"] = target_dim

    return results


def inference_ad_tesseract2_mp(
    data: pd.DataFrame,
    model_path: str | None = None,
    config_path: str = "",
    nsample: int = 15,
    preprocess_model_dir: str | None = None,
    *,
    num_processes: int | None = None,
    gpu_ids: list[int] | None = None,
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    use_dpm_solver: bool = True,
    dpm_steps: int = 20,
) -> dict:
    """
    Multi-GPU inference using subprocess workers and shared-memory windows.

    If ``model_path``/``config_path`` do not exist locally, they are automatically
    downloaded from ``nvidia/nv-tesseract-ad-diffusion`` on Hugging Face Hub.

    Args:
        data: DataFrame with pre-cleaned numeric data.
        model_path: Path to the model checkpoint. If ``None`` or missing locally,
            ``final_model.pth`` is downloaded from Hugging Face.
        config_path: Path to config file. Optional if the config is embedded in the
            checkpoint; otherwise ``curriculum_medium.yaml`` is downloaded from HF.
        nsample: Number of diffusion samples per window.
        preprocess_model_dir: Directory containing preprocessing model (optional).
        num_processes: Number of GPU worker processes (defaults to len(gpu_ids)).
        gpu_ids: List of GPU ids to use (defaults to all visible GPUs).
        seed: Random seed for reproducibility.
        deterministic: If True, enable deterministic behavior in workers.
        use_dpm_solver: If True, use DPM-Solver for 50-100x faster inference
        dpm_steps: Number of DPM-Solver steps (10-50, default: 20)

    Returns:
        dict with residual, residual_l2, target, recon, target_dim keys.

    Note:
        Combining multiprocessing with DPM-Solver provides massive speedup!
        Example: 4 GPUs * 50x DPM speedup = 200x total speedup vs single-GPU standard diffusion.
    """
    # Ensure weights are available locally before spinning up workers (which
    # otherwise each race to download the same files).
    resolved_model, resolved_config = _resolve_model_paths(model_path, config_path)

    total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if total_gpus <= 1:
        return inference_ad_tesseract2(
            data,
            resolved_model,
            config_path=resolved_config,
            nsample=nsample,
            preprocess_model_dir=preprocess_model_dir,
            use_dpm_solver=use_dpm_solver,
            dpm_steps=dpm_steps,
        )

    if gpu_ids is None:
        gpu_ids = list(range(total_gpus))
    if not gpu_ids:
        raise ValueError("gpu_ids is empty; no GPUs available for multiprocessing inference.")

    if num_processes is None:
        num_processes = len(gpu_ids)
    num_processes = max(1, min(num_processes, len(gpu_ids)))

    # Log multiprocessing + DPM strategy
    method_str = f"DPM-Solver ({dpm_steps} steps)" if use_dpm_solver else "Standard Diffusion"
    logger.info("%s", "\n" + "=" * 60)
    logger.info("MULTI-GPU INFERENCE with %s", method_str)
    logger.info("%s", "=" * 60)
    logger.info("Processes: %s", num_processes)
    logger.info("GPU IDs: %s", gpu_ids)
    if use_dpm_solver:
        per_process_speedup = 1000 / dpm_steps
        total_speedup = per_process_speedup * num_processes
        logger.info("Per-GPU speedup: ~%.0fx", per_process_speedup)
        logger.info("Total estimated speedup: ~%.0fx vs single-GPU standard", total_speedup)
    logger.info("%s\n", "=" * 60)

    model_path = Path(resolved_model)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", None)
    if config is None:
        if not resolved_config.strip():
            raise ValueError("Must pass config_path when the model path is a checkpoint")
        config = load_config(resolved_config)

    target_dim = config["model"].get("target_dim", 40)
    scale_factor = config.get("dataset", {}).get("scale_factor", 1.0)
    window_length = config.get("dataset", {}).get("window_length", 100)
    window_split = 1
    split = config.get("dataset", {}).get("split", 4)

    preprocessed = preprocess_dataframe(
        data,
        target_dim,
        scale_factor=scale_factor,
        model_dir=preprocess_model_dir,
    )
    num_rows = len(preprocessed)
    begin_indexes = _build_window_indexes(num_rows, window_length, window_split)
    window_tensor = _build_window_tensor(preprocessed.cpu().numpy(), begin_indexes, window_length)
    shm_info = _create_shared_memory(window_tensor)

    window_indices = list(range(len(begin_indexes)))
    chunks = _split_indexes(window_indices, num_processes)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            processes = []
            result_files: list[tuple[int, Path]] = []

            for chunk_idx, chunk in enumerate(chunks):
                if not chunk:
                    continue
                gpu_id = gpu_ids[chunk_idx]

                args_file = tmpdir / f"args_{chunk_idx}.json"
                result_file = tmpdir / f"result_{chunk_idx}.json"
                result_files.append((chunk_idx, result_file))

                args = {
                    "gpu_id": gpu_id,
                    "data_chunk": {
                        "window_shm": True,
                        "shm_name": shm_info["name"],
                        "shape": list(shm_info["shape"]),
                        "dtype": str(shm_info["dtype"]),
                        "window_indices": chunk,
                        "split": split,
                    },
                    "model_path": str(model_path),
                    "config": config,
                    "target_dim": target_dim,
                    "scale_factor": scale_factor,
                    "nsample": nsample,
                    "seed": seed,
                    "deterministic": deterministic,
                    "preprocess_model_dir": preprocess_model_dir,
                    "use_dpm_solver": use_dpm_solver,
                    "dpm_steps": dpm_steps,
                }
                with open(args_file, "w") as f:
                    json.dump(args, f)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                p = subprocess.Popen(
                    [sys.executable, str(_WORKER_SCRIPT), str(args_file), str(result_file)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                processes.append((chunk_idx, p))

            results_per_chunk: list[dict | None] = [None] * len(chunks)
            for chunk_idx, p in processes:
                _, stderr = p.communicate()
                if p.returncode != 0:
                    raise RuntimeError(f"GPU worker failed (chunk {chunk_idx}, code {p.returncode}): {stderr.decode()}")

            for chunk_idx, result_file in result_files:
                with open(result_file) as f:
                    data = json.load(f)
                if "error" in data:
                    raise RuntimeError(f"GPU worker failed (chunk {chunk_idx}): {data['error']}")
                # Convert JSON lists back to numpy arrays
                results_per_chunk[chunk_idx] = {
                    k: np.array(v) if isinstance(v, list) else v for k, v in data["results"].items()
                }

        return _merge_chunked_results(results_per_chunk, target_dim)
    finally:
        _cleanup_shared_memory(shm_info)
