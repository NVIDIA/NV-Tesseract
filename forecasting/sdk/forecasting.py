# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from huggingface_hub import ModelHubMixin, snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

    class ModelHubMixin:  # type: ignore[no-redef]
        """No-op fallback when huggingface_hub is not installed."""

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__()


# Clean absolute imports - package is installed in editable mode
from backbone import RevIN
from backbone.utils.utils import control_randomness
from channel_stability import (
    PerChannelEmbeddingStabilityReport,
    compute_per_channel_embedding_stability,
)
from dataset_longhorizon import (
    CSVLongHorizonSimpleDataset,
    Standardizer,
)
from interpretability import (
    ForecastExplanation,
    TrajectoryStabilityReport,
    compute_trajectory_stability,
    explain_forecast,
)
from model import build_model

logger = logging.getLogger(__name__)

# Define DEVICE here to avoid import complexity


def _has_mps():
    try:
        return torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    except:
        return False


DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if _has_mps() else torch.device("cpu")
)
CHECKPOINT_CROSS_CHANNEL = "run8_best_model_cr.pt"
CHECKPOINT_BASE = "moment_head_512_6hr.pt"  # non-cross-channel
_KNOWN_CHECKPOINTS = {CHECKPOINT_CROSS_CHANNEL, CHECKPOINT_BASE}
DEFAULT_BACKBONE_NAME = "AutonLab/MOMENT-1-large"
_MODEL_CACHE: dict[str, torch.nn.Module] = {}


@dataclass(frozen=True, slots=True)
class ForecastingConfig:
    """Typed configuration for :func:`perform_forecasting`."""

    timestamp_column: str = "timestamp"
    target_column: str = "target"
    standardizer_pkl: str = "standardizer.pkl"
    ckpt: str = CHECKPOINT_CROSS_CHANNEL
    seq_len: int = 512
    forecast_horizon: int = 72
    model_horizon: int | None = 72
    save_preds: str | None = None
    alpha: float = 0.01
    model_name: str = DEFAULT_BACKBONE_NAME
    batch_size: int = 8
    num_workers: int = 2
    stride: int | None = None
    context_stride: int | None = None
    seed: int = 13
    k: int = 64
    temperature: float = 0.05
    device: str | None = None
    local_files_only: bool = False
    use_cross_channel: bool = True
    cross_channel_heads: int = 8
    cross_channel_dropout: float = 0.1
    return_all_channels: bool = False
    interpretability: bool = False
    interpretability_output: str | None = None
    interpretability_out_dir: str | Path = "interpretability_output"
    interpretability_run_name: str | None = None
    interpretability_top_k: int = 5
    interpretability_dataset_name: str | None = None
    n_lags: int = 128
    softmax_tau: float = 1.0
    channel_output_aware: bool = False
    integrated_gradients: bool = False
    integrated_gradients_baseline: Any = "noise"
    integrated_gradients_steps: int = 64
    integrated_gradients_n_baselines: int = 1
    integrated_gradients_internal_batch_size: int | None = None
    integrated_gradients_reduce: str = "l2"
    integrated_gradients_grad_through_norm: bool = True


_FORECASTING_CONFIG_FIELDS = frozenset(field.name for field in fields(ForecastingConfig))


def load_forecasting_config(config_path: str | Path) -> ForecastingConfig:
    """Load and validate a :class:`ForecastingConfig` from YAML.

    The YAML may be a flat mapping or contain a top-level ``inference`` mapping,
    which allows the TAO specification to carry dataset and training sections.
    """
    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"Forecasting inference config must be a mapping, got {type(raw_config).__name__}.")

    config = raw_config.get("inference", raw_config)
    if config is None:
        return ForecastingConfig()
    if not isinstance(config, dict):
        raise ValueError(
            f"Forecasting inference config 'inference' section must be a mapping, got {type(config).__name__}."
        )

    unknown_keys = sorted(set(config) - _FORECASTING_CONFIG_FIELDS)
    if unknown_keys:
        allowed = ", ".join(sorted(_FORECASTING_CONFIG_FIELDS))
        raise ValueError(f"Unknown forecasting inference config keys: {unknown_keys}. Allowed keys: {allowed}")

    values = dict(config)
    for artifact_field in ("standardizer_pkl", "ckpt"):
        if artifact_field in values and not values[artifact_field]:
            logger.warning(
                "Forecasting config has an empty '%s'; falling back to default %r.",
                artifact_field,
                getattr(ForecastingConfig(), artifact_field),
            )
            values.pop(artifact_field, None)
    return ForecastingConfig(**values)


def _resolve_forecasting_config(
    config: ForecastingConfig | str | Path | None,
) -> ForecastingConfig:
    if config is None:
        resolved = ForecastingConfig()
    elif isinstance(config, ForecastingConfig):
        resolved = config
    elif isinstance(config, (str, Path)):
        resolved = load_forecasting_config(config)
    else:
        raise TypeError("config must be a ForecastingConfig, YAML path, or None")

    artifact_defaults = ForecastingConfig()
    artifact_updates = {}
    for name in ("standardizer_pkl", "ckpt"):
        if not getattr(resolved, name):
            default_value = getattr(artifact_defaults, name)
            logger.warning(
                "Forecasting config has an empty '%s'; falling back to default %r.",
                name,
                default_value,
            )
            artifact_updates[name] = default_value
    return replace(resolved, **artifact_updates) if artifact_updates else resolved


def _checkpoint_uses_cross_channel(state: dict) -> bool:
    return any("cross_channel" in k for k in state)


def _load_standardizer_artifact(path: str) -> Standardizer:
    artifact = joblib.load(path)
    if isinstance(artifact, Standardizer):
        return artifact
    if isinstance(artifact, dict):
        mean = np.array(artifact["mean"], dtype=np.float32)
        std = np.array(artifact["std"], dtype=np.float32)
        std[std < 1e-8] = 1.0
        return Standardizer(mean=mean, std=std)
    if hasattr(artifact, "mean") and hasattr(artifact, "std"):
        mean = np.array(artifact.mean, dtype=np.float32)
        std = np.array(artifact.std, dtype=np.float32)
        std[std < 1e-8] = 1.0
        return Standardizer(mean=mean, std=std)
    raise TypeError(f"Unsupported standardizer artifact type: {type(artifact)!r}")


def _get_model_cache_key(
    model_name: str,
    ckpt: str,
    seq_len: int,
    model_horizon: int,
    device: str,
    cross_channel_heads: int,
    cross_channel_dropout: float,
) -> str:
    try:
        ckpt_mtime = Path(ckpt).stat().st_mtime
    except OSError:
        ckpt_mtime = 0

    cache_data = (
        f"{model_name}_{ckpt}_{seq_len}_{model_horizon}_{device}_{ckpt_mtime}_"
        f"{cross_channel_heads}_{cross_channel_dropout}"
    )
    return hashlib.md5(cache_data.encode()).hexdigest()


def _load_cached_model(
    model_name: str,
    ckpt: str,
    seq_len: int,
    model_horizon: int,
    device: torch.device,
    local_files_only: bool = False,
    use_cross_channel: bool = True,
    cross_channel_heads: int = 8,
    cross_channel_dropout: float = 0.1,
):
    cache_key = _get_model_cache_key(
        model_name,
        ckpt,
        seq_len,
        model_horizon,
        str(device),
        cross_channel_heads,
        cross_channel_dropout,
    )

    if cache_key in _MODEL_CACHE:
        logger.info("Using cached model for %s", ckpt)
        return _MODEL_CACHE[cache_key]

    logger.info("Loading model from checkpoint: %s", ckpt)

    # Inspect state dict before building the model so use_cross_channel always
    # matches what's actually in the checkpoint. We check for "cross_channel"
    # (not "cross_channel_attn.") to remain correct if future checkpoints
    # introduce additional cross-channel components under different prefixes.
    state = torch.load(ckpt, map_location=device)
    ckpt_has_cr = _checkpoint_uses_cross_channel(state)
    if use_cross_channel != ckpt_has_cr:
        logger.warning(
            "use_cross_channel=%s but checkpoint %s cross-channel weights; following checkpoint",
            use_cross_channel,
            "has" if ckpt_has_cr else "lacks",
        )
        use_cross_channel = ckpt_has_cr

    model = build_model(
        model_name=model_name,
        seq_len=seq_len,
        forecast_horizon=model_horizon,
        freeze_encoder=False,
        freeze_embedder=False,
        freeze_head=False,
        use_cross_channel=use_cross_channel,
        cross_channel_heads=cross_channel_heads,
        cross_channel_dropout=cross_channel_dropout,
        local_files_only=local_files_only,
        device=str(device),
    )
    load_result = model.load_state_dict(state, strict=False)
    missing = list(getattr(load_result, "missing_keys", [])) if load_result is not None else []
    unexpected = list(getattr(load_result, "unexpected_keys", [])) if load_result is not None else []
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys for forecasting model: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    model.eval()
    _MODEL_CACHE[cache_key] = model
    logger.info("Model cached with key: %s...", cache_key[:8])
    return model


def clear_model_cache():
    _MODEL_CACHE.clear()
    logger.info("Model cache cleared")


def download_model_weights(
    standardizer_pkl: str = "standardizer.pkl",
    ckpt: str = CHECKPOINT_CROSS_CHANNEL,
    repo_id: str = "nvidia/nv-tesseract-forecasting",
    force_download: bool = False,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> tuple[str, str]:
    """
    Auto-download model weights from Hugging Face if they don't exist locally.

    Args:
        standardizer_pkl: Local path for standardizer pickle file
        ckpt: Local path for model checkpoint
        repo_id: Hugging Face repository ID
        force_download: Force re-download even if files exist

    Returns:
        Tuple of (standardizer_path, checkpoint_path)

    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails

    Note:
        Weights are published on the Hugging Face repository and download
        without authentication. If a download fails with 401/403, accept the
        model license on the repo page or authenticate first:
            1. Install the CLI:  `uv add huggingface_hub[cli]`
            2. Login:            `huggingface-cli login`
            3. Or set a token:   `export HUGGINGFACE_HUB_TOKEN='your_token'`
    """
    standardizer_path = Path(standardizer_pkl)
    checkpoint_path = Path(ckpt)

    # Check if files already exist
    if not force_download and standardizer_path.exists() and checkpoint_path.exists():
        return str(standardizer_path), str(checkpoint_path)

    # Check if huggingface_hub is available
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to download model weights. Install it with: uv add huggingface_hub"
        )

    logger.info("Downloading model weights from Hugging Face...")

    # Download each file to its own parent directory so returned paths always exist
    try:
        for file_path in (standardizer_path, checkpoint_path):
            if force_download or not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading: %s", file_path.name)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(file_path.parent),
                    allow_patterns=[file_path.name],
                    force_download=force_download,
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    local_files_only=local_files_only,
                    token=token,
                    library_name="nv-tesseract",
                )
                logger.info("Downloaded: %s", file_path.name)

    except Exception as e:
        error_msg = f"Failed to download model weights: {e}"
        if "401" in str(e) or "403" in str(e):
            error_msg += (
                "\n\nDownload failed. If you see a 401/403 error, accept the model license on Hugging Face or authenticate:"
                "\n1. Install huggingface-cli: uv add huggingface_hub[cli]"
                "\n2. Login: huggingface-cli login"
                "\n3. Or set token: export HUGGINGFACE_HUB_TOKEN='your_token'"
            )
        raise Exception(error_msg) from e

    return str(standardizer_path), str(checkpoint_path)


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


def _create_temp_csv_path(prefix: str) -> str:
    """Return an exclusively created temporary CSV path for one forecasting call."""
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".csv")
    # pandas writes by path; keep the file but release mkstemp's creator handle.
    os.close(fd)
    return path


class _DARREmbedWrapper(torch.nn.Module):
    """Routes forward() to model.embed() so DataParallel can shard DARR context batches.

    DataParallel only intercepts forward(). The forecasting model's forward() runs
    the forecast head, not the embed path. This wrapper re-maps forward() to embed()
    so that build_context_memory can benefit from multi-GPU batch splitting.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(self, x_enc: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        out = self._model.embed(x_enc=x_enc, input_mask=input_mask)
        return out.embeddings


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
    _use_dp_forward = isinstance(model, (torch.nn.DataParallel, _DARREmbedWrapper))
    for batch in tqdm(context_loader, desc="Building context memory"):
        x_enc, y_future, input_mask = batch[:3]
        x_enc = x_enc.to(device, dtype=torch.float32)
        input_mask = input_mask.to(device)
        if _use_dp_forward:
            emb = model(x_enc, input_mask)
        else:
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


# ---------------------------------------------------------------------------
# Interpretability helpers
#
# These helpers produce interpretability artifacts
# (lag x horizon attribution heatmap, top-k lag tables, explanation JSON, and a self-contained PDF report)
# directly from ``perform_forecasting`` when ``interpretability=True``.
# ---------------------------------------------------------------------------


def _save_lag_horizon_artifacts(
    out_dir: Path,
    attributions: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    na_rep: str = "nan",
) -> Path | None:
    """Write lag x horizon attribution CSVs and a heatmap PNG.

    Returns the path to the heatmap PNG if matplotlib is available, else None.
    """
    K, H = attributions.shape

    lag_cols = [f"horizon_{h}" for h in range(H)]
    df_matrix = pd.DataFrame(attributions, columns=lag_cols)
    df_matrix.insert(0, "lag", [f"lag_{j}" for j in range(K)])
    df_matrix.to_csv(out_dir / "lag_horizon_attributions.csv", index=False, na_rep=na_rep)

    rows = []
    for j in range(K):
        for h in range(H):
            row = {"lag": j, "horizon": h, "attribution": float(attributions[j, h])}
            if scores is not None:
                row["score"] = float(scores[j, h])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "lag_horizon_long.csv", index=False, na_rep=na_rep)

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(max(8, H * 0.15), max(6, K * 0.05)))
    im = ax.imshow(attributions, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Lag (past time step)")
    ax.set_xticks(np.linspace(0, H - 1, min(12, H), dtype=int))
    ax.set_xticklabels(np.linspace(0, H - 1, min(12, H), dtype=int))
    ax.set_yticks(np.linspace(0, K - 1, min(12, K), dtype=int))
    ax.set_yticklabels(np.linspace(0, K - 1, min(12, K), dtype=int))
    plt.colorbar(im, ax=ax, label="Attribution")
    plt.tight_layout()
    heatmap_path = out_dir / "lag_horizon_heatmap.png"
    plt.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    return heatmap_path


def _topk_lag_steps_per_horizon(
    scores: np.ndarray,
    *,
    top_k: int = 5,
) -> tuple[np.ndarray | None, list[list[str]]]:
    """Compute marginal per-step lag weights and produce top-k summary rows."""
    if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] < 1:
        return None, []
    step_scores = np.vstack([scores[0:1, :], np.diff(scores, axis=0)])
    step_scores = step_scores - np.max(step_scores, axis=0, keepdims=True)
    step_probs = np.exp(step_scores)
    step_probs = step_probs / np.maximum(np.sum(step_probs, axis=0, keepdims=True), 1e-12)

    H = step_probs.shape[1]
    rows: list[list[str]] = []
    for h in range(H):
        order = np.argsort(-step_probs[:, h])[:top_k]
        row: list[str] = [f"horizon_{h}"]
        for idx in order:
            lag_step = int(idx) + 1
            w = float(step_probs[idx, h])
            row.append(str(lag_step))
            row.append(f"{w:.4f}")
        while len(row) < 1 + 2 * top_k:
            row.extend(["", ""])
        rows.append(row)
    return step_probs, rows


def _semantic_flow_segments(
    flow: np.ndarray,
    *,
    context_len: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split flow_magnitudes into (history, forecast) using the same convention as
    ``interpretability._flow_segment_ratios``.

    Returns ``(history_flow, forecast_flow, boundary_index)`` where ``boundary_index``
    is the transition index that marks the first window touching the forecast segment.
    """
    f = np.asarray(flow, dtype=np.float32).reshape(-1)
    T_trans = int(f.shape[0])
    L = int(context_len)
    H = int(forecast_horizon)
    boundary = max(0, L - 1)
    hist = f[0:boundary] if T_trans > 0 else f
    fcst = f[boundary : min(T_trans, L + H - 1)] if T_trans > 0 else f
    return hist, fcst, boundary


def _save_semantic_flow_csv(
    out_dir: Path,
    flow: np.ndarray,
    *,
    context_len: int,
    forecast_horizon: int,
    na_rep: str = "nan",
) -> Path:
    """Persist per-transition latent flow magnitudes with a segment label.

    Columns: ``transition_index, segment, flow_magnitude`` where ``segment`` is
    ``"history"`` for transitions whose window sits inside the input context and
    ``"forecast"`` for transitions whose window extends into the model-generated
    future. Transitions outside both segments (only possible when ``T-1 > L+H-1``)
    are labeled ``"tail"``.
    """
    f = np.asarray(flow, dtype=np.float32).reshape(-1)
    T_trans = int(f.shape[0])
    L = int(context_len)
    H = int(forecast_horizon)
    boundary = max(0, L - 1)
    fcst_end = min(T_trans, L + H - 1)

    segments = np.full((T_trans,), "tail", dtype=object)
    segments[0:boundary] = "history"
    segments[boundary:fcst_end] = "forecast"

    df = pd.DataFrame(
        {
            "transition_index": np.arange(T_trans, dtype=np.int64),
            "segment": segments,
            "flow_magnitude": f,
        }
    )
    out_path = out_dir / "semantic_flow.csv"
    df.to_csv(out_path, index=False, na_rep=na_rep)
    return out_path


def _semantic_flow_page(
    pdf: Any,
    plt: Any,
    *,
    explanation: ForecastExplanation,
    context_len: int,
    forecast_horizon: int,
) -> None:
    """Append a single PDF page summarizing semantic flow magnitudes.

    Layout mirrors the trajectory-stability page: title, gray blurb, a primary
    visual (line chart of flow over transitions with the history/forecast split
    annotated), two side-by-side tables (per-segment summary + diagnostics), and
    a monospace reading guide.
    """
    if getattr(explanation, "flow_magnitudes", None) is None:
        return
    flow = np.asarray(explanation.flow_magnitudes, dtype=np.float32).reshape(-1)
    T_trans = int(flow.shape[0])
    if T_trans == 0:
        return

    hist, fcst, boundary = _semantic_flow_segments(
        flow,
        context_len=context_len,
        forecast_horizon=forecast_horizon,
    )

    def _stats(arr: np.ndarray) -> tuple[str, str, str, str, str, str]:
        a = np.asarray(arr, dtype=np.float32)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return ("--", "--", "--", "--", "--", "0")
        return (
            f"{float(np.mean(a)):.4f}",
            f"{float(np.median(a)):.4f}",
            f"{float(np.percentile(a, 95)):.4f}",
            f"{float(np.max(a)):.4f}",
            f"{float(np.var(a)):.4f}",
            f"{int(a.size)}",
        )

    hist_s = _stats(hist)
    fcst_s = _stats(fcst)

    def _fmt(x: Any) -> str:
        if x is None:
            return "n/a"
        try:
            xv = float(x)
        except (TypeError, ValueError):
            return "n/a"
        return f"{xv:.4f}" if np.isfinite(xv) else "n/a"

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.95,
        "Semantic flow magnitudes",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.90,
        r"Per-step latent flow is $m_t = \Vert \mathbf{z}_{t+1} - \mathbf{z}_t \Vert_2$"
        " over the trajectory built on\n"
        "[history; forecast]. This is the temporal signal that drives the lag x horizon\n"
        "heatmap; bigger spikes contribute more to attribution. Compare the history\n"
        "segment against the forecast segment to gauge representation-level stability.",
        ha="left",
        va="top",
        fontsize=9,
        fontfamily="DejaVu Sans",
        math_fontfamily="dejavusans",
        color="gray",
    )

    ax = fig.add_axes((0.08, 0.55, 0.84, 0.23))
    x_axis = np.arange(T_trans)
    ax.plot(x_axis, flow, linewidth=1.0, color="#1f77b4", label="flow magnitude")
    if 0 < boundary < T_trans:
        ax.axvspan(boundary, T_trans - 1, color="#f0f0f0", alpha=0.7, zorder=0)
        ax.axvline(boundary, color="black", linestyle="--", linewidth=0.9, label="history / forecast split")
    if hist.size > 0:
        hist_mean = float(np.nanmean(hist))
        ax.hlines(
            hist_mean,
            0,
            max(0, boundary - 1),
            colors="#2ca02c",
            linestyles=":",
            linewidth=1.2,
            label=f"history mean ({hist_mean:.3f})",
        )
    if fcst.size > 0:
        fcst_mean = float(np.nanmean(fcst))
        fcst_end_idx = max(boundary, min(T_trans, int(context_len) + int(forecast_horizon) - 1) - 1)
        ax.hlines(
            fcst_mean,
            boundary,
            fcst_end_idx,
            colors="#d62728",
            linestyles=":",
            linewidth=1.2,
            label=f"forecast mean ({fcst_mean:.3f})",
        )
    ax.set_xlabel(r"Latent transition index ($t \rightarrow t+1$)")
    ax.set_ylabel(r"$\Vert \mathbf{z}_{t+1} - \mathbf{z}_t \Vert_2$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    seg_rows = [
        ["Mean", hist_s[0], fcst_s[0]],
        ["Median", hist_s[1], fcst_s[1]],
        ["P95", hist_s[2], fcst_s[2]],
        ["Max", hist_s[3], fcst_s[3]],
        ["Variance", hist_s[4], fcst_s[4]],
        ["Transitions", hist_s[5], fcst_s[5]],
    ]
    ax_t1 = fig.add_axes((0.06, 0.32, 0.42, 0.18))
    ax_t1.axis("off")
    table_seg = ax_t1.table(
        cellText=seg_rows,
        colLabels=["Metric", "History", "Forecast"],
        colWidths=[0.4, 0.3, 0.3],
        loc="upper center",
        cellLoc="center",
    )
    table_seg.auto_set_font_size(False)
    table_seg.set_fontsize(8)
    table_seg.scale(1.0, 1.3)
    for c in range(3):
        cell = table_seg[(0, c)]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(weight="bold")

    diag_rows = [
        [
            "Flow ratio (forecast/history)",
            _fmt(explanation.flow_ratio_forecast_vs_history),
            r"$\approx 1$ healthy, $>1.5$ OOD-volatile",
        ],
        [
            "Flow variance ratio",
            _fmt(explanation.flow_variance_ratio_forecast_vs_history),
            r"$\approx 1$ healthy, $>2$ noisy",
        ],
        [
            "Curvature ratio",
            _fmt(explanation.curvature_ratio_forecast_vs_history),
            r"$\approx 1$ healthy, $>1.5$ jaggy",
        ],
        [
            "Latent diag-Mahalanobis ratio",
            _fmt(explanation.latent_diag_mahalanobis_ratio_forecast_vs_history),
            r"$\approx 1$ healthy, $\gg 1$ OOD shift",
        ],
    ]
    ax_t2 = fig.add_axes((0.52, 0.32, 0.42, 0.18))
    ax_t2.axis("off")
    table_diag = ax_t2.table(
        cellText=diag_rows,
        colLabels=["Diagnostic", "Value", "Heuristic"],
        colWidths=[0.42, 0.18, 0.40],
        loc="upper center",
        cellLoc="center",
    )
    table_diag.auto_set_font_size(False)
    table_diag.set_fontsize(8)
    table_diag.scale(1.0, 1.3)
    for c in range(3):
        cell = table_diag[(0, c)]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(weight="bold")

    interp_text = (
        "How to read these metrics\n"
        "-------------------------\n"
        "  - Per-segment table: mean / median / p95 / max / variance of latent\n"
        "    flow magnitudes split into history (transitions fully inside the\n"
        "    input window) and forecast (transitions whose window touches the\n"
        "    model-generated future).\n"
        "  - Flow ratio: average forecast flow divided by average history flow. Values\n"
        "    near 1 mean the model's latent dynamics in the OOD segment match history.\n"
        "    Large values flag attributions in that region as likely noisy.\n"
        "  - Flow variance ratio: same comparison on variance.\n"
        "  - Curvature ratio: second-difference energy of Z; spikes when the\n"
        "    forecast segment becomes much jaggier than history.\n"
        "  - Latent diag-Mahalanobis ratio: per-dim distance of forecast latents\n"
        "    from the history mean (scaled by history variance). Values well above 1 indicate a\n"
        "    representation-level OOD shift.\n"
        "\n"
        "These scalars are also under explanation.diagnostics in explanation.json,\n"
        "and the full series is exported as semantic_flow.csv."
    )
    fig.text(
        0.06,
        0.30,
        interp_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        color="#333333",
        linespacing=1.05,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _save_channel_axis_artifacts(
    out_dir: Path,
    channel_horizon_attributions: np.ndarray,
    channel_labels: list[str] | None = None,
    *,
    na_rep: str = "nan",
) -> Path | None:
    """Write the [C, H] channel x horizon attribution CSV and a heatmap PNG.

    Returns the path to the heatmap PNG if matplotlib is available, else None.
    """
    A = np.asarray(channel_horizon_attributions, dtype=np.float32)
    if A.ndim != 2:
        return None
    C, H = A.shape
    labels = channel_labels if (channel_labels and len(channel_labels) == C) else [f"channel_{c}" for c in range(C)]

    df = pd.DataFrame(A, columns=[f"horizon_{h}" for h in range(H)])
    df.insert(0, "channel", labels)
    df.to_csv(out_dir / "channel_horizon_attributions.csv", index=False, na_rep=na_rep)

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(max(8, H * 0.15), max(4, C * 0.4)))
    im = ax.imshow(A, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Input channel")
    ax.set_yticks(range(C))
    ax.set_yticklabels(labels, fontsize=8)
    xt = np.linspace(0, H - 1, min(12, H), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)
    plt.colorbar(im, ax=ax, label="Attribution")
    plt.tight_layout()
    heatmap_path = out_dir / "channel_horizon_heatmap.png"
    plt.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    return heatmap_path


def _channel_axis_page(
    pdf: Any,
    plt: Any,
    *,
    explanation: ForecastExplanation,
    channel_labels: list[str] | None = None,
) -> None:
    """Append one PDF page for the feature-axis (channel x horizon) attribution.

    Primary visual: a [C, H] heatmap of how much each input channel contributes to
    each forecast step. A side bar summarizes each channel's total attribution
    (summed over horizons) as a single feature-importance score.
    """
    A = getattr(explanation, "channel_horizon_attributions", None)
    if A is None:
        return
    A = np.asarray(A, dtype=np.float32)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        return
    C, H = A.shape
    labels = channel_labels if (channel_labels and len(channel_labels) == C) else [f"channel_{c}" for c in range(C)]
    importance = A.sum(axis=1)
    method = getattr(explanation, "channel_flow_method", None) or "jacobian"
    resid = getattr(explanation, "channel_flow_residual_ratio_mean", None)
    target_jacobian = method == "target_input_jacobian"
    intro = (
        "Which input channel drives each target forecast step. Each cell is the magnitude of\n"
        "the target forecast's local directional response when that channel moves from its\n"
        "local-mean replacement to the observed input. Rows are channels and columns are\n"
        "forecast steps; brighter = a stronger target-specific effect."
        if target_jacobian
        else "Which input channel (feature) drives each forecast step. The model's response is\n"
        "decomposed along the feature axis into per-channel contributions, then aggregated\n"
        "per horizon. Rows are input channels, columns are forecast steps; brighter = that\n"
        "channel matters more there."
    )

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.95,
        "Feature-axis attribution (channel x horizon)",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.90,
        intro,
        ha="left",
        va="top",
        fontsize=9,
        color="gray",
    )

    ax = fig.add_axes((0.08, 0.40, 0.56, 0.42))
    im = ax.imshow(A, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Forecast horizon (0 = first predicted step)")
    ax.set_ylabel("Input channel")
    ax.set_yticks(range(C))
    ax.set_yticklabels(labels, fontsize=8)
    xt = np.linspace(0, H - 1, min(12, H), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Directional effect" if target_jacobian else "Attribution",
    )

    axb = fig.add_axes((0.72, 0.40, 0.22, 0.42))
    y = np.arange(C)
    axb.barh(y, importance, color="#1f77b4")
    axb.set_yticks(y)
    axb.set_yticklabels([])
    axb.invert_yaxis()  # channel 0 at top, aligned with the heatmap rows
    axb.set_title("Channel importance\n(sum over horizons)", fontsize=9)
    axb.set_xlabel("Total effect" if target_jacobian else "Total attribution", fontsize=8)
    axb.tick_params(axis="both", labelsize=7)

    top_c = int(np.argmax(importance))
    resid_line = (
        f"  - Estimator: {method}. Jacobian trust ratio (mean residual): {resid:.3f}\n"
        "    (lower = more reliable first-order split; high values flag strong\n"
        "    cross-channel interaction the first-order decomposition cannot capture).\n"
        if resid is not None
        else f"  - Estimator: {method}.\n"
    )
    heatmap_note = (
        "  - Heatmap cell (channel c, horizon h): magnitude of the target forecast's\n"
        "    first-order directional replacement effect; columns are not normalized.\n"
        if target_jacobian
        else "  - Heatmap cell (channel c, horizon h): share of the forecast at step h\n"
        "    attributable to input channel c (each horizon column sums to 1 over channels).\n"
    )
    bar_note = "total effect" if target_jacobian else "total attribution"
    note = (
        "How to read this page\n"
        "---------------------\n"
        + heatmap_note
        + f"  - Right bar: each channel's {bar_note} summed across horizons -- a single\n"
        "    feature-importance score per input channel.\n"
        f"  - Most influential channel overall: {labels[top_c]}.\n" + resid_line + "\n"
        "Full channel-by-horizon values are exported as channel_horizon_attributions.csv."
    )
    fig.text(
        0.06,
        0.30,
        note,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        color="#333333",
        linespacing=1.05,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _feature_axis_embedding_stability_frame(
    report: PerChannelEmbeddingStabilityReport,
    channel_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return the complete feature-axis embedding-stability report."""
    n_channels = int(report.n_channels)
    labels = (
        channel_labels
        if channel_labels is not None and len(channel_labels) == n_channels
        else [f"channel_{c}" for c in range(n_channels)]
    )
    return pd.DataFrame(
        {
            "feature": labels,
            "lip_ratio_mean": np.asarray(report.lip_ratio_mean, dtype=np.float32),
            "lip_ratio_p50": np.asarray(report.lip_ratio_p50, dtype=np.float32),
            "lip_ratio_p95": np.asarray(report.lip_ratio_p95, dtype=np.float32),
            "lip_ratio_max": np.asarray(report.lip_ratio_max, dtype=np.float32),
            "n_trials_per_channel": np.asarray(report.n_trials_per_channel, dtype=np.int64),
            "n_unique_windows": int(report.n_unique_windows),
            "step_delta_norm_mean": float(report.step_delta_norm_mean),
        }
    )


def _save_feature_axis_embedding_stability_csv(
    out_dir: Path,
    report: PerChannelEmbeddingStabilityReport,
    channel_labels: list[str] | None = None,
) -> Path:
    """Write the complete feature-axis embedding-stability report."""
    out_path = out_dir / "feature_axis_embedding_stability.csv"
    _feature_axis_embedding_stability_frame(report, channel_labels).to_csv(out_path, index=False, na_rep="nan")
    return out_path


def _feature_axis_embedding_stability_page(
    pdf: Any,
    plt: Any,
    *,
    report: PerChannelEmbeddingStabilityReport,
    channel_labels: list[str] | None = None,
) -> None:
    """Append the feature-axis embedding-stability summary."""
    frame = _feature_axis_embedding_stability_frame(report, channel_labels)
    shown = frame.sort_values("lip_ratio_p95", ascending=False, na_position="last").head(10)

    def _fmt(value: Any) -> str:
        numeric = float(value)
        return f"{numeric:.4f}" if np.isfinite(numeric) else "n/a"

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.95,
        "Feature-axis embedding stability",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.90,
        "This shows the delta between two embedding points after a small, controlled change to one feature.\n"
        r"Each sensitivity score is $\Vert \Delta \mathbf{z} \Vert_2"
        r"\,/\, \Vert \Delta \mathbf{x}_c \Vert_2$."
        "\nLower values mean less latent movement per unit feature change; compare features\n"
        "within the same model and dataset because embedding scales are model-specific.",
        ha="left",
        va="top",
        fontsize=9,
        fontfamily="DejaVu Sans",
        math_fontfamily="dejavusans",
        color="gray",
    )

    table_rows = [
        [
            row.feature,
            _fmt(row.lip_ratio_mean),
            _fmt(row.lip_ratio_p50),
            _fmt(row.lip_ratio_p95),
            _fmt(row.lip_ratio_max),
            str(int(row.n_trials_per_channel)),
        ]
        for row in shown.itertuples(index=False)
    ]
    table_ax = fig.add_axes((0.06, 0.50, 0.88, 0.28))
    table_ax.axis("off")
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["Feature", "Mean", "P50", "P95", "Max", "Trials"],
        colWidths=[0.30, 0.14, 0.14, 0.14, 0.14, 0.14],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    for c in range(6):
        cell = table[(0, c)]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(weight="bold")

    interp_text = (
        "How to read these metrics\n"
        "-------------------------\n"
        "  - Mean / P50 / P95 / Max: summaries of the sensitivity score across\n"
        "    small feature changes. Lower values mean the embedding\n"
        "    moves less per unit change in that feature.\n"
        "  - P95: conservative tail sensitivity. Rows are ranked by P95; the highest\n"
        "    values identify features whose small changes can move the embedding most.\n"
        "  - Trials: valid sensitivity measurements retained for that feature.\n"
        f"  - Sample coverage: {int(report.n_unique_windows)} unique windows. Mean\n"
        f"    baseline latent step: {_fmt(report.step_delta_norm_mean)} (reference only).\n"
        "  - Compare values only within the same model and dataset because embedding\n"
        "    scales are model-specific. A larger ratio is sensitivity, not by itself a defect.\n"
        "\n"
        "All features and raw report fields are exported to\n"
        "feature_axis_embedding_stability.csv and explanation.json."
    )
    fig.text(0.06, 0.43, interp_text, ha="left", va="top", fontsize=9, family="monospace", color="#333333")

    pdf.savefig(fig)
    plt.close(fig)


@contextmanager
def _normalization_statistics_gradient(model: torch.nn.Module, *, enabled: bool):
    """Select request-local RevIN statistics autograd without mutating models."""
    normalizer_count = sum(isinstance(module, RevIN) for module in model.modules())
    with RevIN.statistics_gradient_context(enabled=enabled):
        yield normalizer_count if enabled else 0


def _compute_integrated_gradients_report(
    *,
    model: torch.nn.Module,
    x_context_ct: np.ndarray,
    input_mask_l: np.ndarray,
    device: torch.device,
    channel_labels: list[str],
    baseline: Any,
    n_steps: int,
    n_baselines: int,
    internal_batch_size: int | None,
    reduce: str,
    seed: int,
    grad_through_norm: bool,
) -> dict[str, Any]:
    """Run embedding integrated gradients for the current SDK context window."""
    from integrated_gradients import integrated_gradients_embedding

    x_tensor = torch.as_tensor(np.asarray(x_context_ct, dtype=np.float32), device=device)
    input_mask = torch.as_tensor(np.asarray(input_mask_l, dtype=np.int64), device=device)

    def embed_fn(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SDK embedding input must be [batch, channels, context], got {tuple(x.shape)}")
        mask = input_mask.reshape(1, -1).expand(x.shape[0], -1).to(device=x.device)
        with _normalization_statistics_gradient(model, enabled=grad_through_norm):
            output = model.embed(x_enc=x, input_mask=mask)
        embeddings = output.embeddings if hasattr(output, "embeddings") else output
        return embeddings.reshape(embeddings.shape[0], -1)

    attribution = integrated_gradients_embedding(
        embed_fn,
        x_tensor,
        baseline=baseline,
        n_steps=n_steps,
        n_baselines=n_baselines,
        internal_batch_size=internal_batch_size,
        reduce=reduce,
        seed=seed,
        device=device,
    )

    A = np.asarray(attribution.attribution, dtype=np.float32)
    if A.ndim != 2:
        raise ValueError(f"integrated gradients attribution must be [C, L], got shape {A.shape}")

    C, _ = A.shape
    labels = channel_labels if len(channel_labels) == C else [f"channel_{c}" for c in range(C)]
    abs_A = np.abs(A)
    channel_abs_effect = abs_A.sum(axis=1).astype(np.float32)
    channel_signed_effect = A.sum(axis=1).astype(np.float32)
    total_abs = float(channel_abs_effect.sum())
    channel_abs_share = (
        (channel_abs_effect / total_abs).astype(np.float32)
        if total_abs > 1e-12
        else np.zeros_like(channel_abs_effect, dtype=np.float32)
    )
    time_abs_effect = abs_A.sum(axis=0).astype(np.float32)
    top_order = np.argsort(-channel_abs_effect)[: min(10, C)]

    return {
        "attribution": A,
        "channel_labels": labels,
        "channel_abs_effect": channel_abs_effect,
        "channel_signed_effect": channel_signed_effect,
        "channel_abs_share": channel_abs_share,
        "time_abs_effect": time_abs_effect,
        "top_channels": [
            {
                "channel": labels[int(i)],
                "abs_effect": float(channel_abs_effect[int(i)]),
                "signed_effect": float(channel_signed_effect[int(i)]),
                "abs_share": float(channel_abs_share[int(i)]),
            }
            for i in top_order
        ],
        "overall_effect": float(attribution.overall_effect),
        "abs_effect": float(attribution.abs_effect),
        "embedding_delta": float(attribution.embedding_delta),
        "embedding_delta_abs_mean": float(attribution.embedding_delta_abs_mean),
        "convergence_delta": float(attribution.convergence_delta),
        "n_steps": int(attribution.n_steps),
        "n_baselines": int(attribution.n_baselines),
        "baseline": str(attribution.baseline),
        "reduce": str(attribution.reduce),
        "embedding_dim": int(attribution.embedding_dim),
        "grad_through_norm": bool(grad_through_norm),
    }


def _save_integrated_gradients_artifacts(
    out_dir: Path,
    integrated_gradients_report: dict[str, Any],
    *,
    na_rep: str = "nan",
) -> Path | None:
    """Write integrated-gradients CSVs and a signed heatmap PNG."""
    A = np.asarray(integrated_gradients_report.get("attribution"), dtype=np.float32)
    if A.ndim != 2:
        return None
    C, L = A.shape
    labels = integrated_gradients_report.get("channel_labels")
    labels = labels if isinstance(labels, list) and len(labels) == C else [f"channel_{c}" for c in range(C)]

    df_matrix = pd.DataFrame(A.T, columns=labels)
    df_matrix.insert(0, "lag_from_last", np.arange(L - 1, -1, -1, dtype=int))
    df_matrix.insert(0, "context_index", np.arange(L, dtype=int))
    df_matrix.to_csv(out_dir / "integrated_gradients_attributions.csv", index=False, na_rep=na_rep)

    summary = pd.DataFrame(
        {
            "channel": labels,
            "signed_effect": np.asarray(integrated_gradients_report["channel_signed_effect"], dtype=np.float32),
            "abs_effect": np.asarray(integrated_gradients_report["channel_abs_effect"], dtype=np.float32),
            "abs_share": np.asarray(integrated_gradients_report["channel_abs_share"], dtype=np.float32),
        }
    )
    summary.to_csv(out_dir / "integrated_gradients_channel_summary.csv", index=False, na_rep=na_rep)

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    scale = float(np.nanpercentile(np.abs(A), 99)) if np.isfinite(A).any() else 0.0
    if scale <= 1e-12:
        scale = float(np.nanmax(np.abs(A))) if np.isfinite(A).any() else 1.0
    if scale <= 1e-12:
        scale = 1.0

    fig, ax = plt.subplots(figsize=(max(8, L * 0.02), max(4, C * 0.35)))
    im = ax.imshow(A, aspect="auto", cmap="coolwarm", interpolation="nearest", vmin=-scale, vmax=scale)
    ax.set_xlabel("Input context lag (0 = most recent)")
    ax.set_ylabel("Input channel")
    ax.set_yticks(range(C))
    ax.set_yticklabels(labels, fontsize=8)
    xt = np.linspace(0, L - 1, min(12, L), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels((L - 1 - xt).astype(int))
    plt.colorbar(im, ax=ax, label="Signed IG attribution")
    plt.tight_layout()
    heatmap_path = out_dir / "integrated_gradients_heatmap.png"
    plt.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    return heatmap_path


def _integrated_gradients_page(
    pdf: Any,
    plt: Any,
    *,
    integrated_gradients_report: dict[str, Any] | None,
) -> None:
    """Append one PDF page for embedding integrated gradients."""
    if integrated_gradients_report is None:
        return
    A = np.asarray(integrated_gradients_report.get("attribution"), dtype=np.float32)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        return
    C, L = A.shape
    labels = integrated_gradients_report.get("channel_labels")
    labels = labels if isinstance(labels, list) and len(labels) == C else [f"channel_{c}" for c in range(C)]
    abs_effect = np.asarray(integrated_gradients_report.get("channel_abs_effect"), dtype=np.float32)
    if abs_effect.shape != (C,):
        abs_effect = np.abs(A).sum(axis=1)

    scale = float(np.nanpercentile(np.abs(A), 99)) if np.isfinite(A).any() else 0.0
    if scale <= 1e-12:
        scale = float(np.nanmax(np.abs(A))) if np.isfinite(A).any() else 1.0
    if scale <= 1e-12:
        scale = 1.0

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.95,
        "Integrated gradients (embedding sensitivity)",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.90,
        "Signed input attributions for the model embedding along the baseline-to-input path.\n"
        "Rows are input channels and columns are the context timesteps; lag 0 is the most\n"
        "recent input. Positive and negative colors show direction, while magnitude shows strength.",
        ha="left",
        va="top",
        fontsize=9,
        color="gray",
    )

    ax = fig.add_axes((0.08, 0.38, 0.58, 0.43))
    im = ax.imshow(A, aspect="auto", cmap="coolwarm", interpolation="nearest", vmin=-scale, vmax=scale)
    ax.set_xlabel("Input context lag (0 = most recent)")
    ax.set_ylabel("Input channel")
    ax.set_yticks(range(C))
    ax.set_yticklabels(labels, fontsize=8)
    xt = np.linspace(0, L - 1, min(12, L), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels((L - 1 - xt).astype(int))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Signed attribution")

    axb = fig.add_axes((0.74, 0.38, 0.22, 0.43))
    y = np.arange(C)
    axb.barh(y, abs_effect, color="#2a9d8f")
    axb.set_yticks(y)
    axb.set_yticklabels([])
    axb.invert_yaxis()
    axb.set_title(r"Channel effect" "\n" r"($\sum |\mathrm{IG}|$)", fontsize=9, math_fontfamily="dejavusans")
    axb.set_xlabel("Absolute effect", fontsize=8)
    axb.tick_params(axis="both", labelsize=7)

    top_c = int(np.argmax(abs_effect))
    baseline = integrated_gradients_report.get("baseline")
    reduce = integrated_gradients_report.get("reduce")
    steps = integrated_gradients_report.get("n_steps")
    n_baselines = integrated_gradients_report.get("n_baselines")
    convergence = _scalar_to_jsonable(integrated_gradients_report.get("convergence_delta"))
    emb_delta = _scalar_to_jsonable(integrated_gradients_report.get("embedding_delta"))
    grad_norm = bool(integrated_gradients_report.get("grad_through_norm"))
    note = (
        "How to read this page\n"
        "---------------------\n"
        "  - Heatmap cell (channel c, lag l): signed contribution of that input value\n"
        "    to the scalar embedding objective used by integrated gradients.\n"
        "  - Right bar: per-channel absolute effect, summed over all context timesteps.\n"
        f"  - Most influential channel by absolute IG: {labels[top_c]}.\n"
        f"  - Config: baseline={baseline}, steps={steps}, baselines={n_baselines}, reduce={reduce}.\n"
        f"  - RevIN/statistics gradient path enabled: {grad_norm}.\n"
        f"  - Embedding delta: {emb_delta}; convergence delta: {convergence}.\n"
        "\n"
        "Full values are exported as integrated_gradients_attributions.csv and\n"
        "integrated_gradients_channel_summary.csv."
    )
    fig.text(0.06, 0.30, note, ha="left", va="top", fontsize=9, family="monospace", color="#333333")

    pdf.savefig(fig)
    plt.close(fig)


def _build_pdf_report(
    pdf_path: Path,
    *,
    dataset_name: str | None = None,
    forecast_df: pd.DataFrame,
    explanation: ForecastExplanation,
    target_column: str,
    timestamp_column: str,
    heatmap_path: Path | None,
    topk_rows: list[list[str]],
    top_k: int,
    trajectory_report: TrajectoryStabilityReport | None = None,
    feature_axis_embedding_stability_report: PerChannelEmbeddingStabilityReport | None = None,
    context_len: int | None = None,
    forecast_horizon: int | None = None,
    channel_labels: list[str] | None = None,
    integrated_gradients_report: dict[str, Any] | None = None,
) -> Path | None:
    """Compose a multi-page PDF with the forecast and attribution heatmap."""
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return None

    attrib = np.asarray(explanation.lag_horizon_attributions)
    K, H = attrib.shape
    flow_magnitudes = getattr(explanation, "flow_magnitudes", None)
    show_heatmap = heatmap_path is not None and heatmap_path.exists()
    show_topk = bool(topk_rows)
    show_semantic_flow = (
        context_len is not None
        and forecast_horizon is not None
        and flow_magnitudes is not None
        and np.asarray(flow_magnitudes).size > 0
    )
    show_trajectory_stability = trajectory_report is not None
    show_feature_axis = getattr(explanation, "channel_horizon_attributions", None) is not None
    show_feature_axis_stability = feature_axis_embedding_stability_report is not None
    show_integrated_gradients = integrated_gradients_report is not None

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(
            0.5,
            0.94,
            "NV-Tesseract Forecasting Interpretability Report",
            ha="center",
            fontsize=18,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.905,
            datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ha="center",
            fontsize=10,
            color="gray",
        )

        sections = [
            section
            for available, section in [
                (True, ["Forecast preview: line chart of predicted target values."]),
                (
                    show_heatmap,
                    [
                        "Lag x Horizon attribution heatmap: how much each past step",
                        "   contributes to each forecast step.",
                        "   a) lag=0 is the most recent input.",
                        "   b) Softmax-normalized weights; brighter cells = time steps",
                        "      that contribute more to the forecast.",
                    ],
                ),
                (show_topk, ["Top-k lag steps per horizon (marginal contributions)."]),
                (
                    show_semantic_flow,
                    [
                        "Semantic flow magnitudes: per-transition latent flow and",
                        "   forecast-vs-history diagnostics.",
                    ],
                ),
                (
                    show_trajectory_stability,
                    [
                        "Latent trajectory stability: temporal-smoothness metrics",
                        "   over the context window.",
                    ],
                ),
                (
                    show_feature_axis,
                    [
                        "Feature-axis attribution: which input channel drives each",
                        "   forecast step (multivariate inputs only).",
                    ],
                ),
                (
                    show_feature_axis_stability,
                    [
                        "Feature-axis embedding stability: per-channel sensitivity",
                        "   to small, controlled input changes (multivariate inputs only).",
                    ],
                ),
                (
                    show_integrated_gradients,
                    [
                        "Integrated gradients: signed input-channel/time",
                        "   sensitivity for the model embedding.",
                    ],
                ),
            ]
            if available
        ]

        summary = [
            "Overview",
            "--------",
            f"Dataset: {dataset_name or ''}",
            f"Target column: {target_column}",
            f"Timestamp column: {timestamp_column}",
            f"Forecast steps (H): {H}",
            f"Lag context size (K): {K}",
            "",
            "What is in this report",
            "----------------------",
        ]
        for number, section in enumerate(sections, start=1):
            summary.extend([f"{number}. {section[0]}", *section[1:]])
        fig.text(0.08, 0.85, "\n".join(summary), ha="left", va="top", fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        if timestamp_column in forecast_df.columns and f"{target_column}_forecast" in forecast_df.columns:
            ax.plot(
                forecast_df[timestamp_column],
                forecast_df[f"{target_column}_forecast"],
                marker="o",
                linewidth=1.5,
            )
            ax.set_xlabel(timestamp_column)
            ax.set_ylabel(f"{target_column} (forecast, original scale)")
            ax.set_title(f"Forecast preview: {target_column}")
            fig.autofmt_xdate()
        else:
            ax.text(0.5, 0.5, "Forecast columns not found", ha="center", va="center")
            ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if show_heatmap:
            try:
                from matplotlib.image import imread
            except ImportError:
                imread = None
            fig = plt.figure(figsize=(11, 8.5))
            if imread is not None:
                ax = fig.add_axes((0.05, 0.18, 0.9, 0.72))
                ax.imshow(imread(str(heatmap_path)))
                ax.axis("off")
            else:
                ax = fig.add_axes((0.05, 0.18, 0.9, 0.72))
                ax.text(0.5, 0.5, "Heatmap PNG could not be embedded.", ha="center", va="center")
                ax.axis("off")
            fig.text(
                0.5,
                0.94,
                "Lag x Horizon attribution heatmap",
                ha="center",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.10,
                "Vertical axis: lag back from the last observation (0 = most recent).\n"
                "Horizontal axis: forecast step (0 = first predicted step).\n"
                "Brighter = that past step has more influence on that forecast step.",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            pdf.savefig(fig)
            plt.close(fig)

        if show_topk:
            header = ["Horizon"]
            for r in range(1, top_k + 1):
                header.extend([f"Rank-{r}\nLag", f"Rank-{r}\nProb"])

            rows_per_page = 30
            chunks = [topk_rows[i : i + rows_per_page] for i in range(0, len(topk_rows), rows_per_page)]
            num_cols = 1 + 2 * top_k
            horizon_col = 0.12
            data_col = max(0.05, (1.0 - horizon_col) / max(1, num_cols - 1))
            col_widths = [horizon_col] + [data_col] * (num_cols - 1)

            for idx, chunk in enumerate(chunks, start=1):
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(
                    0.5,
                    0.95,
                    f"Top-{top_k} lag steps per horizon (page {idx}/{len(chunks)})",
                    ha="center",
                    fontsize=14,
                    fontweight="bold",
                )
                fig.text(
                    0.06,
                    0.91,
                    "Horizon: which forecast step (0 = first predicted step).\n"
                    "Rank-r Lag: how many steps back from the last observation "
                    "for the r-th most influential past step\n"
                    "    (1 = most recent past step). Ranked from highest to "
                    "lowest contribution.\n"
                    "Rank-r Prob: marginal softmax weight of that ranked lag step "
                    "on that horizon\n"
                    "    -- higher means it contributed more to the forecast.",
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="gray",
                )

                ax = fig.add_axes((0.04, 0.04, 0.92, 0.78))
                ax.axis("off")
                table = ax.table(
                    cellText=chunk,
                    colLabels=header,
                    colWidths=col_widths,
                    loc="upper center",
                    cellLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.25)
                for c in range(len(header)):
                    cell = table[(0, c)]
                    cell.set_facecolor("#dddddd")
                    cell.set_text_props(weight="bold")
                    cell.set_height(cell.get_height() * 1.6)

                pdf.savefig(fig)
                plt.close(fig)

        if show_semantic_flow:
            try:
                _semantic_flow_page(
                    pdf,
                    plt,
                    explanation=explanation,
                    context_len=int(context_len),
                    forecast_horizon=int(forecast_horizon),
                )
            except Exception as e:
                logger.warning("Semantic flow page skipped: %s", e)

        if show_trajectory_stability:
            r = trajectory_report
            rows = [
                [
                    "Zero-crossing rate (mean / p95)",
                    f"{r.zero_crossing_rate_mean:.4f}",
                    f"{r.zero_crossing_rate_p95:.4f}",
                ],
                [
                    "Direction-flip rate (mean / p95)",
                    f"{r.direction_flip_rate_mean:.4f}",
                    f"{r.direction_flip_rate_p95:.4f}",
                ],
                [
                    "Relative jitter (mean / p95)",
                    f"{r.relative_jitter_mean:.4f}",
                    f"{r.relative_jitter_p95:.4f}",
                ],
                [
                    "Occupancy positive / negative",
                    f"{r.occupancy_positive_mean:.4f}",
                    f"{r.occupancy_negative_mean:.4f}",
                ],
                ["Latent shape (T / D)", f"{int(r.n_time_steps)}", f"{int(r.n_dimensions)}"],
            ]

            fig = plt.figure(figsize=(11, 8.5))
            fig.text(
                0.5,
                0.95,
                "Latent trajectory stability",
                ha="center",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.06,
                0.90,
                "Per-dimension temporal-smoothness metrics for the latent trajectory.\n"
                "Lower zero-crossing, direction-flip, and relative-jitter values indicate\n"
                "a smoother embedding (supports the framework's stability assumption).\n"
                r"Relative jitter compares $\mathrm{mean}(|\Delta z|)$ with "
                r"$\mathrm{mean}(|z-z_{\mathrm{center}}|)$.",
                ha="left",
                va="top",
                fontsize=9,
                fontfamily="DejaVu Sans",
                math_fontfamily="dejavusans",
                color="gray",
            )

            ax = fig.add_axes((0.08, 0.45, 0.84, 0.35))
            ax.axis("off")
            table = ax.table(
                cellText=rows,
                colLabels=["Metric", "Value 1", "Value 2"],
                colWidths=[0.5, 0.25, 0.25],
                loc="upper center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.4)
            for c in range(3):
                cell = table[(0, c)]
                cell.set_facecolor("#dddddd")
                cell.set_text_props(weight="bold")

            interp_text = (
                "How to read these metrics\n"
                "-------------------------\n"
                "  - Zero-crossing rate: fraction of consecutive steps where the centered\n"
                "    latent value flips sign. Lower => trajectory stays on one side of the\n"
                "    reference for longer (less oscillation).\n"
                "  - Direction-flip rate: fraction of steps where the latent step direction\n"
                "    reverses sign. Lower => monotone, smoother dynamics.\n"
                "  - Relative jitter: average latent step size divided by average displacement\n"
                "    from the reference. Lower => small smooth steps\n"
                "    relative to overall amplitude.\n"
                "  - Occupancy positive / negative: fraction of time the centered trajectory\n"
                "    sits above / below the deadband. Asymmetry can flag regime drift.\n"
                "  - Latent shape (T / D): number of latent time steps and embedding dim;\n"
                "    sanity-check that T matches the context length used for this run.\n"
                "\n"
                "Pair these with the diagnostics block in explanation.json\n"
                "(flow_ratio_forecast_vs_history, curvature_ratio_forecast_vs_history,\n"
                "latent_diag_mahalanobis_ratio_forecast_vs_history) for the forecast-vs-\n"
                "history comparison."
            )
            fig.text(
                0.06,
                0.40,
                interp_text,
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                color="#333333",
            )
            pdf.savefig(fig)
            plt.close(fig)

        # Feature-axis page last (and only for multivariate inputs), matching
        # its position in the cover summary.
        if show_feature_axis:
            try:
                _channel_axis_page(
                    pdf,
                    plt,
                    explanation=explanation,
                    channel_labels=channel_labels,
                )
            except Exception as e:
                logger.warning("Feature-axis page skipped: %s", e)

        if show_feature_axis_stability:
            try:
                _feature_axis_embedding_stability_page(
                    pdf,
                    plt,
                    report=feature_axis_embedding_stability_report,
                    channel_labels=channel_labels,
                )
            except Exception as e:
                logger.warning("Feature-axis embedding-stability page skipped: %s", e)

        if show_integrated_gradients:
            try:
                _integrated_gradients_page(
                    pdf,
                    plt,
                    integrated_gradients_report=integrated_gradients_report,
                )
            except Exception as e:
                logger.warning("Integrated-gradients page skipped: %s", e)

    return pdf_path


def _array_to_jsonable(arr: Any) -> Any:
    """Convert a numpy array (or torch tensor / scalar / None) into JSON-friendly data."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return np.where(np.isfinite(arr), arr, None).tolist() if arr.dtype.kind == "f" else arr.tolist()
    try:
        return _array_to_jsonable(np.asarray(arr))
    except Exception:
        return None


def _scalar_to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _trajectory_report_to_dict(report: TrajectoryStabilityReport | None) -> dict[str, Any] | None:
    """Convert a TrajectoryStabilityReport into a JSON-friendly dict (or None)."""
    if report is None:
        return None
    return {
        "zero_crossing_rate_mean": _scalar_to_jsonable(report.zero_crossing_rate_mean),
        "zero_crossing_rate_p95": _scalar_to_jsonable(report.zero_crossing_rate_p95),
        "direction_flip_rate_mean": _scalar_to_jsonable(report.direction_flip_rate_mean),
        "direction_flip_rate_p95": _scalar_to_jsonable(report.direction_flip_rate_p95),
        "relative_jitter_mean": _scalar_to_jsonable(report.relative_jitter_mean),
        "relative_jitter_p95": _scalar_to_jsonable(report.relative_jitter_p95),
        "occupancy_positive_mean": _scalar_to_jsonable(report.occupancy_positive_mean),
        "occupancy_negative_mean": _scalar_to_jsonable(report.occupancy_negative_mean),
        "n_time_steps": int(report.n_time_steps),
        "n_dimensions": int(report.n_dimensions),
    }


def _feature_axis_embedding_stability_to_dict(
    report: PerChannelEmbeddingStabilityReport | None,
    *,
    channel_labels: list[str] | None = None,
) -> dict[str, Any] | None:
    """Convert a feature-axis embedding-stability report to JSON."""
    if report is None:
        return None
    frame = _feature_axis_embedding_stability_frame(report, channel_labels)
    return {
        "channel_labels": frame["feature"].tolist(),
        "lip_ratio_mean": _array_to_jsonable(frame["lip_ratio_mean"].to_numpy()),
        "lip_ratio_p50": _array_to_jsonable(frame["lip_ratio_p50"].to_numpy()),
        "lip_ratio_p95": _array_to_jsonable(frame["lip_ratio_p95"].to_numpy()),
        "lip_ratio_max": _array_to_jsonable(frame["lip_ratio_max"].to_numpy()),
        "n_trials_per_channel": _array_to_jsonable(frame["n_trials_per_channel"].to_numpy()),
        "n_unique_windows": int(report.n_unique_windows),
        "step_delta_norm_mean": _scalar_to_jsonable(report.step_delta_norm_mean),
    }


def _integrated_gradients_to_dict(
    integrated_gradients_report: dict[str, Any] | None,
    *,
    include_full_arrays: bool = True,
) -> dict[str, Any] | None:
    """Convert an integrated-gradients report into a JSON-friendly dict."""
    if integrated_gradients_report is None:
        return None

    payload: dict[str, Any] = {
        "baseline": integrated_gradients_report.get("baseline"),
        "reduce": integrated_gradients_report.get("reduce"),
        "n_steps": int(integrated_gradients_report.get("n_steps", 0)),
        "n_baselines": int(integrated_gradients_report.get("n_baselines", 0)),
        "embedding_dim": int(integrated_gradients_report.get("embedding_dim", 0)),
        "grad_through_norm": bool(integrated_gradients_report.get("grad_through_norm", False)),
        "overall_effect": _scalar_to_jsonable(integrated_gradients_report.get("overall_effect")),
        "abs_effect": _scalar_to_jsonable(integrated_gradients_report.get("abs_effect")),
        "embedding_delta": _scalar_to_jsonable(integrated_gradients_report.get("embedding_delta")),
        "embedding_delta_abs_mean": _scalar_to_jsonable(integrated_gradients_report.get("embedding_delta_abs_mean")),
        "convergence_delta": _scalar_to_jsonable(integrated_gradients_report.get("convergence_delta")),
        "channel_labels": list(integrated_gradients_report.get("channel_labels", [])),
        "channel_signed_effect": _array_to_jsonable(integrated_gradients_report.get("channel_signed_effect")),
        "channel_abs_effect": _array_to_jsonable(integrated_gradients_report.get("channel_abs_effect")),
        "channel_abs_share": _array_to_jsonable(integrated_gradients_report.get("channel_abs_share")),
        "top_channels": integrated_gradients_report.get("top_channels", []),
    }
    if include_full_arrays:
        payload["attribution"] = _array_to_jsonable(integrated_gradients_report.get("attribution"))
        payload["time_abs_effect"] = _array_to_jsonable(integrated_gradients_report.get("time_abs_effect"))
    return payload


def _explanation_to_dict(
    forecast_df: pd.DataFrame,
    explanation: ForecastExplanation,
    *,
    target_column: str,
    timestamp_column: str = "timestamp",
    dataset_name: str | None = None,
    include_full_arrays: bool = True,
    trajectory_report: TrajectoryStabilityReport | None = None,
    feature_axis_embedding_stability_report: PerChannelEmbeddingStabilityReport | None = None,
    channel_labels: list[str] | None = None,
    integrated_gradients_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the (forecast, explanation) pair as a JSON-serializable dict."""
    fc = forecast_df.copy()
    if timestamp_column in fc.columns and pd.api.types.is_datetime64_any_dtype(fc[timestamp_column]):
        fc[timestamp_column] = fc[timestamp_column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    forecast_records = fc.to_dict(orient="records")

    base = np.asarray(explanation.baseline_forecast)
    scores = np.asarray(explanation.lag_horizon_scores)
    attrib = np.asarray(explanation.lag_horizon_attributions)
    C, H = base.shape if base.ndim == 2 else (None, None)
    K = attrib.shape[0] if attrib.ndim == 2 else None

    surrogate_block: dict[str, Any] | None
    if explanation.surrogate_coef is not None:
        surrogate_block = {
            "coef": _array_to_jsonable(explanation.surrogate_coef),
            "intercept": _array_to_jsonable(explanation.surrogate_intercept),
            "feature_layout": explanation.surrogate_feature_layout,
        }
    else:
        surrogate_block = None

    explanation_block: dict[str, Any] = {
        "shapes": {
            "C_channels": int(C) if C is not None else None,
            "H_horizon": int(H) if H is not None else None,
            "K_lags": int(K) if K is not None else None,
        },
        "baseline_forecast": _array_to_jsonable(base),
        "lag_horizon_scores": _array_to_jsonable(scores),
        "lag_horizon_attributions": _array_to_jsonable(attrib),
        "surrogate": surrogate_block,
        "diagnostics": {
            "flow_ratio_forecast_vs_history": _scalar_to_jsonable(explanation.flow_ratio_forecast_vs_history),
            "flow_variance_ratio_forecast_vs_history": _scalar_to_jsonable(
                explanation.flow_variance_ratio_forecast_vs_history
            ),
            "curvature_ratio_forecast_vs_history": _scalar_to_jsonable(explanation.curvature_ratio_forecast_vs_history),
            "latent_diag_mahalanobis_ratio_forecast_vs_history": _scalar_to_jsonable(
                explanation.latent_diag_mahalanobis_ratio_forecast_vs_history
            ),
            "latent_trajectory_shape": (
                list(np.asarray(explanation.latent_trajectory).shape)
                if getattr(explanation, "latent_trajectory", None) is not None
                else None
            ),
        },
        "trajectory_stability": _trajectory_report_to_dict(trajectory_report),
    }

    # Feature-axis (channel) attribution -- only present for multivariate inputs.
    chan_attrib = getattr(explanation, "channel_horizon_attributions", None)
    stability_block = _feature_axis_embedding_stability_to_dict(
        feature_axis_embedding_stability_report,
        channel_labels=channel_labels,
    )
    if chan_attrib is not None or stability_block is not None:
        feature_axis: dict[str, Any] = {}
        if chan_attrib is not None:
            feature_axis.update(
                {
                    "method": getattr(explanation, "channel_flow_method", None),
                    "channel_horizon_attributions": _array_to_jsonable(np.asarray(chan_attrib)),
                    "residual_ratio_mean": _scalar_to_jsonable(explanation.channel_flow_residual_ratio_mean),
                    "residual_ratio_p95": _scalar_to_jsonable(explanation.channel_flow_residual_ratio_p95),
                }
            )
        if include_full_arrays and getattr(explanation, "per_channel_flow", None) is not None:
            feature_axis["per_channel_flow"] = _array_to_jsonable(np.asarray(explanation.per_channel_flow))
        if stability_block is not None:
            feature_axis["embedding_stability"] = stability_block
        explanation_block["feature_axis"] = feature_axis

    if include_full_arrays:
        explanation_block["flow_magnitudes"] = _array_to_jsonable(explanation.flow_magnitudes)
        explanation_block["latent_trajectory"] = _array_to_jsonable(explanation.latent_trajectory)

    ig_block = _integrated_gradients_to_dict(
        integrated_gradients_report,
        include_full_arrays=include_full_arrays,
    )
    if ig_block is not None:
        explanation_block["integrated_gradients"] = ig_block

    return {
        "metadata": {
            "dataset_name": dataset_name,
            "target_column": target_column,
            "timestamp_column": timestamp_column,
            "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "include_full_arrays": bool(include_full_arrays),
        },
        "forecast": forecast_records,
        "explanation": explanation_block,
    }


def _save_explanation_json(
    *,
    forecast_df: pd.DataFrame,
    explanation: ForecastExplanation,
    path: str | Path,
    target_column: str,
    timestamp_column: str = "timestamp",
    dataset_name: str | None = None,
    include_full_arrays: bool = True,
    indent: int | None = 2,
    trajectory_report: TrajectoryStabilityReport | None = None,
    feature_axis_embedding_stability_report: PerChannelEmbeddingStabilityReport | None = None,
    channel_labels: list[str] | None = None,
    integrated_gradients_report: dict[str, Any] | None = None,
) -> Path:
    """Persist the (forecast, explanation) pair as a JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _explanation_to_dict(
        forecast_df,
        explanation,
        target_column=target_column,
        timestamp_column=timestamp_column,
        dataset_name=dataset_name,
        include_full_arrays=include_full_arrays,
        trajectory_report=trajectory_report,
        feature_axis_embedding_stability_report=feature_axis_embedding_stability_report,
        channel_labels=channel_labels,
        integrated_gradients_report=integrated_gradients_report,
    )
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, allow_nan=False)
    return out_path


def _run_interpretability(
    *,
    model: torch.nn.Module,
    standardizer: Standardizer,
    working_df: pd.DataFrame,
    columns_to_process: list[str],
    timestamp_column: str,
    target_column: str,
    seq_len: int,
    forecast_horizon: int,
    model_horizon: int,
    device: torch.device,
    n_lags: int,
    softmax_tau: float,
    interpretability_output: str | None,
    interpretability_out_dir: str | Path,
    interpretability_run_name: str | None,
    interpretability_top_k: int,
    dataset_name: str | None,
    channel_output_aware: bool = False,
    return_all_channels: bool = False,
    integrated_gradients: bool = False,
    integrated_gradients_baseline: Any = "noise",
    integrated_gradients_steps: int = 64,
    integrated_gradients_n_baselines: int = 1,
    integrated_gradients_internal_batch_size: int | None = None,
    integrated_gradients_reduce: str = "l2",
    integrated_gradients_seed: int = 0,
    integrated_gradients_grad_through_norm: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """Generate the lag x horizon explanation, write artifacts, return (forecast_df, run_dir).

    The forecast comes from the explanation's ``baseline_forecast`` (single
    forward pass, no autoregressive rollout) so callers should treat it as the
    interpretability-aligned forecast. When ``return_all_channels=True``, the
    returned frame contains one forecast column per processed channel.
    """
    if interpretability_output is not None and interpretability_output not in {"json", "pdf"}:
        raise ValueError(f"interpretability_output must be one of None, 'json', 'pdf'; got {interpretability_output!r}")

    context_df = working_df[[timestamp_column] + columns_to_process].copy().tail(seq_len)
    values_lc = context_df[columns_to_process].to_numpy(dtype=np.float32)
    series_lc = standardizer.transform(values_lc)
    x_context_ct = np.swapaxes(series_lc, 0, 1).copy()
    input_mask_l = np.ones((seq_len,), dtype=np.int64)

    # Feature-axis (channel) attribution only makes sense for multivariate inputs.
    n_channels = int(x_context_ct.shape[0])
    channel_axis = n_channels > 1
    # The target column is always channel 0 of columns_to_process, so output-aware
    # attribution explains the target's own forecast.
    output_aware = bool(channel_output_aware) and channel_axis
    chan_cfg = None
    if channel_axis:
        from channel_flow import ChannelFlowConfig

        # The SDK reports the trailing K lag transitions, so restrict the
        # feature-axis estimator to those transitions. This keeps the artifact
        # values aligned with lag_horizon_attribution while avoiding hundreds of
        # unused Jacobian-flow probes for the rest of the 512-step context.
        lag_count = min(int(n_lags), max(0, int(seq_len) - 1))
        start = max(0, int(seq_len) - 1 - lag_count)
        time_indices = tuple(range(start, int(seq_len) - 1))
        chan_cfg = ChannelFlowConfig(time_indices=time_indices)

    model.eval()
    explanation = explain_forecast(
        model,
        x_context_ct=x_context_ct,
        input_mask_l=input_mask_l,
        model_horizon=model_horizon,
        forecast_horizon=forecast_horizon,
        device=device,
        n_lags=n_lags,
        softmax_tau=softmax_tau,
        surrogate=False,
        channel_axis=channel_axis,
        chan_cfg=chan_cfg,
        channel_output_aware=output_aware,
        channel_target=0 if output_aware else None,
    )

    base_std = explanation.baseline_forecast
    H = base_std.shape[1]
    pred_lc = np.swapaxes(base_std, 0, 1).reshape(-1, base_std.shape[0])
    pred_orig_lc = standardizer.inverse(pred_lc)

    time_diffs = working_df[timestamp_column].diff().dropna()
    inferred_freq = (
        time_diffs.mode()[0]
        if len(time_diffs.mode()) > 0
        else (time_diffs.median() if len(time_diffs) else pd.Timedelta(hours=1))
    )
    last_input_time = working_df[timestamp_column].iloc[-1]
    forecast_timestamps = pd.date_range(start=last_input_time + inferred_freq, periods=H, freq=inferred_freq)

    output_channels = columns_to_process if return_all_channels else [target_column]
    forecast_cols: dict[str, Any] = {timestamp_column: pd.to_datetime(forecast_timestamps)}
    for ch_idx, channel_name in enumerate(output_channels):
        forecast_cols[f"{channel_name}_forecast"] = pred_orig_lc[:, ch_idx].astype(np.float32)
    forecast_df = pd.DataFrame(forecast_cols)

    base_dir = Path(interpretability_out_dir)
    if interpretability_run_name is None:
        interpretability_run_name = datetime.now(tz=timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_dir / interpretability_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    forecast_df.to_csv(run_dir / "forecast.csv", index=False)

    write_json = interpretability_output in (None, "json")
    write_pdf = interpretability_output in (None, "pdf")

    attributions = np.asarray(explanation.lag_horizon_attributions, dtype=np.float32)
    scores = (
        np.asarray(explanation.lag_horizon_scores, dtype=np.float32)
        if getattr(explanation, "lag_horizon_scores", None) is not None
        else None
    )

    heatmap_path: Path | None = None
    topk_rows: list[list[str]] = []
    if write_pdf:
        heatmap_path = _save_lag_horizon_artifacts(run_dir, attributions, scores=scores)
        _, topk_rows = _topk_lag_steps_per_horizon(
            scores if scores is not None else attributions,
            top_k=interpretability_top_k,
        )
        if getattr(explanation, "flow_magnitudes", None) is not None:
            try:
                _save_semantic_flow_csv(
                    run_dir,
                    np.asarray(explanation.flow_magnitudes, dtype=np.float32),
                    context_len=seq_len,
                    forecast_horizon=forecast_horizon,
                )
            except Exception as e:
                logger.warning("semantic_flow.csv skipped: %s", e)
        if getattr(explanation, "channel_horizon_attributions", None) is not None:
            try:
                _save_channel_axis_artifacts(
                    run_dir,
                    np.asarray(explanation.channel_horizon_attributions, dtype=np.float32),
                    channel_labels=columns_to_process,
                )
            except Exception as e:
                logger.warning("channel_horizon artifacts skipped: %s", e)

    trajectory_report: TrajectoryStabilityReport | None = None
    try:
        trajectory_report = compute_trajectory_stability(
            model,
            x_context_ct,
            seq_len=seq_len,
            device=device,
            batch_size=32,
        )
    except Exception as e:
        logger.warning("Trajectory stability skipped: %s", e)

    feature_axis_embedding_stability_report: PerChannelEmbeddingStabilityReport | None = None
    if channel_axis:
        try:
            feature_axis_embedding_stability_report = compute_per_channel_embedding_stability(
                model,
                x_context_ct,
                seq_len=seq_len,
                input_mask_t=input_mask_l,
                device=device,
            )
        except Exception as e:
            logger.warning("Feature-axis embedding stability skipped: %s", e)

    if write_pdf and feature_axis_embedding_stability_report is not None:
        _save_feature_axis_embedding_stability_csv(
            run_dir,
            feature_axis_embedding_stability_report,
            columns_to_process,
        )

    integrated_gradients_report: dict[str, Any] | None = None
    if integrated_gradients:
        model.eval()
        integrated_gradients_report = _compute_integrated_gradients_report(
            model=model,
            x_context_ct=x_context_ct,
            input_mask_l=input_mask_l,
            device=device,
            channel_labels=columns_to_process,
            baseline=integrated_gradients_baseline,
            n_steps=integrated_gradients_steps,
            n_baselines=integrated_gradients_n_baselines,
            internal_batch_size=integrated_gradients_internal_batch_size,
            reduce=integrated_gradients_reduce,
            seed=integrated_gradients_seed,
            grad_through_norm=integrated_gradients_grad_through_norm,
        )
        if write_pdf:
            try:
                _save_integrated_gradients_artifacts(run_dir, integrated_gradients_report)
            except Exception as e:
                logger.warning("integrated_gradients artifacts skipped: %s", e)

    if write_json:
        _save_explanation_json(
            forecast_df=forecast_df,
            explanation=explanation,
            path=run_dir / "explanation.json",
            target_column=target_column,
            timestamp_column=timestamp_column,
            dataset_name=dataset_name,
            trajectory_report=trajectory_report,
            feature_axis_embedding_stability_report=feature_axis_embedding_stability_report,
            channel_labels=columns_to_process,
            integrated_gradients_report=integrated_gradients_report,
        )
        logger.info("Interpretability JSON written to: %s", run_dir / "explanation.json")

    if write_pdf:
        pdf_path = run_dir / "explanation_report.pdf"
        produced = _build_pdf_report(
            pdf_path,
            forecast_df=forecast_df,
            explanation=explanation,
            target_column=target_column,
            timestamp_column=timestamp_column,
            heatmap_path=heatmap_path,
            topk_rows=topk_rows,
            top_k=interpretability_top_k,
            dataset_name=dataset_name,
            trajectory_report=trajectory_report,
            feature_axis_embedding_stability_report=feature_axis_embedding_stability_report,
            context_len=seq_len,
            forecast_horizon=forecast_horizon,
            channel_labels=columns_to_process,
            integrated_gradients_report=integrated_gradients_report,
        )
        if produced is None:
            logger.warning("Interpretability PDF report skipped: matplotlib is not installed.")
        else:
            logger.info("Interpretability PDF report written to: %s", pdf_path)

    return forecast_df, run_dir


def perform_forecasting(
    df: pd.DataFrame,
    *,
    config: ForecastingConfig | str | Path | None = None,
    context_df: pd.DataFrame | None = None,  # Optional context DataFrame for DARR mode
) -> pd.DataFrame:
    """
    Perform time series forecasting using NV-Tesseract with optional context-enhanced mode (DARR).
    Supports autoregressive forecasting for horizons beyond the model's native capability.

    ALWAYS uses InferenceOnlyDataset - only requires seq_len rows for inference.

    When ``return_all_channels=True``, the returned DataFrame contains one
    ``{column}_forecast`` column per processed channel (target column first,
    then the remaining numeric feature columns) instead of only
    ``{target_column}_forecast``. This also applies to interpretability mode.
    """
    cfg = _resolve_forecasting_config(config)

    # Keep locals only for values resolved or transformed during this run.
    standardizer_pkl = cfg.standardizer_pkl
    ckpt = cfg.ckpt
    model_horizon = cfg.model_horizon if cfg.model_horizon is not None else cfg.forecast_horizon
    stride = cfg.stride if cfg.stride is not None else model_horizon
    context_stride = cfg.context_stride if cfg.context_stride is not None else model_horizon
    device = DEVICE if cfg.device is None else torch.device(cfg.device)

    # Validate that model_horizon is reasonable
    if model_horizon <= 0:
        raise ValueError(f"model_horizon must be positive, got {model_horizon}")

    if cfg.forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {cfg.forecast_horizon}")

    # Maximum forecast horizon limit
    MAX_FORECAST_HORIZON = 512
    if cfg.forecast_horizon > MAX_FORECAST_HORIZON:
        raise ValueError(f"forecast_horizon must be <= {MAX_FORECAST_HORIZON}, got {cfg.forecast_horizon}")

    # The model's native horizon limits direct predictions, but DARR retrieves
    # observed continuations from context data. Keep the existing native-sized
    # context windows for shorter requests while retaining the full retrieval
    # trajectory whenever callers ask for a longer forecast.
    darr_context_horizon = max(model_horizon, cfg.forecast_horizon)

    # Input validation
    if df is None or df.empty:
        raise ValueError("Input DataFrame is required and cannot be empty")

    # Validate minimum rows
    if len(df) < cfg.seq_len:
        raise ValueError(f"DataFrame has {len(df)} rows but seq_len requires at least {cfg.seq_len} rows")

    # Validate timestamp column
    if cfg.timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{cfg.timestamp_column}' not found in DataFrame")

    if df[cfg.timestamp_column].isnull().any():
        raise ValueError(f"Timestamp column '{cfg.timestamp_column}' contains NULL values")

    # Try to convert timestamp column to datetime if it's not already
    working_df = df.copy()
    try:
        if not pd.api.types.is_datetime64_any_dtype(working_df[cfg.timestamp_column]):
            working_df[cfg.timestamp_column] = pd.to_datetime(working_df[cfg.timestamp_column])
    except Exception as e:
        raise ValueError(f"Cannot parse timestamp column '{cfg.timestamp_column}' as datetime: {e}")

    # Validate target column
    if cfg.target_column not in df.columns:
        raise ValueError(f"Target column '{cfg.target_column}' not found in DataFrame")

    # Check if target column is numeric
    if not pd.api.types.is_numeric_dtype(working_df[cfg.target_column]):
        raise ValueError(f"Target column '{cfg.target_column}' must contain numeric values")

    # Handle NULL values in target column - fill with zeros
    if working_df[cfg.target_column].isnull().any():
        logger.warning("Found NULL values in '%s', filling with zeros", cfg.target_column)
        working_df[cfg.target_column] = working_df[cfg.target_column].fillna(0)

    # Automatically detect all numeric columns to use as features
    numeric_columns = working_df.select_dtypes(include=[np.number]).columns.tolist()

    # Make sure target column is first in the list
    if cfg.target_column in numeric_columns:
        numeric_columns.remove(cfg.target_column)
    columns_to_process = [cfg.target_column] + numeric_columns

    # return_all_channels emits one {column}_forecast per channel; reject a
    # timestamp column name that would collide with one of them, since the
    # collision would silently overwrite the timestamp column in the output.
    if cfg.return_all_channels:
        colliding = [c for c in columns_to_process if f"{c}_forecast" == cfg.timestamp_column]
        if colliding:
            raise ValueError(
                f"timestamp_column {cfg.timestamp_column!r} collides with the forecast column "
                f"emitted for input column {colliding[0]!r}; rename one of them"
            )

    # Fill NaN values with zeros for all numeric columns
    for col in columns_to_process:
        if working_df[col].isnull().any():
            logger.warning("Found NULL values in '%s', filling with zeros", col)
            working_df[col] = working_df[col].fillna(0)

    # Set random seed
    control_randomness(seed=cfg.seed)

    # For the two published checkpoints, resolve the right one from use_cross_channel.
    # Custom paths are left untouched — state dict inspection in _load_cached_model handles them.
    if ckpt in _KNOWN_CHECKPOINTS:
        ckpt = CHECKPOINT_CROSS_CHANNEL if cfg.use_cross_channel else CHECKPOINT_BASE

    # Auto-download model weights if they don't exist
    try:
        standardizer_pkl, ckpt = download_model_weights(standardizer_pkl=standardizer_pkl, ckpt=ckpt)
    except Exception as e:
        logger.warning("Could not auto-download weights: %s", e)
        logger.warning("Using provided paths as-is. Make sure the files exist locally.")

    # Determine mode
    if context_df is not None:
        mode = "darr"
        logger.info("Using DARR mode (Context-Enhanced Forecasting) with alpha=%s", cfg.alpha)
    else:
        mode = "standard"
        logger.info("Using standard forecasting mode (inference only)")

    # Temporary files for DataFrame conversion
    temp_test_csv = None
    temp_context_csv = None

    try:
        # Create a unique CSV for this invocation. PID-only names collide when
        # multiple requests run concurrently in one service process.
        temp_test_csv = _create_temp_csv_path("nv_tesseract_test_")

        # Save only the necessary columns (timestamp + value columns)
        csv_df = working_df[[cfg.timestamp_column] + columns_to_process].copy()
        csv_df.rename(columns={cfg.timestamp_column: "timestamp"}, inplace=True)
        csv_df.to_csv(temp_test_csv, index=False)
        test_csv_path = temp_test_csv

        # Handle context DataFrame if provided for DARR mode
        if context_df is not None:
            temp_context_csv = _create_temp_csv_path("nv_tesseract_context_")

            # Validate context DataFrame columns
            if cfg.timestamp_column not in context_df.columns:
                raise ValueError(f"Context DataFrame missing timestamp column '{cfg.timestamp_column}'")

            if cfg.target_column not in context_df.columns:
                raise ValueError(f"Context DataFrame missing target column '{cfg.target_column}'")

            # Validate context DataFrame has enough rows for at least one window
            min_context_rows = cfg.seq_len + darr_context_horizon
            if len(context_df) < min_context_rows:
                raise ValueError(
                    f"Context DataFrame has {len(context_df)} rows but requires at least "
                    f"{min_context_rows} rows "
                    f"(seq_len={cfg.seq_len} + context_horizon={darr_context_horizon})"
                )

            # Process context DataFrame similarly to main DataFrame
            context_working = context_df.copy()

            # Convert timestamp if needed
            try:
                if not pd.api.types.is_datetime64_any_dtype(context_working[cfg.timestamp_column]):
                    context_working[cfg.timestamp_column] = pd.to_datetime(context_working[cfg.timestamp_column])
            except Exception as e:
                raise ValueError(f"Cannot parse context timestamp column '{cfg.timestamp_column}' as datetime: {e}")

            # Get numeric columns from context DataFrame
            context_numeric = context_working.select_dtypes(include=[np.number]).columns.tolist()

            # Ensure target column is included
            if cfg.target_column in context_numeric:
                context_numeric.remove(cfg.target_column)

            # COLUMN COMPATIBILITY CHECK AND ALIGNMENT
            # Common feature columns in input-DataFrame order (target excluded;
            # it is always included first). The canonical channel order is
            # anchored to the input DataFrame so the caller/checkpoint-facing
            # layout is preserved for the channels that remain. The context CSV
            # always follows this canonical order (when the column lists already
            # match, common_columns == context_numeric).
            context_numeric_set = set(context_numeric)
            common_columns = [col for col in numeric_columns if col in context_numeric_set]
            context_columns_to_use = [cfg.timestamp_column, cfg.target_column] + common_columns

            # Realign whenever the context column list differs from the input
            # list. A set comparison is not enough: the same columns in a
            # different order pass the downstream shape checks but blend
            # mismatched channels in the hybrid output.
            if numeric_columns != context_numeric:
                if set(numeric_columns) == context_numeric_set:
                    # Order-only difference: every column is kept. The main CSV
                    # is already written in the canonical (input) order, so only
                    # the context CSV needs realigning.
                    logger.warning(
                        "Context dataset lists the same numeric columns in a different order; "
                        "realigning the context to the input column order: %s",
                        common_columns,
                    )
                else:
                    logger.warning("Column mismatch detected between input and context datasets")
                    logger.warning("  Input dataset columns: %s", numeric_columns)
                    logger.warning("  Context dataset columns: %s", context_numeric)
                    logger.warning("  Common columns (input order): %s", common_columns)

                    if len(common_columns) == 0:
                        raise ValueError(
                            f"No common numeric columns found between input and context datasets.\n"
                            f"Input dataset has: {numeric_columns}\n"
                            f"Context dataset has: {context_numeric}\n"
                            f"For DARR mode to work, both datasets must share at least some numeric columns."
                        )

                    logger.warning("  Using only common columns for consistent predictions: %s", common_columns)

                    # The channel set narrowed: rewrite the main CSV in the same
                    # canonical order as the context CSV.
                    columns_to_process = [cfg.target_column] + common_columns
                    csv_df = working_df[[cfg.timestamp_column] + columns_to_process].copy()
                    csv_df.rename(columns={cfg.timestamp_column: "timestamp"}, inplace=True)
                    csv_df.to_csv(temp_test_csv, index=False)

            # Create CSV with selected columns
            context_csv_df = context_working[context_columns_to_use].copy()
            context_csv_df.rename(columns={cfg.timestamp_column: "timestamp"}, inplace=True)

            # Fill NULLs in context data
            for col in context_csv_df.columns:
                if col != "timestamp" and context_csv_df[col].isnull().any():
                    logger.warning("Found NULL values in context DataFrame column '%s', filling with zeros", col)
                    context_csv_df[col] = context_csv_df[col].fillna(0)

            context_csv_df.to_csv(temp_context_csv, index=False)
            context_csv_path = temp_context_csv

        # Load standardizer
        standardizer = _load_standardizer_artifact(standardizer_pkl)

        # ALWAYS use InferenceOnlyDataset for the main test data
        test_dataset = InferenceOnlyDataset(csv_path=test_csv_path, seq_len=cfg.seq_len, standardizer=standardizer)

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Always 1 for inference-only
            shuffle=False,
            num_workers=0,  # Set to 0 for single sample
            pin_memory=torch.cuda.is_available(),
        )

        model = _load_cached_model(
            model_name=cfg.model_name,
            ckpt=ckpt,
            seq_len=cfg.seq_len,
            model_horizon=model_horizon,
            device=device,
            local_files_only=cfg.local_files_only,
            use_cross_channel=cfg.use_cross_channel,
            cross_channel_heads=cfg.cross_channel_heads,
            cross_channel_dropout=cfg.cross_channel_dropout,
        )

        # Interpretability path: produce explanation artifacts (heatmap, top-k
        # tables, JSON, PDF) using the same loaded model. The forecast returned
        # comes from the explanation's baseline (single forward pass, no AR
        # rollout) so that the persisted forecast.csv aligns 1:1 with the
        # attribution matrix.
        if cfg.interpretability:
            result_df, run_dir = _run_interpretability(
                model=model,
                standardizer=standardizer,
                working_df=working_df,
                columns_to_process=columns_to_process,
                timestamp_column=cfg.timestamp_column,
                target_column=cfg.target_column,
                seq_len=cfg.seq_len,
                forecast_horizon=cfg.forecast_horizon,
                model_horizon=model_horizon,
                device=device,
                n_lags=cfg.n_lags,
                softmax_tau=cfg.softmax_tau,
                interpretability_output=cfg.interpretability_output,
                interpretability_out_dir=cfg.interpretability_out_dir,
                interpretability_run_name=cfg.interpretability_run_name,
                interpretability_top_k=cfg.interpretability_top_k,
                dataset_name=cfg.interpretability_dataset_name,
                channel_output_aware=cfg.channel_output_aware,
                return_all_channels=cfg.return_all_channels,
                integrated_gradients=cfg.integrated_gradients,
                integrated_gradients_baseline=cfg.integrated_gradients_baseline,
                integrated_gradients_steps=cfg.integrated_gradients_steps,
                integrated_gradients_n_baselines=cfg.integrated_gradients_n_baselines,
                integrated_gradients_internal_batch_size=cfg.integrated_gradients_internal_batch_size,
                integrated_gradients_reduce=cfg.integrated_gradients_reduce,
                integrated_gradients_seed=cfg.seed,
                integrated_gradients_grad_through_norm=cfg.integrated_gradients_grad_through_norm,
            )
            logger.info("Interpretability bundle written to: %s", run_dir)
            if cfg.save_preds:
                result_df.to_csv(cfg.save_preds, index=False)
                logger.info("Saved predictions to %s", cfg.save_preds)
            return result_df

        # Perform inference based on mode
        if mode == "darr":
            # Context-Enhanced Forecasting (DARR Mode)

            # Retrieve observed context continuations for the entire requested
            # horizon. The native model horizon only constrains the direct
            # component, which is extended autoregressively below.
            context_dataset = CSVLongHorizonSimpleDataset(
                csv_path=context_csv_path,
                data_split="train",
                seq_len=cfg.seq_len,
                forecast_horizon=darr_context_horizon,
                standardizer=None,
                standardize=False,
                stride=context_stride,
            )
            context_dataset.standardizer = standardizer
            context_dataset.series = context_dataset.standardizer.transform(context_dataset.values)

            context_loader = DataLoader(
                context_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
            )

            # Build context memory with observed continuations long enough for
            # the requested retrieval forecast.
            # Wrap in DataParallel when multiple GPUs are available: the context
            # loader has real batch depth so splitting across GPUs gives a meaningful
            # speedup, especially with cross-channel attention enabled.
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            darr_embedder = torch.nn.DataParallel(_DARREmbedWrapper(model)) if n_gpus > 1 else model
            if n_gpus > 1:
                logger.info("DARR context build: using %d GPUs via DataParallel", n_gpus)
            DB_E, DB_Y = build_context_memory(darr_embedder, context_loader, device, cosine=True)

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
                    model, timeseries, input_mask, model_horizon, cfg.forecast_horizon, standardizer, device
                )
                preds_direct.append(batch_preds)

            Q_E = np.concatenate(Q_E_list, axis=0)
            Q_E = np.nan_to_num(Q_E, nan=0.0, posinf=0.0, neginf=0.0)
            Q_E = l2_normalize(Q_E)

            preds_direct = np.concatenate(preds_direct, axis=0)

            # kNN retrieval forecast
            preds_knn = knn_forecast(DB_E, DB_Y, Q_E, k=cfg.k, temperature=cfg.temperature)
            preds_knn = preds_knn[:, :, : cfg.forecast_horizon]

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
            preds_hybrid = cfg.alpha * preds_direct + (1 - cfg.alpha) * preds_knn

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
                    model, timeseries, input_mask, model_horizon, cfg.forecast_horizon, standardizer, device
                )
                preds.append(batch_preds)

            preds = np.concatenate(preds, axis=0)

        # Convert to original scale
        B, C, H = preds.shape
        P_flat = preds.transpose(0, 2, 1).reshape(-1, C)

        P_orig = test_dataset.inverse_transform(P_flat)

        # Reshape back to [B, C, H]
        P_orig_reshaped = P_orig.reshape(B * H, C).reshape(B, H, C).transpose(0, 2, 1)

        # Infer frequency from timestamp column
        time_diffs = working_df[cfg.timestamp_column].diff().dropna()
        if len(time_diffs) > 0:
            inferred_freq = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        else:
            inferred_freq = pd.Timedelta(hours=1)

        last_input_time = working_df[cfg.timestamp_column].iloc[-1]
        forecast_timestamps = pd.date_range(
            start=last_input_time + inferred_freq, periods=cfg.forecast_horizon, freq=inferred_freq
        )

        # Emit the target channel only (default), or one {column}_forecast per
        # processed channel when return_all_channels=True. P_orig_reshaped is
        # [B, C, H] with B == 1 (single inference window) and channels ordered
        # as columns_to_process (target column first, then the remaining
        # numeric features in input-column order). The backbone predicts every
        # channel anyway, so emitting them all avoids re-running the SDK once
        # per column (V calls -> 1) for multivariate use cases.
        output_channels = columns_to_process if cfg.return_all_channels else [cfg.target_column]
        cols = {cfg.timestamp_column: forecast_timestamps}
        for ch_idx, channel_name in enumerate(output_channels):
            cols[f"{channel_name}_forecast"] = P_orig_reshaped[0, ch_idx, : cfg.forecast_horizon].tolist()
        result_df = pd.DataFrame(cols)

        # Save predictions if requested
        if cfg.save_preds:
            result_df.to_csv(cfg.save_preds, index=False)
            logger.info("Saved predictions to %s", cfg.save_preds)

        if mode == "darr":
            logger.info("%s", "\n" + "=" * 60)
            logger.info("DARR Mode Results")
            logger.info("%s", "=" * 60)
        else:
            logger.info("%s", "\n" + "=" * 60)
            logger.info("Results")
            logger.info("%s", "=" * 60)
        logger.info("Added columns: %s", ", ".join(f"{c}_forecast" for c in output_channels))

        return result_df

    finally:
        # Clean up temporary files
        for temp_csv in (temp_test_csv, temp_context_csv):
            if temp_csv:
                Path(temp_csv).unlink(missing_ok=True)


class NVTesseractForecasting(
    ModelHubMixin,
    library_name="nv-tesseract",
    tags=["time-series", "forecasting"],
    repo_url="https://github.com/NVIDIA/NV-Tesseract",
    docs_url="https://huggingface.co/nvidia/nv-tesseract-forecasting",
):
    """NV-Tesseract Forecasting model with HuggingFace Hub integration.

    Example::

        model = NVTesseractForecasting.from_pretrained("nvidia/nv-tesseract-forecasting")
        predictions = model.forecast(df, forecast_horizon=72)

        # Save weights locally or push to Hub
        model.save_pretrained("./my-forecasting-model")
        model.push_to_hub("username/my-forecasting-model")
    """

    def __init__(self, *, standardizer_pkl: str, ckpt: str) -> None:
        self.standardizer_pkl = standardizer_pkl
        self.ckpt = ckpt

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **model_kwargs,
    ) -> "NVTesseractForecasting":
        standardizer_name = model_kwargs.get("standardizer_pkl", "standardizer.pkl")
        ckpt_name = model_kwargs.get("ckpt", CHECKPOINT_CROSS_CHANNEL)
        local_path = Path(model_id)
        if local_path.is_dir():
            return cls(
                standardizer_pkl=str(local_path / standardizer_name),
                ckpt=str(local_path / ckpt_name),
            )
        standardizer_pkl, ckpt = download_model_weights(
            standardizer_pkl=standardizer_name,
            ckpt=ckpt_name,
            repo_id=model_id,
            force_download=force_download,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        return cls(standardizer_pkl=standardizer_pkl, ckpt=ckpt)

    def _save_pretrained(self, save_directory: Path) -> None:
        import shutil

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.standardizer_pkl, save_directory / Path(self.standardizer_pkl).name)
        shutil.copy2(self.ckpt, save_directory / Path(self.ckpt).name)

    def forecast(
        self,
        df: "pd.DataFrame",
        *,
        config: "ForecastingConfig | str | Path | None" = None,
        context_df: "pd.DataFrame | None" = None,
    ) -> "pd.DataFrame":
        """Forecast a DataFrame using this instance's artifacts."""
        resolved = replace(
            _resolve_forecasting_config(config),
            standardizer_pkl=self.standardizer_pkl,
            ckpt=self.ckpt,
        )
        return perform_forecasting(df, config=resolved, context_df=context_df)
