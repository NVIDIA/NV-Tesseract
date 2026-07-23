#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fine-tune NV-Tesseract AD Diffusion on normal windows from a user CSV.

Run from the ad_diffusion directory:
    uv run python examples/finetune_example.py \
        --csv /path/to/normal_training_data.csv \
        --timestamp-col timestamp \
        --label-col is_anomaly \
        --epochs 10 \
        --output-dir artifacts/finetune_my_data

The training CSV should contain mostly normal behavior. Non-numeric columns are
ignored; pass --timestamp-col, --label-col, or --drop-cols to exclude numeric
metadata from the feature matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from sklearn.decomposition import PCA
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.main_model import TSDiffuser_Generic
from sdk.inference_ad import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_MODEL_FILENAME,
    HF_REPO_ID,
    download_model_weights,
    load_config,
)

LOGGER = logging.getLogger("ad_diffusion_finetune")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_distributed(rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class FeatureAdapter:
    """Fit train-time dimensionality adaptation and min-max scaling."""

    def __init__(self, target_dim: int, scale_factor: float, seed: int) -> None:
        self.target_dim = target_dim
        self.scale_factor = scale_factor
        self.seed = seed
        self.pca: PCA | None = None
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.input_dim: int | None = None

    def fit(self, data: np.ndarray) -> None:
        self.input_dim = int(data.shape[1])
        if data.shape[1] > self.target_dim:
            if data.shape[0] < self.target_dim:
                raise ValueError(
                    f"PCA needs at least target_dim rows; got {data.shape[0]} rows for target_dim={self.target_dim}."
                )
            self.pca = PCA(n_components=self.target_dim, random_state=self.seed)
            data = self.pca.fit_transform(data)
        elif data.shape[1] < self.target_dim:
            data = self._pad(data)

        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)

    def transform(self, data: np.ndarray) -> torch.Tensor:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("FeatureAdapter.fit must be called before transform.")

        if self.pca is not None:
            data = self.pca.transform(data)
        elif data.shape[1] < self.target_dim:
            data = self._pad(data)
        elif data.shape[1] > self.target_dim:
            data = data[:, : self.target_dim]

        denom = np.where((self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        data = (data - self.min_) / denom
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.tensor(data, dtype=torch.float32) * self.scale_factor

    def metadata(self) -> dict[str, Any]:
        return {
            "target_dim": self.target_dim,
            "input_dim": self.input_dim,
            "scale_factor": self.scale_factor,
            "uses_pca": self.pca is not None,
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
        }

    def _pad(self, data: np.ndarray) -> np.ndarray:
        pad_width = self.target_dim - data.shape[1]
        return np.pad(data, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


class MaskedWindowDataset(Dataset):
    """Sliding windows with complementary segment masks used by TSDiffuser_Generic."""

    def __init__(self, data: torch.Tensor, window_length: int, stride: int, split: int) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be positive.")
        if split <= 0:
            raise ValueError("split must be positive.")
        if window_length % split != 0:
            raise ValueError("window_length must be divisible by split.")
        if len(data) < window_length:
            raise ValueError(f"Need at least {window_length} rows, got {len(data)}.")

        self.data = data
        self.window_length = window_length
        self.stride = max(1, stride)
        self.split = split
        self.begin_indexes = list(range(0, len(data) - window_length + 1, self.stride))

    def __len__(self) -> int:
        return len(self.begin_indexes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start_idx = self.begin_indexes[idx]
        observed_data = self.data[start_idx : start_idx + self.window_length]
        observed_mask = torch.ones_like(observed_data)
        strategy_type = torch.tensor(random.randint(0, 1), dtype=torch.long)

        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": self._create_mask(observed_mask, int(strategy_type.item())),
            "timepoints": torch.arange(self.window_length, dtype=torch.float32),
            "strategy_type": strategy_type,
        }

    def _create_mask(self, observed_mask: torch.Tensor, strategy_type: int) -> torch.Tensor:
        mask = torch.zeros_like(observed_mask)
        segment = self.window_length // self.split
        for split_idx in range(self.split):
            start = split_idx * segment
            end = min(start + segment, self.window_length)
            if (split_idx % 2 == 0 and strategy_type == 0) or (split_idx % 2 == 1 and strategy_type == 1):
                mask[start:end] = 1
        return mask


def parse_column_list(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_numeric_frame(path: str, args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    drop_cols = set(parse_column_list(args.drop_cols))
    for col in (args.timestamp_col, args.label_col):
        if col:
            drop_cols.add(col)
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError(f"No numeric feature columns found in {path}.")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return numeric_df.to_numpy(dtype=np.float32), list(numeric_df.columns)


def split_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_raw, columns = load_numeric_frame(args.csv, args)
    if args.val_csv:
        val_raw, val_columns = load_numeric_frame(args.val_csv, args)
        if val_columns != columns:
            raise ValueError("Validation CSV numeric feature columns must match the training CSV.")
        return train_raw, val_raw, columns

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1 when --val-csv is not used.")
    split_idx = int(round(len(train_raw) * (1.0 - args.val_ratio)))
    if split_idx <= 0 or split_idx >= len(train_raw):
        raise ValueError("Validation split produced an empty train or validation partition.")
    return train_raw[:split_idx], train_raw[split_idx:], columns


def resolve_model_assets(args: argparse.Namespace) -> tuple[str, str]:
    model_path = args.pretrained_model or DEFAULT_MODEL_FILENAME
    config_path = args.config or DEFAULT_CONFIG_FILENAME

    model_exists = Path(model_path).exists()
    config_exists = Path(config_path).exists()
    if model_exists and config_exists:
        return model_path, config_path
    if args.no_download:
        missing = [path for path, exists in ((model_path, model_exists), (config_path, config_exists)) if not exists]
        raise FileNotFoundError(f"Missing model assets and --no-download was set: {missing}")

    LOGGER.info("Downloading missing pretrained assets from %s", HF_REPO_ID)
    return download_model_weights(model_path=model_path, config_path=config_path, repo_id=args.repo_id)


def load_checkpoint_and_config(model_path: str, config_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = load_config(config_path)
    return checkpoint, config


def load_state_dict(model: torch.nn.Module, checkpoint: dict[str, Any] | Any) -> None:
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing:
        LOGGER.info("Missing checkpoint keys initialized from current model config: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected checkpoint keys skipped: %s", unexpected)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def unpack_loss(loss_result: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(loss_result, dict):
        return loss_result["total_loss"]
    return loss_result


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        loss = unpack_loss(model(batch_to_device(batch, device)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in tqdm(loader, desc="val", leave=False):
        loss = unpack_loss(model(batch_to_device(batch, device)))
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    train_loss: float,
    val_loss: float,
    preprocessing: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "args": vars(args),
            "preprocessing": preprocessing,
        },
        path,
    )


def _load_run_config(path: str) -> dict[str, Any]:
    """Load a JSON or YAML run config written by AutoMLRunner at {config_path}."""
    p = Path(path)
    with p.open() as f:
        if p.suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f) or {}


def _run_config_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested run config into argparse dest names."""
    ds = cfg.get("dataset", {})
    tr = cfg.get("train", {})
    mapping = {
        "csv": ds.get("csv"),
        "val_ratio": ds.get("val_ratio"),
        "timestamp_col": ds.get("timestamp_col"),
        "mask_ratio": ds.get("mask_ratio"),
        "window_length": ds.get("window_length"),
        "window_stride": ds.get("window_stride"),
        "split": ds.get("split"),
        "lr": tr.get("lr"),
        "batch_size": tr.get("batch_size"),
        "weight_decay": tr.get("weight_decay"),
        "grad_clip": tr.get("grad_clip"),
        "epochs": tr.get("epochs"),
        "seed": tr.get("seed"),
        "output_dir": tr.get("output_dir"),
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune NV-Tesseract AD Diffusion on a CSV dataset.")
    parser.add_argument(
        "--run-config",
        default=None,
        help="JSON/YAML config file generated by AutoMLRunner. CLI flags override any value in this file.",
    )

    parser.add_argument("--csv", default=None, help="Training CSV, ideally containing normal behavior.")
    parser.add_argument("--val-csv", default=None, help="Optional validation CSV. Otherwise --csv is split temporally.")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--label-col", default=None, help="Optional label column to drop from the feature matrix.")
    parser.add_argument("--drop-cols", default=None, help="Comma-separated extra columns to drop before training.")

    parser.add_argument("--pretrained-model", default=None, help="Pretrained checkpoint. Defaults to final_model.pth.")
    parser.add_argument("--config", default=None, help="Model config YAML. Defaults to curriculum_medium.yaml.")
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--no-download", action="store_true", help="Do not auto-download missing pretrained assets.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use. Defaults to all available GPUs. Set to 1 to force single-GPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/finetune")

    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--split", type=int, default=None, help="Number of alternating mask segments per window.")
    parser.add_argument("--mask-ratio", type=float, default=0.7, help="Random masking ratio used during training.")
    parser.add_argument("--scale-factor", type=float, default=None)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.3,
        help="Temporal validation fraction when --val-csv is not used. Keep validation rows >= --window-length.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = _build_parser()

    # First pass: extract --run-config only, ignore everything else.
    preliminary, _ = parser.parse_known_args()
    if preliminary.run_config:
        cfg = _load_run_config(preliminary.run_config)
        parser.set_defaults(**_run_config_defaults(cfg))

    args = parser.parse_args()
    if not args.csv:
        parser.error("--csv is required (either directly or via --run-config dataset.csv)")
    return args


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    is_main = rank == 0
    init_distributed(rank, world_size)

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(args.seed + rank)
    device = get_device(rank if world_size > 1 else 0)
    output_dir = Path(args.output_dir)

    model_path, config_path = resolve_model_assets(args)
    checkpoint, config = load_checkpoint_and_config(model_path, config_path)
    target_dim = int(config.get("model", {}).get("target_dim", 40))
    dataset_config = config.get("dataset", {})
    window_length = args.window_length or int(dataset_config.get("window_length", 100))
    split = args.split or int(dataset_config.get("split", 4))
    scale_factor = args.scale_factor or float(dataset_config.get("scale_factor", 20))

    train_raw, val_raw, columns = split_arrays(args)
    adapter = FeatureAdapter(target_dim=target_dim, scale_factor=scale_factor, seed=args.seed)
    adapter.fit(train_raw)
    train_tensor = adapter.transform(train_raw)
    val_tensor = adapter.transform(val_raw)

    train_dataset = MaskedWindowDataset(train_tensor, window_length, args.window_stride, split)
    val_dataset = MaskedWindowDataset(val_tensor, window_length, args.window_stride, split)

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if is_main:
        LOGGER.info("Device: %s | world_size: %s", device, world_size)
        LOGGER.info("Target dimension: %s", target_dim)
        LOGGER.info("Feature columns: %s", columns)
        LOGGER.info("Train windows: %s", len(train_dataset))
        LOGGER.info("Val windows: %s", len(val_dataset))

    model = TSDiffuser_Generic(config, device=device, target_dim=target_dim, ratio=args.mask_ratio).to(device)
    load_state_dict(model, checkpoint)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    preprocessing = adapter.metadata()
    preprocessing["columns"] = columns
    preprocessing["dropped_columns"] = parse_column_list(args.drop_cols)

    best_val = float("inf")
    metrics: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_loss = validate(model, val_loader, device)
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if is_main:
            LOGGER.info("Epoch %s/%s | train loss %.6f | val loss %.6f", epoch, args.epochs, train_loss, val_loss)

        if is_main and val_loss < best_val:
            best_val = val_loss
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            save_checkpoint(
                output_dir / "best_finetuned_model.pth",
                raw_model,
                optimizer,
                config,
                args,
                epoch,
                train_loss,
                val_loss,
                preprocessing,
            )
            LOGGER.info("Saved new best checkpoint to %s", output_dir / "best_finetuned_model.pth")

    if is_main:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        save_checkpoint(
            output_dir / "final_finetuned_model.pth",
            raw_model,
            optimizer,
            config,
            args,
            args.epochs,
            metrics[-1]["train_loss"],
            metrics[-1]["val_loss"],
            preprocessing,
        )
        # Per-epoch log for human inspection.
        with (output_dir / "epoch_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        # Scalar summary consumed by TAO AutoML runner metric extraction.
        best_val_loss = min(m["val_loss"] for m in metrics) if metrics else float("inf")
        with (output_dir / "metrics.json").open("w") as f:
            json.dump({"val_loss": best_val_loss}, f)
        with (output_dir / "finetune_config.yaml").open("w") as f:
            yaml.safe_dump(config, f)
        LOGGER.info("Fine-tuning complete. Best val loss %.6f", best_val)
        LOGGER.info("Artifacts written to %s", output_dir)

    cleanup_distributed()


def main() -> None:
    args = parse_args()

    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = args.num_gpus if args.num_gpus is not None else max(available, 1)
    world_size = max(1, world_size)

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()
