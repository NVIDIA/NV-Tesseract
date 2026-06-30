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
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.main_model import TSDiffuser_Generic
from sdk.feature_adapter import FeatureAdapter
from sdk.inference_ad import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_MODEL_FILENAME,
    HF_REPO_ID,
    download_model_weights,
    load_config,
)

LOGGER = logging.getLogger("ad_diffusion_finetune")


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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune NV-Tesseract AD Diffusion on a CSV dataset.")
    parser.add_argument("--csv", required=True, help="Training CSV, ideally containing normal behavior.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)

    model_path, config_path = resolve_model_assets(args)
    checkpoint, config = load_checkpoint_and_config(model_path, config_path)
    target_dim = int(config.get("model", {}).get("target_dim", 40))
    dataset_config = config.setdefault("dataset", {})
    window_length = args.window_length or int(dataset_config.get("window_length", 100))
    split = args.split or int(dataset_config.get("split", 4))
    scale_factor = args.scale_factor or float(dataset_config.get("scale_factor", 20))
    dataset_config["window_length"] = window_length
    dataset_config["split"] = split
    dataset_config["scale_factor"] = scale_factor

    train_raw, val_raw, columns = split_arrays(args)
    adapter = FeatureAdapter(target_dim=target_dim, scale_factor=scale_factor, seed=args.seed)
    adapter.fit(train_raw)
    train_tensor = adapter.transform(train_raw)
    val_tensor = adapter.transform(val_raw)

    train_dataset = MaskedWindowDataset(train_tensor, window_length, args.window_stride, split)
    val_dataset = MaskedWindowDataset(val_tensor, window_length, args.window_stride, split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    LOGGER.info("Device: %s", device)
    LOGGER.info("Target dimension: %s", target_dim)
    LOGGER.info("Feature columns: %s", columns)
    LOGGER.info("Train windows: %s", len(train_dataset))
    LOGGER.info("Val windows: %s", len(val_dataset))

    model = TSDiffuser_Generic(config, device=device, target_dim=target_dim, ratio=args.mask_ratio).to(device)
    load_state_dict(model, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    preprocessing = adapter.metadata()
    preprocessing["columns"] = columns
    preprocessing["dropped_columns"] = parse_column_list(args.drop_cols)
    preprocessing["window_length"] = window_length
    preprocessing["window_stride"] = args.window_stride
    preprocessing["split"] = split

    best_val = float("inf")
    metrics: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_loss = validate(model, val_loader, device)
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        LOGGER.info("Epoch %s/%s | train loss %.6f | val loss %.6f", epoch, args.epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output_dir / "best_finetuned_model.pth",
                model,
                optimizer,
                config,
                args,
                epoch,
                train_loss,
                val_loss,
                preprocessing,
            )
            LOGGER.info("Saved new best checkpoint to %s", output_dir / "best_finetuned_model.pth")

    save_checkpoint(
        output_dir / "final_finetuned_model.pth",
        model,
        optimizer,
        config,
        args,
        args.epochs,
        metrics[-1]["train_loss"],
        metrics[-1]["val_loss"],
        preprocessing,
    )
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "finetune_config.yaml").open("w") as f:
        yaml.safe_dump(config, f)

    LOGGER.info("Fine-tuning complete. Best val loss %.6f", best_val)
    LOGGER.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
