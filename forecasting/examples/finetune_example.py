#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fine-tune the NV-Tesseract forecasting head on a user CSV.

Run from the forecasting directory:
    uv run python examples/finetune_example.py \
        --csv /path/to/timeseries.csv \
        --timestamp-col timestamp \
        --target-cols target \
        --forecast-horizon 72 \
        --seq-len 512 \
        --epochs 5 \
        --output-dir artifacts/finetune_my_data

The CSV must contain a timestamp column and one or more numeric time-series columns.
By default the script trains the forecasting head while keeping the pretrained
encoder and embedder frozen.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_longhorizon import CSVLongHorizonDataset, CSVLongHorizonSimpleDataset
from model import build_model, count_trainable_params
from sdk.forecasting import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_CROSS_CHANNEL_CHECKPOINT_NAME,
    download_model_weights,
)

LOGGER = logging.getLogger("forecasting_finetune")


def parse_column_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
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


def build_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
    target_cols = parse_column_list(args.target_cols)

    if args.csv is not None:
        train_dataset = CSVLongHorizonDataset(
            csv_path=args.csv,
            data_split="train",
            seq_len=args.seq_len,
            forecast_horizon=args.forecast_horizon,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
            standardize=not args.no_standardize,
            stride=args.stride,
            timestamp_col=args.timestamp_col,
            usecols=target_cols,
        )
        val_dataset = CSVLongHorizonDataset(
            csv_path=args.csv,
            data_split="val",
            seq_len=args.seq_len,
            forecast_horizon=args.forecast_horizon,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
            standardize=not args.no_standardize,
            stride=args.stride,
            timestamp_col=args.timestamp_col,
            usecols=target_cols,
        )
        return train_dataset, val_dataset

    if args.train_csv is None or args.val_csv is None:
        raise ValueError("Pass either --csv or both --train-csv and --val-csv.")

    train_dataset = CSVLongHorizonSimpleDataset(
        csv_path=args.train_csv,
        data_split="train",
        seq_len=args.seq_len,
        forecast_horizon=args.forecast_horizon,
        standardizer=None,
        standardize=not args.no_standardize,
        stride=args.stride,
        timestamp_col=args.timestamp_col,
        usecols=target_cols,
    )
    val_dataset = CSVLongHorizonSimpleDataset(
        csv_path=args.val_csv,
        data_split="val",
        seq_len=args.seq_len,
        forecast_horizon=args.forecast_horizon,
        standardizer=train_dataset.standardizer,
        standardize=not args.no_standardize,
        stride=args.stride,
        timestamp_col=args.timestamp_col,
        usecols=target_cols,
    )
    return train_dataset, val_dataset


def resolve_checkpoint_init(args: argparse.Namespace) -> str | None:
    if args.ckpt_init is None or args.ckpt_init.lower() in {"none", "false", "0"}:
        return None
    if args.ckpt_init.lower() != "auto":
        return args.ckpt_init

    ckpt_name = DEFAULT_CROSS_CHANNEL_CHECKPOINT_NAME if args.use_cross_channel else DEFAULT_CHECKPOINT_NAME
    _, ckpt_path = download_model_weights(
        standardizer_pkl=args.standardizer_init,
        ckpt=ckpt_name,
        repo_id=args.repo_id,
    )
    return ckpt_path


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append(key)

    load_result = model.load_state_dict(compatible_state, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if skipped:
        LOGGER.info("Skipped incompatible checkpoint keys: %s", skipped)
    if unexpected:
        LOGGER.warning("Unexpected checkpoint keys: %s", unexpected)
    if missing:
        LOGGER.info("Missing checkpoint keys initialized from the base model: %s", missing)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    scaler: torch.cuda.amp.GradScaler,
    criterion: torch.nn.Module,
    device: torch.device,
    max_norm: float,
) -> float:
    model.train()
    losses: list[float] = []
    use_amp = device.type == "cuda"

    for timeseries, forecast, input_mask in tqdm(loader, desc="train", leave=False):
        timeseries = timeseries.to(device, dtype=torch.float32)
        forecast = forecast.to(device, dtype=torch.float32)
        input_mask = input_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(x_enc=timeseries, input_mask=input_mask)
            loss = criterion(output.forecast, forecast)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    maes: list[float] = []
    use_amp = device.type == "cuda"

    for timeseries, forecast, input_mask in tqdm(loader, desc="val", leave=False):
        timeseries = timeseries.to(device, dtype=torch.float32)
        forecast = forecast.to(device, dtype=torch.float32)
        input_mask = input_mask.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(x_enc=timeseries, input_mask=input_mask)
            loss = criterion(output.forecast, forecast)

        losses.append(float(loss.detach().cpu()))
        maes.append(float(torch.mean(torch.abs(output.forecast - forecast)).detach().cpu()))

    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(maes)) if maes else float("nan"),
    )


def save_artifacts(
    model: torch.nn.Module,
    output_dir: Path,
    standardizer: Any,
    args: argparse.Namespace,
    channels: list[str],
    metrics: list[dict[str, float]],
    best_epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    if standardizer is not None:
        joblib.dump({"mean": standardizer.mean, "std": standardizer.std}, output_dir / "standardizer.pkl")

    metadata = {
        "best_epoch": best_epoch,
        "seq_len": args.seq_len,
        "forecast_horizon": args.forecast_horizon,
        "model_name": args.model_name,
        "channels": channels,
        "use_cross_channel": args.use_cross_channel,
        "cross_channel_heads": args.cross_channel_heads,
        "cross_channel_dropout": args.cross_channel_dropout,
        "args": vars(args),
        "metrics": metrics,
    }
    with (output_dir / "finetune_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune NV-Tesseract forecasting on a CSV dataset.")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--csv", type=str, help="Single CSV to split temporally into train/val/test.")
    data_group.add_argument("--train-csv", type=str, help="Training CSV. Requires --val-csv.")
    parser.add_argument("--val-csv", type=str, help="Validation CSV when --train-csv is used.")
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument(
        "--target-cols",
        type=str,
        default=None,
        help="Comma-separated numeric columns. Defaults to all numeric columns.",
    )

    parser.add_argument("--model-name", type=str, default="AutonLab/MOMENT-1-large")
    parser.add_argument(
        "--ckpt-init",
        type=str,
        default="auto",
        help="Checkpoint to warm-start from. Use 'auto' for published NV-Tesseract weights or 'none' for a fresh head.",
    )
    parser.add_argument("--standardizer-init", type=str, default="standardizer.pkl")
    parser.add_argument("--repo-id", type=str, default="nvidia/nv-tesseract-forecasting")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--forecast-horizon", type=int, default=72)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--no-standardize", action="store_true")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--max-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=str, default="artifacts/finetune")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--unfreeze-embedder", action="store_true")
    parser.add_argument("--use-cross-channel", action="store_true")
    parser.add_argument("--cross-channel-heads", type=int, default=8)
    parser.add_argument("--cross-channel-dropout", type=float, default=0.1)

    parser.add_argument("--run-config", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def _load_run_config(path: str) -> dict:
    import yaml

    text = Path(path).read_text()
    return yaml.safe_load(text) or {}


def _run_config_defaults(cfg: dict) -> list[str]:
    """Flatten a nested YAML run-config into CLI-style --key value pairs."""
    _BOOL_FLAGS = {
        "no_standardize",
        "local_files_only",
        "unfreeze_encoder",
        "unfreeze_embedder",
        "use_cross_channel",
    }
    mapping = {
        "dataset": {
            "csv": "--csv",
            "train_csv": "--train-csv",
            "val_csv": "--val-csv",
            "timestamp_col": "--timestamp-col",
            "target_cols": "--target-cols",
            "val_ratio": "--val-ratio",
            "test_ratio": "--test-ratio",
            "stride": "--stride",
            "no_standardize": "--no-standardize",
        },
        "model": {
            "seq_len": "--seq-len",
            "forecast_horizon": "--forecast-horizon",
            "ckpt_init": "--ckpt-init",
            "use_cross_channel": "--use-cross-channel",
            "cross_channel_heads": "--cross-channel-heads",
            "cross_channel_dropout": "--cross-channel-dropout",
            "unfreeze_encoder": "--unfreeze-encoder",
            "unfreeze_embedder": "--unfreeze-embedder",
        },
        "train": {
            "epochs": "--epochs",
            "batch_size": "--batch-size",
            "lr": "--lr",
            "weight_decay": "--weight-decay",
            "head_dropout": "--head-dropout",
            "max_norm": "--max-norm",
            "seed": "--seed",
            "output_dir": "--output-dir",
            "local_files_only": "--local-files-only",
        },
    }
    extras: list[str] = []
    for section, keys in mapping.items():
        section_cfg = cfg.get(section, {}) or {}
        for key, flag in keys.items():
            val = section_cfg.get(key)
            if val is None or val == "":
                continue
            if key in _BOOL_FLAGS:
                if val:
                    extras.append(flag)
            else:
                extras.extend([flag, str(val)])
    return extras


def parse_args() -> argparse.Namespace:
    parser = _build_parser()
    # First pass: detect --run-config without requiring --csv/--train-csv yet.
    pre, _ = parser.parse_known_args()
    defaults: list[str] = []
    if pre.run_config:
        cfg = _load_run_config(pre.run_config)
        defaults = _run_config_defaults(cfg)
    # Second pass: config defaults first, then CLI overrides.
    args = parser.parse_args(defaults + sys.argv[1:])
    if args.csv is None and args.train_csv is None:
        parser.error("one of --csv or --train-csv is required (or supply both via --run-config)")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)

    train_dataset, val_dataset = build_datasets(args)
    if len(train_dataset) == 0:
        raise ValueError("No training windows were created. Reduce --seq-len/--forecast-horizon or use more data.")
    if len(val_dataset) == 0:
        raise ValueError("No validation windows were created. Reduce --seq-len/--forecast-horizon or use more data.")

    LOGGER.info("Device: %s", device)
    LOGGER.info("Train windows: %s", len(train_dataset))
    LOGGER.info("Val windows: %s", len(val_dataset))
    LOGGER.info("Channels: %s", train_dataset.channels)

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

    model = build_model(
        model_name=args.model_name,
        forecast_horizon=args.forecast_horizon,
        seq_len=args.seq_len,
        head_dropout=args.head_dropout,
        weight_decay=args.weight_decay,
        freeze_encoder=not args.unfreeze_encoder,
        freeze_embedder=not args.unfreeze_embedder,
        freeze_head=False,
        use_cross_channel=args.use_cross_channel,
        cross_channel_heads=args.cross_channel_heads,
        cross_channel_dropout=args.cross_channel_dropout,
        local_files_only=args.local_files_only,
        device=str(device),
    )
    ckpt_init = resolve_checkpoint_init(args)
    if ckpt_init:
        LOGGER.info("Warm-starting from checkpoint: %s", ckpt_init)
        load_checkpoint(model, ckpt_init, device)

    LOGGER.info("Trainable parameters: %s", f"{count_trainable_params(model):,}")

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, len(train_loader) * args.epochs),
        pct_start=0.3,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    metrics: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device, args.max_norm)
        val_mse, val_mae = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "val_mae": val_mae}
        metrics.append(row)
        LOGGER.info(
            "Epoch %s/%s | train MSE %.6f | val MSE %.6f | val MAE %.6f",
            epoch,
            args.epochs,
            train_mse,
            val_mse,
            val_mae,
        )

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            save_artifacts(
                model=model,
                output_dir=output_dir,
                standardizer=getattr(train_dataset, "standardizer", None),
                args=args,
                channels=train_dataset.channels,
                metrics=metrics,
                best_epoch=best_epoch,
            )
            LOGGER.info("Saved new best checkpoint to %s", output_dir / "best_model.pt")

    LOGGER.info("Fine-tuning complete. Best val MSE %.6f at epoch %s", best_val, best_epoch)
    # Per-epoch log for human inspection.
    with (output_dir / "epoch_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    # Scalar summary consumed by TAO AutoML runner metric extraction.
    best_row = min(metrics, key=lambda r: r["val_mse"]) if metrics else {}
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(
            {"val_mse": best_row.get("val_mse", float("inf")), "val_mae": best_row.get("val_mae", float("inf"))}, f
        )
    LOGGER.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
