# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from datasetsforecast.long_horizon import LongHorizon
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]


def _is_ydf(df: Any) -> bool:
    if not isinstance(df, pd.DataFrame):
        return False
    cols = set(map(str.lower, df.columns))
    return {"unique_id", "ds", "y"}.issubset(cols)


class NixtlaLongHorizonDataset(Dataset):
    """
    Backbone-ready windows from Nixtla Long Horizon.

    Policy:
      - Train uses Nixtla TRAIN split if available, else prefix (temporal).
      - Val uses the SAME segment as Test.
      - Test uses Nixtla TEST split if available, else temporal tail.

    Returns per item:
      timeseries: FloatTensor [C, L]
      forecast:   FloatTensor [C, H]
      input_mask: BoolTensor  [L]
    """

    def __init__(
        self,
        directory: str = "./data",
        group: str = "ETTh1",
        data_split: Split = "train",  # 'train' | 'val' | 'test'
        seq_len: int = 512,
        forecast_horizon: int = 96,
        random_seed: int = 13,
        fillna: str | None = "ffill",  # 'ffill'|'bfill'|'zero'|None
        test_frac: float = 0.20,  # used only when we must fabricate splits
    ) -> None:
        super().__init__()
        assert data_split in {"train", "val", "test"}
        self.L = int(seq_len)
        self.H = int(forecast_horizon)

        # --- 1) Load (supports both signatures) ---
        a, b, c = LongHorizon.load(directory=directory, group=group)
        # Case A: (Y_train, Y_val, Y_test) like the blog shows
        if _is_ydf(a) and (b is None or _is_ydf(b)) and (c is None or _is_ydf(c)) and (b is not None or c is not None):
            Y_train, Y_val, Y_test = a, b, c
            # Build wide slices from the chosen split.
            if data_split == "train":
                use_df = Y_train
            else:
                # val == test policy
                use_df = Y_test if Y_test is not None and len(Y_test) else Y_val
                if use_df is None or use_df.empty:
                    raise RuntimeError("No test/val data provided by LongHorizon; cannot honor val==test policy.")

            use_df = use_df.copy()
            use_df["ds"] = pd.to_datetime(use_df["ds"])
            wide = (
                use_df.sort_values(["ds", "unique_id"]).pivot(index="ds", columns="unique_id", values="y").sort_index()
            )
        else:
            # Case B: (Y, X, S) — fabricate train/test; val==test
            Y_df = a
            if not _is_ydf(Y_df):
                raise RuntimeError("Unexpected LongHorizon.load() return format.")
            Y_df = Y_df.copy()
            Y_df["ds"] = pd.to_datetime(Y_df["ds"])
            wide_all = (
                Y_df.sort_values(["ds", "unique_id"]).pivot(index="ds", columns="unique_id", values="y").sort_index()
            )

            T = wide_all.shape[0]
            n_test = int(math.floor(test_frac * T))
            if n_test <= 0 or n_test >= T:
                raise RuntimeError(f"Invalid test_frac={test_frac} for T={T} -> n_test={n_test}")

            train_range = (0, T - n_test)
            eval_range = (T - n_test, T)  # val == test

            start, end = train_range if data_split == "train" else eval_range
            wide = wide_all.iloc[start:end]

        # --- 2) Fill NA if requested ---
        if fillna == "ffill":
            wide = wide.ffill().bfill()
        elif fillna == "bfill":
            wide = wide.bfill().ffill()
        elif fillna == "zero":
            wide = wide.fillna(0.0)

        self.columns = list(wide.columns)
        self.series = wide.to_numpy(dtype=np.float32)  # [T_split, C]
        self.C = self.series.shape[1]

        # --- 3) Build valid window centers t (context [t-L, t), forecast [t, t+H)) ---
        self.idx = []
        t_min = self.L
        t_max = self.series.shape[0] - self.H
        if t_max > t_min:
            self.idx = list(range(t_min, t_max))

        # Shuffle only train windows
        if data_split == "train" and self.idx:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(self.idx)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.idx[i]
        x = self.series[t - self.L : t, :]  # [L, C]
        y = self.series[t : t + self.H, :]  # [H, C]
        x = torch.from_numpy(x.T.copy())  # -> [C, L]
        y = torch.from_numpy(y.T.copy())  # -> [C, H]
        input_mask = torch.ones(self.L, dtype=torch.bool)
        return x, y, input_mask


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean


class CSVLongHorizonDataset:
    """
    Drop-in replacement for NixtlaLongHorizonDataset,
    but loads from a custom CSV file.

    Arguments:
        csv_path: path to your CSV with 'timestamp' + numeric columns
        data_split: 'train', 'val', or 'test'
        seq_len: length of history window
        forecast_horizon: length of forecast horizon
        val_ratio: fraction of data for validation
        test_ratio: fraction of data for test
        random_seed: random seed for reproducibility
        standardize: whether to z-normalize based on train split
        stride: sliding window stride (defaults to forecast_horizon)
    """

    def __init__(
        self,
        csv_path: str,
        data_split: str,
        seq_len: int,
        forecast_horizon: int,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int = 13,
        standardize: bool = True,
        stride: int | None = None,
        timestamp_col: str = "timestamp",
        usecols: List[str] | None = None,
    ):
        assert data_split in ["train", "val", "test"], "Invalid split"
        self.split = data_split
        self.seq_len = int(seq_len)
        self.h = int(forecast_horizon)
        self.stride = stride or self.h

        # load series
        df = pd.read_csv(csv_path)
        if timestamp_col not in df.columns:
            raise ValueError(f"timestamp column '{timestamp_col}' not found")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(timestamp_col)
        self.raw_timestamps = df[timestamp_col].values

        if usecols is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            num_cols = [c for c in usecols if c in df.columns and c != timestamp_col]
        if len(num_cols) == 0:
            raise ValueError("No numeric columns found in dataset")

        df = df[[timestamp_col] + num_cols].dropna()
        self.values = df[num_cols].to_numpy(dtype=np.float32)  # [T, C]
        self.times = pd.DatetimeIndex(df[timestamp_col].values)
        self.channels = num_cols

        # determine splits
        T = self.values.shape[0]
        t_train = int(round(T * (1.0 - val_ratio - test_ratio)))
        t_val = int(round(T * val_ratio))
        t_test = T - t_train - t_val

        if t_train < (seq_len + forecast_horizon):
            raise ValueError("Training split too short for given seq_len+horizon")

        # compute stats from training only
        if standardize:
            mean = self.values[:t_train].mean(axis=0)
            std = self.values[:t_train].std(axis=0)
            std[std < 1e-8] = 1.0
            self.standardizer = Standardizer(mean=mean, std=std)
            self.series = self.standardizer.transform(self.values)
        else:
            self.standardizer = None
            self.series = self.values.copy()

        # assign split ranges
        if data_split == "train":
            self.start, self.stop = 0, t_train
        elif data_split == "val":
            self.start, self.stop = t_train, t_train + t_val
        else:
            self.start, self.stop = t_train + t_val, T

        # build indices
        self._starts = []
        for s in range(self.start, self.stop - (self.seq_len + self.h) + 1, self.stride):
            self._starts.append(s)

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, idx):
        s = self._starts[idx]
        e_hist = s + self.seq_len
        e_fore = e_hist + self.h
        hist = self.series[s:e_hist]  # [L, C]
        fut = self.series[e_hist:e_fore]  # [H, C]
        hist = np.swapaxes(hist, 0, 1).copy()
        fut = np.swapaxes(fut, 0, 1).copy()
        mask = np.ones((self.seq_len,), dtype=np.float32)
        return hist, fut, mask


class CSVLongHorizonSimpleDataset:
    """
    A dataset loader when train/val/test splits are provided as separate CSV files.

    Arguments:
        csv_path: path to the CSV file for the given split
        data_split: 'train', 'val', or 'test'
        seq_len: history window length
        forecast_horizon: forecast horizon length
        standardizer: Standardizer object (fitted on train split), or None
        standardize: whether to apply standardization
        stride: sliding window stride (default = forecast_horizon)
        timestamp_col: name of timestamp column
        usecols: optional subset of numeric columns
    """

    def __init__(
        self,
        csv_path: str,
        data_split: str,
        seq_len: int,
        forecast_horizon: int,
        standardizer: Standardizer | None = None,
        standardize: bool = True,
        stride: int | None = None,
        timestamp_col: str = "timestamp",
        usecols: List[str] | None = None,
    ):
        assert data_split in ["train", "val", "test"], "Invalid split"
        self.split = data_split
        self.seq_len = int(seq_len)
        self.h = int(forecast_horizon)
        self.stride = stride or self.h

        # load series
        df = pd.read_csv(csv_path)
        if timestamp_col not in df.columns:
            raise ValueError(f"timestamp column '{timestamp_col}' not found")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(timestamp_col)

        if usecols is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            num_cols = [c for c in usecols if c in df.columns and c != timestamp_col]
        if len(num_cols) == 0:
            raise ValueError("No numeric columns found in dataset")

        df = df[[timestamp_col] + num_cols].dropna()
        self.values = df[num_cols].to_numpy(dtype=np.float32)  # [T, C]
        self.times = pd.DatetimeIndex(df[timestamp_col].values)
        self.channels = num_cols

        # train split: compute stats
        if standardizer is None and standardize and data_split == "train":
            mean = self.values.mean(axis=0)
            std = self.values.std(axis=0)
            std[std < 1e-8] = 1.0
            self.standardizer = Standardizer(mean=mean, std=std)
        else:
            self.standardizer = standardizer

        # apply transform if required
        if self.standardizer is not None and standardize:
            self.series = self.standardizer.transform(self.values)
        else:
            self.series = self.values.copy()

        # build indices
        T = self.series.shape[0]
        if (self.seq_len + self.h) > T:
            raise ValueError("Series too short for given seq_len+horizon")
        self._starts = []
        for s in range(0, T - (self.seq_len + self.h) + 1, self.stride):
            self._starts.append(s)

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, idx):
        s = self._starts[idx]
        e_hist = s + self.seq_len
        e_fore = e_hist + self.h
        hist = self.series[s:e_hist]  # [L, C]
        fut = self.series[e_hist:e_fore]  # [H, C]
        hist = np.swapaxes(hist, 0, 1).copy()
        fut = np.swapaxes(fut, 0, 1).copy()
        mask = np.ones((self.seq_len,), dtype=np.float32)
        return hist, fut, mask

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.standardizer is None:
            return x
        return self.standardizer.inverse(x)
