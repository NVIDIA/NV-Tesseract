# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable feature preprocessing shared by AD fine-tuning and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


class FeatureAdapter:
    """Fit and replay dimensionality adaptation plus min-max scaling."""

    METADATA_VERSION = 1

    def __init__(self, target_dim: int, scale_factor: float, seed: int) -> None:
        if target_dim <= 0:
            raise ValueError("target_dim must be positive.")
        if not np.isfinite(scale_factor):
            raise ValueError("scale_factor must be finite.")
        self.target_dim = target_dim
        self.scale_factor = scale_factor
        self.seed = seed
        self.pca: PCA | None = None
        self.pca_components_: np.ndarray | None = None
        self.pca_mean_: np.ndarray | None = None
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.input_dim: int | None = None
        self.dtype_: np.dtype | None = None

    @property
    def uses_pca(self) -> bool:
        return self.pca is not None or self.pca_components_ is not None

    def fit(self, data: np.ndarray) -> None:
        data = self._validate_array(data)
        self.input_dim = int(data.shape[1])
        if self.input_dim > self.target_dim:
            if data.shape[0] < self.target_dim:
                raise ValueError(
                    f"PCA needs at least target_dim rows; got {data.shape[0]} rows for target_dim={self.target_dim}."
                )
            self.pca = PCA(n_components=self.target_dim, random_state=self.seed)
            data = self.pca.fit_transform(data)
            self.pca_components_ = np.asarray(self.pca.components_)
            self.pca_mean_ = np.asarray(self.pca.mean_)
        elif self.input_dim < self.target_dim:
            data = self._pad(data)

        self.dtype_ = data.dtype
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)

    def transform(self, data: np.ndarray) -> torch.Tensor:
        if self.input_dim is None or self.dtype_ is None or self.min_ is None or self.max_ is None:
            raise RuntimeError("FeatureAdapter.fit must be called before transform.")

        data = self._validate_array(data).astype(self.dtype_, copy=False)
        if data.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {data.shape[1]}.")

        if self.pca is not None:
            data = self.pca.transform(data)
        elif self.pca_components_ is not None and self.pca_mean_ is not None:
            data = (data - self.pca_mean_) @ self.pca_components_.T
        elif self.input_dim < self.target_dim:
            data = self._pad(data)

        denom = np.where((self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        data = (data - self.min_) / denom
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.tensor(data, dtype=torch.float32) * self.scale_factor

    def metadata(self) -> dict[str, Any]:
        if self.input_dim is None or self.dtype_ is None or self.min_ is None or self.max_ is None:
            raise RuntimeError("FeatureAdapter.fit must be called before metadata serialization.")

        return {
            "version": self.METADATA_VERSION,
            "target_dim": self.target_dim,
            "input_dim": self.input_dim,
            "dtype": self.dtype_.name,
            "scale_factor": self.scale_factor,
            "seed": self.seed,
            "uses_pca": self.uses_pca,
            "min": self.min_.tolist(),
            "max": self.max_.tolist(),
            "pca_components": self.pca_components_.tolist() if self.pca_components_ is not None else None,
            "pca_mean": self.pca_mean_.tolist() if self.pca_mean_ is not None else None,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> FeatureAdapter:
        version = metadata.get("version")
        if version is not None and version != cls.METADATA_VERSION:
            raise ValueError(f"Unsupported checkpoint preprocessing metadata version: {version!r}")
        required = ("target_dim", "input_dim", "scale_factor", "min", "max", "uses_pca")
        missing = [key for key in required if metadata.get(key) is None]
        if missing:
            raise ValueError(f"Checkpoint preprocessing metadata is missing: {', '.join(missing)}")

        adapter = cls(
            target_dim=int(metadata["target_dim"]),
            scale_factor=float(metadata["scale_factor"]),
            seed=int(metadata.get("seed", 42)),
        )
        adapter.input_dim = int(metadata["input_dim"])
        if adapter.input_dim <= 0:
            raise ValueError("Checkpoint preprocessing input_dim must be positive.")
        try:
            # Legacy checkpoints came from load_numeric_frame(), which always emits float32.
            adapter.dtype_ = np.dtype(metadata.get("dtype", "float32"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint preprocessing dtype is invalid: {metadata.get('dtype')!r}") from exc
        if adapter.dtype_.kind != "f":
            raise ValueError(f"Checkpoint preprocessing dtype must be floating point, got {adapter.dtype_.name}.")
        adapter.min_ = np.asarray(metadata["min"], dtype=adapter.dtype_)
        adapter.max_ = np.asarray(metadata["max"], dtype=adapter.dtype_)

        if adapter.min_.shape != (adapter.target_dim,) or adapter.max_.shape != (adapter.target_dim,):
            raise ValueError("Checkpoint preprocessing min/max shapes do not match target_dim.")
        if not np.all(np.isfinite(adapter.min_)) or not np.all(np.isfinite(adapter.max_)):
            raise ValueError("Checkpoint preprocessing min/max values must be finite.")

        uses_pca = metadata["uses_pca"]
        if not isinstance(uses_pca, (bool, np.bool_)):
            raise ValueError("Checkpoint preprocessing uses_pca must be boolean.")
        if uses_pca:
            components = metadata.get("pca_components")
            mean = metadata.get("pca_mean")
            if components is None or mean is None:
                raise ValueError(
                    "This checkpoint used PCA but does not contain the fitted PCA state. "
                    "Re-run fine-tuning with the current FeatureAdapter format."
                )
            adapter.pca_components_ = np.asarray(components, dtype=adapter.dtype_)
            adapter.pca_mean_ = np.asarray(mean, dtype=adapter.dtype_)
            if adapter.pca_components_.shape != (adapter.target_dim, adapter.input_dim):
                raise ValueError("Checkpoint PCA component shape does not match target_dim and input_dim.")
            if adapter.pca_mean_.shape != (adapter.input_dim,):
                raise ValueError("Checkpoint PCA mean shape does not match input_dim.")
            if not np.all(np.isfinite(adapter.pca_components_)) or not np.all(np.isfinite(adapter.pca_mean_)):
                raise ValueError("Checkpoint PCA state must be finite.")
        elif adapter.input_dim > adapter.target_dim:
            raise ValueError("Checkpoint preprocessing cannot reduce input_dim without fitted PCA state.")

        return adapter

    def _pad(self, data: np.ndarray) -> np.ndarray:
        pad_width = self.target_dim - data.shape[1]
        return np.pad(data, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    @staticmethod
    def _validate_array(data: np.ndarray) -> np.ndarray:
        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D feature array, got shape {array.shape}.")
        if array.shape[1] == 0:
            raise ValueError("Feature array has no columns.")
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float64)
        return array


def transform_dataframe_from_metadata(data: pd.DataFrame, metadata: dict[str, Any]) -> torch.Tensor:
    """Apply a fine-tuned checkpoint's fitted preprocessing to an inference frame."""
    adapter = FeatureAdapter.from_metadata(metadata)
    columns = metadata.get("columns")
    if columns is not None:
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            raise ValueError("Checkpoint preprocessing columns must be a list of strings.")
    if columns:
        if len(columns) != adapter.input_dim:
            raise ValueError("Checkpoint preprocessing column count does not match input_dim.")
        if len(set(columns)) != len(columns):
            raise ValueError("Checkpoint preprocessing columns must be unique.")
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"Inference data is missing fine-tuning feature columns: {missing}")
        feature_frame = data.loc[:, columns].copy()
    else:
        feature_frame = data.select_dtypes(include=[np.number]).copy()

    if feature_frame.empty:
        raise ValueError("Inference data has no numeric feature columns.")
    non_numeric = [
        column for column in feature_frame.columns if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if non_numeric:
        raise ValueError(f"Fine-tuning feature columns must be numeric: {non_numeric}")

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return adapter.transform(feature_frame.to_numpy(dtype=adapter.dtype_))
