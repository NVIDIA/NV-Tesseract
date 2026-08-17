# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JSON serialization of sklearn preprocessing models."""

import tempfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from utils.adaptive_normalizer import AdaptiveNormalizer
from utils.json_utils import (
    _deserialize_sklearn_model,
    _serialize_sklearn_model,
    deserialize_adaptive_normalizer,
    load_preprocessing_models,
    save_preprocessing_models,
    serialize_adaptive_normalizer,
)

RNG = np.random.default_rng(42)
X = RNG.normal(loc=50, scale=10, size=(200, 5))
X_TEST = RNG.normal(loc=50, scale=10, size=(50, 5))


def test_robust_scaler() -> None:
    scaler = RobustScaler().fit(X)
    restored = _deserialize_sklearn_model(_serialize_sklearn_model(scaler))

    assert np.allclose(scaler.transform(X_TEST), restored.transform(X_TEST))


def test_standard_scaler() -> None:
    scaler = StandardScaler().fit(X)
    restored = _deserialize_sklearn_model(_serialize_sklearn_model(scaler))

    assert np.allclose(scaler.transform(X_TEST), restored.transform(X_TEST))


def test_quantile_transformer() -> None:
    transformer = QuantileTransformer(n_quantiles=100, output_distribution="normal").fit(X)
    restored = _deserialize_sklearn_model(_serialize_sklearn_model(transformer))

    assert np.allclose(transformer.transform(X_TEST), restored.transform(X_TEST))


def test_power_transformer() -> None:
    transformer = PowerTransformer(method="yeo-johnson").fit(X)
    restored = _deserialize_sklearn_model(_serialize_sklearn_model(transformer))

    assert np.allclose(transformer.transform(X_TEST), restored.transform(X_TEST))


def test_pca() -> None:
    pca = PCA(n_components=3).fit(X)
    restored = _deserialize_sklearn_model(_serialize_sklearn_model(pca))

    assert np.allclose(pca.transform(X_TEST), restored.transform(X_TEST))


def test_adaptive_normalizer() -> None:
    normalizer = AdaptiveNormalizer(target_range=100)
    normalizer.fit(X[:, 0])
    restored = deserialize_adaptive_normalizer(
        serialize_adaptive_normalizer(normalizer),
        AdaptiveNormalizer,
    )

    assert np.allclose(normalizer.transform(X_TEST[:, 0]), restored.transform(X_TEST[:, 0]))


def test_save_load_preprocessing_models() -> None:
    normalizer = AdaptiveNormalizer(target_range=100)
    normalizer.fit(X[:, 0])
    pca = PCA(n_components=3).fit(X)
    models = {
        "normalizers": {"test_domain": {"features_5": normalizer}},
        "pca_models": {"test_domain_features_5": pca},
        "target_dim": 25,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "preprocessing_models.json"
        save_preprocessing_models(models, path)
        loaded = load_preprocessing_models(path, normalizer_class=AdaptiveNormalizer)

    loaded_normalizer = loaded["normalizers"]["test_domain"]["features_5"]
    assert np.allclose(normalizer.transform(X_TEST[:, 0]), loaded_normalizer.transform(X_TEST[:, 0]))
    assert np.allclose(pca.transform(X_TEST), loaded["pca_models"]["test_domain_features_5"].transform(X_TEST))
    assert loaded["target_dim"] == 25
