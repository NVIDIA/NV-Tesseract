# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples import finetune_example
from sdk import inference_ad
from sdk.feature_adapter import FeatureAdapter, transform_dataframe_from_metadata


def test_checkpoint_transform_preserves_training_scale_on_shifted_serving_batch() -> None:
    train = np.array(
        [[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]],
        dtype=np.float32,
    )
    serving = pd.DataFrame(
        {
            "is_anomaly": [0, 1, 0],
            "sensor_b": [200.0, 210.0, 220.0],
            "sensor_a": [100.0, 105.0, 110.0],
        }
    )

    adapter = FeatureAdapter(target_dim=2, scale_factor=20.0, seed=42)
    adapter.fit(train)
    metadata = adapter.metadata()
    metadata["columns"] = ["sensor_a", "sensor_b"]

    expected = adapter.transform(serving[["sensor_a", "sensor_b"]].to_numpy(dtype=np.float32))
    replayed = transform_dataframe_from_metadata(serving, metadata)
    batch_refit = inference_ad.preprocess_dataframe(
        serving[["sensor_a", "sensor_b"]],
        target_dim=2,
        scale_factor=20.0,
    )

    assert torch.equal(replayed, expected)
    assert torch.max(torch.abs(batch_refit - expected)).item() == 200.0


def test_checkpoint_transform_exactly_replays_pca_with_reordered_and_extra_columns() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(64, 5)).astype(np.float32)
    serving_array = rng.normal(loc=3.0, size=(8, 5)).astype(np.float32)
    columns = [f"sensor_{index}" for index in range(5)]

    adapter = FeatureAdapter(target_dim=3, scale_factor=20.0, seed=42)
    adapter.fit(train)
    metadata = adapter.metadata()
    metadata["columns"] = columns

    serving = pd.DataFrame(serving_array, columns=columns)
    serving.insert(0, "numeric_label", np.ones(len(serving)))
    serving = serving[["numeric_label", *reversed(columns)]]

    expected = adapter.transform(serving[columns].to_numpy(dtype=np.float32))
    replayed = transform_dataframe_from_metadata(serving, metadata)

    assert torch.allclose(replayed, expected, rtol=1e-5, atol=1e-5)
    assert replayed.shape == (len(serving), 3)


def test_metadata_replay_matches_fitted_adapter_across_dimension_modes() -> None:
    for input_dim, target_dim in ((2, 4), (4, 4), (7, 3)):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            train = rng.normal(size=(64, input_dim)).astype(np.float32)
            serving_array = rng.normal(loc=3.0, size=(17, input_dim)).astype(np.float32)
            columns = [f"sensor_{index}" for index in range(input_dim)]

            adapter = FeatureAdapter(target_dim=target_dim, scale_factor=20.0, seed=seed)
            adapter.fit(train)
            metadata = adapter.metadata()
            metadata["columns"] = columns

            expected = adapter.transform(serving_array)
            replayed = transform_dataframe_from_metadata(pd.DataFrame(serving_array, columns=columns), metadata)

            assert torch.allclose(replayed, expected, rtol=1e-5, atol=1e-5)


def test_legacy_non_pca_metadata_remains_replayable() -> None:
    train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    adapter = FeatureAdapter(target_dim=4, scale_factor=20.0, seed=42)
    adapter.fit(train)
    metadata = adapter.metadata()
    for key in ("version", "dtype", "seed", "pca_components", "pca_mean"):
        metadata.pop(key)
    metadata["columns"] = ["sensor_a", "sensor_b"]

    serving = pd.DataFrame({"sensor_a": [1.5, 2.5], "sensor_b": [15.0, 25.0]})

    assert torch.equal(
        transform_dataframe_from_metadata(serving, metadata),
        adapter.transform(serving.to_numpy(dtype=np.float32)),
    )


def test_legacy_pca_metadata_fails_instead_of_silently_refitting() -> None:
    train = np.arange(48, dtype=np.float32).reshape(12, 4)
    adapter = FeatureAdapter(target_dim=2, scale_factor=20.0, seed=42)
    adapter.fit(train)
    metadata = adapter.metadata()
    metadata["pca_components"] = None
    metadata["pca_mean"] = None

    with pytest.raises(ValueError, match="used PCA.*Re-run fine-tuning"):
        FeatureAdapter.from_metadata(metadata)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 99, "Unsupported checkpoint preprocessing metadata version"),
        ("uses_pca", "false", "uses_pca must be boolean"),
        ("min", [np.nan, 0.0], "min/max values must be finite"),
    ],
)
def test_invalid_checkpoint_metadata_fails_closed(field, value, message) -> None:
    adapter = FeatureAdapter(target_dim=2, scale_factor=20.0, seed=42)
    adapter.fit(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    metadata = adapter.metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match=message):
        FeatureAdapter.from_metadata(metadata)


def test_checkpoint_metadata_survives_serialization_and_rejects_missing_features(tmp_path) -> None:
    train = np.array([[0.0, 10.0], [5.0, 15.0], [10.0, 20.0]], dtype=np.float32)
    adapter = FeatureAdapter(target_dim=2, scale_factor=20.0, seed=42)
    adapter.fit(train)
    metadata = adapter.metadata()
    metadata["columns"] = ["sensor_a", "sensor_b"]

    checkpoint_path = tmp_path / "fine-tuned.pth"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    finetune_example.save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config={"model": {"target_dim": 2}},
        args=SimpleNamespace(seed=42),
        epoch=1,
        train_loss=0.2,
        val_loss=0.1,
        preprocessing=metadata,
    )
    restored_metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["preprocessing"]

    serving = pd.DataFrame({"sensor_a": [2.5], "sensor_b": [12.5]})
    assert torch.equal(
        transform_dataframe_from_metadata(serving, restored_metadata),
        adapter.transform(serving.to_numpy(dtype=np.float32)),
    )

    with pytest.raises(ValueError, match="missing fine-tuning feature columns.*sensor_b"):
        transform_dataframe_from_metadata(serving.drop(columns="sensor_b"), restored_metadata)


def test_explicit_external_preprocessor_overrides_checkpoint_metadata() -> None:
    adapter = FeatureAdapter(target_dim=2, scale_factor=20.0, seed=42)
    adapter.fit(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    checkpoint = {"preprocessing": adapter.metadata()}

    assert (
        inference_ad._apply_checkpoint_preprocessing(
            pd.DataFrame({"sensor_a": [1.0], "sensor_b": [2.0]}),
            checkpoint,
            config={"dataset": {}},
            target_dim=2,
            preprocess_model_dir="external/preprocessor",
        )
        is None
    )


def test_public_inference_automatically_uses_checkpoint_transform(monkeypatch) -> None:
    train = np.array([[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]], dtype=np.float32)
    serving = pd.DataFrame(
        {
            "sensor_b": [200.0, 210.0, 220.0],
            "sensor_a": [100.0, 105.0, 110.0],
            "numeric_label": [0, 1, 0],
        }
    )
    adapter = FeatureAdapter(target_dim=2, scale_factor=20.0, seed=42)
    adapter.fit(train)
    preprocessing = adapter.metadata()
    preprocessing.update({"columns": ["sensor_a", "sensor_b"], "window_length": 2, "split": 2})
    checkpoint = {
        "model": {},
        "config": {"model": {"target_dim": 2}, "dataset": {"scale_factor": 999.0}},
        "preprocessing": preprocessing,
    }

    class DummyModel:
        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    captured: dict[str, object] = {}

    def fake_evaluate(_model, loader1, _loader2, **_kwargs):
        captured["data"] = loader1.dataset.data.clone()
        captured["window_length"] = loader1.dataset.window_length
        captured["split"] = loader1.dataset.split
        return {"residual": np.zeros(len(serving))}

    monkeypatch.setattr(inference_ad, "_resolve_model_paths", lambda *_args: ("fine-tuned.pth", ""))
    monkeypatch.setattr(inference_ad.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(inference_ad, "TSDiffuser_Generic", lambda *_args, **_kwargs: DummyModel())
    monkeypatch.setattr(inference_ad, "evaluate_ad_tesseract2", fake_evaluate)
    monkeypatch.setattr(
        inference_ad,
        "get_dataloader",
        lambda *_args, **_kwargs: pytest.fail("generic batch-fitted preprocessing should not be used"),
    )

    result = inference_ad.inference_ad_tesseract2(serving, model_path="fine-tuned.pth")

    expected = adapter.transform(serving[["sensor_a", "sensor_b"]].to_numpy(dtype=np.float32))
    assert torch.equal(captured["data"], expected)
    assert captured["window_length"] == 2
    assert captured["split"] == 2
    assert result["target_dim"] == 2


def test_public_inference_without_metadata_keeps_generic_preprocessing(monkeypatch) -> None:
    checkpoint = {
        "model": {},
        "config": {"model": {"target_dim": 2}, "dataset": {"scale_factor": 7.0}},
    }

    class DummyModel:
        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    captured: dict[str, object] = {}

    def fake_get_dataloader(*args: object, **kwargs: object):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object(), object()

    monkeypatch.setattr(inference_ad, "_resolve_model_paths", lambda *_args: ("pretrained.pth", ""))
    monkeypatch.setattr(inference_ad.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(inference_ad, "TSDiffuser_Generic", lambda *_args, **_kwargs: DummyModel())
    monkeypatch.setattr(inference_ad, "get_dataloader", fake_get_dataloader)
    monkeypatch.setattr(inference_ad, "evaluate_ad_tesseract2", lambda *_args, **_kwargs: {})

    data = pd.DataFrame({"sensor_a": [1.0], "sensor_b": [2.0]})
    result = inference_ad.inference_ad_tesseract2(data, model_path="pretrained.pth")

    assert captured["args"] == (data, 2)
    assert captured["kwargs"] == {"scale_factor": 7.0, "model_dir": None}
    assert result["target_dim"] == 2
