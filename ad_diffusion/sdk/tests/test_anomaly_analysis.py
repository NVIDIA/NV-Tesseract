# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk import anomaly_analysis


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [2.5, 3.5, 4.5, 5.5, 6.5],
            "feature_3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


@pytest.fixture
def inference_results() -> dict[str, np.ndarray]:
    return {
        "residual": np.array([0.1, 0.5, 0.2, 0.8, 0.3]),
        "residual_l2": np.array([0.2, 0.6, 0.3, 0.9, 0.4]),
        "target": np.array(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
                [1.4, 2.4],
            ]
        ),
        "recon": np.array(
            [
                [0.9, 1.9],
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
            ]
        ),
        "target_dim": np.array(2),
    }


@pytest.fixture(autouse=True)
def patch_model_paths(monkeypatch):
    monkeypatch.setattr(
        anomaly_analysis, "_resolve_model_paths", lambda model_path, config_path: ("model.pth", "config.yaml")
    )
    monkeypatch.setattr(anomaly_analysis, "get_model_target_dim", lambda model_path, config_path: 2)


def test_perform_anomaly_analysis_with_scs_strategy(monkeypatch, numeric_df, inference_results):
    mock_inference = Mock(return_value=inference_results)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", mock_inference)

    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([False, True, False, True, False])
    mock_strategy = Mock()
    mock_strategy.scs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", Mock(return_value=mock_strategy))

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        numeric_df,
        threshold_strategy="scs",
        model_path="model.pth",
        config_path="config.yaml",
        nsample=7,
    )

    assert list(result.columns) == ["feature_1", "feature_2", "feature_3", "Anomaly", "MAE"]
    assert result["Anomaly"].tolist() == [False, True, False, True, False]
    assert np.array_equal(result["MAE"].to_numpy(), inference_results["residual"])
    _, kwargs = mock_inference.call_args
    assert kwargs["model_path"] == "model.pth"
    assert kwargs["config_path"] == "config.yaml"
    assert kwargs["nsample"] == 7
    mock_thresholder.detect_anomalies.assert_called_once_with(
        inference_results["residual"], inference_results["target"]
    )


def test_perform_anomaly_analysis_rejects_non_numeric_columns(numeric_df):
    mixed_df = numeric_df.copy()
    mixed_df["machine_id"] = ["a", "b", "c", "d", "e"]

    with pytest.raises(ValueError, match="contain non-numeric values"):
        anomaly_analysis.perform_anomaly_analysis_with_diffusion(
            mixed_df,
            threshold_strategy="scs",
            model_path="model.pth",
            config_path="config.yaml",
        )


def test_perform_anomaly_analysis_rejects_insufficient_rows(monkeypatch):
    monkeypatch.setattr(anomaly_analysis, "get_model_target_dim", lambda model_path, config_path: 10)
    short_df = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Insufficient data samples"):
        anomaly_analysis.perform_anomaly_analysis_with_diffusion(
            short_df,
            threshold_strategy="scs",
            model_path="model.pth",
            config_path="config.yaml",
        )


def test_perform_anomaly_analysis_rejects_unknown_threshold(monkeypatch, numeric_df, inference_results):
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", Mock(return_value=inference_results))

    with pytest.raises(ValueError, match="Unknown threshold strategy"):
        anomaly_analysis.perform_anomaly_analysis_with_diffusion(
            numeric_df,
            threshold_strategy="not-a-strategy",
            model_path="model.pth",
            config_path="config.yaml",
        )


def test_perform_anomaly_analysis_truncates_long_model_outputs(monkeypatch, numeric_df, inference_results):
    long_results = dict(inference_results)
    long_results["residual"] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    long_results["target"] = np.array([[1.0, 2.0]] * 7)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", Mock(return_value=long_results))

    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([True] * 7)
    mock_strategy = Mock()
    mock_strategy.macs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "MACSThresholdStrategy", Mock(return_value=mock_strategy))

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        numeric_df,
        threshold_strategy="macs",
        model_path="model.pth",
        config_path="config.yaml",
    )

    assert len(result) == len(numeric_df)
    assert result["MAE"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_args, _ = mock_thresholder.detect_anomalies.call_args
    assert len(threshold_args[0]) == len(numeric_df)
    assert len(threshold_args[1]) == len(numeric_df)
