# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sdk import forecasting


class DummyModel:
    def __init__(self, model_horizon: int):
        self.model_horizon = max(1, int(model_horizon))

    def eval(self):
        return self

    def embed(self, x_enc: torch.Tensor, input_mask: torch.Tensor):
        batch_size = x_enc.size(0)
        embeddings = torch.ones((batch_size, 4), dtype=torch.float32, device=x_enc.device)
        return SimpleNamespace(embeddings=embeddings)

    def __call__(self, x_enc: torch.Tensor, input_mask: torch.Tensor):
        batch_size, channels, _ = x_enc.shape
        forecast = torch.ones((batch_size, channels, self.model_horizon), dtype=torch.float32)
        forecast = forecast.to(x_enc.device)
        return SimpleNamespace(forecast=forecast)

    def load_state_dict(self, state: dict, strict: bool = False):
        return


@pytest.fixture(autouse=True)
def patch_external_dependencies(monkeypatch):
    build_calls = []

    def build_model(*, forecast_horizon, **kwargs):
        build_calls.append({"forecast_horizon": forecast_horizon, **kwargs})
        return DummyModel(model_horizon=forecast_horizon)

    monkeypatch.setattr(forecasting, "build_model", build_model)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        forecasting.joblib,
        "load",
        lambda path: {
            "mean": np.array([0.0], dtype=np.float32),
            "std": np.array([1.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(forecasting, "control_randomness", lambda seed=0: None)
    forecasting.clear_model_cache()
    yield build_calls
    forecasting.clear_model_cache()


def make_timeseries(num_rows: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=num_rows, freq="H")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "target": np.arange(num_rows, dtype=np.float32),
            "feature": np.linspace(0.0, 1.0, num_rows, dtype=np.float32),
        }
    )


def make_timeseries_with_columns(num_rows: int = 10, columns: list[str] | None = None) -> pd.DataFrame:
    """Create a timeseries DataFrame with specified columns."""
    if columns is None:
        columns = ["feature1", "feature2"]

    timestamps = pd.date_range("2024-01-01", periods=num_rows, freq="H")
    data = {
        "timestamp": timestamps,
        "target": np.arange(num_rows, dtype=np.float32),
    }

    # Add the specified feature columns
    for i, col in enumerate(columns):
        data[col] = np.linspace(i, i + 1.0, num_rows, dtype=np.float32)

    return pd.DataFrame(data)


def test_perform_forecasting_generates_forecast_when_horizon_exceeds_model_horizon():
    df = make_timeseries(num_rows=10)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=2,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert "timestamp" in result.columns
    assert "target_forecast" in result.columns
    assert len(result) == 3
    assert result["target_forecast"].tolist() == [1.0, 1.0, 1.0]


def test_perform_forecasting_uses_cross_channel_model_by_default(patch_external_dependencies):
    df = make_timeseries(num_rows=10)
    forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=2,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
    )

    assert patch_external_dependencies
    build_kwargs = patch_external_dependencies[0]
    assert build_kwargs["use_cross_channel"] is True
    assert build_kwargs["cross_channel_heads"] == 8
    assert build_kwargs["cross_channel_dropout"] == 0.1


def test_perform_forecasting_allows_cross_channel_configuration_override(patch_external_dependencies):
    df = make_timeseries(num_rows=10)
    forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=2,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        use_cross_channel=False,
        cross_channel_heads=4,
        cross_channel_dropout=0.25,
    )

    build_kwargs = patch_external_dependencies[0]
    assert build_kwargs["use_cross_channel"] is False
    assert build_kwargs["cross_channel_heads"] == 4
    assert build_kwargs["cross_channel_dropout"] == 0.25


def test_perform_forecasting_requires_timestamp_column():
    df = make_timeseries(num_rows=6).drop(columns=["timestamp"])
    with pytest.raises(ValueError, match="Timestamp column 'timestamp' not found"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=2,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_requires_numeric_target():
    df = make_timeseries(num_rows=6)
    df["target"] = df["target"].astype(str)
    with pytest.raises(ValueError, match="Target column 'target' must contain numeric values"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=2,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_validates_forecast_horizon():
    df = make_timeseries(num_rows=6)
    with pytest.raises(ValueError, match="forecast_horizon must be positive"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=0,
            model_horizon=4,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_rejects_forecast_horizon_above_max():
    """forecast_horizon above 512 should raise ValueError."""
    df = make_timeseries(num_rows=520)
    with pytest.raises(ValueError, match="forecast_horizon must be <= 512"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=513,
            model_horizon=72,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_accepts_max_forecast_horizon():
    """forecast_horizon of exactly 512 should be accepted."""
    df = make_timeseries(num_rows=520)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=512,
        model_horizon=72,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
    )

    assert len(result) == 512
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_rejects_forecast_horizon_above_max():
    """DARR mode also enforces the 512 max forecast_horizon limit."""
    df = make_timeseries(num_rows=520)
    context_df = make_timeseries(num_rows=520)
    with pytest.raises(ValueError, match="forecast_horizon must be <= 512"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=600,
            model_horizon=72,
            context_df=context_df,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_validates_model_horizon():
    df = make_timeseries(num_rows=6)
    with pytest.raises(ValueError, match="model_horizon must be positive"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=0,
            model_horizon=0,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_requires_minimum_rows():
    df = make_timeseries(num_rows=4)
    with pytest.raises(ValueError, match="DataFrame has 4 rows but seq_len requires at least 5 rows"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=2,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_context_df_missing_timestamp():
    df = make_timeseries(num_rows=7)
    context_df = make_timeseries(num_rows=7).drop(columns=["timestamp"])
    with pytest.raises(ValueError, match="Context DataFrame missing timestamp column 'timestamp'"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=2,
            context_df=context_df,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_darr_mode_returns_hybrid_forecast():
    """DARR mode returns hybrid forecast combining direct and kNN predictions."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=10)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert "timestamp" in result.columns
    assert "target_forecast" in result.columns
    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_with_custom_alpha():
    """DARR mode respects alpha parameter for blending direct and kNN predictions."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=10)

    # Test with alpha=0.5 (50% direct, 50% kNN)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        alpha=0.5,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_with_alpha_zero():
    """DARR mode with alpha=0 uses only kNN predictions."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=10)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        alpha=0.0,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_with_alpha_one():
    """DARR mode with alpha=1 uses only direct predictions."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=10)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        alpha=1.0,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_forecast_horizon_smaller_than_model_horizon():
    """DARR mode works when forecast_horizon < model_horizon (truncates output)."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=10)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,  # Smaller than model_horizon
        model_horizon=4,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_forecast_horizon_equals_model_horizon():
    """DARR mode works when forecast_horizon == model_horizon (no autoregression needed)."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=10)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=3,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 3
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_forecast_horizon_larger_than_model_horizon():
    """DARR mode works with autoregressive forecasting when forecast_horizon > model_horizon."""
    df = make_timeseries(num_rows=12)
    context_df = make_timeseries(num_rows=12)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=6,  # Larger than model_horizon
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 6
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_retrieves_full_requested_horizon():
    """DARR retrieval keeps the historical continuation beyond the model's native horizon."""
    df = make_timeseries(num_rows=12)
    context_df = make_timeseries(num_rows=12)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=6,
        model_horizon=2,
        context_df=context_df,
        alpha=0.0,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    # There is exactly one six-step context continuation: target values 5..10.
    # The old implementation retrieved only 5..6, then repeated 6 for steps 3..6.
    np.testing.assert_allclose(result["target_forecast"], np.arange(5, 11, dtype=np.float32))


def test_perform_forecasting_darr_mode_blends_full_retrieval_horizon():
    """The hybrid output uses the complete retrieval trajectory for long forecasts."""
    df = make_timeseries(num_rows=12)
    context_df = make_timeseries(num_rows=12)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=6,
        model_horizon=2,
        context_df=context_df,
        alpha=0.25,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    # DummyModel forecasts ones, while the retrieved continuation is 5..10.
    expected = 0.25 + 0.75 * np.arange(5, 11, dtype=np.float32)
    np.testing.assert_allclose(result["target_forecast"], expected)


def test_perform_forecasting_darr_mode_forecast_horizon_much_larger_than_model_horizon():
    """DARR mode works with many autoregressive iterations."""
    df = make_timeseries(num_rows=15)
    context_df = make_timeseries(num_rows=15)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=10,  # 5x model_horizon, requires 5 iterations
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 10
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_forecast_horizon_not_multiple_of_model_horizon():
    """DARR mode works when forecast_horizon is not a clean multiple of model_horizon."""
    df = make_timeseries(num_rows=12)
    context_df = make_timeseries(num_rows=12)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=7,  # Not a multiple of model_horizon (3)
        model_horizon=3,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 7
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_requires_target_column():
    df = make_timeseries(num_rows=6).drop(columns=["target"])
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=2,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_null_timestamps_rejected():
    df = make_timeseries(num_rows=6)
    df.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="Timestamp column 'timestamp' contains NULL values"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=2,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_allows_custom_timestamp_column():
    df = make_timeseries(num_rows=10).rename(columns={"timestamp": "ts"})
    result = forecasting.perform_forecasting(
        df,
        timestamp_column="ts",
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
    )

    assert "ts" in result.columns
    assert "timestamp" not in result.columns


def test_perform_forecasting_fills_null_targets():
    df = make_timeseries(num_rows=7)
    df.loc[0, "target"] = np.nan
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_context_df_missing_target():
    df = make_timeseries(num_rows=7)
    context_df = make_timeseries(num_rows=7).drop(columns=["target"])
    with pytest.raises(ValueError, match="Context DataFrame missing target column 'target'"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=2,
            context_df=context_df,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
        )


def test_perform_forecasting_darr_mode_with_custom_k():
    """DARR mode respects k parameter for kNN retrieval."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=10)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        k=16,  # Custom k value
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_with_custom_temperature():
    """DARR mode respects temperature parameter for kNN softmax weights."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=10)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        temperature=0.1,  # Custom temperature
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_with_larger_context():
    """DARR mode works with context_df larger than input df."""
    df = make_timeseries(num_rows=10)
    context_df = make_timeseries(num_rows=50)

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_custom_timestamp_column():
    """DARR mode works with custom timestamp column name."""
    df = make_timeseries(num_rows=10).rename(columns={"timestamp": "ts"})
    context_df = make_timeseries(num_rows=10).rename(columns={"timestamp": "ts"})

    result = forecasting.perform_forecasting(
        df,
        timestamp_column="ts",
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert "ts" in result.columns
    assert "timestamp" not in result.columns
    assert len(result) == 2


def test_perform_forecasting_darr_mode_custom_target_column():
    """DARR mode works with custom target column name."""
    df = make_timeseries(num_rows=10).rename(columns={"target": "value"})
    context_df = make_timeseries(num_rows=10).rename(columns={"target": "value"})

    result = forecasting.perform_forecasting(
        df,
        target_column="value",
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert "value_forecast" in result.columns
    assert len(result) == 2


def test_perform_forecasting_darr_mode_context_df_insufficient_rows():
    """DARR mode raises when context_df cannot provide its required continuation."""
    df = make_timeseries(num_rows=10)
    # context_df with 6 rows, but seq_len=5 + model_horizon=3 = 8 rows needed
    context_df = make_timeseries(num_rows=6)

    with pytest.raises(ValueError, match="Context DataFrame has .* rows but requires at least .* rows"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=3,
            context_df=context_df,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
            seed=42,
        )


def test_perform_forecasting_darr_mode_requires_full_retrieval_horizon():
    """Long DARR forecasts require context through the requested retrieval horizon."""
    df = make_timeseries(num_rows=12)
    context_df = make_timeseries(num_rows=10)

    with pytest.raises(ValueError, match=r"seq_len=5 \+ context_horizon=6"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=6,
            model_horizon=2,
            context_df=context_df,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
            seed=42,
        )


def test_perform_forecasting_darr_mode_column_mismatch_input_more_columns():
    """DARR mode handles case where input_df has more columns than context_df."""
    # Input dataset with 4 feature columns
    df = make_timeseries_with_columns(num_rows=10, columns=["feat1", "feat2", "feat3", "feat4"])
    # Context dataset with only 2 feature columns (subset of input)
    context_df = make_timeseries_with_columns(num_rows=10, columns=["feat1", "feat2"])

    # Should succeed by using only common columns
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_column_mismatch_context_more_columns():
    """DARR mode handles case where context_df has more columns than input_df."""
    # Input dataset with 2 feature columns
    df = make_timeseries_with_columns(num_rows=10, columns=["feat1", "feat2"])
    # Context dataset with 4 feature columns (superset of input)
    context_df = make_timeseries_with_columns(num_rows=10, columns=["feat1", "feat2", "feat3", "feat4"])

    # Should succeed by using only common columns
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_column_mismatch_partial_overlap():
    """DARR mode handles case where datasets have partial column overlap."""
    # Input dataset with columns [feat1, feat2, feat3]
    df = make_timeseries_with_columns(num_rows=10, columns=["feat1", "feat2", "feat3"])
    # Context dataset with columns [feat2, feat3, feat4] - partial overlap
    context_df = make_timeseries_with_columns(num_rows=10, columns=["feat2", "feat3", "feat4"])

    # Should succeed by using only common columns (feat2, feat3)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_column_mismatch_no_common_columns():
    """DARR mode raises error when datasets have no common feature columns."""
    # Input dataset with columns [feat1, feat2]
    df = make_timeseries_with_columns(num_rows=10, columns=["feat1", "feat2"])
    # Context dataset with completely different columns [feat3, feat4]
    context_df = make_timeseries_with_columns(num_rows=10, columns=["feat3", "feat4"])

    # Should raise error due to no common columns
    with pytest.raises(ValueError, match="No common numeric columns found between input and context datasets"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=2,
            model_horizon=2,
            context_df=context_df,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
            seed=42,
        )


def test_perform_forecasting_darr_mode_column_mismatch_many_vs_few():
    """DARR mode handles extreme case of many columns vs few columns."""
    # Input dataset with many columns (7 columns like ETTh.csv)
    df = make_timeseries_with_columns(num_rows=10, columns=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "OT", "extra"])
    # Context dataset with only 3 columns (like ETTh_short.csv)
    context_df = make_timeseries_with_columns(num_rows=10, columns=["HUFL", "HULL", "MUFL"])

    # Should succeed by using only common columns
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 2
    assert result["target_forecast"].notna().all()


def test_perform_forecasting_darr_mode_column_alignment_with_autoregression():
    """DARR mode column alignment works with autoregressive forecasting."""
    # Input dataset with more columns
    df = make_timeseries_with_columns(num_rows=12, columns=["feat1", "feat2", "feat3", "feat4"])
    # Context dataset with fewer columns
    context_df = make_timeseries_with_columns(num_rows=12, columns=["feat1", "feat2"])

    # Test with autoregressive forecasting (forecast_horizon > model_horizon)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=6,  # Larger than model_horizon
        model_horizon=2,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
    )

    assert len(result) == 6
    assert result["target_forecast"].notna().all()
