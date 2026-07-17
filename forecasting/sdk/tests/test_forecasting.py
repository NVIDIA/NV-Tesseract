# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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


class ChannelIndexedModel(DummyModel):
    """DummyModel variant whose forecast value equals the channel index.

    Makes channel/column mix-ups observable: every output column must carry
    the value of the channel it claims to represent.
    """

    def __call__(self, x_enc: torch.Tensor, input_mask: torch.Tensor):
        batch_size, channels, _ = x_enc.shape
        forecast = torch.arange(channels, dtype=torch.float32).reshape(1, channels, 1)
        forecast = forecast.repeat(batch_size, 1, self.model_horizon).to(x_enc.device)
        return SimpleNamespace(forecast=forecast)


class LastValueModel(DummyModel):
    """DummyModel variant that repeats each channel's last observed value.

    Makes the direct-prediction path sensitive to the channel layout of the
    (rewritten) input CSV: each forecast channel reproduces the constant of
    the input channel it was built from.
    """

    def __call__(self, x_enc: torch.Tensor, input_mask: torch.Tensor):
        forecast = x_enc[:, :, -1:].repeat(1, 1, self.model_horizon)
        return SimpleNamespace(forecast=forecast)


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


def make_constant_feature_df(num_rows: int, feature_values: dict[str, float]) -> pd.DataFrame:
    """Create a timeseries DataFrame whose feature columns are constants.

    With uniform stub embeddings, the kNN retrieval forecast is the mean of the
    context continuations, so each forecast channel reproduces the constant of
    the context channel it was built from — which makes channel misalignment
    observable as a wrong constant in the output. Column order follows the
    insertion order of ``feature_values``.
    """
    timestamps = pd.date_range("2024-01-01", periods=num_rows, freq="h")
    data = {
        "timestamp": timestamps,
        "target": np.zeros(num_rows, dtype=np.float32),
    }
    for name, value in feature_values.items():
        data[name] = np.full(num_rows, value, dtype=np.float32)
    return pd.DataFrame(data)


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


def test_perform_forecasting_uses_distinct_temp_csvs_for_concurrent_calls(monkeypatch):
    """Same-process calls must not select the same temporary input path."""

    class StopAfterTempCsvError(RuntimeError):
        pass

    paths = []
    paths_lock = threading.Lock()
    both_calls_reached_csv = threading.Barrier(2)

    def capture_temp_csv(self, path, *args: object, **kwargs: object):
        with paths_lock:
            paths.append(str(path))
        both_calls_reached_csv.wait(timeout=5)
        raise StopAfterTempCsvError

    def invoke(_):
        with pytest.raises(StopAfterTempCsvError):
            forecasting.perform_forecasting(
                make_timeseries(num_rows=10),
                seq_len=5,
                forecast_horizon=2,
                model_horizon=2,
                standardizer_pkl="fake_std.pkl",
                ckpt="fake_ckpt.pt",
            )

    monkeypatch.setattr(pd.DataFrame, "to_csv", capture_temp_csv)
    monkeypatch.setattr(
        forecasting,
        "download_model_weights",
        lambda standardizer_pkl, ckpt: (standardizer_pkl, ckpt),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(invoke, range(2)))

    assert len(paths) == 2
    assert len(set(paths)) == 2


@pytest.mark.parametrize("with_context, expected_temp_files", [(False, 1), (True, 2)])
def test_perform_forecasting_uses_and_cleans_owned_temp_csvs(monkeypatch, tmp_path, with_context, expected_temp_files):
    """Standard and DARR calls clean up every unique CSV allocated for them."""
    created_paths = []

    def create_temp_csv_path(prefix: str) -> str:
        path = tmp_path / f"{prefix}{len(created_paths)}.csv"
        path.touch()
        created_paths.append(path)
        return str(path)

    monkeypatch.setattr(forecasting, "_create_temp_csv_path", create_temp_csv_path)

    result = forecasting.perform_forecasting(
        make_timeseries(num_rows=10),
        context_df=make_timeseries(num_rows=10) if with_context else None,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        num_workers=0,
    )

    assert len(result) == 2
    assert len(created_paths) == expected_temp_files
    assert all(not path.exists() for path in created_paths)


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


def test_perform_forecasting_return_all_channels_emits_per_channel_forecasts(monkeypatch):
    monkeypatch.setattr(
        forecasting,
        "build_model",
        lambda *, forecast_horizon, **kwargs: ChannelIndexedModel(model_horizon=forecast_horizon),
    )
    forecasting.clear_model_cache()

    df = make_timeseries(num_rows=10)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=3,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
        return_all_channels=True,
    )

    # Channel order matches columns_to_process: target column first, then the
    # remaining numeric features in input-column order.
    assert list(result.columns) == ["timestamp", "target_forecast", "feature_forecast"]
    assert len(result) == 3
    # The stub model encodes the channel index in the forecast value, so each
    # column must carry its own channel's value (a mix-up fails here).
    assert result["target_forecast"].tolist() == [0.0, 0.0, 0.0]
    assert result["feature_forecast"].tolist() == [1.0, 1.0, 1.0]


def test_perform_forecasting_return_all_channels_with_autoregressive_horizon():
    df = make_timeseries(num_rows=10)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=6,
        model_horizon=2,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
        return_all_channels=True,
    )

    assert list(result.columns) == ["timestamp", "target_forecast", "feature_forecast"]
    assert len(result) == 6
    assert result["target_forecast"].notna().all()
    assert result["feature_forecast"].notna().all()


def test_perform_forecasting_return_all_channels_darr_mode():
    df = make_timeseries(num_rows=12)
    context_df = make_timeseries(num_rows=12)
    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=3,
        context_df=context_df,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
        return_all_channels=True,
    )

    assert list(result.columns) == ["timestamp", "target_forecast", "feature_forecast"]
    assert len(result) == 3
    assert result["target_forecast"].notna().all()
    assert result["feature_forecast"].notna().all()


def test_perform_forecasting_return_all_channels_rejects_interpretability():
    df = make_timeseries(num_rows=10)
    with pytest.raises(ValueError, match="return_all_channels"):
        forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=3,
            model_horizon=3,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
            return_all_channels=True,
            interpretability=True,
        )


def test_interpretability_sdk_defaults_match_algorithms():
    from integrated_gradients import integrated_gradients_embedding

    sdk_params = inspect.signature(forecasting.perform_forecasting).parameters
    explain_params = inspect.signature(forecasting.explain_forecast).parameters
    ig_params = inspect.signature(integrated_gradients_embedding).parameters

    assert sdk_params["n_lags"].default == explain_params["n_lags"].default
    assert sdk_params["softmax_tau"].default == explain_params["softmax_tau"].default
    assert sdk_params["channel_output_aware"].default == explain_params["channel_output_aware"].default
    assert sdk_params["integrated_gradients_baseline"].default == ig_params["baseline"].default
    assert sdk_params["integrated_gradients_steps"].default == ig_params["n_steps"].default
    assert sdk_params["integrated_gradients_n_baselines"].default == ig_params["n_baselines"].default
    assert sdk_params["integrated_gradients_internal_batch_size"].default == ig_params["internal_batch_size"].default
    assert sdk_params["integrated_gradients_reduce"].default == ig_params["reduce"].default


def test_perform_forecasting_forwards_output_aware_mode(monkeypatch, tmp_path):
    captured = {}

    def run_interpretability(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-02", periods=2, freq="h"),
                "target_forecast": np.ones(2, dtype=np.float32),
            }
        ), tmp_path

    monkeypatch.setattr(forecasting, "_run_interpretability", run_interpretability)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "target": np.arange(10, dtype=np.float32),
            "feature": np.linspace(0.0, 1.0, 10, dtype=np.float32),
        }
    )
    forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=2,
        model_horizon=2,
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        interpretability=True,
        channel_output_aware=True,
    )

    assert captured["channel_output_aware"] is True


def test_run_interpretability_integrated_gradients_reaches_report(monkeypatch, tmp_path):
    import integrated_gradients as ig

    explanation = SimpleNamespace(
        baseline_forecast=np.ones((2, 2), dtype=np.float32),
        lag_horizon_scores=np.ones((3, 2), dtype=np.float32),
        lag_horizon_attributions=np.ones((3, 2), dtype=np.float32) / 3.0,
        flow_magnitudes=None,
        latent_trajectory=None,
        surrogate_coef=None,
        surrogate_intercept=None,
        surrogate_feature_layout=None,
        flow_ratio_forecast_vs_history=None,
        flow_variance_ratio_forecast_vs_history=None,
        curvature_ratio_forecast_vs_history=None,
        latent_diag_mahalanobis_ratio_forecast_vs_history=None,
        per_channel_flow=None,
        lag_channel_horizon_attributions=None,
        channel_horizon_attributions=None,
        channel_coupling_matrix=None,
        channel_flow_method=None,
        channel_coupling_off_diag_norm=None,
    )
    captured = {}

    def moment_embed_fn(model, *, input_mask=None, grad_through_norm=False):
        captured["input_mask"] = input_mask.detach().cpu().tolist()
        captured["grad_through_norm"] = grad_through_norm
        return lambda x: x.reshape(x.shape[0], -1).sum(dim=1, keepdim=True)

    def integrated_gradients_embedding(embed_fn, x, **kwargs):
        captured["ig_shape"] = tuple(x.shape)
        captured["ig_kwargs"] = kwargs
        assert tuple(embed_fn(torch.zeros((2, *x.shape), dtype=torch.float32)).shape) == (2, 1)
        return SimpleNamespace(
            attribution=np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]], dtype=np.float32),
            overall_effect=0.0,
            abs_effect=20.0,
            embedding_delta=0.5,
            embedding_delta_abs_mean=0.5,
            convergence_delta=0.1,
            n_steps=kwargs["n_steps"],
            n_baselines=kwargs["n_baselines"],
            baseline=kwargs["baseline"],
            reduce=kwargs["reduce"],
            embedding_dim=1,
        )

    def build_pdf_report(pdf_path, **kwargs):
        captured["pdf_report"] = kwargs["integrated_gradients_report"]
        captured["channel_labels"] = kwargs["channel_labels"]
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        return pdf_path

    monkeypatch.setattr(forecasting, "explain_forecast", lambda *args, **kwargs: explanation)
    monkeypatch.setattr(forecasting, "compute_trajectory_stability", lambda *args, **kwargs: None)
    monkeypatch.setattr(ig, "moment_embed_fn", moment_embed_fn)
    monkeypatch.setattr(ig, "integrated_gradients_embedding", integrated_gradients_embedding)
    monkeypatch.setattr(forecasting, "_save_lag_horizon_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(forecasting, "_build_pdf_report", build_pdf_report)

    model = SimpleNamespace(eval=lambda: None)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "target": np.arange(6, dtype=np.float32),
            "feature": np.linspace(0.0, 1.0, 6, dtype=np.float32),
        }
    )
    standardizer = forecasting.Standardizer(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )
    _, run_dir = forecasting._run_interpretability(
        model=model,
        standardizer=standardizer,
        working_df=df,
        columns_to_process=["target", "feature"],
        timestamp_column="timestamp",
        target_column="target",
        seq_len=4,
        forecast_horizon=2,
        model_horizon=2,
        device=torch.device("cpu"),
        n_lags=3,
        softmax_tau=1.0,
        interpretability_output=None,
        interpretability_out_dir=tmp_path,
        interpretability_run_name="integrated_gradients",
        interpretability_top_k=2,
        dataset_name="unit",
        integrated_gradients=True,
        integrated_gradients_steps=7,
        integrated_gradients_n_baselines=2,
        integrated_gradients_internal_batch_size=3,
        integrated_gradients_reduce="mean",
        integrated_gradients_seed=123,
        interpretability_channel_axis=False,
    )

    payload = json.loads((run_dir / "explanation.json").read_text(encoding="utf-8"))
    ig_payload = payload["explanation"]["integrated_gradients"]
    assert ig_payload["channel_labels"] == ["target", "feature"]
    assert ig_payload["channel_abs_effect"] == [10.0, 10.0]
    assert captured["input_mask"] == [1, 1, 1, 1]
    assert captured["ig_shape"] == (2, 4)
    assert captured["ig_kwargs"]["internal_batch_size"] == 3
    assert captured["ig_kwargs"]["seed"] == 123
    assert captured["grad_through_norm"] is True
    assert captured["channel_labels"] == ["target", "feature"]
    assert captured["pdf_report"]["attribution"].shape == (2, 4)
    assert (run_dir / "integrated_gradients_attributions.csv").exists()
    assert (run_dir / "integrated_gradients_channel_summary.csv").exists()


def test_perform_forecasting_return_all_channels_rejects_timestamp_collision():
    df = make_timeseries(num_rows=10).rename(columns={"timestamp": "feature_forecast"})
    with pytest.raises(ValueError, match="collides"):
        forecasting.perform_forecasting(
            df,
            timestamp_column="feature_forecast",
            seq_len=5,
            forecast_horizon=3,
            model_horizon=3,
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
            return_all_channels=True,
        )


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


def test_perform_forecasting_darr_mode_same_columns_different_order(caplog):
    """An order-only difference must trigger realignment of the context CSV.

    The target channel is first in both layouts, so this test observes the
    realignment through the warning log; the value-level alignment is covered
    by the return_all_channels tests below.
    """
    df = make_constant_feature_df(12, {"f1": 10.0, "f2": 20.0})
    context_df = make_constant_feature_df(12, {"f2": 20.0, "f1": 10.0})

    with caplog.at_level(logging.WARNING):
        result = forecasting.perform_forecasting(
            df,
            seq_len=5,
            forecast_horizon=3,
            model_horizon=3,
            context_df=context_df,
            alpha=0.0,  # pure kNN: forecasts reproduce the context channel constants
            standardizer_pkl="fake_std.pkl",
            ckpt="fake_ckpt.pt",
            seed=42,
        )

    assert len(result) == 3
    assert result["target_forecast"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    # A set-based comparison would treat the two layouts as identical and skip
    # realignment silently; the order-only warning proves the branch fired.
    assert any("same numeric columns in a different order" in rec.getMessage() for rec in caplog.records)


def test_perform_forecasting_darr_mode_different_order_return_all_channels_aligned():
    """With return_all_channels=True, every output column must carry the value of
    its own channel even when the context lists the same columns in another order."""
    df = make_constant_feature_df(12, {"f1": 10.0, "f2": 20.0})
    context_df = make_constant_feature_df(12, {"f2": 20.0, "f1": 10.0})

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=3,
        context_df=context_df,
        alpha=0.0,  # pure kNN: forecasts reproduce the context channel constants
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
        return_all_channels=True,
    )

    assert list(result.columns) == ["timestamp", "target_forecast", "f1_forecast", "f2_forecast"]
    # A channel mix-up would surface the other channel's constant here.
    assert result["f1_forecast"].tolist() == pytest.approx([10.0, 10.0, 10.0])
    assert result["f2_forecast"].tolist() == pytest.approx([20.0, 20.0, 20.0])


def test_perform_forecasting_darr_mode_column_mismatch_uses_input_order_common_subset():
    """Missing/extra context features fall back to the aligned common subset,
    anchored to the input-DataFrame column order (not sorted order)."""
    # Input feature order [zz, beta, alpha] differs from sorted order; the
    # context is missing `zz` and lists the common features in another order.
    df = make_constant_feature_df(12, {"zz": 30.0, "beta": 20.0, "alpha": 10.0})
    context_df = make_constant_feature_df(12, {"alpha": 10.0, "beta": 20.0})

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=3,
        context_df=context_df,
        alpha=0.0,  # pure kNN: forecasts reproduce the context channel constants
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
        return_all_channels=True,
    )

    # Only the aligned common subset survives, in input-DataFrame order.
    assert list(result.columns) == ["timestamp", "target_forecast", "beta_forecast", "alpha_forecast"]
    assert result["beta_forecast"].tolist() == pytest.approx([20.0, 20.0, 20.0])
    assert result["alpha_forecast"].tolist() == pytest.approx([10.0, 10.0, 10.0])


def test_perform_forecasting_darr_mode_column_mismatch_direct_predictions_aligned(monkeypatch):
    """With alpha=1.0 (pure direct) the forecasts come from the rewritten input
    CSV, so this guards the main-CSV side of the canonical-order rewrite (the
    kNN-side tests above run with alpha=0.0 and cannot observe it)."""
    monkeypatch.setattr(
        forecasting,
        "build_model",
        lambda *, forecast_horizon, **kwargs: LastValueModel(model_horizon=forecast_horizon),
    )
    forecasting.clear_model_cache()

    df = make_constant_feature_df(12, {"zz": 30.0, "beta": 20.0, "alpha": 10.0})
    context_df = make_constant_feature_df(12, {"alpha": 10.0, "beta": 20.0})

    result = forecasting.perform_forecasting(
        df,
        seq_len=5,
        forecast_horizon=3,
        model_horizon=3,
        context_df=context_df,
        alpha=1.0,  # pure direct: forecasts reproduce the input channel constants
        standardizer_pkl="fake_std.pkl",
        ckpt="fake_ckpt.pt",
        seed=42,
        return_all_channels=True,
    )

    assert list(result.columns) == ["timestamp", "target_forecast", "beta_forecast", "alpha_forecast"]
    # If the rewritten input CSV used a different order than the output naming,
    # each column would carry the other channel's constant.
    assert result["beta_forecast"].tolist() == pytest.approx([20.0, 20.0, 20.0])
    assert result["alpha_forecast"].tolist() == pytest.approx([10.0, 10.0, 10.0])
