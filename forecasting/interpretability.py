# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Model-agnostic interpretability utilities for time series forecasting.

Pipeline structure (latent interface -> semantic flow -> lag x horizon matrix ->
local surrogates), implemented as a stable, model-agnostic approximation:
  - Build a latent trajectory by embedding a rolling window at each time index.
  - Define a scalar "flow magnitude" per step from consecutive latent states
    (delta norm + optional scale term).
  - Aggregate these magnitudes into a lag x horizon influence matrix and
    normalize per horizon via softmax.
  - Fit horizon-specific *weighted sparse linear* surrogates using
    time-series-preserving perturbations.

The result is horizon-resolved, temporally continuous, model-agnostic
explanations without retraining the forecaster.

Stability / quality notes:
- The pipeline assumes temporal smoothness of the embedding: small input
  changes should yield proportionally small latent changes. If not, flow
  magnitudes may reflect representation noise.
- Use compute_embedding_stability() to run an empirical stability test
  (effective Lipschitz ratio under small perturbations). Optionally use
  SemanticFlowConfig.smooth_latent_alpha for temporal smoothing of Z.
- The latent trajectory is built over [history; forecast]. The forecast
  segment is out-of-distribution (model-generated). Flow there can be noisier;
  use flow_ratio_forecast_vs_history and flow_variance_ratio_forecast_vs_history
  on ForecastExplanation to gauge whether attributions that depend on the
  extended segment are likely to be volatile (ratio > 1 suggests more noise).
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import torch
from integrated_gradients import normalization_statistics_gradient

if TYPE_CHECKING:
    from channel_flow import ChannelFlowConfig

Array = np.ndarray


class ForecastModel(Protocol):
    """Minimal protocol for MOMENT-like forecasting models."""

    def __call__(self, *, x_enc: torch.Tensor, input_mask: torch.Tensor) -> Any:
        """Forward pass returning an object with a `.forecast` tensor [B, C, H]."""

    def embed(self, *, x_enc: torch.Tensor, input_mask: torch.Tensor) -> Any:
        """Embedding pass returning an object with `.embeddings` tensor [B, D]."""


@dataclass(frozen=True)
class SemanticFlowConfig:
    """
    Configuration for semantic-flow computation.

    Args:
      flow_kind:
        - 'delta_l2': m_t = ||z_{t+1} - z_t||_2 (latent displacement; no output sensitivity).
        - 'output_aware': m_t = |(∂y/∂Z_t)(Z_{t+1}-Z_t)| (first-order forecast effect).
          Requires dy_dZ in compute_semantic_flow_magnitudes.
      scale_weight:
        Optional extra term based on latent norm changes (delta_l2 only):
          scale_weight * clip(|log(||z_{t+1}|| / ||z_t||)|, 0, 10)
        The log-ratio term is capped at 10 to limit the influence of extreme norm jumps.
      smooth_latent_alpha:
        If in (0, 1), smooth the latent trajectory with EMA before computing flow:
          Z_smooth[t] = alpha * Z_smooth[t-1] + (1 - alpha) * Z[t].
        Can improve robustness when embeddings are noisy (Lipschitz-like regularization).
      output_aware_aggregation:
        When flow_kind is 'output_aware', how to aggregate over horizons if dy_dZ has shape (T, H, D):
        'l2' => m_t = sqrt(sum_h (grad_t_h @ delta_t)^2), 'sum' => m_t = sum_h |grad_t_h @ delta_t|.
      eps:
        Numerical stability.
    """

    flow_kind: Literal["delta_l2", "output_aware"] = "delta_l2"
    scale_weight: float = 0.0
    smooth_latent_alpha: float = 0.0
    output_aware_aggregation: Literal["l2", "sum"] = "l2"
    eps: float = 1e-12


@dataclass(frozen=True)
class EmbeddingStabilityReport:
    """
    Result of empirical embedding stability test.

    Under small input perturbations, we measure ||Z' - Z|| and ||x' - x|| and report the
    effective ratio (Lipschitz-style). Lower ratios indicate more stable embeddings.
    `n_trials` counts perturbation runs that contributed a ratio; `n_unique_windows`
    counts the number of distinct time windows covered by those runs.
    """

    lip_ratio_mean: float
    lip_ratio_max: float
    lip_ratio_p50: float
    lip_ratio_p95: float
    n_trials: int
    n_unique_windows: int
    step_delta_norm_mean: float  # mean ||Z_{t+1} - Z_t|| on unperturbed trajectory (for reference)


@dataclass(frozen=True)
class TrajectoryStabilityReport:
    """
    Summary of temporal smoothness for a latent trajectory.

    The trajectory is centered by a reference latent state and optionally scaled by the
    per-dimension standard deviation of the reference segment. Lower zero-crossing,
    direction-flip, and relative-jitter values indicate smoother latent evolution.
    """

    zero_crossing_rate_mean: float
    zero_crossing_rate_p95: float
    direction_flip_rate_mean: float
    direction_flip_rate_p95: float
    relative_jitter_mean: float
    relative_jitter_p95: float
    occupancy_positive_mean: float
    occupancy_negative_mean: float
    n_time_steps: int
    n_dimensions: int


@dataclass(frozen=True)
class PerturbationConfig:
    """
    Structure-preserving perturbations for local surrogates.

    block-bootstrap: resamples contiguous blocks to preserve local autocorrelation.
    fourier-phase-preserving noise: keeps Fourier phase but perturbs amplitude mildly to preserve seasonality.
    """

    n_samples: int = 128
    block_len: int = 32
    fourier_amp_sigma: float = 0.05
    mix_original: float = 0.25  # 0 = fully perturbed, 1 = fully original
    seed: int = 13


@dataclass(frozen=True)
class SurrogateConfig:
    """
    Horizon-specific local surrogate settings.

    We fit a weighted L1-regularized linear model (ISTA) independently for each horizon h.
    """

    n_lags: int = 64
    l1_alpha: float = 5e-3
    distance_lambda: float = 1.0
    max_iter: int = 300
    tol: float = 1e-5


@dataclass(frozen=True)
class ForecastExplanation:
    """
    Explanation outputs.

    Shapes:
      - baseline_forecast: [C, H]
      - lag_horizon_scores: [K, H] (unnormalized, larger means more influence)
      - lag_horizon_attributions: [K, H] (softmax-normalized over lags for each horizon)
      - surrogate_coef: [C, H, P] (may be None if surrogates disabled; P = C * surrogate_n_lags)
      - surrogate_intercept: [C, H] (may be None if surrogates disabled)

    Segment quality (forecast vs history):
      - flow_ratio_forecast_vs_history: mean flow in forecast segment / mean flow in history.
        >1 means the extended (OOD) segment is more volatile; attributions that weight by
        flow in that region may be noisier.
      - flow_variance_ratio_forecast_vs_history: variance of flow in forecast / variance in history.
      - Set to None if segment split is invalid (e.g. too short).

    Additional diagnostics:
      - curvature_ratio_forecast_vs_history: compares second-difference energy of Z in forecast vs history.
      - latent_diag_mahalanobis_ratio_forecast_vs_history: compares a diagonal-Mahalanobis latent distance
        in forecast vs history (OOD shift indicator).

    Feature-axis (v2) outputs (populated only when ``explain_forecast`` is
    called with ``channel_axis=True``):
      - per_channel_flow: [T-1, C] -- per-channel decomposition of latent
        flow (phi_tau(c)).
      - lag_channel_horizon_scores: [K, C, H] -- joint score tensor before
        normalization.
      - lag_channel_horizon_attributions: [K, C, H] -- softmax-normalized
        joint lag x channel x horizon attribution.
      - channel_horizon_attributions: [C, H] -- channel-marginal of the
        joint attribution; suitable for channel-axis faithfulness benchmarks.
      - channel_coupling_matrix: [C, C] -- Harsanyi dividend matrix
        diagnosing cross-channel interaction; symmetric, zero diagonal.
      - channel_flow_method: which estimator produced the above
        ("jacobian" / "shapley_exact" / "shapley_kernel").
      - channel_flow_residual_ratio_mean / _p95: trust-region diagnostic for
        the Jacobian variant. ``|| Delta Z - sum_c Delta Z^(c) || / ||Delta Z||``
        averaged over transitions; lower is better.
      - channel_coupling_off_diag_norm: Frobenius norm of the off-diagonal
        coupling matrix; sharply distinguishes channel-mixing models (large)
        from channel-independent ones (~0).
    """

    baseline_forecast: Array
    lag_horizon_scores: Array
    lag_horizon_attributions: Array
    flow_magnitudes: Array
    latent_trajectory: Array
    surrogate_coef: Array | None = None
    surrogate_intercept: Array | None = None
    surrogate_feature_layout: str | None = None
    flow_ratio_forecast_vs_history: float | None = None
    flow_variance_ratio_forecast_vs_history: float | None = None
    curvature_ratio_forecast_vs_history: float | None = None
    latent_diag_mahalanobis_ratio_forecast_vs_history: float | None = None
    # v2 feature-axis outputs (None when channel_axis=False)
    per_channel_flow: Array | None = None
    lag_channel_horizon_scores: Array | None = None
    lag_channel_horizon_attributions: Array | None = None
    channel_horizon_attributions: Array | None = None
    channel_coupling_matrix: Array | None = None
    channel_flow_method: str | None = None
    channel_flow_residual_ratio_mean: float | None = None
    channel_flow_residual_ratio_p95: float | None = None
    channel_coupling_off_diag_norm: float | None = None


def _flow_segment_ratios(
    flow_magnitudes: Array,  # [T-1]
    context_len: int,
    forecast_horizon: int,
    eps: float = 1e-12,
) -> tuple[float | None, float | None]:
    """
    Compare flow in history vs forecast (OOD) segment to gauge attribution quality.

    The latent trajectory is built over [history; forecast]. Flow in the forecast segment
    (windows that include model-generated future) can be noisier; higher ratio means
    attributions that depend on that segment are more likely to be volatile.

    Returns:
      (flow_ratio, variance_ratio): mean(flow_forecast)/mean(flow_history) and
      var(flow_forecast)/var(flow_history). None if segment split is invalid.
    """
    m = np.asarray(flow_magnitudes, dtype=np.float32)
    m = np.where(np.isfinite(m), m, np.nan)
    L = int(context_len)
    H = int(forecast_horizon)
    if L <= 0 or H <= 0 or m.shape[0] < L + H - 1:
        return None, None
    # History: transitions 0..L-2 (windows purely in history)
    m_hist = m[0 : L - 1]
    # Forecast: transitions L-1..L+H-2 (windows extending into forecast)
    m_fcst = m[L - 1 : L + H - 1]
    if m_hist.size == 0 or m_fcst.size == 0:
        return None, None
    mean_hist = float(np.nanmean(m_hist)) + eps
    mean_fcst = float(np.nanmean(m_fcst)) + eps
    flow_ratio = mean_fcst / mean_hist
    var_hist = float(np.nanvar(m_hist)) + eps
    var_fcst = float(np.nanvar(m_fcst)) + eps
    var_ratio = var_fcst / var_hist
    return flow_ratio, var_ratio


def _curvature_segment_ratio(
    Z: Array,  # [T, D]
    context_len: int,
    forecast_horizon: int,
    eps: float = 1e-12,
) -> float | None:
    """
    Curvature/jitter diagnostic: ratio of second-difference energy in forecast vs history.

    Define curvature per step:
      c_t = ||Z_{t+2} - 2 Z_{t+1} + Z_t||_2, t=0..T-3.

    Returns:
      mean(c in forecast segment) / mean(c in history segment), or None if invalid split.
    """
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2 or Z.shape[0] < 3:
        return None
    T = Z.shape[0]
    L = int(context_len)
    H = int(forecast_horizon)
    if L <= 1 or H <= 0 or T < L + H:
        return None
    # second differences live on indices 0..T-3; segment boundary at latent index L-1.
    dd = Z[2:] - 2.0 * Z[1:-1] + Z[:-2]  # [T-2, D]
    c = np.linalg.norm(dd, ord=2, axis=1)
    c = np.where(np.isfinite(c), c, np.nan)

    # history curvature uses steps fully before boundary: t=0..(L-3)
    c_hist = c[0 : max(0, L - 2)]
    # forecast curvature uses steps starting at boundary: t=(L-2)..(L+H-3) (clipped)
    c_fcst = c[max(0, L - 2) : min(c.shape[0], L + H - 2)]
    if c_hist.size == 0 or c_fcst.size == 0:
        return None
    return (float(np.nanmean(c_fcst)) + eps) / (float(np.nanmean(c_hist)) + eps)


def _latent_diag_mahalanobis_ratio(
    Z: Array,  # [T, D]
    context_len: int,
    forecast_horizon: int,
    eps: float = 1e-12,
) -> float | None:
    """
    OOD shift diagnostic: compare diagonal-Mahalanobis distance of latents in forecast vs history.

    Fit (mu, var) on history latents Z[0:L], then compute:
      d2(z) = sum_i ( (z_i - mu_i)^2 / (var_i + eps) )
    Return mean(d2 in forecast latents) / mean(d2 in history latents).
    """
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2:
        return None
    T = Z.shape[0]
    L = int(context_len)
    H = int(forecast_horizon)
    if L <= 1 or H <= 0 or T < L + H:
        return None

    Z_hist = Z[:L]
    Z_fcst = Z[L : L + H]
    if Z_hist.shape[0] < 2 or Z_fcst.shape[0] < 1:
        return None

    mu = np.nanmean(np.where(np.isfinite(Z_hist), Z_hist, np.nan), axis=0)
    var = np.nanvar(np.where(np.isfinite(Z_hist), Z_hist, np.nan), axis=0)
    var = np.where(np.isfinite(var) & (var > 1e-8), var, 1.0).astype(np.float32)

    def d2(A: Array) -> Array:
        A = np.where(np.isfinite(A), A, np.nan)
        return np.nansum(((A - mu[None, :]) ** 2) / (var[None, :] + eps), axis=1)

    d2_hist = d2(Z_hist)
    d2_fcst = d2(Z_fcst)
    if not np.isfinite(d2_hist).any() or not np.isfinite(d2_fcst).any():
        return None
    return (float(np.nanmean(d2_fcst)) + eps) / (float(np.nanmean(d2_hist)) + eps)


def _softmax(x: Array, axis: int = 0, eps: float = 1e-12) -> Array:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    den = np.sum(ex, axis=axis, keepdims=True)
    return ex / np.maximum(den, eps)


def _as_numpy(x: Any) -> Array:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@torch.no_grad()
def _forecast_autoregressive(
    model: ForecastModel,
    x_enc: torch.Tensor,
    input_mask: torch.Tensor,
    *,
    model_horizon: int,
    target_horizon: int,
) -> Array:
    """
    Autoregressive forecasting (batched) to extend beyond model's native horizon.

    Returns: numpy array [B, C, target_horizon]
    """
    if target_horizon <= 0:
        raise ValueError("target_horizon must be positive")
    if model_horizon <= 0:
        raise ValueError("model_horizon must be positive")

    B, C, seq_len = x_enc.shape
    device = x_enc.device

    if target_horizon <= model_horizon:
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(x_enc=x_enc, input_mask=input_mask)
        forecast = _as_numpy(out.forecast[:, :, :target_horizon]).astype(np.float32, copy=False)
        return np.nan_to_num(forecast, nan=0.0, posinf=0.0, neginf=0.0)

    num_iters = int(np.ceil(target_horizon / model_horizon))
    cur_x = x_enc.clone()
    cur_mask = input_mask.clone()
    preds_chunks: list[Array] = []
    remaining = target_horizon

    for _ in range(num_iters):
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(x_enc=cur_x, input_mask=cur_mask)
        chunk = torch.nan_to_num(out.forecast.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        steps = min(model_horizon, remaining)
        chunk_trimmed = chunk[:, :, :steps]
        preds_chunks.append(chunk_trimmed.to(device="cpu", dtype=torch.float32).numpy())
        remaining -= steps
        if remaining <= 0:
            break

        extended = torch.cat([cur_x, chunk_trimmed], dim=2)
        cur_x = extended[:, :, -seq_len:]
        cur_mask = torch.ones(B, seq_len, dtype=cur_mask.dtype, device=device)

    return np.concatenate(preds_chunks, axis=2).astype(np.float32, copy=False)


def _channel_replacement_rows(arr_cl: Array, *, replacement: str) -> Array:
    arr = np.asarray(arr_cl, dtype=np.float32)
    C, L = arr.shape
    if replacement == "zero":
        return np.zeros((C, L), dtype=np.float32)
    if replacement == "local_mean":
        return np.repeat(arr.mean(axis=1, keepdims=True).astype(np.float32), L, axis=1)
    raise ValueError(f"Unsupported replacement={replacement!r}")


def compute_target_channel_input_jacobian_importance(
    model: ForecastModel,
    *,
    x_context_ct: Array,
    input_mask_l: Array,
    device: torch.device,
    target_channel: int,
    model_horizon: int,
    forecast_horizon: int,
    replacement: Literal["local_mean", "zero"] = "local_mean",
    forecast_batch_size: int = 8,
    finite_difference_scale: float = 0.1,
    prefer_autograd: bool = True,
) -> Array:
    """Target-specific directional replacement effect by channel and horizon."""
    arr = np.asarray(x_context_ct, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected x_context_ct [C,L], got {arr.shape}")
    C, L = arr.shape
    H = int(forecast_horizon)
    target = int(target_channel)
    if not (0 <= target < C):
        raise ValueError(f"target_channel out of range for C={C}: {target}")
    mask = np.asarray(input_mask_l, dtype=np.int64)
    if mask.shape != (L,):
        raise ValueError(f"Expected input_mask_l shape {(L,)}, got {mask.shape}")

    baseline = _channel_replacement_rows(arr, replacement=str(replacement))
    delta = arr - baseline
    active_direction = np.any(np.abs(delta) > 1e-12, axis=1)

    if prefer_autograd:
        delta_t = torch.from_numpy(delta[None]).to(device=device, dtype=torch.float32)
        x_t = torch.from_numpy(arr[None]).to(device=device, dtype=torch.float32).requires_grad_(True)
        m_t = torch.from_numpy(mask[None]).to(device=device, dtype=torch.long)
        with normalization_statistics_gradient(model, "frozen"):
            with torch.enable_grad():
                out = model(x_enc=x_t, input_mask=m_t)
                forecast = out.forecast[:, target, :H]
                if forecast.shape[-1] < H:
                    raise ValueError(f"Model returned horizon={forecast.shape[-1]}, expected at least {H}")
                if forecast.requires_grad:
                    importance = np.zeros((C, H), dtype=np.float32)
                    input_is_connected = True
                    for h in range(H):
                        grad = torch.autograd.grad(
                            forecast[:, h].sum(),
                            x_t,
                            retain_graph=h < H - 1,
                            create_graph=False,
                            allow_unused=True,
                        )[0]
                        if grad is None:
                            input_is_connected = False
                            break
                        score_c = (grad * delta_t).sum(dim=-1).abs()[0]
                        active_t = torch.as_tensor(active_direction, dtype=torch.bool, device=score_c.device)
                        score_c = torch.where(active_t, score_c, torch.zeros_like(score_c))
                        importance[:, h] = score_c.detach().cpu().numpy().astype(np.float32)
                    if input_is_connected:
                        importance[target, :] = 0.0
                        if not np.isfinite(importance).all():
                            n_bad = int((~np.isfinite(importance)).sum())
                            raise FloatingPointError(f"Jacobian attribution produced {n_bad} non-finite value(s)")
                        return importance.astype(np.float32, copy=False)

    eps = float(finite_difference_scale)
    if not (0.0 < eps <= 1.0):
        raise ValueError(f"finite_difference_scale must be in (0, 1], got {finite_difference_scale}")
    importance = np.zeros((C, H), dtype=np.float32)
    chans = [c for c in range(C) if c != target]
    channel_batch = max(1, int(forecast_batch_size) // 2)
    for start in range(0, len(chans), channel_batch):
        chunk = chans[start : start + channel_batch]
        stack = np.broadcast_to(arr[None], (2 * len(chunk), C, L)).copy()
        for i, c in enumerate(chunk):
            stack[i, c, :] = arr[c, :] - eps * delta[c, :]
            stack[len(chunk) + i, c, :] = arr[c, :] + eps * delta[c, :]
        masks = np.broadcast_to(mask[None], (2 * len(chunk), L)).copy()
        xb = torch.from_numpy(stack).to(device=device, dtype=torch.float32)
        mb = torch.from_numpy(masks).to(device=device, dtype=torch.long)
        yb = _forecast_autoregressive(
            model,
            xb,
            mb,
            model_horizon=int(model_horizon),
            target_horizon=H,
        )[:, target, :]
        effects = np.abs(yb[len(chunk) :] - yb[: len(chunk)]) / (2.0 * eps)
        for i, c in enumerate(chunk):
            importance[c, :] = effects[i]
    if not np.isfinite(importance).all():
        n_bad = int((~np.isfinite(importance)).sum())
        raise FloatingPointError(f"Finite-difference attribution produced {n_bad} non-finite value(s)")
    return importance.astype(np.float32, copy=False)


@torch.no_grad()
def _embed_batch(model: ForecastModel, x_enc: torch.Tensor, input_mask: torch.Tensor) -> Array:
    out = model.embed(x_enc=x_enc, input_mask=input_mask)
    if not hasattr(out, "embeddings") or out.embeddings is None:
        raise RuntimeError("model.embed(...) did not return `.embeddings`")
    return _as_numpy(out.embeddings).astype(np.float32)


def _sliding_window_view(a: Array, window_shape: int, *, axis: int = 0) -> Array:
    try:
        from numpy.lib.stride_tricks import sliding_window_view  # type: ignore[attr-defined]
    except Exception:
        from numpy.lib.stride_tricks import as_strided

        def sliding_window_view(a: Array, window_shape: int, *, axis: int = 0) -> Array:
            if axis != 0:
                raise ValueError("fallback sliding_window_view only supports axis=0")
            a = np.asarray(a)
            n = a.shape[0]
            w = int(window_shape)
            if w <= 0 or w > n:
                raise ValueError("invalid window_shape")
            shape = (n - w + 1, w) + a.shape[1:]
            strides = (a.strides[0],) + a.strides
            return as_strided(a, shape=shape, strides=strides)

    return sliding_window_view(a, window_shape=window_shape, axis=axis)


def _rolling_window_sources(
    series_ct: Array,  # [C, T]
    *,
    seq_len: int,
    input_mask_t: Array | None = None,  # [T] optional observed mask (1=real,0=pad/missing)
) -> tuple[Array, Array, int]:
    """
    Prepare rolling-window views used to materialize batches on demand.
    """
    if series_ct.ndim != 2:
        raise ValueError(f"Expected series [C,T], got shape {series_ct.shape}")
    _, T = series_ct.shape
    L = int(seq_len)
    if L <= 0:
        raise ValueError("seq_len must be positive")

    mask_t = None
    if input_mask_t is not None:
        mask_t = np.asarray(input_mask_t, dtype=np.int64).reshape(-1)
        if mask_t.shape != (T,):
            raise ValueError(f"Expected input_mask_t shape {(T,)}, got {mask_t.shape}")
    if T == 0:
        C = int(series_ct.shape[0])
        return np.zeros((0, C, L), dtype=np.float32), np.zeros((0, L), dtype=np.int64), 0

    x = np.asarray(series_ct, dtype=np.float32)
    pad = L - 1
    if pad > 0:
        x_pad = np.pad(x, ((0, 0), (pad, 0)), mode="constant", constant_values=np.float32(0.0))
    else:
        x_pad = x
    x_view = _sliding_window_view(x_pad, window_shape=L, axis=1)
    xw_view = np.transpose(x_view, (1, 0, 2))

    if mask_t is None:
        mask_src = np.ones((T,), dtype=np.int64)
    else:
        mask_src = mask_t.astype(np.int64, copy=False)
    if pad > 0:
        mask_pad = np.pad(mask_src, (pad, 0), mode="constant", constant_values=np.int64(0))
    else:
        mask_pad = mask_src
    m_view = _sliding_window_view(mask_pad, window_shape=L, axis=0)
    return xw_view, m_view, T


def _rolling_window_batch(
    xw_view: Array,
    m_view: Array,
    *,
    start: int,
    stop: int,
) -> tuple[Array, Array]:
    """
    Build only the requested rolling windows [start:stop].

    Returns sliced views when possible so callers can defer materialization until
    they actually need contiguous/writable arrays (for example at torch conversion).
    """
    start_i = max(0, int(start))
    stop_i = max(start_i, int(stop))
    batch = stop_i - start_i
    C = int(xw_view.shape[1]) if xw_view.ndim >= 2 else 0
    L = int(xw_view.shape[2]) if xw_view.ndim >= 3 else 0
    if batch == 0:
        return np.zeros((0, C, L), dtype=np.float32), np.zeros((0, L), dtype=np.int64)
    return (
        np.asarray(xw_view[start_i:stop_i], dtype=np.float32),
        np.asarray(m_view[start_i:stop_i], dtype=np.int64),
    )


def _rolling_windows(
    series_ct: Array,  # [C, T]
    *,
    seq_len: int,
    input_mask_t: Array | None = None,  # [T] optional observed mask (1=real,0=pad/missing)
) -> tuple[Array, Array]:
    """
    Build rolling model-input windows for every time index.

    Returns:
      x_windows: [T, C, seq_len]
      masks:     [T, seq_len]
    """
    L = int(seq_len)
    xw_view, m_view, T = _rolling_window_sources(series_ct, seq_len=L, input_mask_t=input_mask_t)
    if T == 0:
        C = int(np.asarray(series_ct).shape[0])
        return np.zeros((0, C, L), dtype=np.float32), np.zeros((0, L), dtype=np.int64)
    x_batch, m_batch = _rolling_window_batch(xw_view, m_view, start=0, stop=T)
    return np.ascontiguousarray(x_batch), np.ascontiguousarray(m_batch)


def extract_latent_trajectory(
    model: ForecastModel,
    series_ct: Array,  # [C, T]
    *,
    seq_len: int,
    input_mask_t: Array | None = None,  # [T] optional observed mask for the series
    device: torch.device,
    batch_size: int = 32,
) -> Array:
    """
    Latent Representation Interface (model-agnostic):
    - Builds a latent state Z_t for each time t by embedding the last `seq_len` values up to t.

    Returns: Z [T, D]
    """
    xw_view, m_view, T = _rolling_window_sources(series_ct, seq_len=seq_len, input_mask_t=input_mask_t)
    if T == 0:
        return np.zeros((0, 0), dtype=np.float32)
    Z_chunks: list[Array] = []
    for i in range(0, T, batch_size):
        x_batch, m_batch = _rolling_window_batch(xw_view, m_view, start=i, stop=i + batch_size)
        xb = torch.from_numpy(np.ascontiguousarray(x_batch)).to(device=device, dtype=torch.float32)
        mb = torch.from_numpy(np.ascontiguousarray(m_batch)).to(device=device, dtype=torch.long)
        Z_chunks.append(_embed_batch(model, xb, mb))
    Z = np.concatenate(Z_chunks, axis=0)
    if np.any(~np.isfinite(Z)):
        warnings.warn(
            "Latent trajectory Z contains non-finite values (NaN/inf); replacing with 0 before downstream computation.",
            UserWarning,
            stacklevel=2,
        )
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Z


def _signed_states(values_td: Array, *, deadband: float) -> np.ndarray:
    states = np.zeros(values_td.shape, dtype=np.int8)
    states[values_td > deadband] = 1
    states[values_td < -deadband] = -1
    return states


def _state_change_rate(states_td: np.ndarray) -> np.ndarray:
    if states_td.ndim != 2:
        raise ValueError(f"Expected states [T,D], got shape {states_td.shape}")
    if states_td.shape[0] < 2:
        return np.full((states_td.shape[1],), np.nan, dtype=np.float32)
    prev = states_td[:-1]
    cur = states_td[1:]
    valid = (prev != 0) & (cur != 0)
    changed = (prev != cur) & valid
    denom = np.sum(valid, axis=0).astype(np.float32)
    numer = np.sum(changed, axis=0).astype(np.float32)
    out = np.full((states_td.shape[1],), np.nan, dtype=np.float32)
    nonzero = denom > 0
    out[nonzero] = numer[nonzero] / denom[nonzero]
    return out


def _safe_mean_and_p95(values_d: Array) -> tuple[float, float]:
    vals = np.asarray(values_d, dtype=np.float32).reshape(-1)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.percentile(finite, 95))


def summarize_trajectory_stability(
    Z_td: Array,
    *,
    reference_mask_t: Array | None = None,
    reference_mode: Literal["mean", "median"] = "mean",
    normalize: bool = True,
    zero_band: float = 0.05,
    eps: float = 1e-8,
) -> TrajectoryStabilityReport:
    """
    Quantify how often a latent trajectory oscillates around its reference state.

    We center each latent dimension by a reference vector, optionally divide by its
    reference-segment standard deviation, then compute:
      - zero-crossing rate: how often the trajectory changes sign around the reference
      - direction-flip rate: how often consecutive step directions reverse
      - relative jitter: mean absolute step size divided by mean absolute level
      - occupancy: fraction of time spent above/below the centered reference
    """
    Z = np.asarray(Z_td, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Expected latent trajectory [T,D], got shape {Z.shape}")
    if Z.shape[0] < 2:
        raise ValueError("Need at least two latent states to summarize trajectory stability")
    if np.any(~np.isfinite(Z)):
        warnings.warn(
            "Latent trajectory Z contains non-finite values (NaN/inf); replacing with 0 for trajectory stability.",
            UserWarning,
            stacklevel=2,
        )
        Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)

    T, D = Z.shape
    if reference_mask_t is None:
        ref_mask = np.ones((T,), dtype=bool)
    else:
        ref_mask = np.asarray(reference_mask_t, dtype=bool).reshape(-1)
        if ref_mask.shape != (T,):
            raise ValueError(f"Expected reference_mask_t shape {(T,)}, got {ref_mask.shape}")
        if not np.any(ref_mask):
            ref_mask = np.ones((T,), dtype=bool)
    Z_ref = Z[ref_mask]

    if reference_mode == "mean":
        center = np.mean(Z_ref, axis=0, dtype=np.float32)
    else:
        center = np.median(Z_ref, axis=0).astype(np.float32)

    Z_centered = Z - center[None, :]
    if normalize:
        scale = np.std(Z_ref, axis=0).astype(np.float32)
        scale[scale < eps] = 1.0
        Z_centered = Z_centered / scale[None, :]

    states = _signed_states(Z_centered, deadband=float(zero_band))
    zero_crossing_rate_d = _state_change_rate(states)

    deltas = Z_centered[1:] - Z_centered[:-1]
    delta_states = _signed_states(deltas, deadband=float(zero_band))
    direction_flip_rate_d = _state_change_rate(delta_states)

    level_mag = np.mean(np.abs(Z_centered), axis=0)
    step_mag = np.mean(np.abs(deltas), axis=0)
    relative_jitter_d = step_mag / (level_mag + float(eps))

    return TrajectoryStabilityReport(
        zero_crossing_rate_mean=_safe_mean_and_p95(zero_crossing_rate_d)[0],
        zero_crossing_rate_p95=_safe_mean_and_p95(zero_crossing_rate_d)[1],
        direction_flip_rate_mean=_safe_mean_and_p95(direction_flip_rate_d)[0],
        direction_flip_rate_p95=_safe_mean_and_p95(direction_flip_rate_d)[1],
        relative_jitter_mean=_safe_mean_and_p95(relative_jitter_d)[0],
        relative_jitter_p95=_safe_mean_and_p95(relative_jitter_d)[1],
        occupancy_positive_mean=float(np.mean(Z_centered > float(zero_band))),
        occupancy_negative_mean=float(np.mean(Z_centered < -float(zero_band))),
        n_time_steps=int(T),
        n_dimensions=int(D),
    )


@torch.no_grad()
def compute_trajectory_stability(
    model: ForecastModel,
    series_ct: Array,  # [C, T]
    *,
    seq_len: int,
    input_mask_t: Array | None = None,
    device: torch.device,
    batch_size: int = 32,
    reference_mode: Literal["mean", "median"] = "mean",
    normalize: bool = True,
    zero_band: float = 0.05,
    eps: float = 1e-8,
) -> TrajectoryStabilityReport:
    """
    Build the latent trajectory for a series and summarize its temporal smoothness.
    """
    Z = extract_latent_trajectory(
        model,
        series_ct,
        seq_len=seq_len,
        input_mask_t=input_mask_t,
        device=device,
        batch_size=batch_size,
    )
    return summarize_trajectory_stability(
        Z,
        reference_mask_t=input_mask_t,
        reference_mode=reference_mode,
        normalize=normalize,
        zero_band=zero_band,
        eps=eps,
    )


@torch.no_grad()
def compute_embedding_stability(
    model: ForecastModel,
    series_ct: Array,  # [C, T]
    *,
    seq_len: int,
    input_mask_t: Array | None = None,  # [T] optional observed mask for the series
    device: torch.device,
    n_trials: int = 50,
    noise_scale: float = 0.01,
    time_indices: Array | None = None,
    batch_size: int = 16,
    seed: int = 42,
) -> EmbeddingStabilityReport:
    """
    Empirical stability test for the latent mapping Z_t = embed(x_{t-L+1:t}).

    For each trial we perturb the input window slightly (Gaussian noise scaled by input std),
    embed both original and perturbed windows, and compute the ratio:
      ratio = ||Z' - Z||_2 / (||x' - x||_F + eps)
    A bounded ratio (e.g. low mean/max) suggests Lipschitz-like behaviour; large ratios
    indicate that small input changes cause large latent jumps (representation noise).

    Returns:
      EmbeddingStabilityReport with lip_ratio stats and mean step delta ||Z_{t+1}-Z_t||
      on the unperturbed trajectory for comparison. `n_trials` counts perturbation
      runs; `n_unique_windows` counts distinct valid windows used.
    """
    xw_view, m_view, T = _rolling_window_sources(series_ct, seq_len=seq_len, input_mask_t=input_mask_t)
    if T < 2:
        raise ValueError("Need at least 2 time steps for stability test")

    rng = np.random.default_rng(seed)
    if time_indices is None:
        # Prefer indices where we have full windows. If we have fewer candidates than n_trials
        # (e.g. single context window [C,L] gives only one full-window index), we repeat
        # with different random perturbations to get n_trials ratios.
        candidates = np.arange(seq_len - 1, T, dtype=np.int64)
        if len(candidates) == 0:
            candidates = np.arange(T, dtype=np.int64)
        if len(candidates) >= n_trials:
            time_indices = rng.choice(candidates, size=n_trials, replace=False)
        else:
            # Repeat indices so we can run n_trials perturbations (different noise each time)
            n_repeats = int(np.ceil(n_trials / len(candidates)))
            time_indices = np.tile(candidates, n_repeats)[:n_trials]
            rng.shuffle(time_indices)
    else:
        time_indices = np.asarray(time_indices, dtype=np.int64).ravel()

    ratios: list[float] = []
    used_time_indices: set[int] = set()
    for t in time_indices:
        if t < 0 or t >= T:
            continue
        x_window, m_window = _rolling_window_batch(xw_view, m_view, start=int(t), stop=int(t) + 1)
        x = x_window[0]  # [C, L]
        win_mask = m_window[0].astype(np.float32, copy=False)  # [L]
        obs = win_mask.astype(bool)
        if not np.any(obs):
            continue

        # Perturb: additive Gaussian noise scaled by per-channel std (observed positions only)
        std = np.std(x[:, obs], axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        noise = rng.standard_normal(x.shape).astype(np.float32) * float(noise_scale) * std
        noise = noise * win_mask[None, :]
        x_pert = x + noise

        dx_norm = float(np.linalg.norm((x_pert - x) * win_mask[None, :], ord="fro")) + 1e-12
        if dx_norm < 1e-12:
            continue

        x_batch = np.stack([x, x_pert], axis=0)  # [2, C, L]
        m_batch = np.repeat(m_window, repeats=2, axis=0)
        x_t = torch.from_numpy(x_batch).to(device=device, dtype=torch.float32)
        m_t = torch.from_numpy(m_batch).to(device=device, dtype=torch.long)
        Z_batch = _embed_batch(model, x_t, m_t)  # [2, D]
        dz_norm = float(np.linalg.norm(Z_batch[1] - Z_batch[0], ord=2)) + 1e-12
        ratios.append(dz_norm / dx_norm)
        used_time_indices.add(int(t))

    if not ratios:
        return EmbeddingStabilityReport(
            lip_ratio_mean=float("nan"),
            lip_ratio_max=float("nan"),
            lip_ratio_p50=float("nan"),
            lip_ratio_p95=float("nan"),
            n_trials=0,
            n_unique_windows=0,
            step_delta_norm_mean=float("nan"),
        )

    ratios_arr = np.array(ratios, dtype=np.float64)
    # Unperturbed trajectory step deltas (consecutive pairs). Use nanmean in case some Z are NaN.
    Z_full = extract_latent_trajectory(
        model,
        series_ct,
        seq_len=seq_len,
        input_mask_t=input_mask_t,
        device=device,
        batch_size=batch_size,
    )
    step_deltas = np.linalg.norm(Z_full[1:] - Z_full[:-1], ord=2, axis=1)
    step_delta_mean = float(np.nanmean(step_deltas))

    return EmbeddingStabilityReport(
        lip_ratio_mean=float(np.mean(ratios_arr)),
        lip_ratio_max=float(np.max(ratios_arr)),
        lip_ratio_p50=float(np.percentile(ratios_arr, 50)),
        lip_ratio_p95=float(np.percentile(ratios_arr, 95)),
        n_trials=len(ratios),
        n_unique_windows=len(used_time_indices),
        step_delta_norm_mean=step_delta_mean,
    )


def _smooth_latent_trajectory(Z: Array, alpha: float) -> Array:
    """
    EMA along time (returns new array):
      Z_smooth[t] = alpha * Z_smooth[t-1] + (1-alpha) * Z[t]
    with the first row unchanged.

    Uses a vectorized closed form for speed on large [T, D] arrays and falls
    back to the sequential recurrence if alpha is so extreme that the power
    scaling becomes numerically unstable.
    """
    if Z.shape[0] < 2 or alpha <= 0 or alpha >= 1:
        return Z

    out = np.asarray(Z, dtype=np.float32).copy()
    T = out.shape[0]
    alpha_f = float(alpha)

    # Closed form:
    #   y[t] = alpha^t * (x[0] + (1-alpha) * sum_{k=1}^t x[k] / alpha^k)
    # This avoids the Python O(T) recurrence loop while preserving the exact EMA.
    idx = np.arange(1, T, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        pow_alpha = np.power(alpha_f, idx)
        inv_pow_alpha = np.power(alpha_f, -idx)

    if np.all(np.isfinite(pow_alpha)) and np.all(np.isfinite(inv_pow_alpha)):
        scaled = out[1:].astype(np.float64, copy=False) * inv_pow_alpha[:, None]
        csum = np.cumsum(scaled, axis=0, dtype=np.float64)
        out[1:] = (
            pow_alpha[:, None] * (out[0].astype(np.float64, copy=False)[None, :] + (1.0 - alpha_f) * csum)
        ).astype(np.float32)
        return out

    for t in range(1, T):
        out[t] = alpha_f * out[t - 1] + (1.0 - alpha_f) * out[t]
    return out


def compute_semantic_flow_magnitudes(
    Z: Array,
    cfg: SemanticFlowConfig,
    dy_dZ: Array | None = None,
) -> Array:
    """
    Semantic Flow Computation Unit:
    - Computes a per-step scalar flow magnitude from consecutive latent states.

    If SemanticFlowConfig.smooth_latent_alpha is in (0, 1), applies EMA smoothing to Z
    before computing flow (Lipschitz-like regularization when embeddings are noisy).

    For flow_kind 'output_aware': requires dy_dZ. Then
      m_t = |(∂y/∂Z_t)(Z_{t+1}-Z_t)|  (first-order effect of latent step on forecast).
    dy_dZ must have shape (T, D) or (T, H, D). T must match Z.shape[0]. If (T, H, D),
    magnitudes are aggregated over horizons using output_aware_aggregation. Assumptions:
    forecast (or surrogate) is differentiable w.r.t. Z_t and depends on more than Z_{L-1}
    for non-degenerate lag resolution.

    Returns: m [T-1]
    """
    if Z.ndim != 2:
        raise ValueError(f"Expected Z [T,D], got shape {Z.shape}")
    if Z.shape[0] < 2:
        raise ValueError("Need at least two latent states to compute flow.")

    T, D = Z.shape

    # Optional temporal smoothing of latent trajectory
    if 0 < cfg.smooth_latent_alpha < 1:
        Z = _smooth_latent_trajectory(Z, cfg.smooth_latent_alpha)

    # Sanitize NaN in latent trajectory (e.g. from embed on out-of-distribution windows)
    if np.any(~np.isfinite(Z)):
        warnings.warn(
            "Latent trajectory Z contains non-finite values (NaN/inf); replacing with 0 for flow computation. "
            "This can happen when the model embeds windows that include forecast values.",
            UserWarning,
            stacklevel=2,
        )
        Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)

    d = Z[1:] - Z[:-1]  # [T-1, D]

    if cfg.flow_kind == "output_aware":
        if dy_dZ is None:
            raise ValueError("flow_kind='output_aware' requires dy_dZ (gradient of forecast w.r.t. Z_t).")
        dy_dZ = np.asarray(dy_dZ, dtype=np.float32)
        if dy_dZ.shape[0] != T:
            raise ValueError(f"dy_dZ.shape[0] must equal Z.shape[0] ({T}), got {dy_dZ.shape[0]}")
        if dy_dZ.ndim == 2:
            # [T, D] => scalar effect per step: m_t = |dy_dZ[t] @ delta_t|
            if dy_dZ.shape[1] != D:
                raise ValueError(f"dy_dZ shape [T,D] expects D={D}, got {dy_dZ.shape[1]}")
            # m_t = |dy_dZ[t] @ d[t]| for t = 0..T-2; we need dy_dZ at indices 0..T-2 (first T-1 rows)
            grad_t = dy_dZ[:-1]  # [T-1, D]
            dot = np.sum(grad_t * d, axis=1)  # [T-1]
            m = np.abs(dot).astype(np.float32)
        elif dy_dZ.ndim == 3:
            # [T, H, D] => aggregate over horizons
            H = dy_dZ.shape[1]
            if dy_dZ.shape[2] != D:
                raise ValueError(f"dy_dZ shape [T,H,D] expects D={D}, got {dy_dZ.shape[2]}")
            grad_t = dy_dZ[:-1]  # [T-1, H, D]
            # dot[t,h] = grad_t[t,h] @ d[t]
            dot = np.einsum("thd,td->th", grad_t, d)  # [T-1, H]
            if cfg.output_aware_aggregation == "l2":
                m = np.sqrt(np.sum(dot**2, axis=1)).astype(np.float32)
            else:
                m = np.sum(np.abs(dot), axis=1).astype(np.float32)
        else:
            raise ValueError(f"dy_dZ must be 2D [T,D] or 3D [T,H,D], got shape {dy_dZ.shape}")
        if np.any(~np.isfinite(m)):
            m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
        return m

    if cfg.flow_kind != "delta_l2":
        raise ValueError(f"Unsupported flow_kind={cfg.flow_kind}")

    delta = np.linalg.norm(d, ord=2, axis=1).astype(np.float32)
    if np.any(~np.isfinite(delta)):
        delta = np.where(np.isfinite(delta), delta, 0.0).astype(np.float32)

    if cfg.scale_weight > 0:
        n0 = np.linalg.norm(Z[:-1], ord=2, axis=1)
        n1 = np.linalg.norm(Z[1:], ord=2, axis=1)
        ratio = (n1 + cfg.eps) / (n0 + cfg.eps)
        ratio = np.clip(ratio, 1e-6, 1e6)
        log_scale = np.abs(np.log(ratio)).astype(np.float32)
        log_scale = np.clip(log_scale, 0.0, 10.0)
        return delta + float(cfg.scale_weight) * log_scale

    return delta


def lag_horizon_attribution(
    flow_magnitudes: Array,  # [T-1]
    *,
    t_index: int,
    n_lags: int,
    horizon: int,
    softmax_tau: float = 1.0,
    horizon_kernel: Literal["exp", "none"] = "exp",
    kernel_min_scale: float = 4.0,
    kernel_max_scale: float | None = None,
) -> tuple[Array, Array]:
    """
    Lag-Horizon Attribution Engine.

    We compute a lag x horizon score matrix from *history* flow magnitudes (transitions up to t_index),
    then normalize horizon-wise with softmax over lags.

    Why history-only?
    - The latent trajectory is built over [history; forecast]. The forecast part is model-generated (OOD).
      Including forecast-segment flow in the lag-score adds a horizon-dependent *constant* across lags,
      which makes the horizon-wise softmax identical across horizons (a degeneracy).
    - For horizon-resolved explanations, we instead weight the history flow with a horizon-dependent kernel
      so that far horizons can distribute weight over longer temporal scales.

    Returns:
      scores: [K, H] (cumulative-flow score; larger => more influence)
      attributions: [K, H] (softmax-normalized over lags for each horizon)

    Note: The attribution matrix is in general *full*, not upper triangular. Each lag j can
    influence every horizon h, because we sum flow along the path from t-j to t+h; there is no
    structural constraint that lag j only affects horizon h when h >= j.
    """
    m = np.asarray(flow_magnitudes, dtype=np.float32).copy()
    if np.any(~np.isfinite(m)):
        warnings.warn(
            "flow_magnitudes contain non-finite values; replacing with 0 for attribution.",
            UserWarning,
            stacklevel=2,
        )
        m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)

    Tm = m.shape[0]  # transitions count (Z length - 1)
    K = int(n_lags)
    H = int(horizon)
    if K <= 0 or H <= 0:
        raise ValueError("n_lags and horizon must be positive")
    if not (0 <= t_index <= Tm):
        raise ValueError(f"t_index must be in [0,{Tm}], got {t_index}")

    # History transitions are indices 0..t_index-1 inclusive (count = t_index).
    # A lag j uses the last j history transitions: p = (t_index-j) .. (t_index-1).
    if t_index < 1:
        raise ValueError("t_index must be >= 1 to attribute history flow to lags.")
    if t_index < K:
        raise ValueError(f"n_lags must be <= t_index={t_index}, got {K}")

    # Horizon-dependent kernel over transition age a=0..K-1 (0=most recent history transition).
    if horizon_kernel not in ("exp", "none"):
        raise ValueError(f"Unsupported horizon_kernel={horizon_kernel}")
    min_s = float(kernel_min_scale)
    max_s = float(kernel_max_scale) if kernel_max_scale is not None else float(K)
    if min_s <= 0 or max_s <= 0:
        raise ValueError("kernel_min_scale and kernel_max_scale must be positive")
    max_s = max(max_s, min_s)

    scores = np.zeros((K, H), dtype=np.float32)
    # Pre-slice history flow (length t_index). We only need last K steps for attribution.
    hist = m[:t_index]  # [t_index]
    hist_tail = hist[max(0, t_index - K) : t_index]  # [<=K], oldest..newest
    # Reverse so index 0 = most recent, consistent with "age"
    hist_rev = hist_tail[::-1].astype(np.float32, copy=False)  # [<=K], most recent..older

    # Vectorized horizon kernels. The previous implementation recomputed ages/weights/cumsum per horizon.
    # Here we broadcast weights across horizons and compute all cumulative sums in one pass (chunked to
    # cap memory when K*H is large).
    L = int(hist_rev.shape[0])
    if H == 1:
        scales_h = np.array([min_s], dtype=np.float32)
    else:
        u_h = (np.arange(H, dtype=np.float32) / float(H - 1)).astype(np.float32)  # 0..1
        scales_h = (min_s + u_h * (max_s - min_s)).astype(np.float32)

    if horizon_kernel == "none":
        csum = np.cumsum(hist_rev, dtype=np.float32)  # [L]
        scores[:L, :] = csum[:, None]
    else:
        scales_h = np.maximum(scales_h, np.float32(1e-6))
        ages_l = np.arange(L, dtype=np.float32)[:, None]  # [L,1]

        # Choose a horizon chunk size to keep the intermediate [L,chunk] weight matrix bounded.
        # (float32 => 4 bytes per entry)
        target_bytes = 64 * 1024 * 1024  # ~64MiB
        bytes_per_h = max(1, L * 4)
        chunk = int(max(1, min(H, target_bytes // bytes_per_h)))

        for h0 in range(0, H, chunk):
            h1 = min(H, h0 + chunk)
            s = scales_h[h0:h1][None, :]  # [1,chunk] float32
            w_lh = np.exp(-ages_l / s)  # [L,chunk] float32
            hw_lh = hist_rev[:, None] * w_lh
            csum_lh = np.cumsum(hw_lh, axis=0, dtype=np.float32)  # [L,chunk]
            scores[:L, h0:h1] = csum_lh

    # Horizon-wise softmax over lags.
    tau = max(float(softmax_tau), 1e-6)
    zero_score_cols = np.all(scores == 0.0, axis=0)
    if np.any(zero_score_cols):
        warnings.warn(
            f"{int(np.sum(zero_score_cols))} horizon(s) have all-zero lag scores; "
            "returning uniform lag attributions for those columns.",
            UserWarning,
            stacklevel=2,
        )
    attributions = _softmax(scores / tau, axis=0)
    # If any column is all zero or had NaN, softmax can yield NaN; fallback to uniform
    if np.any(~np.isfinite(attributions)):
        uniform = np.ones((K, H), dtype=np.float32) / float(K)
        attributions = np.where(np.isfinite(attributions), attributions, uniform)
    return scores, attributions


def _block_bootstrap(series_ct: Array, *, block_len: int, rng: np.random.Generator) -> Array:
    C, L = series_ct.shape
    b = max(1, int(block_len))
    n_blocks = int(np.ceil(L / b))
    out = np.empty((C, n_blocks * b), dtype=np.float32)
    for i in range(n_blocks):
        s = int(rng.integers(0, max(1, L - b + 1)))
        out[:, i * b : (i + 1) * b] = series_ct[:, s : s + b]
    return out[:, :L]


def _fourier_phase_preserving_noise(series_ct: Array, *, amp_sigma: float, rng: np.random.Generator) -> Array:
    if amp_sigma <= 0:
        return series_ct.astype(np.float32, copy=True)

    C, L = series_ct.shape
    out = np.empty_like(series_ct, dtype=np.float32)
    for c in range(C):
        x = series_ct[c].astype(np.float32)
        X = np.fft.rfft(x)
        amp = np.abs(X)
        phase = np.angle(X)

        # multiplicative log-amplitude noise keeps phase unchanged
        noise = rng.normal(loc=0.0, scale=float(amp_sigma), size=amp.shape).astype(np.float32)
        amp2 = amp * np.exp(noise)
        X2 = amp2 * np.exp(1j * phase)
        x2 = np.fft.irfft(X2, n=L).real.astype(np.float32)

        # Match mean/std to original to avoid drift.
        mu0, sd0 = float(x.mean()), float(x.std() + 1e-8)
        mu1, sd1 = float(x2.mean()), float(x2.std() + 1e-8)
        x2 = (x2 - mu1) * (sd0 / sd1) + mu0
        out[c] = x2
    return out


def generate_structure_preserving_perturbations(x_context_ct: Array, cfg: PerturbationConfig) -> Array:
    """
    Generate perturbations that preserve temporal continuity and seasonality.

    Returns: samples [N, C, L]
    """
    x0 = np.asarray(x_context_ct, dtype=np.float32)
    if x0.ndim != 2:
        raise ValueError(f"Expected x_context [C,L], got {x0.shape}")
    C, L = x0.shape

    rng = np.random.default_rng(int(cfg.seed))
    out = np.empty((int(cfg.n_samples), C, L), dtype=np.float32)
    mix = float(cfg.mix_original)
    for i in range(int(cfg.n_samples)):
        xb = _block_bootstrap(x0, block_len=cfg.block_len, rng=rng)
        xf = _fourier_phase_preserving_noise(xb, amp_sigma=cfg.fourier_amp_sigma, rng=rng)
        out[i] = mix * x0 + (1.0 - mix) * xf
    return out


def _weighted_standardize(X: Array, w: Array, eps: float = 1e-12) -> tuple[Array, Array, Array]:
    """
    Weighted standardization column-wise.

    Returns Xs, mean, std.
    """
    w = w.reshape(-1, 1).astype(np.float32)
    wsum = float(np.sum(w)) + eps
    mu = np.sum(w * X, axis=0) / wsum
    var = np.sum(w * (X - mu) ** 2, axis=0) / wsum
    sd = np.sqrt(np.maximum(var, eps))
    return (X - mu) / sd, mu.astype(np.float32), sd.astype(np.float32)


def _ista_weighted_lasso(
    X: Array,
    y: Array,
    w: Array,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[Array, float]:
    """
    Weighted Lasso via ISTA:
      min_b  1/2 ||sqrt(w) (X b - y)||^2 + alpha ||b||_1

    Returns:
      coef: [P]
      intercept: scalar (computed in original space)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, p = X.shape
    if y.shape[0] != n or w.shape[0] != n:
        raise ValueError("Mismatched shapes between X, y, w")

    # Weighted standardize X; center y
    Xs, mu_x, sd_x = _weighted_standardize(X, w)
    wsum = float(np.sum(w)) + 1e-12
    mu_y = float(np.sum(w * y) / wsum)
    yc = y - mu_y

    # Precompute for gradient with weights
    sw = np.sqrt(np.maximum(w, 0.0)).astype(np.float32)
    Xw = Xs * sw[:, None]
    yw = yc * sw

    # Lipschitz constant of grad = ||Xw||_2^2 (spectral norm squared).
    # Use power iteration on A=(Xw^T Xw) with a convergence check.
    # Underestimating L makes ISTA steps too large; we also add backtracking below.
    v = np.ones((p,), dtype=np.float32) / np.sqrt(max(1, p))
    power_tol = 1e-4
    max_power_iter = 100
    for _ in range(max_power_iter):
        Av = Xw.T @ (Xw @ v)
        nAv = float(np.linalg.norm(Av) + 1e-12)
        v_next = (Av / nAv).astype(np.float32)
        if float(np.linalg.norm(v_next - v)) <= power_tol:
            v = v_next
            break
        v = v_next
    Av = Xw.T @ (Xw @ v)
    L = float(v @ Av)
    if not np.isfinite(L) or L <= 0:
        # Safe fallback bound: ||Xw||_F^2 >= ||Xw||_2^2
        L = float(np.sum(Xw.astype(np.float64) ** 2)) + 1e-12
    else:
        L = L + 1e-12
    step = 1.0 / L

    def soft_threshold(a: Array, thr: float) -> Array:
        return np.sign(a) * np.maximum(np.abs(a) - thr, 0.0)

    b = np.zeros((p,), dtype=np.float32)
    last_obj = np.inf
    max_backtrack = 25
    for _ in range(int(max_iter)):
        r = Xw @ b - yw
        grad = Xw.T @ r
        obj_cur = 0.5 * float(r @ r) + float(alpha) * float(np.sum(np.abs(b)))

        # ISTA step; backtrack if objective fails to decrease (guards against L underestimation).
        step_local = step
        b_new = soft_threshold(b - step_local * grad, step_local * float(alpha))

        # objective (for convergence check)
        r2 = Xw @ b_new - yw
        obj = 0.5 * float(r2 @ r2) + float(alpha) * float(np.sum(np.abs(b_new)))
        bt = 0
        while (not np.isfinite(obj) or obj > obj_cur) and bt < max_backtrack:
            step_local *= 0.5
            b_new = soft_threshold(b - step_local * grad, step_local * float(alpha))
            r2 = Xw @ b_new - yw
            obj = 0.5 * float(r2 @ r2) + float(alpha) * float(np.sum(np.abs(b_new)))
            bt += 1

        step = min(step, step_local)
        if abs(last_obj - obj) <= float(tol) * max(1.0, last_obj):
            b = b_new
            break
        b = b_new
        last_obj = obj

    # Unstandardize: y ≈ mu_y + sum_j ( (x_j - mu_xj)/sd_xj ) * b_j
    # => y ≈ (mu_y - sum_j mu_xj/sd_xj * b_j) + sum_j x_j * (b_j/sd_xj)
    coef = (b / sd_x).astype(np.float32)
    intercept = float(mu_y - np.sum((mu_x / sd_x) * b))
    return coef, intercept


def fit_horizon_surrogates(
    model: ForecastModel,
    *,
    x_context_ct: Array,  # [C, L]
    input_mask_l: Array,  # [L]
    model_horizon: int,
    forecast_horizon: int,
    device: torch.device,
    pert_cfg: PerturbationConfig,
    surr_cfg: SurrogateConfig,
    batch_size: int = 16,
    n_jobs: int | None = None,
) -> tuple[Array, Array, str]:
    """
    Local Surrogate Modeling Module.

    We fit a weighted sparse linear model per horizon h and channel c:
      y_hat[c,h] ≈ intercept[c,h] + X_lags · coef[c,h,:]

    Features are lagged values of the context window:
      X = flatten([C, K]) for last K lags (oldest->newest within that slice).

    Weights:
      w_i = exp(-distance_lambda * ||Z_i - Z_0||^2)
    where Z_i is an embedding of the (perturbed) context window.
    """
    x0 = np.asarray(x_context_ct, dtype=np.float32)
    C, L = x0.shape
    K = int(surr_cfg.n_lags)
    if K <= 0 or K > L:
        raise ValueError(f"Surrogate n_lags must be in [1,{L}], got {K}")

    # Baseline embedding for weighting
    x0_t = torch.from_numpy(x0[None, :, :]).to(device=device, dtype=torch.float32)
    m0_t = torch.from_numpy(np.asarray(input_mask_l[None, :], dtype=np.int64)).to(device=device, dtype=torch.long)
    z0 = _embed_batch(model, x0_t, m0_t)[0]  # [D]

    # Perturbations
    samples = generate_structure_preserving_perturbations(x0, pert_cfg)  # [N, C, L]
    N = samples.shape[0]

    # Build features from last K lags
    X = samples[:, :, -K:]  # [N,C,K]
    X = X.reshape(N, C * K).astype(np.float32)  # [N, P]
    feature_layout = f"flatten(C={C},lags={K}) with ordering [c0_lag0..lagK-1, c1_lag0..]"

    # Embed perturbed samples for weights, and forecast to build y targets.
    Z_list: list[Array] = []
    Y_list: list[Array] = []
    mask = np.asarray(input_mask_l, dtype=np.int64)
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(samples[i : i + batch_size]).to(device=device, dtype=torch.float32)
        mb = torch.from_numpy(np.tile(mask[None, :], (xb.shape[0], 1))).to(device=device, dtype=torch.long)
        Z_list.append(_embed_batch(model, xb, mb))
        Yb = _forecast_autoregressive(
            model,
            xb,
            mb,
            model_horizon=model_horizon,
            target_horizon=forecast_horizon,
        )  # [b,C,H]
        Y_list.append(Yb)

    Z = np.concatenate(Z_list, axis=0)  # [N,D]
    Y = np.concatenate(Y_list, axis=0)  # [N,C,H]

    # Local weights around baseline
    dz = Z - z0[None, :]
    dist2 = np.sum(dz * dz, axis=1).astype(np.float32)
    w = np.exp(-float(surr_cfg.distance_lambda) * dist2).astype(np.float32) + 1e-12

    # Fit per horizon+channel. Each (c, h) regression is independent and
    # CPU-bound (NumPy / BLAS), so we farm them out to a thread pool when
    # ``n_jobs > 1``. ISTA's inner loops call into NumPy linalg which
    # releases the GIL, so threads scale better than processes here and
    # avoid pickling the (potentially large) shared X / w buffers.
    H = int(forecast_horizon)
    P = X.shape[1]
    coef = np.zeros((C, H, P), dtype=np.float32)  # [C,H,P]
    intercept = np.zeros((C, H), dtype=np.float32)  # [C,H]

    alpha = float(surr_cfg.l1_alpha)
    max_iter = int(surr_cfg.max_iter)
    tol = float(surr_cfg.tol)

    def _solve_one(args: tuple[int, int]) -> tuple[int, int, Array, float]:
        c_idx, h_idx = args
        b, a0 = _ista_weighted_lasso(
            X,
            Y[:, c_idx, h_idx],
            w,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
        return c_idx, h_idx, b, a0

    work_items = [(c, h) for c in range(C) for h in range(H)]

    # Default: 1 thread (matches historical behaviour). Callers that want
    # the parallel path opt in via ``n_jobs`` (typically the OS CPU count
    # divided by the number of GPU workers, so threads do not oversubscribe
    # cores in a multi-GPU run).
    if n_jobs is not None and int(n_jobs) > 1 and len(work_items) > 1:
        from parallel import parallel_apply_threads

        for c_idx, h_idx, b, a0 in parallel_apply_threads(_solve_one, work_items, n_jobs=int(n_jobs)):
            coef[c_idx, h_idx] = b
            intercept[c_idx, h_idx] = a0
    else:
        for item in work_items:
            c_idx, h_idx, b, a0 = _solve_one(item)
            coef[c_idx, h_idx] = b
            intercept[c_idx, h_idx] = a0

    return coef, intercept, feature_layout


def explain_forecast(
    model: ForecastModel,
    *,
    x_context_ct: Array,  # [C, L]
    input_mask_l: Array,  # [L]
    model_horizon: int,
    forecast_horizon: int,
    device: torch.device,
    n_lags: int = 128,
    flow_cfg: SemanticFlowConfig | None = None,
    softmax_tau: float = 1.0,
    surrogate: bool = True,
    pert_cfg: PerturbationConfig | None = None,
    surr_cfg: SurrogateConfig | None = None,
    latent_batch_size: int = 32,
    surrogate_n_jobs: int | None = None,
    # v2 feature-axis switches
    channel_axis: bool = False,
    chan_cfg: ChannelFlowConfig | None = None,  # forward ref; imported lazily
    channel_softmax_tau: float | None = None,
    channel_output_aware: bool = False,
    channel_target: int | None = None,
) -> ForecastExplanation:
    """
    End-to-end interpretability pipeline:
      1) Forecast (for extension into the future)
      2) Latent trajectory extraction over [history + forecast]
      3) Semantic flow magnitudes
      4) Lag x Horizon attribution matrix + horizon-wise softmax
      5) Optional horizon-specific local surrogates with structure-preserving perturbations

    When ``channel_axis=True`` (and the input has C > 1 channels) we additionally
    run the v2 feature-axis stages: per-channel flow decomposition, joint
    lag x channel x horizon attribution, and -- for the Shapley variant --
    the cross-channel coupling matrix.

    Pass ``chan_cfg`` to control the per-channel flow estimator. When
    ``chan_cfg`` is None we default to ``ChannelFlowConfig()`` (Jacobian
    variant), which is the fast O(C) estimator suitable for routine use.
    Use ``ChannelFlowConfig(method="shapley")`` when you need the axiomatic
    decomposition or the coupling matrix for a paper figure.
    """
    flow_cfg = flow_cfg or SemanticFlowConfig()
    pert_cfg = pert_cfg or PerturbationConfig()
    if surr_cfg is None:
        # Preserve existing behavior: cap surrogate lag features to 64 by default.
        # Emit a warning so callers who set n_lags>64 aren't surprised.
        if int(n_lags) > 64:
            warnings.warn(
                f"explain_forecast default SurrogateConfig caps n_lags to 64 "
                f"(got n_lags={int(n_lags)}). Pass surr_cfg=SurrogateConfig(n_lags=...) "
                f"to override.",
                UserWarning,
                stacklevel=2,
            )
        surr_cfg = SurrogateConfig(n_lags=min(64, int(n_lags)))
    K = int(n_lags)

    x0 = np.asarray(x_context_ct, dtype=np.float32)
    if x0.ndim != 2:
        raise ValueError(f"Expected x_context [C,L], got {x0.shape}")
    C, L = x0.shape
    if L < 2:
        raise ValueError(f"x_context must have length >= 2 for lag attribution, got L={L}")
    max_lags = L - 1
    if K <= 0 or max_lags < K:
        raise ValueError(f"n_lags must be in [1,{max_lags}], got {K}")

    mask = np.asarray(input_mask_l, dtype=np.int64)
    if mask.shape != (L,):
        raise ValueError(f"Expected input_mask [L], got {mask.shape}")

    # --- 1) baseline forecast ---
    x_t = torch.from_numpy(x0[None, :, :]).to(device=device, dtype=torch.float32)
    m_t = torch.from_numpy(mask[None, :]).to(device=device, dtype=torch.long)
    base = _forecast_autoregressive(
        model,
        x_t,
        m_t,
        model_horizon=int(model_horizon),
        target_horizon=int(forecast_horizon),
    )[0]  # [C,H]

    # --- 2) build extended series and latent trajectory ---
    series_ext = np.concatenate([x0, base], axis=1)  # [C, L+H]
    mask_ext = np.concatenate([mask, np.ones((int(forecast_horizon),), dtype=np.int64)], axis=0)  # [L+H]
    Z = extract_latent_trajectory(
        model,
        series_ext,
        seq_len=L,
        input_mask_t=mask_ext,
        device=device,
        batch_size=latent_batch_size,
    )  # [T,D]

    # --- 3) semantic flow magnitudes ---
    m = compute_semantic_flow_magnitudes(Z, flow_cfg)  # [T-1]

    # --- 3b) segment quality: forecast vs history flow (OOD volatility) ---
    flow_ratio, flow_var_ratio = _flow_segment_ratios(m, context_len=L, forecast_horizon=int(forecast_horizon))
    curvature_ratio = _curvature_segment_ratio(Z, context_len=L, forecast_horizon=int(forecast_horizon))
    maha_ratio = _latent_diag_mahalanobis_ratio(Z, context_len=L, forecast_horizon=int(forecast_horizon))

    # --- 4) lag×horizon attribution ---
    # t corresponds to last observed index in the context window.
    t_index = L - 1
    scores, attrib = lag_horizon_attribution(
        m,
        t_index=t_index,
        n_lags=K,
        horizon=int(forecast_horizon),
        softmax_tau=float(softmax_tau),
    )

    # --- 5) horizon-specific surrogates ---
    coef = None
    intercept = None
    feature_layout = None
    if surrogate:
        coef, intercept, feature_layout = fit_horizon_surrogates(
            model,
            x_context_ct=x0,
            input_mask_l=mask,
            model_horizon=int(model_horizon),
            forecast_horizon=int(forecast_horizon),
            device=device,
            pert_cfg=pert_cfg,
            surr_cfg=surr_cfg,
            n_jobs=surrogate_n_jobs,
        )

    # --- 6) v2 feature-axis stages (optional) ---
    per_chan_flow = None
    lag_chan_scores = None
    lag_chan_attrib = None
    chan_horizon_attrib = None
    coupling_mat = None
    chan_method = None
    chan_resid_mean = None
    chan_resid_p95 = None
    coupling_norm = None
    if channel_axis:
        from channel_flow import (  # local import avoids circular dependency at module load
            ChannelFlowConfig,
            channel_horizon_marginal,
            compute_per_channel_flow,
            lag_channel_horizon_attribution,
            make_target_forecast_value_fn,
        )

        chan_cfg_eff = chan_cfg if chan_cfg is not None else ChannelFlowConfig()
        if channel_output_aware and chan_cfg_eff.method == "jacobian":
            if channel_target is None:
                raise ValueError(
                    "channel_output_aware=True requires channel_target (the forecast "
                    "channel whose prediction the attribution should explain)."
                )
            chan_horizon_attrib = compute_target_channel_input_jacobian_importance(
                model,
                x_context_ct=x0,
                input_mask_l=mask,
                device=device,
                target_channel=int(channel_target),
                model_horizon=int(model_horizon),
                forecast_horizon=int(forecast_horizon),
            )
            chan_method = "target_input_jacobian"
        else:
            chan_value_fn = None
            if channel_output_aware:
                if channel_target is None:
                    raise ValueError(
                        "channel_output_aware=True requires channel_target (the forecast "
                        "channel whose prediction the attribution should explain)."
                    )
                chan_value_fn = make_target_forecast_value_fn(
                    target_channel=int(channel_target),
                    model_horizon=int(model_horizon),
                    forecast_horizon=int(forecast_horizon),
                )
            chan_protect = (int(channel_target),) if channel_output_aware and channel_target is not None else None
            report = compute_per_channel_flow(
                model,
                series_ext,
                seq_len=L,
                input_mask_t=mask_ext,
                device=device,
                cfg=chan_cfg_eff,
                value_fn=chan_value_fn,
                protect_channels=chan_protect,
            )
            per_chan_flow = report.per_channel_flow
            chan_method = report.method
            chan_resid_mean = report.residual_ratio_mean
            chan_resid_p95 = report.residual_ratio_p95
            coupling_mat = report.coupling_matrix
            coupling_norm = report.coupling_off_diag_norm

            chan_tau = float(channel_softmax_tau) if channel_softmax_tau is not None else float(softmax_tau)
            lag_chan_scores, lag_chan_attrib = lag_channel_horizon_attribution(
                per_chan_flow,
                t_index=t_index,
                n_lags=K,
                horizon=int(forecast_horizon),
                softmax_tau=chan_tau,
                normalize="joint",
            )
            chan_horizon_attrib = channel_horizon_marginal(lag_chan_attrib)

    return ForecastExplanation(
        baseline_forecast=base,
        lag_horizon_scores=scores,
        lag_horizon_attributions=attrib,
        flow_magnitudes=m,
        latent_trajectory=Z,
        surrogate_coef=coef,
        surrogate_intercept=intercept,
        surrogate_feature_layout=feature_layout,
        flow_ratio_forecast_vs_history=flow_ratio,
        flow_variance_ratio_forecast_vs_history=flow_var_ratio,
        curvature_ratio_forecast_vs_history=curvature_ratio,
        latent_diag_mahalanobis_ratio_forecast_vs_history=maha_ratio,
        per_channel_flow=per_chan_flow,
        lag_channel_horizon_scores=lag_chan_scores,
        lag_channel_horizon_attributions=lag_chan_attrib,
        channel_horizon_attributions=chan_horizon_attrib,
        channel_coupling_matrix=coupling_mat,
        channel_flow_method=chan_method,
        channel_flow_residual_ratio_mean=chan_resid_mean,
        channel_flow_residual_ratio_p95=chan_resid_p95,
        channel_coupling_off_diag_norm=coupling_norm,
    )
