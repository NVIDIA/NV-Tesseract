# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Per-channel Jacobian-flow decomposition for feature-axis interpretability.

This module extends the temporal-axis framework in :mod:`interpretability` to
multivariate inputs. It decomposes each latent trajectory step into per-channel
contributions with a directional central-secant approximation of the
channel-block Jacobian.

The resulting per-channel flow feeds :func:`lag_channel_horizon_attribution`,
which produces a joint `[K, C, H]` lag-by-channel-by-horizon attribution
tensor. :func:`channel_horizon_marginal` reduces that tensor to the `[C, H]`
feature-axis attribution used by the SDK report.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

from interpretability import (
    Array,
    ForecastModel,
    _embed_batch,
    _rolling_window_sources,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelFlowConfig:
    """Configuration for per-channel Jacobian-flow decomposition.

    Args:
      jacobian_secant_scale: Multiplicative scale applied to the per-channel
        input increment for the central-difference probe.
      time_indices: Optional trajectory transition indices to evaluate. By
        default every consecutive transition is processed.
      batch_size: Number of probe windows evaluated per forward pass.
      transition_batch: Number of transitions stacked into one embed call
        before the probe windows are chunked by `batch_size`.
    """

    jacobian_secant_scale: float = 0.5
    time_indices: Sequence[int] | None = None
    batch_size: int = 32
    transition_batch: int = 1


@dataclass(frozen=True)
class ChannelFlowReport:
    """Per-channel Jacobian-flow decomposition outputs."""

    method: str
    per_channel_flow: Array
    flow_total: Array
    residual_ratio_per_step: Array | None = None
    residual_ratio_mean: float | None = None
    residual_ratio_p95: float | None = None
    n_time_steps: int = 0
    n_channels: int = 0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_step_indices(
    *,
    time_indices: Sequence[int] | None,
    n_windows: int,
) -> np.ndarray:
    """Return the array of trajectory transition indices to evaluate.

    A "transition" at index ``tau`` is the pair ``(tau, tau+1)``; valid range
    is ``[0, n_windows - 2]``.
    """
    if n_windows < 2:
        raise ValueError(f"Need at least 2 rolling windows for flow, got {n_windows}")
    if time_indices is None:
        return np.arange(n_windows - 1, dtype=np.int64)
    arr = np.asarray(time_indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError("`time_indices` must contain at least one transition index.")
    invalid = (arr < 0) | (arr >= n_windows - 1)
    if np.any(invalid):
        raise ValueError(f"`time_indices` contains values outside [0, {n_windows - 2}]: {arr[invalid].tolist()}")
    return np.unique(arr)


class _ChannelFlowEmbedDP(torch.nn.Module):
    """Routes forward() to model.embed() so DataParallel shards the
    probe-embedding batches across GPUs in channel-flow Jacobians.
    DataParallel only intercepts forward(), not arbitrary named methods.
    """

    def __init__(self, model: "ForecastModel") -> None:
        super().__init__()
        self._m = model

    def forward(self, x_enc: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        out = self._m.embed(x_enc=x_enc, input_mask=input_mask)
        return out.embeddings if hasattr(out, "embeddings") else out


def _embed_windows_batched(
    model: ForecastModel,
    windows: Array,
    masks: Array,
    *,
    device: torch.device,
    batch_size: int,
) -> Array:
    """Batched embedding of a stack of ``[N, C, L]`` windows.

    NaN/inf in the model's embedding output are replaced with zero, mirroring
    the sanitization that ``interpretability.extract_latent_trajectory``
    applies to the v1 path. Without this, non-finite activations from
    out-of-distribution forecast-extension windows can propagate
    into ``per_channel_flow`` and break the v1 reduction (Prop 1) on real
    weights even though the in-context portion is clean.
    """
    n = int(windows.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    out_chunks: list[Array] = []
    _is_dp = isinstance(model, torch.nn.DataParallel)
    for i in range(0, n, batch_size):
        x = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).to(device=device, dtype=torch.float32)
        m = torch.from_numpy(np.ascontiguousarray(masks[i : i + batch_size])).to(device=device, dtype=torch.long)
        if _is_dp:
            out_chunks.append(model(x, m).detach().cpu().numpy().astype(np.float32))
        else:
            out_chunks.append(_embed_batch(model, x, m))
    z = np.concatenate(out_chunks, axis=0)
    if np.any(~np.isfinite(z)):
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return z


ValueFn = Callable[..., Array]


# ---------------------------------------------------------------------------
# Jacobian-flow
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_per_channel_flow_jacobian(
    model: ForecastModel,
    series_ct: Array,  # [C, T]
    *,
    seq_len: int,
    input_mask_t: Array | None = None,
    device: torch.device,
    cfg: ChannelFlowConfig | None = None,
    value_fn: ValueFn | None = None,
) -> ChannelFlowReport:
    """First-order per-channel decomposition of latent flow.

    For each transition ``tau -> tau+1`` we approximate

        Delta Z_tau^(c) ~= 0.5 * [ embed(X_tau + s * bump_c) - embed(X_tau - s * bump_c) ]
                          / s

    where ``bump_c`` is the input-tensor with row ``c`` set to the per-channel
    increment ``Delta x_tau^(c) = X_{tau+1}[c] - X_tau[c]`` and zeros elsewhere,
    and ``s = cfg.jacobian_secant_scale``. The per-channel flow magnitude is
    ``phi_tau(c) = || Delta Z_tau^(c) ||_2``.

    The residual ``r_tau = Delta Z_tau - sum_c Delta Z_tau^(c)`` is used to
    populate the trust ratio diagnostic.

    Cost: For each of ``T - 1`` transitions we evaluate ``2C + 1`` embeddings
    (``C`` "+s" probes, ``C`` "-s" probes, plus ``Z_{tau+1}`` -- ``Z_tau`` is
    re-used from the previous iteration). We batch over channels so this is
    one batched forward of size ``2C + 1`` per transition.
    """
    cfg = cfg or ChannelFlowConfig()
    # When multiple GPUs are available and no custom value_fn is set,
    # wrap the model so DataParallel splits the (2+2*C)-probe embedding
    # batch across devices, cutting wall-clock roughly in half per
    # additional GPU without any change to the transition loop logic.
    _n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if _n_gpus > 1 and value_fn is None:
        import logging as _log
        _log.getLogger(__name__).info(
            "channel flow Jacobian: using %d GPUs via DataParallel", _n_gpus
        )
        model = torch.nn.DataParallel(_ChannelFlowEmbedDP(model))
    vf = value_fn if value_fn is not None else _embed_windows_batched

    xw_view, m_view, T = _rolling_window_sources(series_ct, seq_len=seq_len, input_mask_t=input_mask_t)
    if T < 2:
        raise ValueError(f"Need at least 2 time steps for flow decomposition, got T={T}")
    transitions = _resolve_step_indices(time_indices=cfg.time_indices, n_windows=T)

    C = int(np.asarray(series_ct).shape[0])
    L = int(seq_len)
    s = float(cfg.jacobian_secant_scale)
    if s <= 0:
        raise ValueError(f"jacobian_secant_scale must be > 0, got {s}")

    phi = np.full((T - 1, C), np.nan, dtype=np.float32)
    flow_total = np.full((T - 1,), np.nan, dtype=np.float32)
    residual = np.full((T - 1,), np.nan, dtype=np.float32)

    # Number of "probe windows" per transition: Z_tau, Z_{tau+1}, +/- bumps.
    probes_per_trans = 2 + 2 * C
    trans_batch = max(1, int(cfg.transition_batch))

    for chunk_start in range(0, len(transitions), trans_batch):
        chunk = transitions[chunk_start : chunk_start + trans_batch]
        m_chunk = len(chunk)

        # Pre-fetch all windows referenced by this chunk in one slice each
        # (avoids materialising the rolling-window view C+1 times per
        # transition like the per-tau path used to do).
        starts = chunk.astype(np.int64)
        win_a = np.asarray(xw_view[starts], dtype=np.float32)  # [m, C, L]
        win_b = np.asarray(xw_view[starts + 1], dtype=np.float32)  # [m, C, L]
        mask_a = np.asarray(m_view[starts], dtype=np.int64)  # [m, L]
        mask_b = np.asarray(m_view[starts + 1], dtype=np.int64)  # [m, L]
        delta_cl = (win_b - win_a).astype(np.float32, copy=False)  # [m, C, L]

        # Build the [m * probes_per_trans, C, L] probe stack in one shot.
        # Layout per transition is [a, b, +probe_0..+probe_{C-1}, -probe_0..].
        stack = np.empty((m_chunk, probes_per_trans, C, L), dtype=np.float32)
        stack[:, 0] = win_a
        stack[:, 1] = win_b

        # Bumps share the original window's content except on the bumped
        # channel row, which is shifted by +/- s * delta_x. We construct
        # them by tiling x_a across C "probes" and then overwriting the
        # diagonal (c, c) row with the per-channel shifted value.
        plus = np.broadcast_to(win_a[:, None, :, :], (m_chunk, C, C, L)).copy()
        minus = plus.copy()
        # diag indexing: for each transition m and probe c, row c is
        # x_a[m, c] +/- s * delta_cl[m, c].
        diag_idx = np.arange(C)
        plus[:, diag_idx, diag_idx, :] = win_a[:, diag_idx, :] + s * delta_cl[:, diag_idx, :]
        minus[:, diag_idx, diag_idx, :] = win_a[:, diag_idx, :] - s * delta_cl[:, diag_idx, :]
        stack[:, 2 : 2 + C] = plus
        stack[:, 2 + C :] = minus

        # Mask discipline: position 1 (Z_{tau+1}) uses window B's mask;
        # everything else uses window A's mask. This matches the per-window
        # mask discipline used by ``extract_latent_trajectory`` and is what
        # makes the v1 reduction (Prop 1) hold exactly when C=1.
        masks_full = np.broadcast_to(mask_a[:, None, :], (m_chunk, probes_per_trans, L)).copy()
        masks_full[:, 1] = mask_b

        flat_stack = stack.reshape(m_chunk * probes_per_trans, C, L)
        flat_masks = masks_full.reshape(m_chunk * probes_per_trans, L)

        z_flat = vf(
            model,
            flat_stack,
            flat_masks,
            device=device,
            batch_size=int(cfg.batch_size),
        )  # [m * probes_per_trans, Dv]
        D = int(z_flat.shape[1])
        z = z_flat.reshape(m_chunk, probes_per_trans, D)

        z_tau = z[:, 0, :]  # [m, D]
        z_tau_plus = z[:, 1, :]  # [m, D]
        delta_z = z_tau_plus - z_tau  # [m, D]

        # Per-channel directional derivative, central-secant form:
        # (z_+ - z_-) / (2s). Reshape pulls out [m, C, D].
        delta_z_per_chan = (z[:, 2 : 2 + C, :] - z[:, 2 + C :, :]) / (2.0 * s)  # [m, C, D]

        # Per-tau scalar norms. We deliberately call ``np.linalg.norm`` once
        # per row (instead of an axis-aware norm across the m-batch) to
        # match the float32 summation order of the original per-tau loop
        # bit-for-bit -- otherwise BLAS picks a slightly different reduction
        # tree for [m, D] vs [D] inputs and the scalar flow_total / residual
        # drift by ~1 ULP. The scalar work is negligible compared to the
        # (already-vectorised) embed pass that produced ``z``.
        for k, tau in enumerate(chunk):
            dz_k = delta_z[k]  # [D]
            dzpc_k = delta_z_per_chan[k]  # [C, D]
            phi[int(tau)] = np.linalg.norm(dzpc_k, ord=2, axis=1).astype(np.float32)
            flow_total[int(tau)] = float(np.linalg.norm(dz_k, ord=2))
            denom_k = float(np.linalg.norm(dz_k, ord=2)) + 1e-12
            r_k = dz_k - dzpc_k.sum(axis=0)
            residual[int(tau)] = float(np.linalg.norm(r_k, ord=2)) / denom_k

    finite = np.isfinite(residual)
    res_mean = float(np.nanmean(residual[finite])) if np.any(finite) else None
    res_p95 = float(np.nanpercentile(residual[finite], 95)) if np.any(finite) else None

    return ChannelFlowReport(
        method="jacobian",
        per_channel_flow=phi,
        flow_total=flow_total,
        residual_ratio_per_step=residual,
        residual_ratio_mean=res_mean,
        residual_ratio_p95=res_p95,
        n_time_steps=int(T - 1),
        n_channels=C,
    )


def compute_per_channel_flow(
    model: ForecastModel,
    series_ct: Array,
    *,
    seq_len: int,
    input_mask_t: Array | None = None,
    device: torch.device,
    cfg: ChannelFlowConfig | None = None,
    value_fn: ValueFn | None = None,
) -> ChannelFlowReport:
    """Compute the configured per-channel Jacobian-flow report."""
    return compute_per_channel_flow_jacobian(
        model,
        series_ct,
        seq_len=seq_len,
        input_mask_t=input_mask_t,
        device=device,
        cfg=cfg,
        value_fn=value_fn,
    )


# ---------------------------------------------------------------------------
# Lag x Channel x Horizon attribution
# ---------------------------------------------------------------------------


def lag_channel_horizon_attribution(
    per_channel_flow: Array,  # [T-1, C]
    *,
    t_index: int,
    n_lags: int,
    horizon: int,
    softmax_tau: float = 1.0,
    horizon_kernel: Literal["exp", "none"] = "exp",
    kernel_min_scale: float = 4.0,
    kernel_max_scale: float | None = None,
    normalize: Literal["joint", "per_channel", "none"] = "joint",
) -> tuple[Array, Array]:
    """Joint lag x channel x horizon attribution.

    This is the natural generalisation of :func:`lag_horizon_attribution` to
    a per-channel flow input. The lag-horizon kernel ``w_h(a) = exp(-a / s_h)``
    is reused unchanged from v1; the only difference is that the cumulative
    kernel-weighted sum is taken per channel, producing a ``[K, C, H]``
    score tensor instead of v1's ``[K, H]``.

    The ``normalize`` flag controls how the softmax over the score tensor is
    taken:

    * ``"joint"`` (default): softmax over the *joint* ``(lag, channel)`` axis
      for each horizon. Marginalising over channels recovers v1's
      lag-horizon attribution as a sanity check; marginalising over lags
      gives a per-channel-per-horizon attribution.
    * ``"per_channel"``: softmax over the lag axis, separately for each
      ``(c, h)`` pair. Useful when the absolute scale of each channel's
      contribution is known to vary widely.
    * ``"none"``: return raw scores without softmax normalization.

    Returns:
      scores: ``[K, C, H]``
      attributions: ``[K, C, H]``
    """
    flow_tc = np.asarray(per_channel_flow, dtype=np.float32)
    if flow_tc.ndim != 2:
        raise ValueError(f"Expected per_channel_flow shape [T-1, C], got {flow_tc.shape}")
    flow_tc = np.where(np.isfinite(flow_tc), flow_tc, 0.0).astype(np.float32)

    Tm1, C = flow_tc.shape
    K = int(n_lags)
    H = int(horizon)
    if K <= 0 or H <= 0:
        raise ValueError("n_lags and horizon must be positive")
    if not (1 <= int(t_index) <= Tm1):
        raise ValueError(f"t_index must be in [1, {Tm1}], got {t_index}")
    if int(t_index) < K:
        raise ValueError(f"n_lags must be <= t_index={int(t_index)}, got {K}")

    # Most-recent-first slice of history flow per channel.
    hist_tail = flow_tc[max(0, int(t_index) - K) : int(t_index), :]  # [<=K, C]
    hist_rev = hist_tail[::-1, :]  # most recent first
    L = hist_rev.shape[0]

    # Horizon scales (identical to v1).
    min_s = float(kernel_min_scale)
    max_s = float(kernel_max_scale) if kernel_max_scale is not None else float(K)
    if min_s <= 0 or max_s <= 0:
        raise ValueError("kernel scales must be positive")
    max_s = max(max_s, min_s)
    if H == 1:
        scales_h = np.array([min_s], dtype=np.float32)
    else:
        u_h = np.arange(H, dtype=np.float32) / float(H - 1)
        scales_h = (min_s + u_h * (max_s - min_s)).astype(np.float32)

    scores = np.zeros((K, C, H), dtype=np.float32)
    if horizon_kernel == "none":
        csum = np.cumsum(hist_rev, axis=0, dtype=np.float32)  # [L, C]
        scores[:L, :, :] = csum[:, :, None]
    elif horizon_kernel == "exp":
        ages_l = np.arange(L, dtype=np.float32)[:, None]  # [L, 1]
        target_bytes = 64 * 1024 * 1024
        bytes_per_h = max(1, L * C * 4)
        chunk = max(1, min(H, target_bytes // bytes_per_h))
        for h0 in range(0, H, chunk):
            h1 = min(H, h0 + chunk)
            s = np.maximum(scales_h[h0:h1][None, :], np.float32(1e-6))
            w_lh = np.exp(-ages_l / s).astype(np.float32)  # [L, chunk]
            # hist_rev: [L, C]; w_lh: [L, chunk] -> contribution per (l, c, h)
            hw_lhc = hist_rev[:, :, None] * w_lh[:, None, :]  # [L, C, chunk]
            csum = np.cumsum(hw_lhc, axis=0, dtype=np.float32)  # [L, C, chunk]
            scores[:L, :, h0:h1] = csum
    else:
        raise ValueError(f"Unsupported horizon_kernel={horizon_kernel!r}")

    tau = max(float(softmax_tau), 1e-6)
    if normalize == "none":
        return scores, scores.copy()
    if normalize == "per_channel":
        attrib = np.empty_like(scores)
        for c in range(C):
            for h in range(H):
                col = scores[:, c, h]
                if not np.any(col):
                    attrib[:, c, h] = 1.0 / float(K)
                else:
                    e = np.exp((col - col.max()) / tau)
                    s = e.sum()
                    attrib[:, c, h] = e / s if s > 0 else 1.0 / float(K)
        return scores, attrib
    if normalize != "joint":
        raise ValueError(f"Unsupported normalize={normalize!r}")

    # Joint softmax over (lag, channel) per horizon.
    attrib = np.empty_like(scores)
    flat = scores.reshape(K * C, H)
    for h in range(H):
        col = flat[:, h]
        if not np.any(col):
            attrib[:, :, h] = 1.0 / float(K * C)
            continue
        e = np.exp((col - col.max()) / tau)
        s = e.sum()
        if s <= 0:
            attrib[:, :, h] = 1.0 / float(K * C)
        else:
            attrib[:, :, h] = (e / s).reshape(K, C)
    return scores, attrib


def channel_horizon_marginal(attribution_kch: Array) -> Array:
    """Sum a ``[K, C, H]`` attribution tensor over the lag axis."""
    return np.asarray(attribution_kch, dtype=np.float32).sum(axis=0).astype(np.float32)


def lag_horizon_marginal(attribution_kch: Array) -> Array:
    """Sum a ``[K, C, H]`` attribution tensor over the channel axis.

    Matches the v1 lag-horizon attribution shape ``[K, H]`` so it can be
    plugged directly into :func:`evaluate_lag_faithfulness`.
    """
    return np.asarray(attribution_kch, dtype=np.float32).sum(axis=1).astype(np.float32)
