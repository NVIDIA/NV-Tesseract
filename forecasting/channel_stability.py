# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Per-channel embedding stability (v2 feature-axis stability).

This module contains the channel-axis analogue of v1's embedding stability
primitive in :mod:`interpretability`:

* :func:`compute_embedding_stability` (v1) reports a single scalar Lipschitz
  ratio per (model, series), aggregating over many input perturbation trials.
  Its v2 counterpart :func:`compute_per_channel_embedding_stability` reports
  the same ratio *per input channel*, by perturbing only one row of the input
  window at a time. Answers: "which input channels does the embedding map
  treat in a Lipschitz-safe way?"

Mathematical definitions
========================

**Per-channel embedding stability (Lipschitz-style).**
For an input context ``X_t in R^{C x L}`` and a channel ``c in {0, ..., C-1}``,
define the per-channel-perturbed input

    X_t^(c, eps) = X_t + epsilon e_c x_t^T ,    epsilon ~ N(0, (alpha sigma_c)^2 I_L),

i.e. additive Gaussian noise with scale ``alpha * sigma_c`` (where
``sigma_c = std(X_t[c, observed])`` and ``alpha = noise_scale``) restricted to
row ``c`` of the input. Only row ``c`` differs from ``X_t``, so the Frobenius
norm reduces to a row-vector L2 norm:

    || X_t^(c, eps) - X_t ||_F  =  || epsilon ||_2  =  || x_t'[c, :] - x_t[c, :] ||_2 .

The per-channel Lipschitz-style ratio is

    R_t^(c) = || embed(X_t^(c, eps)) - embed(X_t) ||_2 / ( || X_t^(c, eps) - X_t ||_F + eta ),

where ``eta = 1e-12`` is a numerical floor on the denominator. We aggregate
over ``n_trials`` perturbation trials per channel and per time index, then
report mean / max / p50 / p95 per channel.

**Sanity invariance.** If the embedding is channel-independent --
``embed(X) = sum_c phi_c(x_c)`` for some per-channel maps ``phi_c`` -- then
perturbing only row ``c`` changes ``embed`` by ``phi_c(x_c + n) - phi_c(x_c)``
alone. The ratio ``R_t^(c)`` therefore probes the local Lipschitz constant of
``phi_c`` in isolation. In the limit of vanishing noise scale,
``R_t^(c) -> ||J_{phi_c}||_op``. For the linear channel-independent mock model
the ratio is *exactly* ``||W^c||_op`` of that channel's projection -- this is
the invariance checked by
``test_channel_independent_model_recovers_jacobian_norm`` in the test suite.

Cost summary
============

* :func:`compute_per_channel_embedding_stability`:
  ``n_trials * C`` embed calls per time index, ``+1`` baseline embed call.
  All batched into one forward pass per time index via ``_embed_windows_batched``.
"""

from dataclasses import dataclass

import numpy as np
import torch
from channel_flow import _embed_windows_batched
from interpretability import (
    Array,
    ForecastModel,
    _rolling_window_sources,
    extract_latent_trajectory,
)

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerChannelEmbeddingStabilityReport:
    """Per-channel result of the empirical embedding stability test.

    Shapes (with ``C`` input channels):

    * ``lip_ratio_mean``, ``lip_ratio_max``, ``lip_ratio_p50``, ``lip_ratio_p95``:
      ``[C]`` -- aggregates over the ratios collected for that channel.
    * ``n_trials_per_channel``: ``[C]`` -- number of perturbation trials that
      contributed a finite ratio for that channel (may differ across channels
      if some perturbations were skipped, e.g. a fully-zero noise draw on a
      constant channel).
    * ``n_unique_windows``: ``int`` -- count of distinct time indices that
      contributed to at least one channel's ratio set.
    * ``step_delta_norm_mean``: ``float`` -- mean ``||Z_{t+1} - Z_t||`` on the
      unperturbed trajectory (a v1-style reference scale; identical to the
      scalar returned by :func:`compute_embedding_stability`).
    """

    lip_ratio_mean: Array  # [C]
    lip_ratio_max: Array  # [C]
    lip_ratio_p50: Array  # [C]
    lip_ratio_p95: Array  # [C]
    n_trials_per_channel: Array  # [C], int
    n_unique_windows: int
    step_delta_norm_mean: float
    n_channels: int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _select_time_indices(
    *,
    seq_len: int,
    T: int,
    n_trials: int,
    rng: np.random.Generator,
    time_indices: Array | None = None,
) -> np.ndarray:
    """Mirror the sampling discipline used in v1 :func:`compute_embedding_stability`.

    Prefers indices ``[seq_len - 1, T)`` (full-window region) and repeats with
    a shuffle if the candidate pool is smaller than ``n_trials``. Different
    perturbation seeds across the repeats keep the trials statistically
    distinct even when the underlying windows are reused.
    """
    if time_indices is not None:
        return np.asarray(time_indices, dtype=np.int64).reshape(-1)
    candidates = np.arange(seq_len - 1, T, dtype=np.int64)
    if len(candidates) == 0:
        candidates = np.arange(T, dtype=np.int64)
    if len(candidates) >= n_trials:
        return rng.choice(candidates, size=n_trials, replace=False)
    n_repeats = int(np.ceil(n_trials / len(candidates)))
    out = np.tile(candidates, n_repeats)[:n_trials]
    rng.shuffle(out)
    return out


def _per_channel_std(window_cl: Array, win_mask: Array) -> Array:
    """Per-channel std of ``window_cl[:, observed]``.

    Channels with std ``< 1e-8`` (numerically constant on the observed slice)
    fall back to ``1.0`` to match the v1 convention -- they get the *raw*
    noise scale and we measure ``embed`` sensitivity to that absolute scale
    rather than to a normalised-by-zero scale.
    """
    obs = np.asarray(win_mask, dtype=bool)
    if not np.any(obs):
        return np.ones((window_cl.shape[0],), dtype=np.float32)
    std = np.std(window_cl[:, obs], axis=1).astype(np.float32, copy=False)
    std = np.where(std < 1e-8, np.float32(1.0), std)
    return std


# ---------------------------------------------------------------------------
# Per-channel embedding stability
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_per_channel_embedding_stability(
    model: ForecastModel,
    series_ct: Array,  # [C, T]
    *,
    seq_len: int,
    input_mask_t: Array | None = None,
    device: torch.device,
    n_trials: int = 30,
    noise_scale: float = 0.01,
    time_indices: Array | None = None,
    batch_size: int = 16,
    trajectory_batch_size: int = 32,
    seed: int = 42,
) -> PerChannelEmbeddingStabilityReport:
    """Channel-wise Lipschitz-style stability test.

    For each channel ``c``, perturbs only row ``c`` of the input window and
    measures the ratio ``|| embed(X') - embed(X) ||_2 / || X' - X ||_F``
    averaged over ``n_trials`` perturbation trials. The result is the v2
    feature-axis analogue of :func:`compute_embedding_stability` (which
    perturbs all channels jointly and returns a single scalar).

    Args:
        model: forecast model exposing the ``embed`` interface.
        series_ct: ``[C, T]`` series; only the trailing time indices matter.
        seq_len: context length ``L`` of the model.
        input_mask_t: optional ``[T]`` observed-mask for the series.
        device: torch device for the embed calls.
        n_trials: number of perturbation trials *per channel and per time
            index*. Total perturbation embeds = ``n_trials * |time_indices| * C``.
        noise_scale: noise amplitude as a fraction of per-channel std.
        time_indices: optional explicit ``[T_idx]`` time-index list. If None,
            samples ``n_trials`` indices from the full-window region.
        batch_size: chunk size for batched embed of perturbation windows.
        trajectory_batch_size: chunk size for the reference-trajectory embed
            (separate budget because the reference is computed once over the
            full series).
        seed: RNG seed for the per-channel noise.

    Returns:
        :class:`PerChannelEmbeddingStabilityReport` with shape-``[C]`` summaries.
    """
    xw_view, m_view, T = _rolling_window_sources(series_ct, seq_len=seq_len, input_mask_t=input_mask_t)
    if T < 2:
        raise ValueError(f"Need at least 2 time steps for per-channel stability test, got T={T}")
    C = int(np.asarray(series_ct).shape[0])
    L = int(seq_len)

    rng = np.random.default_rng(int(seed))
    sampled_indices = _select_time_indices(seq_len=L, T=T, n_trials=int(n_trials), rng=rng, time_indices=time_indices)
    sampled_indices = np.unique(np.clip(sampled_indices, 0, T - 1))
    if sampled_indices.size == 0:
        raise ValueError("No valid time indices to sample for per-channel stability.")

    # Collected ratios are gathered into per-channel lists across all
    # (time_index, trial) draws so the aggregation is purely vectorised at the
    # end. Each accepted draw contributes one float per channel.
    per_channel_ratios: list[list[float]] = [[] for _ in range(C)]
    n_unique = 0

    # Number of trials per time index: keep at least 1; rotate noise seeds so
    # the trials are not identical across time indices.
    trials_per_index = max(1, int(np.ceil(n_trials / sampled_indices.size)))

    for t_idx, t in enumerate(sampled_indices):
        win = np.asarray(xw_view[int(t)], dtype=np.float32)  # [C, L]
        mask = np.asarray(m_view[int(t)], dtype=np.int64)  # [L]
        obs_mask = mask.astype(bool)
        if not np.any(obs_mask):
            continue

        std_per_chan = _per_channel_std(win, mask)  # [C]
        win_mask_f = mask.astype(np.float32, copy=False)  # [L]

        # Build a [1 + C * trials_per_index, C, L] probe stack:
        # slot 0 = baseline; slots 1..C, C+1..2C, ... = trial-major,
        # channel-minor perturbations (channel c of trial r at slot 1 + r*C + c).
        n_perturbs = C * trials_per_index
        stack = np.empty((1 + n_perturbs, C, L), dtype=np.float32)
        stack[0] = win

        # Draw noise once per (trial, channel) at scale alpha * std_c restricted
        # to observed positions. Padded positions contribute 0 to both
        # numerator (embed sees the same masked input) and denominator (we
        # measure Frobenius norm of x' - x, which is zero in masked positions).
        noise_unit = rng.standard_normal((trials_per_index, C, L)).astype(np.float32)
        # Apply the observation mask (zero perturbation where mask=0).
        noise_unit = noise_unit * win_mask_f[None, None, :]
        # Per-channel scale.
        noise_unit *= (float(noise_scale) * std_per_chan)[None, :, None]  # [r, C, L]

        # Frobenius norm equals the row-c L2 norm of the per-channel noise.
        # Drop trials where every channel ended up with a numerically zero
        # noise vector (e.g. constant + tiny noise_scale on a fully-padded
        # window) -- they would inject inf into the ratio aggregation.
        dx_norm = np.linalg.norm(noise_unit.reshape(trials_per_index * C, L), ord=2, axis=1).astype(np.float32)
        dx_norm = dx_norm.reshape(trials_per_index, C)

        # Materialise perturbed windows: copy ``win`` then overwrite row c with
        # ``win[c] + noise[r, c, :]``.
        perturbed = np.broadcast_to(win[None, None, :, :], (trials_per_index, C, C, L)).copy()
        chan_idx = np.arange(C)
        perturbed[:, chan_idx, chan_idx, :] = win[chan_idx, :][None, :, :] + noise_unit  # [r, C, L]

        # Flatten trial/channel dims into the leading "slot" axis.
        stack[1:] = perturbed.reshape(n_perturbs, C, L)
        masks_stack = np.broadcast_to(mask[None, :], (1 + n_perturbs, L)).copy()

        z = _embed_windows_batched(
            model,
            stack,
            masks_stack,
            device=device,
            batch_size=int(batch_size),
        )  # [1 + n_perturbs, D]
        z_base = z[0]  # [D]
        z_pert = z[1:].reshape(trials_per_index, C, -1)  # [r, C, D]

        dz_norm = np.linalg.norm(z_pert - z_base[None, None, :], ord=2, axis=2).astype(np.float32)  # [r, C]

        # Aggregate per-channel ratios. Skip (trial, channel) cells where the
        # noise vector was numerically zero.
        valid_mask = dx_norm > 1e-12
        eps = np.float32(1e-12)
        ratios = dz_norm / (dx_norm + eps)
        for c in range(C):
            valid_c = valid_mask[:, c]
            if not np.any(valid_c):
                continue
            for r_val in ratios[valid_c, c]:
                per_channel_ratios[c].append(float(r_val))
        n_unique += 1

    # Reference scale (matches v1 step_delta_norm_mean) - computed once over
    # the full series. Failure-tolerant: if the trajectory is degenerate, fall
    # back to NaN.
    try:
        Z_full = extract_latent_trajectory(
            model,
            series_ct,
            seq_len=L,
            input_mask_t=input_mask_t,
            device=device,
            batch_size=int(trajectory_batch_size),
        )
        if Z_full.shape[0] >= 2:
            step_deltas = np.linalg.norm(Z_full[1:] - Z_full[:-1], ord=2, axis=1)
            step_delta_mean = float(np.nanmean(step_deltas))
        else:
            step_delta_mean = float("nan")
    except Exception:
        step_delta_mean = float("nan")

    lip_mean = np.full((C,), np.nan, dtype=np.float32)
    lip_max = np.full((C,), np.nan, dtype=np.float32)
    lip_p50 = np.full((C,), np.nan, dtype=np.float32)
    lip_p95 = np.full((C,), np.nan, dtype=np.float32)
    n_per_chan = np.zeros((C,), dtype=np.int64)
    for c in range(C):
        if not per_channel_ratios[c]:
            continue
        arr = np.asarray(per_channel_ratios[c], dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        lip_mean[c] = float(np.mean(finite))
        lip_max[c] = float(np.max(finite))
        lip_p50[c] = float(np.percentile(finite, 50))
        lip_p95[c] = float(np.percentile(finite, 95))
        n_per_chan[c] = int(finite.size)

    return PerChannelEmbeddingStabilityReport(
        lip_ratio_mean=lip_mean,
        lip_ratio_max=lip_max,
        lip_ratio_p50=lip_p50,
        lip_ratio_p95=lip_p95,
        n_trials_per_channel=n_per_chan,
        n_unique_windows=int(n_unique),
        step_delta_norm_mean=step_delta_mean,
        n_channels=int(C),
    )


__all__ = [
    "PerChannelEmbeddingStabilityReport",
    "compute_per_channel_embedding_stability",
]
