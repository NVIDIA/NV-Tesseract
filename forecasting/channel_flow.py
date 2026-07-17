# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Per-channel semantic-flow decomposition (v2 feature-axis interpretability).

This module extends the temporal-axis framework in :mod:`interpretability` to
the *feature axis* of multivariate forecasting. It decomposes the v1 scalar
flow magnitude

    m_tau = || Z_{tau+1} - Z_tau ||_2

into a vector of per-channel contributions

    phi_tau(c),   c = 1..C

so that, modulo a higher-order residual that is bounded by the embedding's
empirical Lipschitz constant, ``sum_c phi_tau(c) ~= m_tau``. Two complementary
estimators are provided:

* **Jacobian-flow** (``compute_per_channel_flow_jacobian``):
  first-order Taylor decomposition of the latent step using a directional
  finite-difference (a.k.a. "secant") evaluation of the embedding's
  channel-block Jacobian. O(C) extra batched forward passes per time step;
  comes with an analytic trust ratio ``||residual|| / ||Delta Z||``.

* **Shapley-flow** (``compute_per_channel_flow_shapley``):
  Shapley value of the channel-coalition flow game ``v_tau(S)``, computed
  exactly when ``C <= shapley_max_exact_c`` and via KernelSHAP-style sampled
  coalitions otherwise. Inherits the SHAP axioms (efficiency, null-player,
  symmetry, linearity) on the flow value function. The same coalition
  evaluations populate the Harsanyi *channel coupling matrix*
  ``G(c, c')`` measuring genuine cross-channel interactions.

The aggregated outputs feed :func:`lag_channel_horizon_attribution`, which is
the natural generalisation of v1's :func:`lag_horizon_attribution` to a
``[K, C, H]`` joint attribution tensor (the v1 ``[K, H]`` matrix is recovered
as the channel-marginal).

Theoretical correspondence with v1:

* For ``C == 1`` and any time-invariant baseline (mean / zero / last-value),
  Shapley-flow reduces *exactly* to the v1 magnitude ``m_tau``: the empty
  coalition embeds a constant window so ``v_tau(empty) == 0`` and
  ``phi_tau(1) == v_tau({1}) == m_tau``.
* For ``C == 1``, Jacobian-flow recovers ``m_tau`` up to the second-order
  Taylor residual whose norm is bounded by the empirical Lipschitz constant
  reported by :func:`compute_embedding_stability`.

These reductions are codified in the unit tests under ``tests/``.
"""

from dataclasses import dataclass
from itertools import combinations
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
    """Configuration for per-channel flow decomposition.

    Args:
      method: Decomposition strategy.
        ``"jacobian"`` is the fast O(C) directional-finite-difference
        decomposition; ``"shapley"`` is the axiomatic O(2^C) exact (or
        sampled) Shapley decomposition.
      jacobian_secant_scale: Multiplicative scale applied to the per-channel
        input increment when computing the central-difference probe. Smaller
        values approach the analytic Jacobian-vector product (lower bias,
        higher variance from float noise); larger values measure a finite
        secant (lower variance, higher bias). The default of 0.5 corresponds
        to a symmetric ``+/- 0.5 * Delta_x`` probe centered on the window,
        which is exact for linear models and matches v1 exactly when ``C=1``
        in the linear regime.
      shapley_baseline: Channel-replacement baseline used to define the
        empty-coalition window.

        * ``"global_mean"`` (recommended): per-channel mean computed once
          over the full series passed to :func:`compute_per_channel_flow_shapley`.
          *Time-invariant*: the empty-coalition window is the same constant
          tensor at every transition, so ``v_tau(empty) == 0`` exactly and
          Shapley-flow reduces to the v1 magnitude when ``C == 1``
          (Proposition 1 in the v2 paper).
        * ``"zero"`` (also time-invariant): all-zero baseline. Matches the
          standardized-zero replacement convention used by
          ``run_framework_evaluation``. Same Proposition-1 guarantee.
        * ``"mean"``: per-window per-channel mean. *Not time-invariant* --
          the baseline changes as the window slides, so ``v_tau(empty)``
          can be nonzero and Shapley-flow measures "flow above the
          baseline drift" rather than the raw latent step. Use this when
          you want attributions that are normalised against a slowly
          drifting reference (e.g., highly nonstationary regimes).
        * ``"last_value"``: per-channel last-observed value at each
          window. Like ``"mean"``, *not time-invariant*; same caveat.
      shapley_max_exact_c: Maximum number of channels for which exact Shapley
        is computed (cost ~ ``2^C`` embeddings per time step). Above this,
        falls back to KernelSHAP with ``shapley_n_samples`` coalitions.
      shapley_n_samples: Number of sampled coalitions for KernelSHAP. ``None``
        falls back to ``2 * C + 256``.
      compute_coupling: When using the Shapley method, also compute the
        Harsanyi channel coupling matrix ``G(c, c')`` from the size <=2
        coalitions. These coalitions are computed regardless when
        ``method == "shapley"`` since they are needed for the singleton
        Shapley estimates, so the marginal cost is small.
      coupling_aggregation: How to aggregate per-step coupling matrices over
        time when reporting ``coupling_matrix``. ``"mean"`` is the default
        and matches the headline figure in the v2 paper.
      time_indices: Optional explicit list of trajectory indices at which to
        compute the per-channel flow. ``None`` (the default) processes every
        consecutive transition in the rolling-window trajectory.
      batch_size: Number of windows / coalitions evaluated per forward pass.
      transition_batch: Number of *transitions* whose probe windows are
        stacked into a single embed call before being chunked by
        ``batch_size``. Defaults to ``1`` for backward compatibility.
        Setting this > 1 amortises Python / kernel-launch overhead across
        many transitions, which is the main throughput win for the
        Jacobian variant on a single GPU. The Shapley variant honors it
        too by stacking coalition sets across transitions when the
        baseline is time-invariant ("zero" / "global_mean").
      seed: Seed for KernelSHAP coalition sampling.
    """

    method: Literal["jacobian", "shapley"] = "jacobian"
    jacobian_secant_scale: float = 0.5
    shapley_baseline: Literal["global_mean", "zero", "mean", "last_value"] = "global_mean"
    shapley_max_exact_c: int = 8
    shapley_n_samples: int | None = None
    compute_coupling: bool = True
    coupling_aggregation: Literal["mean", "sum"] = "mean"
    time_indices: Sequence[int] | None = None
    batch_size: int = 32
    transition_batch: int = 1
    seed: int = 13


@dataclass(frozen=True)
class ChannelFlowReport:
    """Per-channel flow decomposition outputs.

    Shapes:
      - per_channel_flow: ``[T-1, C]`` -- ``phi_tau(c)`` for each transition.
      - flow_total: ``[T-1]`` -- the v1 scalar magnitude ``m_tau`` for
        cross-checking. For Shapley with constant baselines this equals
        ``v_tau({1..C})``; for Jacobian it equals ``|| Delta Z_tau ||_2``.
      - residual_ratio_per_step: ``[T-1]`` -- only populated for the Jacobian
        method. ``|| Delta Z_tau - sum_c J_tau^(c) Delta x_tau^(c) || /
        || Delta Z_tau ||``. NaN means flow was numerically zero.
      - coupling_matrix: ``[C, C]`` -- only populated for the Shapley method
        when ``cfg.compute_coupling`` is True. Symmetric; the diagonal is
        zero by construction (a Harsanyi dividend of a singleton with itself
        is undefined / zero).
    """

    method: str
    per_channel_flow: Array
    flow_total: Array
    residual_ratio_per_step: Array | None = None
    residual_ratio_mean: float | None = None
    residual_ratio_p95: float | None = None
    coupling_matrix: Array | None = None
    coupling_off_diag_norm: float | None = None
    n_time_steps: int = 0
    n_channels: int = 0
    # Internal: raw coupling accumulator (sum of Harsanyi dividends across the
    # transitions covered by *this* shard) and the count of transitions that
    # contributed to it. Used by ``merge_shapley_reports`` to combine partial
    # reports from sharded execution into a single global report whose
    # coupling_matrix is bit-identical to the un-sharded computation
    # (assuming the shards are merged in transition order). Only populated
    # by the Shapley path when ``compute_coupling=True``; left ``None`` for
    # the Jacobian path and for any caller that does not opt in.
    coupling_partial_sum: Array | None = None
    coupling_partial_n: int = 0


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


def _baseline_window(
    window_cl: Array,
    *,
    kind: Literal["global_mean", "zero", "mean", "last_value"],
    global_mean_c: Array | None = None,
) -> Array:
    """Return a ``[C, L]`` baseline window used to mask channels in coalitions.

    When ``kind == "global_mean"`` the caller must provide ``global_mean_c``
    (per-channel mean computed once across the full series). The other
    baselines are derived from the per-window content of ``window_cl``.
    """
    if kind == "zero":
        return np.zeros_like(window_cl, dtype=np.float32)
    if kind == "global_mean":
        if global_mean_c is None:
            raise ValueError("`global_mean_c` is required when shapley_baseline='global_mean'.")
        means = np.asarray(global_mean_c, dtype=np.float32).reshape(-1, 1)
        if means.shape[0] != window_cl.shape[0]:
            raise ValueError(f"global_mean_c has {means.shape[0]} channels, expected {window_cl.shape[0]}")
        return np.broadcast_to(means, window_cl.shape).astype(np.float32, copy=True)
    if kind == "mean":
        means = window_cl.mean(axis=1, keepdims=True).astype(np.float32)
        return np.broadcast_to(means, window_cl.shape).astype(np.float32, copy=True)
    if kind == "last_value":
        last = window_cl[:, -1:].astype(np.float32)
        return np.broadcast_to(last, window_cl.shape).astype(np.float32, copy=True)
    raise ValueError(f"Unsupported shapley_baseline={kind!r}")


def _compose_coalition_window(
    base_window_cl: Array,
    baseline_cl: Array,
    coalition: tuple[int, ...],
) -> Array:
    """Return ``window`` with non-coalition channels overwritten by the baseline."""
    out = baseline_cl.copy()
    if coalition:
        idx = np.asarray(coalition, dtype=np.int64)
        out[idx, :] = base_window_cl[idx, :]
    return out


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
    applies to the v1 path. Without this, MOMENT's tendency to emit non-finite
    activations for out-of-distribution forecast-extension windows propagates
    into ``per_channel_flow`` and breaks the v1 reduction (Prop 1) on real
    weights even though the in-context portion is clean.
    """
    n = int(windows.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    out_chunks: list[Array] = []
    for i in range(0, n, batch_size):
        x = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).to(device=device, dtype=torch.float32)
        m = torch.from_numpy(np.ascontiguousarray(masks[i : i + batch_size])).to(device=device, dtype=torch.long)
        out_chunks.append(_embed_batch(model, x, m))
    z = np.concatenate(out_chunks, axis=0)
    if np.any(~np.isfinite(z)):
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return z


ValueFn = Callable[..., Array]


def make_target_forecast_value_fn(
    *,
    target_channel: int,
    model_horizon: int,
    forecast_horizon: int,
) -> ValueFn:
    """Build a channel-flow value function over the target forecast."""
    from interpretability import _forecast_autoregressive

    def _value_fn(model, windows, masks, *, device, batch_size) -> Array:
        n = int(windows.shape[0])
        if n == 0:
            return np.zeros((0, int(forecast_horizon)), dtype=np.float32)
        out_chunks: list[Array] = []
        for i in range(0, n, int(batch_size)):
            xb = torch.from_numpy(np.ascontiguousarray(windows[i : i + int(batch_size)])).to(
                device=device,
                dtype=torch.float32,
            )
            mb = torch.from_numpy(np.ascontiguousarray(masks[i : i + int(batch_size)])).to(
                device=device,
                dtype=torch.long,
            )
            forecast = _forecast_autoregressive(
                model,
                xb,
                mb,
                model_horizon=int(model_horizon),
                target_horizon=int(forecast_horizon),
            )
            out_chunks.append(np.asarray(forecast[:, int(target_channel), :], dtype=np.float32))
        values = np.concatenate(out_chunks, axis=0)
        if np.any(~np.isfinite(values)):
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return values

    return _value_fn


# ---------------------------------------------------------------------------
# Variant A: Jacobian-flow (fast, first-order)
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
    cfg = cfg or ChannelFlowConfig(method="jacobian")
    if cfg.method != "jacobian":
        raise ValueError(f"compute_per_channel_flow_jacobian requires method='jacobian', got {cfg.method!r}")
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
        coupling_matrix=None,
        coupling_off_diag_norm=None,
        n_time_steps=int(T - 1),
        n_channels=C,
    )


# ---------------------------------------------------------------------------
# Variant B: Shapley-flow (axiomatic, expensive)
# ---------------------------------------------------------------------------


def _enumerate_coalitions(C: int) -> list[tuple[int, ...]]:
    """All coalitions ``S \\subseteq {0..C-1}`` ordered by size then lex."""
    return [tuple(combo) for size in range(C + 1) for combo in combinations(range(C), size)]


def _kernelshap_weight(C: int, size: int) -> float:
    """Standard SHAP kernel weight for a coalition of given size in {1..C-1}.

    Empty / full coalitions receive infinite weight in KernelSHAP and are
    enforced as equality constraints rather than weighted residuals; we
    handle them separately in the solve.
    """
    if size == 0 or size == C:
        return 0.0  # handled as equality constraints
    from math import comb

    return (C - 1) / (comb(C, size) * size * (C - size))


def _exact_shapley_from_v(
    v: dict[tuple[int, ...], float],
    *,
    C: int,
) -> np.ndarray:
    """Exact Shapley value of ``c`` from the cooperative game ``v``."""
    from math import factorial

    phi = np.zeros((C,), dtype=np.float64)
    fact = [factorial(k) for k in range(C + 1)]
    for c in range(C):
        for size in range(C):
            others = [j for j in range(C) if j != c]
            for S in combinations(others, size):
                w = fact[size] * fact[C - size - 1] / fact[C]
                phi[c] += w * (v[tuple(sorted(S + (c,)))] - v[tuple(S)])
    return phi.astype(np.float32)


def _kernelshap_from_v(
    v: dict[tuple[int, ...], float],
    *,
    C: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate Shapley values via constrained KernelSHAP regression.

    Solves:
        min_phi  sum_S w_S (v(S) - v(empty) - phi.T @ z_S)^2
        s.t.     sum_c phi_c = v(full) - v(empty)
    """
    del n_samples, rng
    coalitions = sorted(v, key=lambda s: (len(s), s))
    values = np.asarray([v[S] for S in coalitions], dtype=np.float64)
    return _kernelshap_from_values(coalitions, values, C=C)


def _kernelshap_from_values(
    coalitions: Sequence[tuple[int, ...]],
    values: np.ndarray,
    *,
    C: int,
) -> np.ndarray:
    """Constrained KernelSHAP for one or many outputs."""
    coalition_list = [tuple(int(c) for c in S) for S in coalitions]
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim < 1 or vals.shape[0] != len(coalition_list):
        raise ValueError(
            "values must have one leading row per coalition; "
            f"got values.shape={vals.shape}, n_coalitions={len(coalition_list)}"
        )
    if C < 1:
        raise ValueError(f"C must be positive, got {C}")

    try:
        empty_idx = coalition_list.index(())
        full_idx = coalition_list.index(tuple(range(C)))
    except ValueError as exc:
        raise ValueError("KernelSHAP requires both empty and full coalitions") from exc

    trailing_shape = vals.shape[1:]
    flat = vals.reshape(vals.shape[0], -1)
    v_empty = flat[empty_idx]
    total_effect = flat[full_idx] - v_empty
    if C == 1:
        return total_effect.reshape((1, *trailing_shape)).astype(np.float32)

    interior = [(i, S) for i, S in enumerate(coalition_list) if 0 < len(S) < C]
    if not interior:
        uniform = np.broadcast_to(total_effect[None, :] / C, (C, total_effect.size)).copy()
        return uniform.reshape((C, *trailing_shape)).astype(np.float32)

    size_counts: dict[int, int] = {}
    for _, S in interior:
        size_counts[len(S)] = size_counts.get(len(S), 0) + 1

    Z = np.zeros((len(interior), C), dtype=np.float64)
    y = np.empty((len(interior), flat.shape[1]), dtype=np.float64)
    w = np.empty((len(interior),), dtype=np.float64)
    for row, (value_idx, S) in enumerate(interior):
        size = len(S)
        Z[row, list(S)] = 1.0
        y[row] = flat[value_idx] - v_empty
        size_mass = (C - 1) / (size * (C - size))
        w[row] = size_mass / size_counts[size]

    gram = Z.T @ (w[:, None] * Z)
    rhs = Z.T @ (w[:, None] * y)
    ridge = max(1e-12, 1e-10 * float(np.trace(gram)) / max(1, C))
    gram = gram + ridge * np.eye(C, dtype=np.float64)
    ones = np.ones((C, 1), dtype=np.float64)
    kkt = np.block([[gram, ones], [ones.T, np.zeros((1, 1), dtype=np.float64)]])
    target = np.concatenate([rhs, total_effect[None, :]], axis=0)
    try:
        sol = np.linalg.solve(kkt, target)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(kkt, target, rcond=None)
    return sol[:C].reshape((C, *trailing_shape)).astype(np.float32)


def _build_step_coalition_set(
    C: int,
    *,
    method: Literal["exact", "kernel"],
    n_samples: int,
    rng: np.random.Generator,
    include_pairs: bool,
) -> list[tuple[int, ...]]:
    """Coalitions that need to be embedded for one transition.

    For exact mode this is all 2^C subsets. For kernel mode we always include
    the empty set, the full set, all singletons (for completeness/diagnostics
    and to bound variance on per-channel attributions), and a sample of
    intermediate-sized coalitions. Optionally also include all size-2
    coalitions when the coupling matrix is requested (so the size <=2
    coalitions are sufficient to compute G in addition to phi).
    """
    if method == "exact":
        return _enumerate_coalitions(C)

    coalitions: set[tuple[int, ...]] = {()}
    coalitions.add(tuple(range(C)))
    for c in range(C):
        coalitions.add((c,))
        coalitions.add(tuple(i for i in range(C) if i != c))
    if include_pairs:
        for pair in combinations(range(C), 2):
            coalitions.add(pair)
    sizes = np.arange(1, C, dtype=np.int64)
    if sizes.size:
        target_total = max(len(coalitions), int(n_samples))
        represented_sizes = {len(coalition) for coalition in coalitions}
        for size in range(2, (C // 2) + 1):
            complement_size = C - size
            needed = int(size not in represented_sizes) + int(
                complement_size not in represented_sizes and complement_size != size
            )
            if needed == 0 or len(coalitions) + needed > target_total:
                continue
            coalition = tuple(sorted(rng.choice(C, size=size, replace=False).tolist()))
            complement = tuple(i for i in range(C) if i not in set(coalition))
            coalitions.add(coalition)
            if complement_size != size:
                coalitions.add(complement)
            represented_sizes.add(size)
            represented_sizes.add(complement_size)

        p_sizes = 1.0 / (sizes.astype(np.float64) * (C - sizes).astype(np.float64))
        p_sizes = p_sizes / p_sizes.sum()
        attempts = 0
        max_attempts = max(target_total * 20, target_total + 64)
        while len(coalitions) < target_total and attempts < max_attempts:
            attempts += 1
            size = int(rng.choice(sizes, p=p_sizes))
            S = tuple(sorted(rng.choice(C, size=size, replace=False).tolist()))
            coalitions.add(S)
            if len(coalitions) < target_total:
                coalitions.add(tuple(i for i in range(C) if i not in set(S)))
    # Stable ordering by (size, lex) so that batches are deterministic.
    return sorted(coalitions, key=lambda s: (len(s), s))


@torch.no_grad()
def compute_per_channel_flow_shapley(
    model: ForecastModel,
    series_ct: Array,
    *,
    seq_len: int,
    input_mask_t: Array | None = None,
    device: torch.device,
    cfg: ChannelFlowConfig | None = None,
    value_fn: ValueFn | None = None,
    protect_channels: Sequence[int] | None = None,
    _per_transition_rng: bool = False,
) -> ChannelFlowReport:
    """Axiomatic Shapley decomposition of latent flow.

    For each transition ``tau -> tau+1`` we define the cooperative game

        v_tau(S) = || Z_tau^(S) - Z_{tau-1}^(S) ||_2,

    where ``Z_tau^(S)`` is the embedding of the rolling window with channels
    not in ``S`` replaced by the configured baseline. The per-channel flow
    contribution is the Shapley value ``phi_tau(c) = Sh(v_tau)(c)``. By
    construction ``sum_c phi_tau(c) = v_tau({1..C}) - v_tau(empty)`` (Shapley
    efficiency); when the baseline is constant in time, ``v_tau(empty) = 0``
    and the sum equals the v1 magnitude exactly.

    When ``cfg.compute_coupling`` is True the same forward passes furnish the
    per-step Harsanyi coupling

        G_tau(c, c') = | v_tau({c, c'}) - v_tau({c}) - v_tau({c'}) + v_tau(empty) |,

    aggregated across time according to ``cfg.coupling_aggregation``.

    Parameters
    ----------
    _per_transition_rng:
        Internal flag (leading underscore signals "not part of the stable
        public API"). When ``False`` (default), kernel-Shapley sample
        selection inside the per-transition loop draws from a single RNG
        seeded with ``cfg.seed`` -- this is the original behaviour and the
        result is bit-identical to historic runs. When ``True``, each
        transition ``tau`` uses its own RNG seeded with ``(cfg.seed,
        int(tau))`` so the per-transition Shapley estimate is independent
        of which other transitions are computed alongside it. The
        ``True`` setting is what the sharded multi-GPU dispatcher uses so
        that ``--shapley-workers N`` produces the same numbers regardless
        of ``N``. Coupling-matrix outputs and exact-mode Shapley values
        are bit-identical between the two settings (they do not consume
        the RNG); only kernel-mode per-channel ``phi`` values differ, and
        only by Monte-Carlo sampling noise within KernelSHAP variance
        (``O(1/sqrt(n_samples))``). Shapley efficiency
        (``sum_c phi_tau(c) == flow_total[tau]``) is preserved exactly.
    """
    cfg = cfg or ChannelFlowConfig(method="shapley")
    if cfg.method != "shapley":
        raise ValueError(f"compute_per_channel_flow_shapley requires method='shapley', got {cfg.method!r}")
    vf = value_fn if value_fn is not None else _embed_windows_batched

    xw_view, m_view, T = _rolling_window_sources(series_ct, seq_len=seq_len, input_mask_t=input_mask_t)
    if T < 2:
        raise ValueError(f"Need at least 2 time steps for flow decomposition, got T={T}")
    transitions = _resolve_step_indices(time_indices=cfg.time_indices, n_windows=T)

    C = int(np.asarray(series_ct).shape[0])
    L = int(seq_len)
    rng = np.random.default_rng(int(cfg.seed))

    protect = tuple(sorted({int(p) for p in (protect_channels or ()) if 0 <= int(p) < C}))
    free = [c for c in range(C) if c not in set(protect)]
    n_free = len(free)
    if n_free < 1:
        raise ValueError("compute_per_channel_flow_shapley: all channels protected leaves no coalition game")
    compute_coupling = bool(cfg.compute_coupling) and not protect

    use_exact = int(cfg.shapley_max_exact_c) >= n_free
    method_label = "exact" if use_exact else "kernel"
    n_samples = int(cfg.shapley_n_samples) if cfg.shapley_n_samples is not None else (2 * n_free + 256)

    coalitions = _build_step_coalition_set(
        n_free,
        method=method_label,
        n_samples=n_samples,
        rng=rng,
        include_pairs=compute_coupling and not use_exact,
    )
    coalition_idx = {S: i for i, S in enumerate(coalitions)}

    # Pre-compute the global per-channel mean once when needed. This is the
    # *time-invariant* baseline that satisfies Proposition 1 (Shapley-flow
    # reduces to v1 exactly for ``C == 1``).
    global_mean_c: Array | None = None
    if cfg.shapley_baseline == "global_mean":
        finite_series = np.where(np.isfinite(series_ct), series_ct, np.nan)
        global_mean_c = np.nanmean(finite_series, axis=1).astype(np.float32)
        global_mean_c = np.where(np.isfinite(global_mean_c), global_mean_c, 0.0).astype(np.float32)

    phi = np.full((T - 1, C), np.nan, dtype=np.float32)
    flow_total = np.full((T - 1,), np.nan, dtype=np.float32)
    coupling_acc = np.zeros((C, C), dtype=np.float64)
    coupling_n = 0

    n_coal = len(coalitions)
    ones_mask = np.ones((L,), dtype=np.int64)
    trans_batch = max(1, int(cfg.transition_batch))

    # Pre-build the [n_coal, C] coalition membership mask once -- it is
    # shared across every transition and lets us vectorise the per-window
    # composition step (we just blend ``baseline`` and ``window`` according
    # to the channel's coalition membership).
    coal_mask = np.zeros((n_coal, C), dtype=bool)
    if protect:
        coal_mask[:, list(protect)] = True
    for i, S in enumerate(coalitions):
        if S:
            coal_mask[i, [free[j] for j in S]] = True
    coal_keep = coal_mask.astype(np.float32)[:, :, None]  # [n_coal, C, 1]
    coal_drop = (1.0 - coal_keep).astype(np.float32)  # [n_coal, C, 1]
    nonempty_coal = coal_mask.any(axis=1)  # [n_coal] -- True iff S != empty

    for chunk_start in range(0, len(transitions), trans_batch):
        chunk = transitions[chunk_start : chunk_start + trans_batch]
        m_chunk = len(chunk)
        starts = chunk.astype(np.int64)
        x_a_batch = np.asarray(xw_view[starts], dtype=np.float32)  # [m, C, L]
        x_b_batch = np.asarray(xw_view[starts + 1], dtype=np.float32)  # [m, C, L]
        mask_a_batch = np.asarray(m_view[starts], dtype=np.int64)  # [m, L]
        mask_b_batch = np.asarray(m_view[starts + 1], dtype=np.int64)  # [m, L]

        # Per-transition baselines. We allow them to differ across the
        # chunk (``mean`` / ``last_value`` are window-dependent), then
        # combine with the coalition mask.
        baseline_a_batch = np.stack(
            [
                _baseline_window(x_a_batch[k], kind=cfg.shapley_baseline, global_mean_c=global_mean_c)
                for k in range(m_chunk)
            ],
            axis=0,
        )  # [m, C, L]
        baseline_b_batch = np.stack(
            [
                _baseline_window(x_b_batch[k], kind=cfg.shapley_baseline, global_mean_c=global_mean_c)
                for k in range(m_chunk)
            ],
            axis=0,
        )  # [m, C, L]

        # Compose coalition windows: out[i, c, l] = keep_S[c] * window[c, l]
        # + drop_S[c] * baseline[c, l]. Broadcast does the heavy lifting --
        # we build an [m_chunk * n_coal, C, L] tensor in one go.
        a_coals = (
            coal_keep[None, :, :, :] * x_a_batch[:, None, :, :]
            + coal_drop[None, :, :, :] * baseline_a_batch[:, None, :, :]
        ).astype(np.float32)  # [m, n_coal, C, L]
        b_coals = (
            coal_keep[None, :, :, :] * x_b_batch[:, None, :, :]
            + coal_drop[None, :, :, :] * baseline_b_batch[:, None, :, :]
        ).astype(np.float32)  # [m, n_coal, C, L]

        # Interleave to get pair order [a_S0, b_S0, a_S1, b_S1, ...] per
        # transition, matching the original layout.
        stack = np.empty((m_chunk, 2 * n_coal, C, L), dtype=np.float32)
        stack[:, 0::2] = a_coals
        stack[:, 1::2] = b_coals

        masks = np.empty((m_chunk, 2 * n_coal, L), dtype=np.int64)
        # For non-empty coalitions, propagate the corresponding window mask;
        # for the empty coalition, attach the all-ones mask (Prop 1 hygiene).
        for k in range(m_chunk):
            for i, is_nonempty in enumerate(nonempty_coal):
                if is_nonempty:
                    masks[k, 2 * i] = mask_a_batch[k]
                    masks[k, 2 * i + 1] = mask_b_batch[k]
                else:
                    masks[k, 2 * i] = ones_mask
                    masks[k, 2 * i + 1] = ones_mask

        flat_stack = stack.reshape(m_chunk * 2 * n_coal, C, L)
        flat_masks = masks.reshape(m_chunk * 2 * n_coal, L)

        z_flat = vf(
            model,
            flat_stack,
            flat_masks,
            device=device,
            batch_size=int(cfg.batch_size),
        )  # [m * 2 * n_coal, Dv]
        D = int(z_flat.shape[1])
        z = z_flat.reshape(m_chunk, 2 * n_coal, D)
        z_a = z[:, 0::2, :]  # [m, n_coal, D]
        z_b = z[:, 1::2, :]  # [m, n_coal, D]
        delta_z_chunk = (z_b - z_a).astype(np.float64)  # [m, n_coal, D]
        v_vals_chunk = np.linalg.norm(delta_z_chunk, ord=2, axis=2)  # [m, n_coal]

        for k, tau in enumerate(chunk):
            v_vals = v_vals_chunk[k]
            v: dict[tuple[int, ...], float] = {S: float(v_vals[coalition_idx[S]]) for S in coalitions}
            if use_exact:
                phi_free = _exact_shapley_from_v(v, C=n_free)
            else:
                # Per-transition RNG: pin the kernel-Shapley sampler to a
                # deterministic state derived from (seed, tau) so that the
                # Shapley estimate at this transition does not depend on the
                # iteration order of other transitions in the same call.
                # This is what makes sharded execution produce identical
                # results regardless of how many workers we split across.
                rng_for_tau = np.random.default_rng((int(cfg.seed), int(tau))) if _per_transition_rng else rng
                phi_free = _kernelshap_from_v(
                    v,
                    C=n_free,
                    n_samples=n_samples,
                    rng=rng_for_tau,
                )
            phi_tau = np.zeros((C,), dtype=np.float32)
            for j, c in enumerate(free):
                phi_tau[c] = float(phi_free[j])
            phi[int(tau)] = phi_tau
            flow_total[int(tau)] = float(v[tuple(range(n_free))])

            if compute_coupling:
                delta_z = delta_z_chunk[k]
                d_empty = delta_z[coalition_idx[()]]
                singletons = {c: delta_z[coalition_idx[(c,)]] for c in range(C)}
                for c1, c2 in combinations(range(C), 2):
                    pair_key = (c1, c2)
                    pair_idx = coalition_idx.get(pair_key)
                    if pair_idx is None:
                        continue
                    interaction = delta_z[pair_idx] - singletons[c1] - singletons[c2] + d_empty
                    g = float(np.linalg.norm(interaction, ord=2))
                    coupling_acc[c1, c2] += g
                    coupling_acc[c2, c1] += g
                coupling_n += 1

    coupling_matrix = None
    coupling_off_diag = None
    coupling_partial_sum: np.ndarray | None = None
    coupling_partial_n = 0
    if compute_coupling and coupling_n > 0:
        if cfg.coupling_aggregation == "mean":
            coupling_matrix = (coupling_acc / float(coupling_n)).astype(np.float32)
        elif cfg.coupling_aggregation == "sum":
            coupling_matrix = coupling_acc.astype(np.float32)
        else:
            raise ValueError(f"Unsupported coupling_aggregation={cfg.coupling_aggregation!r}")
        np.fill_diagonal(coupling_matrix, 0.0)
        coupling_off_diag = float(np.linalg.norm(coupling_matrix, ord="fro"))
        # Carry the raw accumulator + count so merge_shapley_reports can
        # reduce across shards bit-identically.
        coupling_partial_sum = coupling_acc.copy()
        coupling_partial_n = int(coupling_n)

    return ChannelFlowReport(
        method="shapley_exact" if use_exact else "shapley_kernel",
        per_channel_flow=phi,
        flow_total=flow_total,
        residual_ratio_per_step=None,
        residual_ratio_mean=None,
        residual_ratio_p95=None,
        coupling_matrix=coupling_matrix,
        coupling_off_diag_norm=coupling_off_diag,
        n_time_steps=int(T - 1),
        n_channels=C,
        coupling_partial_sum=coupling_partial_sum,
        coupling_partial_n=coupling_partial_n,
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
    protect_channels: Sequence[int] | None = None,
) -> ChannelFlowReport:
    """Dispatch to the configured per-channel flow estimator."""
    cfg = cfg or ChannelFlowConfig()
    if cfg.method == "jacobian":
        return compute_per_channel_flow_jacobian(
            model,
            series_ct,
            seq_len=seq_len,
            input_mask_t=input_mask_t,
            device=device,
            cfg=cfg,
            value_fn=value_fn,
        )
    if cfg.method == "shapley":
        return compute_per_channel_flow_shapley(
            model,
            series_ct,
            seq_len=seq_len,
            input_mask_t=input_mask_t,
            device=device,
            cfg=cfg,
            value_fn=value_fn,
            protect_channels=protect_channels,
        )
    raise ValueError(f"Unsupported ChannelFlowConfig.method={cfg.method!r}")


def merge_shapley_reports(
    reports: Sequence[ChannelFlowReport],
    *,
    coupling_aggregation: Literal["mean", "sum"] = "mean",
) -> ChannelFlowReport:
    """Combine partial Shapley reports from sharded execution.

    Each report in ``reports`` must come from a call to
    :func:`compute_per_channel_flow_shapley` over a *disjoint* slice of
    transitions of the same overall trajectory (same ``T`` and ``C``).
    The reports are merged into a single global report whose:

    * ``per_channel_flow`` and ``flow_total`` are scattered into the
      union of the shards' transition slices (entries outside any
      shard's slice remain ``NaN``, matching the un-sharded behaviour
      when ``cfg.time_indices`` is set).
    * ``coupling_matrix`` is reduced from the per-shard
      ``coupling_partial_sum`` accumulators -- this preserves the
      bit-exact reduction order of the un-sharded computation provided
      the shards are passed in transition order.

    The shards must have been produced with ``_per_transition_rng=True``
    so that each transition's Shapley estimate is independent of which
    other transitions ran on the same worker.
    """
    if not reports:
        raise ValueError("merge_shapley_reports requires at least one report")
    first = reports[0]
    if first.per_channel_flow.ndim != 2:
        raise ValueError("per_channel_flow must be 2-D [T-1, C]")
    Tm1, C = first.per_channel_flow.shape
    for r in reports[1:]:
        if r.per_channel_flow.shape != (Tm1, C):
            raise ValueError(f"shape mismatch across shards: {r.per_channel_flow.shape} vs {(Tm1, C)}")
        if r.n_channels != C:
            raise ValueError(f"n_channels mismatch across shards: {r.n_channels} vs {C}")

    # Disjoint scatter for phi and flow_total. We assert shards do not
    # overlap on a transition by checking that every transition where
    # multiple shards are finite produces the same value -- if they
    # disagree it means the caller's shard slices overlap, which is a bug.
    merged_phi = np.full((Tm1, C), np.nan, dtype=first.per_channel_flow.dtype)
    merged_total = np.full((Tm1,), np.nan, dtype=first.flow_total.dtype)
    for r in reports:
        finite_phi = np.isfinite(r.per_channel_flow)
        # Sanity: any cell already set must agree (catches overlapping shards).
        if finite_phi.any():
            preset = np.isfinite(merged_phi)
            overlap = finite_phi & preset
            if overlap.any():
                if not np.allclose(merged_phi[overlap], r.per_channel_flow[overlap], atol=0.0, rtol=0.0):
                    raise ValueError(
                        "merge_shapley_reports: overlapping shards disagree on "
                        "per_channel_flow values; ensure shard slices are disjoint."
                    )
            merged_phi[finite_phi] = r.per_channel_flow[finite_phi]
        finite_tot = np.isfinite(r.flow_total)
        if finite_tot.any():
            merged_total[finite_tot] = r.flow_total[finite_tot]

    coupling_matrix: np.ndarray | None = None
    coupling_off_diag: float | None = None
    coupling_acc_total: np.ndarray | None = None
    coupling_n_total = 0
    have_partials = all(r.coupling_partial_sum is not None for r in reports)
    if have_partials:
        coupling_acc_total = np.zeros((C, C), dtype=np.float64)
        for r in reports:
            coupling_acc_total = coupling_acc_total + np.asarray(r.coupling_partial_sum, dtype=np.float64)
            coupling_n_total += int(r.coupling_partial_n)
        if coupling_n_total > 0:
            if coupling_aggregation == "mean":
                coupling_matrix = (coupling_acc_total / float(coupling_n_total)).astype(np.float32)
            elif coupling_aggregation == "sum":
                coupling_matrix = coupling_acc_total.astype(np.float32)
            else:
                raise ValueError(f"Unsupported coupling_aggregation={coupling_aggregation!r}")
            np.fill_diagonal(coupling_matrix, 0.0)
            coupling_off_diag = float(np.linalg.norm(coupling_matrix, ord="fro"))

    return ChannelFlowReport(
        method=first.method,
        per_channel_flow=merged_phi,
        flow_total=merged_total,
        residual_ratio_per_step=None,
        residual_ratio_mean=None,
        residual_ratio_p95=None,
        coupling_matrix=coupling_matrix,
        coupling_off_diag_norm=coupling_off_diag,
        n_time_steps=int(Tm1),
        n_channels=int(C),
        coupling_partial_sum=(coupling_acc_total.copy() if coupling_acc_total is not None else None),
        coupling_partial_n=int(coupling_n_total),
    )


def shard_transitions(n_transitions: int, n_shards: int) -> list[np.ndarray]:
    """Split ``[0, 1, ..., n_transitions-1]`` into roughly-equal contiguous chunks.

    Used by the multi-GPU dispatcher so each shard sees a contiguous slice of
    the transition trajectory; this preserves the original sequential
    reduction order when the per-shard Harsanyi accumulators are summed
    back in shard order.
    """
    if n_transitions <= 0:
        raise ValueError("n_transitions must be positive")
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    n_shards = min(n_shards, n_transitions)
    base = n_transitions // n_shards
    extra = n_transitions % n_shards
    out: list[np.ndarray] = []
    cursor = 0
    for i in range(n_shards):
        size = base + (1 if i < extra else 0)
        out.append(np.arange(cursor, cursor + size, dtype=np.int64))
        cursor += size
    return out


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
      gives a per-channel-per-horizon attribution suitable for direct
      comparison to TimeSHAP / IntGrad / ChInf baselines.
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
