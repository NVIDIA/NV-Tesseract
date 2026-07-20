# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU interpretability passes (Pass A Jacobian + Pass B Shapley/coupling).

* Pass A: ``explain_forecast(channel_axis=True, chan_cfg=jacobian)``
* Pass B: ``compute_per_channel_flow(method=shapley, compute_coupling=True)``
* Optional: Pass A on device 0 while Pass B is sharded across extra GPUs via
  ``shard_transitions`` + ``parallel_map_with_init`` + ``merge_shapley_reports``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from channel_flow import (
    ChannelFlowConfig,
    ChannelFlowReport,
    compute_per_channel_flow,
    compute_per_channel_flow_shapley,
    merge_shapley_reports,
    shard_transitions,
)
from interpretability import ForecastExplanation, explain_forecast
from model import build_model
from parallel import parallel_map_with_init, parse_devices_arg

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterpretabilityPassConfig:
    """Knobs for v1+v2 interpretability passes (Pass A Jacobian + Pass B Shapley)."""

    seq_len: int
    forecast_horizon: int
    model_horizon: int
    n_lags: int = 128
    softmax_tau: float = 1.0
    surrogate: bool = False
    surrogate_n_jobs: int | None = None
    latent_batch_size: int = 32
    coupling_transitions: int = 16
    shapley_n_samples: int = 64
    shapley_baseline: str = "zero"
    transition_batch: int = 8
    channel_batch_size: int = 64
    devices: str | None = "auto"
    parallel_passes: bool = False
    shapley_workers: int = 0
    run_coupling: bool = True
    channel_output_aware: bool = False
    channel_target: int | None = None


@dataclass(frozen=True)
class InterpretabilityPassResult:
    explanation: ForecastExplanation
    coupling_report: ChannelFlowReport | None
    pass_a_seconds: float
    pass_b_seconds: float
    parallel: bool


@dataclass(frozen=True)
class _WorkerInit:
    ckpt_path: str
    model_name: str
    seq_len: int
    model_horizon: int
    use_cross_channel: bool
    cross_channel_heads: int
    cross_channel_dropout: float

    def __call__(self, rank: int, device: torch.device) -> dict:
        model = build_model(
            model_name=self.model_name,
            seq_len=self.seq_len,
            forecast_horizon=int(self.model_horizon),
            freeze_encoder=False,
            freeze_embedder=False,
            freeze_head=False,
            use_cross_channel=self.use_cross_channel,
            cross_channel_heads=self.cross_channel_heads,
            cross_channel_dropout=self.cross_channel_dropout,
            local_files_only=False,
            device=str(device),
        )
        state = torch.load(self.ckpt_path, map_location=device)
        load_result = model.load_state_dict(state, strict=False)
        missing = list(getattr(load_result, "missing_keys", [])) if load_result else []
        unexpected = list(getattr(load_result, "unexpected_keys", [])) if load_result else []
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
        non_cr_missing = [k for k in missing if "cross_channel" not in k]
        if non_cr_missing:
            raise RuntimeError(f"Missing non-cross-channel keys: {non_cr_missing}")
        model.eval()
        return {"model": model}


@dataclass(frozen=True)
class _PassAJob:
    x_ct: np.ndarray
    input_mask_l: np.ndarray
    model_horizon: int
    forecast_horizon: int
    n_lags: int
    softmax_tau: float
    surrogate: bool
    latent_batch_size: int
    surrogate_n_jobs: int | None
    chan_cfg: ChannelFlowConfig
    channel_output_aware: bool
    channel_target: int | None


@dataclass(frozen=True)
class _PassBJob:
    series_ext: np.ndarray
    seq_len: int
    mask_ext: np.ndarray
    chan_cfg: ChannelFlowConfig


@dataclass(frozen=True)
class _PassBShardJob:
    series_ext: np.ndarray
    seq_len: int
    mask_ext: np.ndarray
    chan_cfg: ChannelFlowConfig
    shard_index: int
    n_shards: int


def _run_pass_worker(ctx: dict, device: torch.device, job) -> dict:
    model = ctx["model"]
    if isinstance(job, _PassAJob):
        t0 = time.time()
        expl = explain_forecast(
            model,
            x_context_ct=job.x_ct,
            input_mask_l=job.input_mask_l,
            model_horizon=int(job.model_horizon),
            forecast_horizon=int(job.forecast_horizon),
            device=device,
            n_lags=int(job.n_lags),
            softmax_tau=float(job.softmax_tau),
            surrogate=bool(job.surrogate),
            latent_batch_size=int(job.latent_batch_size),
            surrogate_n_jobs=job.surrogate_n_jobs,
            channel_axis=True,
            chan_cfg=job.chan_cfg,
            channel_output_aware=job.channel_output_aware,
            channel_target=job.channel_target,
        )
        return {"expl": expl, "elapsed": time.time() - t0}
    if isinstance(job, _PassBJob):
        t0 = time.time()
        report = compute_per_channel_flow(
            model,
            job.series_ext,
            seq_len=int(job.seq_len),
            input_mask_t=job.mask_ext,
            device=device,
            cfg=job.chan_cfg,
        )
        return {"report": report, "elapsed": time.time() - t0}
    if isinstance(job, _PassBShardJob):
        t0 = time.time()
        report = compute_per_channel_flow_shapley(
            model,
            job.series_ext,
            seq_len=int(job.seq_len),
            input_mask_t=job.mask_ext,
            device=device,
            cfg=job.chan_cfg,
            _per_transition_rng=True,
        )
        return {
            "report": report,
            "elapsed": time.time() - t0,
            "shard_index": int(job.shard_index),
            "n_shards": int(job.n_shards),
        }
    raise TypeError(f"Unknown pass job type: {type(job).__name__}")


def build_extended_series(
    x_ct: np.ndarray,
    *,
    forecast_horizon: int,
    baseline_forecast: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(series_ext [C,L+H], mask_ext [L+H])`` for Pass B."""
    c_use, seq_len = x_ct.shape
    if baseline_forecast is None:
        tail = np.zeros((c_use, forecast_horizon), dtype=np.float32)
    else:
        tail = np.asarray(baseline_forecast, dtype=np.float32)
        if tail.shape != (c_use, forecast_horizon):
            raise ValueError(f"baseline_forecast shape {tail.shape} != ({c_use}, {forecast_horizon})")
    series_ext = np.concatenate([x_ct, tail], axis=1)
    mask_ext = np.concatenate(
        [np.ones((seq_len,), dtype=np.int64), np.ones((forecast_horizon,), dtype=np.int64)],
        axis=0,
    )
    return series_ext, mask_ext


def build_pass_configs(
    x_ct: np.ndarray,
    *,
    cfg: InterpretabilityPassConfig,
    baseline_forecast: np.ndarray | None = None,
) -> tuple[ChannelFlowConfig, ChannelFlowConfig, np.ndarray, np.ndarray, list[int]]:
    """Build Jacobian (Pass A) and Shapley (Pass B) configs plus extended series."""
    seq_len = int(cfg.seq_len)
    t_index = seq_len - 1
    lag_lo = max(0, t_index - int(cfg.n_lags))
    chan_cfg_jac = ChannelFlowConfig(
        method="jacobian",
        batch_size=int(cfg.channel_batch_size),
        transition_batch=int(cfg.transition_batch),
        compute_coupling=False,
        time_indices=list(range(lag_lo, t_index)),
    )

    n_t = max(1, int(cfg.coupling_transitions))
    coup_hi = max(1, t_index)
    coup_transitions = list(range(max(0, coup_hi - n_t), coup_hi))

    series_ext, mask_ext = build_extended_series(
        x_ct,
        forecast_horizon=int(cfg.forecast_horizon),
        baseline_forecast=baseline_forecast,
    )
    chan_cfg_sh = ChannelFlowConfig(
        method="shapley",
        shapley_baseline=cfg.shapley_baseline,  # type: ignore[arg-type]
        shapley_n_samples=int(cfg.shapley_n_samples),
        compute_coupling=True,
        time_indices=coup_transitions,
        batch_size=int(cfg.channel_batch_size),
        transition_batch=int(cfg.transition_batch),
    )
    return chan_cfg_jac, chan_cfg_sh, series_ext, mask_ext, coup_transitions


def should_run_parallel_passes(
    devices: Sequence[torch.device],
    *,
    use_channel_axis: bool,
    run_coupling: bool,
    parallel_passes: bool,
    shapley_workers: int,
) -> bool:
    """Whether to dispatch Pass A and Pass B across multiple worker processes."""
    if not use_channel_axis or not run_coupling:
        return False
    return (
        bool(parallel_passes)
        or int(shapley_workers) >= 2
        or (len(devices) >= 2 and all(d.type == "cuda" for d in devices[:2]))
    )


def merge_coupling_into_explanation(
    explanation: ForecastExplanation,
    coupling_report: ChannelFlowReport | None,
) -> ForecastExplanation:
    """Attach Pass B coupling matrix / norm to a Pass A ``ForecastExplanation``."""
    if coupling_report is None or coupling_report.coupling_matrix is None:
        return explanation
    return replace(
        explanation,
        channel_coupling_matrix=coupling_report.coupling_matrix,
        channel_coupling_off_diag_norm=coupling_report.coupling_off_diag_norm,
    )


def _run_pass_b_serial(
    model: torch.nn.Module,
    *,
    series_ext: np.ndarray,
    seq_len: int,
    mask_ext: np.ndarray,
    chan_cfg_sh: ChannelFlowConfig,
    device: torch.device,
) -> tuple[ChannelFlowReport, float]:
    t0 = time.time()
    report = compute_per_channel_flow(
        model,
        series_ext,
        seq_len=seq_len,
        input_mask_t=mask_ext,
        device=device,
        cfg=chan_cfg_sh,
    )
    return report, time.time() - t0


def _run_parallel_passes(
    *,
    init_fn: _WorkerInit,
    x_ct: np.ndarray,
    input_mask_l: np.ndarray,
    series_ext: np.ndarray,
    mask_ext: np.ndarray,
    cfg: InterpretabilityPassConfig,
    chan_cfg_jac: ChannelFlowConfig,
    chan_cfg_sh: ChannelFlowConfig,
    coup_transitions: list[int],
    devices: list[torch.device],
    n_pb_shards: int,
) -> tuple[ForecastExplanation, ChannelFlowReport, float, float]:
    if n_pb_shards >= 2:
        shard_slices = shard_transitions(len(coup_transitions), n_pb_shards)
        n_pb_shards = len(shard_slices)
        pass_jobs: list = [
            _PassAJob(
                x_ct=x_ct,
                input_mask_l=input_mask_l,
                model_horizon=int(cfg.model_horizon),
                forecast_horizon=int(cfg.forecast_horizon),
                n_lags=int(cfg.n_lags),
                softmax_tau=float(cfg.softmax_tau),
                surrogate=bool(cfg.surrogate),
                latent_batch_size=int(cfg.latent_batch_size),
                surrogate_n_jobs=cfg.surrogate_n_jobs,
                chan_cfg=chan_cfg_jac,
                channel_output_aware=bool(cfg.channel_output_aware),
                channel_target=cfg.channel_target,
            ),
        ]
        for shard_idx, sl in enumerate(shard_slices):
            shard_time_indices = [int(coup_transitions[int(i)]) for i in sl]
            cfg_shard = ChannelFlowConfig(
                method=chan_cfg_sh.method,
                shapley_baseline=chan_cfg_sh.shapley_baseline,
                shapley_n_samples=chan_cfg_sh.shapley_n_samples,
                compute_coupling=chan_cfg_sh.compute_coupling,
                coupling_aggregation=chan_cfg_sh.coupling_aggregation,
                time_indices=shard_time_indices,
                batch_size=chan_cfg_sh.batch_size,
                transition_batch=chan_cfg_sh.transition_batch,
                seed=chan_cfg_sh.seed,
            )
            pass_jobs.append(
                _PassBShardJob(
                    series_ext=series_ext,
                    seq_len=int(cfg.seq_len),
                    mask_ext=mask_ext,
                    chan_cfg=cfg_shard,
                    shard_index=shard_idx,
                    n_shards=n_pb_shards,
                )
            )
        n_workers = 1 + n_pb_shards
        worker_devices = devices[:n_workers]
        logger.info(
            "[interpretability] Pass A on %s; Pass B sharded into %d on %s",
            worker_devices[0],
            n_pb_shards,
            [str(d) for d in worker_devices[1:]],
        )
    else:
        pass_jobs = [
            _PassAJob(
                x_ct=x_ct,
                input_mask_l=input_mask_l,
                model_horizon=int(cfg.model_horizon),
                forecast_horizon=int(cfg.forecast_horizon),
                n_lags=int(cfg.n_lags),
                softmax_tau=float(cfg.softmax_tau),
                surrogate=bool(cfg.surrogate),
                latent_batch_size=int(cfg.latent_batch_size),
                surrogate_n_jobs=cfg.surrogate_n_jobs,
                chan_cfg=chan_cfg_jac,
                channel_output_aware=bool(cfg.channel_output_aware),
                channel_target=cfg.channel_target,
            ),
            _PassBJob(
                series_ext=series_ext,
                seq_len=int(cfg.seq_len),
                mask_ext=mask_ext,
                chan_cfg=chan_cfg_sh,
            ),
        ]
        n_workers = 2
        worker_devices = devices[:2]
        logger.info(
            "[interpretability] Pass A on %s; Pass B on %s",
            worker_devices[0],
            worker_devices[1],
        )

    t0 = time.time()
    results = parallel_map_with_init(
        init_fn,
        _run_pass_worker,
        pass_jobs,
        devices=worker_devices,
        num_workers=n_workers,
        progress=True,
        progress_desc="interpretability",
    )
    _ = time.time() - t0
    t_jac = float(results[0]["elapsed"])
    expl = results[0]["expl"]

    if n_pb_shards >= 2:
        shard_results = sorted(results[1:], key=lambda r: int(r["shard_index"]))
        coup_report = merge_shapley_reports(
            [r["report"] for r in shard_results],
            coupling_aggregation=chan_cfg_sh.coupling_aggregation,
        )
        t_sh = max(float(r["elapsed"]) for r in shard_results)
    else:
        coup_report = results[1]["report"]
        t_sh = float(results[1]["elapsed"])

    logger.info("  pass A: %.1fs  pass B: %.1fs", t_jac, t_sh)
    return expl, coup_report, t_jac, t_sh


def run_interpretability_passes(
    *,
    x_ct: np.ndarray,
    input_mask_l: np.ndarray,
    cfg: InterpretabilityPassConfig,
    use_channel_axis: bool,
    # In-process path (single GPU / no parallel dispatch)
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    # Multi-process path reloads the checkpoint per worker
    ckpt_path: str | None = None,
    model_name: str = "AutonLab/MOMENT-1-large",
    use_cross_channel: bool = True,
    cross_channel_heads: int = 8,
    cross_channel_dropout: float = 0.1,
) -> InterpretabilityPassResult:
    """Run Pass A (+ optional Pass B), using multi-process dispatch when enabled."""
    devices = parse_devices_arg(cfg.devices)
    if device is None:
        device = devices[0]

    run_coupling = bool(cfg.run_coupling) and use_channel_axis
    chan_cfg_jac, chan_cfg_sh, series_ext, mask_ext, coup_transitions = build_pass_configs(x_ct, cfg=cfg)

    pp_enabled = should_run_parallel_passes(
        devices,
        use_channel_axis=use_channel_axis,
        run_coupling=run_coupling,
        parallel_passes=bool(cfg.parallel_passes),
        shapley_workers=int(cfg.shapley_workers),
    )

    if pp_enabled and len(devices) < 2:
        logger.warning(
            "[interpretability] parallel passes requested but only one device; running sequentially on %s.",
            devices[0],
        )
        pp_enabled = False

    if pp_enabled:
        if ckpt_path is None:
            raise ValueError("ckpt_path is required when running parallel interpretability passes")
        n_extra = max(0, len(devices) - 1)
        requested = max(0, int(cfg.shapley_workers))
        n_pb_shards = min(requested, n_extra) if requested >= 2 else 1
        init_fn = _WorkerInit(
            ckpt_path=str(ckpt_path),
            model_name=model_name,
            seq_len=int(cfg.seq_len),
            model_horizon=int(cfg.model_horizon),
            use_cross_channel=use_cross_channel,
            cross_channel_heads=cross_channel_heads,
            cross_channel_dropout=cross_channel_dropout,
        )
        expl, coup_report, t_jac, t_sh = _run_parallel_passes(
            init_fn=init_fn,
            x_ct=x_ct,
            input_mask_l=input_mask_l,
            series_ext=series_ext,
            mask_ext=mask_ext,
            cfg=cfg,
            chan_cfg_jac=chan_cfg_jac,
            chan_cfg_sh=chan_cfg_sh,
            coup_transitions=coup_transitions,
            devices=devices,
            n_pb_shards=n_pb_shards,
        )
        expl = merge_coupling_into_explanation(expl, coup_report)
        return InterpretabilityPassResult(
            explanation=expl,
            coupling_report=coup_report,
            pass_a_seconds=t_jac,
            pass_b_seconds=t_sh,
            parallel=True,
        )

    if model is None:
        if ckpt_path is None:
            raise ValueError("model or ckpt_path is required for sequential interpretability passes")
        init_fn = _WorkerInit(
            ckpt_path=str(ckpt_path),
            model_name=model_name,
            seq_len=int(cfg.seq_len),
            model_horizon=int(cfg.model_horizon),
            use_cross_channel=use_cross_channel,
            cross_channel_heads=cross_channel_heads,
            cross_channel_dropout=cross_channel_dropout,
        )
        model = init_fn(0, device)["model"]

    model.eval()
    t0 = time.time()
    expl = explain_forecast(
        model,
        x_context_ct=x_ct,
        input_mask_l=input_mask_l,
        model_horizon=int(cfg.model_horizon),
        forecast_horizon=int(cfg.forecast_horizon),
        device=device,
        n_lags=int(cfg.n_lags),
        softmax_tau=float(cfg.softmax_tau),
        surrogate=bool(cfg.surrogate),
        latent_batch_size=int(cfg.latent_batch_size),
        surrogate_n_jobs=cfg.surrogate_n_jobs,
        channel_axis=use_channel_axis,
        chan_cfg=chan_cfg_jac if use_channel_axis else None,
        channel_output_aware=bool(cfg.channel_output_aware),
        channel_target=cfg.channel_target,
    )
    t_jac = time.time() - t0

    coup_report: ChannelFlowReport | None = None
    t_sh = 0.0
    if run_coupling:
        series_ext, mask_ext = build_extended_series(
            x_ct,
            forecast_horizon=int(cfg.forecast_horizon),
            baseline_forecast=np.asarray(expl.baseline_forecast, dtype=np.float32),
        )
        coup_report, t_sh = _run_pass_b_serial(
            model,
            series_ext=series_ext,
            seq_len=int(cfg.seq_len),
            mask_ext=mask_ext,
            chan_cfg_sh=chan_cfg_sh,
            device=device,
        )
        expl = merge_coupling_into_explanation(expl, coup_report)
        logger.info("  pass A: %.1fs  pass B: %.1fs", t_jac, t_sh)

    return InterpretabilityPassResult(
        explanation=expl,
        coupling_report=coup_report,
        pass_a_seconds=t_jac,
        pass_b_seconds=t_sh,
        parallel=False,
    )
