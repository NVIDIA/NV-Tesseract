# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Integrated Gradients / Expected Gradients over the embedding layer.

What this measures
==================
This is a *general*, model-agnostic interpretability primitive. Given any model
that encodes a time-series context window into an embedding,

    e = Enc(x) ,    x in R^{...} (the input window),  e in R^{D} (the encoding),

it measures the effect of the input on the encoding by **accumulating the
gradients of the embedding layer along a straight-line path from a baseline to
the input** (Integrated Gradients; Sundararajan et al., 2017):

    A_i = (x_i - x0_i) * \\int_0^1  d g(Enc(x0 + a*(x - x0))) / d x_i  da ,

where ``g`` reduces the embedding vector to a scalar (default: its L2 norm) and
``x0`` is the baseline (default: zeros, i.e. the "absence of input"). The path
``x0 -> x`` progressively *imputes* the input values into the encoder, and the
accumulated gradient quantifies how strongly those values drive the encoding.

By the completeness axiom,

    sum_i A_i = g(Enc(x)) - g(Enc(x0)) ,

which is reported as a convergence diagnostic.

Baseline choice (important for instance-normalized encoders)
============================================================
Instance normalization ``(x - mean) / std`` is invariant to shifting and scaling
the input. A straight-line path from a *zero or constant* baseline to ``x`` can
therefore become the same normalized series at every step, making the embedding
flat along the path and degrading IG completeness at the degenerate endpoint.
The default white-noise baseline is matched to the input's global mean/std, which
breaks that invariance. Averaging several such baselines (``n_baselines > 1``)
produces the Expected-Gradients estimator and reduces variance.

Design
======
* **Model-agnostic.** The core depends only on a differentiable
  ``embed_fn: Tensor[n, *feat] -> Tensor[n, D]`` that maps a *batch* of inputs
  to a batch of embeddings. Nothing here knows about channels, patches,
  missingness, or any specific architecture.
* **Real gradients.** Unlike the gradient-free probes elsewhere in
  this package, this uses autograd: interpolation points are pushed through the
  encoder in one or more batches and ``torch.autograd.grad`` yields the
  gradients with respect to the input path only (each step's embedding depends
  only on its own row).
* **Global, not per-axis.** The headline output is a single scalar effect over
  the whole embedding layer plus a saliency map in the input's own shape. No
  channel/feature-axis decomposition is imposed by this primitive.

Cost: one forward/gradient pass per interpolation chunk.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

Array = np.ndarray
EmbedFn = Callable[[torch.Tensor], torch.Tensor]
Reduce = Literal["l2", "sum", "mean"]
BaselineMode = Literal["noise", "zero", "mean"]


@dataclass(frozen=True)
class EmbeddingAttribution:
    """Result of Integrated Gradients over the embedding layer.

    * ``attribution``: per-input-element Integrated Gradients, in the input's
      own shape (a model-agnostic saliency map over the context window).
    * ``overall_effect``: ``sum_i attribution_i`` -- the net signed effect of
      the input on the (reduced) embedding. Equal to ``embedding_delta`` up to
      the Riemann approximation error.
    * ``abs_effect``: ``sum_i |attribution_i|`` -- total saliency mass, a
      magnitude summary that does not cancel across positions.
    * ``embedding_delta``: ``g(Enc(x)) - g(Enc(x0))`` measured directly.
    * ``embedding_delta_abs_mean``: mean absolute endpoint change across
      Expected-Gradients baselines. This is the stable scale used below.
    * ``convergence_delta``: ``|overall_effect - embedding_delta| /
      (embedding_delta_abs_mean + eps)``; the Integrated-Gradients completeness
      residual (-> 0 as ``n_steps`` grows). Using ``|mean(delta)|`` in the
      denominator, as the old implementation did, explodes whenever endpoint
      changes from different noise baselines cancel.
    """

    attribution: Array
    overall_effect: float
    abs_effect: float
    embedding_delta: float
    embedding_delta_abs_mean: float
    convergence_delta: float
    n_steps: int
    n_baselines: int
    baseline: str
    reduce: str
    embedding_dim: int


def _reduce_embedding(emb: torch.Tensor, reduce: Reduce) -> torch.Tensor:
    """Reduce an ``[n, D]`` (or ``[n, ...]``) embedding batch to a scalar per row."""
    flat = emb.reshape(emb.shape[0], -1)
    if reduce == "l2":
        return torch.linalg.vector_norm(flat, dim=1)
    if reduce == "sum":
        return flat.sum(dim=1)
    if reduce == "mean":
        return flat.mean(dim=1)
    raise ValueError(f"Unknown reduce {reduce!r}; expected 'l2' | 'sum' | 'mean'.")


def _ig_single(
    embed_fn: EmbedFn,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    *,
    n_steps: int,
    reduce: Reduce,
    internal_batch_size: int | None = None,
) -> tuple[Array, float, int]:
    """One Integrated-Gradients pass for a fixed baseline ``x0``.

    Returns ``(attribution, embedding_delta, embedding_dim)``.
    """
    delta = x_t - x0  # [*feat]
    dev = x_t.device

    chunk = int(internal_batch_size or n_steps)
    if chunk < 1:
        raise ValueError(f"internal_batch_size must be >= 1 when provided, got {internal_batch_size}")
    chunk = min(chunk, int(n_steps))

    grad_sum = torch.zeros_like(x_t)
    expand_prefix = (1,) * x_t.ndim
    for start in range(0, int(n_steps), chunk):
        stop = min(int(n_steps), start + chunk)
        count = stop - start

        # Midpoint Riemann nodes for this chunk.
        alphas = (torch.arange(start, stop, device=dev, dtype=torch.float32) + 0.5) / int(n_steps)
        x_interp = x0.unsqueeze(0) + alphas.reshape((count,) + expand_prefix) * delta.unsqueeze(0)
        x_interp = x_interp.clone().requires_grad_(True)

        with torch.enable_grad():
            emb = embed_fn(x_interp)  # [count, D]
            if emb.shape[0] != count:
                raise RuntimeError(
                    f"embed_fn must preserve the leading batch dim: got {emb.shape[0]} rows for {count} inputs."
                )
            g = _reduce_embedding(emb, reduce)  # [count]
            # Request only input-path gradients. Using .backward() here would
            # also materialize parameter gradients for large encoders and keep
            # them resident across integration batches.
            grad = torch.autograd.grad(g.sum(), x_interp, retain_graph=False, create_graph=False)[0]

        grad_sum += grad.detach().sum(dim=0)
        del emb, g, grad, x_interp

    avg_grad = grad_sum / float(n_steps)  # [*feat]
    attribution = (delta * avg_grad).detach().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        ends = torch.stack([x0, x_t], dim=0)
        emb_ends = embed_fn(ends)
        g_ends = _reduce_embedding(emb_ends, reduce)
        emb_dim = int(emb_ends.reshape(emb_ends.shape[0], -1).shape[1])
    embedding_delta = float((g_ends[1] - g_ends[0]).item())
    return attribution, embedding_delta, emb_dim


def integrated_gradients_embedding(
    embed_fn: EmbedFn,
    x: Array | torch.Tensor,
    *,
    baseline: BaselineMode | Array | torch.Tensor | float = "noise",
    n_steps: int = 64,
    n_baselines: int = 1,
    internal_batch_size: int | None = None,
    reduce: Reduce = "l2",
    seed: int = 0,
    device: torch.device | str | None = None,
) -> EmbeddingAttribution:
    """Accumulate embedding-layer gradients along the baseline->input path.

    Args:
        embed_fn: differentiable map from a *batch* of inputs ``[n, *feat]`` to a
            batch of embeddings ``[n, D]`` (or any ``[n, ...]`` reduced to a
            scalar per row). This callable is the only model-specific input.
        x: a single input window, shape ``[*feat]`` (no batch dim).
        baseline: path start. Either a mode string or an explicit reference:

            * ``"noise"`` (default) -- white noise matched to ``x``'s global
              mean/std. Recommended for instance-normalized models; with
              ``n_baselines > 1`` this becomes the Expected-Gradients estimator.
            * ``"zero"`` / ``"mean"`` / a constant float -- a *constant* baseline.
              **Degenerate for instance-normalized backbones** (emits a warning).
            * an array shaped like ``x`` -- an explicit baseline.
        n_steps: number of Riemann (midpoint) steps for the path integral.
        n_baselines: number of independent ``"noise"`` baselines to average over
            (Expected Gradients). Ignored (forced to 1) for explicit/constant
            baselines.
        internal_batch_size: maximum number of interpolation points to embed at
            once. ``None`` keeps the original one-batch behavior; smaller values
            trade a little runtime for much lower activation memory.
        reduce: how the embedding vector is reduced to the scalar objective
            ``g`` -- ``'l2'`` (default), ``'sum'`` or ``'mean'``.
        seed: RNG seed for the noise baseline(s).
        device: device for the computation; defaults to ``x``'s device (or CPU).

    Returns:
        :class:`EmbeddingAttribution`.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_baselines < 1:
        raise ValueError(f"n_baselines must be >= 1, got {n_baselines}")

    if isinstance(x, torch.Tensor):
        dev = torch.device(device) if device is not None else x.device
        x_t = x.detach().to(device=dev, dtype=torch.float32)
    else:
        dev = torch.device(device) if device is not None else torch.device("cpu")
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)

    # Build the list of baseline tensors.
    baselines: list[torch.Tensor] = []
    if isinstance(baseline, str) and baseline == "noise":
        mu = float(x_t.mean().item())
        sigma = float(x_t.std().item())
        sigma = sigma if sigma > 1e-8 else 1.0
        rng = np.random.default_rng(seed)
        for _ in range(n_baselines):
            noise = rng.standard_normal(tuple(x_t.shape)).astype(np.float32)
            baselines.append(torch.as_tensor(mu + sigma * noise, device=dev))
        baseline_name = "noise"
    else:
        if n_baselines != 1:
            warnings.warn("n_baselines is ignored for non-'noise' baselines; using 1.", stacklevel=2)
        if isinstance(baseline, str) and baseline in ("zero", "mean"):
            const = 0.0 if baseline == "zero" else float(x_t.mean().item())
            baselines.append(torch.full_like(x_t, const))
            baseline_name = baseline
            warnings.warn(
                f"A constant '{baseline}' baseline is degenerate for instance-normalized models "
                "(scale/shift invariance) and typically yields ~0 attribution; prefer baseline='noise'.",
                stacklevel=2,
            )
        elif isinstance(baseline, (int, float)):
            baselines.append(torch.full_like(x_t, float(baseline)))
            baseline_name = f"const({float(baseline):g})"
            warnings.warn(
                "A constant baseline is degenerate for instance-normalized models; prefer baseline='noise'.",
                stacklevel=2,
            )
        else:
            x0 = (
                baseline.detach().to(device=dev, dtype=torch.float32)
                if isinstance(baseline, torch.Tensor)
                else torch.as_tensor(np.asarray(baseline, dtype=np.float32), device=dev)
            )
            if x0.shape != x_t.shape:
                raise ValueError(f"baseline shape {tuple(x0.shape)} must match x shape {tuple(x_t.shape)}")
            baselines.append(x0)
            baseline_name = "array"

    attrs: list[Array] = []
    deltas: list[float] = []
    emb_dim = 0
    for x0 in baselines:
        attr, edelta, emb_dim = _ig_single(
            embed_fn,
            x_t,
            x0,
            n_steps=n_steps,
            reduce=reduce,
            internal_batch_size=internal_batch_size,
        )
        attrs.append(attr)
        deltas.append(edelta)

    attribution = np.mean(np.stack(attrs, axis=0), axis=0).astype(np.float32)
    embedding_delta = float(np.mean(deltas))
    embedding_delta_abs_mean = float(np.mean(np.abs(deltas)))
    overall = float(attribution.sum())
    abs_effect = float(np.abs(attribution).sum())
    eps = 1e-12
    convergence = float(abs(overall - embedding_delta) / (embedding_delta_abs_mean + eps))

    return EmbeddingAttribution(
        attribution=attribution,
        overall_effect=overall,
        abs_effect=abs_effect,
        embedding_delta=embedding_delta,
        embedding_delta_abs_mean=embedding_delta_abs_mean,
        convergence_delta=convergence,
        n_steps=int(n_steps),
        n_baselines=len(baselines),
        baseline=baseline_name,
        reduce=str(reduce),
        embedding_dim=int(emb_dim),
    )


__all__ = [
    "EmbeddingAttribution",
    "integrated_gradients_embedding",
]
