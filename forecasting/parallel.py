# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

"""
Parallel-execution utilities for the interpretability framework.

This module provides reusable parallel-execution building blocks:

* **Multi-GPU data parallelism**: a small fork-join helper
  (:func:`parallel_map_with_init`) that distributes a list of coarse work
  units across one process per visible GPU. Each worker initialises its own
  model on its assigned device and pulls items off a shared queue.

* **CPU threading**: :func:`parallel_apply_threads` runs independent NumPy
  workloads concurrently and is used by the optional local surrogate fit.

Device discovery, worker-count selection, and generic chunking helpers are
also available to callers that need to orchestrate larger workloads.

Design notes
------------
The multi-GPU helper is intentionally minimal. It supports torch's "spawn"
start method (required for CUDA contexts) and falls back to in-process
execution when only one device is available, so the same code path works
on CPU/MPS/single-GPU laptops without spawning subprocesses.

We deliberately avoid ``DataParallel`` / ``DistributedDataParallel`` because
the worker-pool API targets coarse, independent work units. A per-process
model copy and shared work queue fit that workload without DDP coordination.
"""

import contextlib
import os
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------


def available_devices() -> list[torch.device]:
    """Return all GPUs visible to PyTorch, falling back to ``[mps]`` or ``[cpu]``.

    The order matches ``torch.cuda.device_count()`` so it is stable across
    runs given the same ``CUDA_VISIBLE_DEVICES``.
    """
    if torch.cuda.is_available():
        n = int(torch.cuda.device_count())
        if n > 0:
            return [torch.device(f"cuda:{i}") for i in range(n)]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return [torch.device("mps")]
    return [torch.device("cpu")]


def parse_devices_arg(spec: str | None) -> list[torch.device]:
    """Parse a CLI ``--devices`` argument.

    Accepted forms:

    * ``""`` / ``None`` / ``"auto"`` -- use :func:`available_devices`.
    * ``"cpu"`` -- single CPU device (forces sequential execution).
    * ``"cuda"`` -- all visible CUDA devices.
    * ``"cuda:0"`` -- one specific CUDA device.
    * ``"cuda:0,cuda:2"`` / ``"0,2"`` -- explicit list of CUDA devices.

    Unknown / invalid devices fall back to :func:`available_devices` with a
    printed warning so a misconfigured CLI does not silently mask a real
    multi-GPU environment.
    """
    s = (spec or "").strip().lower()
    if not s or s == "auto":
        return available_devices()
    if s == "cpu":
        return [torch.device("cpu")]
    if s == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return [torch.device("mps")]
        logger.warning("requested 'mps' but it is not available; falling back to auto.")
        return available_devices()
    if s == "cuda":
        return [torch.device(f"cuda:{i}") for i in range(int(torch.cuda.device_count()))] or available_devices()

    parts = [p.strip() for p in s.split(",") if p.strip()]
    devs: list[torch.device] = []
    for p in parts:
        if p.isdigit():
            devs.append(torch.device(f"cuda:{int(p)}"))
        else:
            devs.append(torch.device(p))
    if not devs:
        return available_devices()
    return devs


def recommend_num_workers(
    n_items: int,
    *,
    devices: Sequence[torch.device] | None = None,
    requested: int | None = None,
) -> int:
    """Suggest a worker count given item count, devices, and a user override.

    Heuristic:
      * If ``requested`` is positive, honour it but cap at ``n_items``.
      * Otherwise, use ``len(devices)`` (or 1 for non-CUDA, since spawning
        a subprocess for CPU work has no benefit and forks can fight for
        the same cores).
    """
    if devices is None:
        devices = available_devices()
    if not devices:
        return 1
    if requested is not None and requested > 0:
        return max(1, min(int(requested), max(1, n_items)))
    if devices[0].type != "cuda":
        return 1
    return max(1, min(len(devices), max(1, n_items)))


# ---------------------------------------------------------------------------
# Generic chunking
# ---------------------------------------------------------------------------


def chunk_indices(n: int, *, chunk: int) -> list[tuple[int, int]]:
    """Split ``range(n)`` into ``[start, stop)`` pairs of length ``<= chunk``."""
    chunk = max(1, int(chunk))
    return [(i, min(n, i + chunk)) for i in range(0, max(0, int(n)), chunk)]


def split_for_workers(items: Sequence[T], n_workers: int) -> list[list[T]]:
    """Split ``items`` into ``n_workers`` contiguous shards of near-equal size.

    Contiguous (rather than round-robin) shards keep each worker's slice in
    cache-friendly order and make per-worker progress logs interpretable.
    """
    n_workers = max(1, int(n_workers))
    n = len(items)
    if n_workers == 1 or n == 0:
        return [list(items)]
    base, extra = divmod(n, n_workers)
    out: list[list[T]] = []
    cursor = 0
    for w in range(n_workers):
        size = base + (1 if w < extra else 0)
        out.append(list(items[cursor : cursor + size]))
        cursor += size
    return out


# ---------------------------------------------------------------------------
# Multi-GPU worker pool
# ---------------------------------------------------------------------------


@dataclass
class _WorkerSpec:
    rank: int
    device_str: str
    init_fn: Callable[[int, torch.device], Any]
    work_fn: Callable[[Any, torch.device, Any], Any]


# Per-process worker state used inside spawned subprocesses. Wrapped in a
# mutable container so we can update it from inside the worker loop without
# tripping ``global`` statements (each spawned process gets its own copy
# anyway because the ``spawn`` start method does not share memory).
_WORKER_STATE: dict[str, Any] = {"ctx": None, "device": None}


def _shutdown_workers(in_queue, procs, *, join_timeout: float = 5.0) -> None:
    """Best-effort shutdown of spawned workers.

    Sends one ``None`` sentinel per worker down the shared input queue,
    swallowing any error (the queue may already be closed if a worker
    died), and then joins each subprocess. Errors are intentionally
    suppressed because this is only ever called on a teardown path where
    we are about to raise the real failure to the caller anyway.
    """
    for _ in procs:
        with contextlib.suppress(Exception):
            in_queue.put(None)
    for p in procs:
        p.join(timeout=join_timeout)


def _gpu_worker_loop(
    spec: _WorkerSpec,
    in_queue,
    out_queue,
) -> None:
    """Worker entrypoint executed in a subprocess.

    The loop initialises a per-worker context (typically a model loaded
    onto ``spec.device_str``), then drains ``(item_id, item)`` tuples from
    ``in_queue`` until it sees a ``None`` sentinel, putting
    ``(item_id, "result", result)`` or ``(item_id, "error", traceback)`` on
    ``out_queue`` for each one.
    """
    try:
        device = torch.device(spec.device_str)
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device.index)
        _WORKER_STATE["device"] = device
        _WORKER_STATE["ctx"] = spec.init_fn(spec.rank, device)
    except Exception:
        out_queue.put((-1, "init_error", traceback.format_exc()))
        return

    out_queue.put((-1, "ready", spec.rank))

    while True:
        msg = in_queue.get()
        if msg is None:
            break
        item_id, item = msg
        try:
            result = spec.work_fn(_WORKER_STATE["ctx"], _WORKER_STATE["device"], item)
            out_queue.put((item_id, "result", result))
        except Exception:
            out_queue.put((item_id, "error", traceback.format_exc()))

    out_queue.put((-1, "done", spec.rank))


def parallel_map_with_init(
    init_fn: Callable[[int, torch.device], Any],
    work_fn: Callable[[Any, torch.device, T], R],
    items: Sequence[T],
    *,
    devices: Sequence[torch.device] | None = None,
    num_workers: int | None = None,
    progress: bool = False,
    progress_desc: str = "parallel",
    on_result: Callable[[int, R], None] | None = None,
) -> list[R]:
    """Run ``work_fn(ctx, device, item)`` over ``items``, distributed across workers.

    Each worker calls ``init_fn(rank, device)`` exactly once and reuses the
    returned ``ctx`` (typically a loaded model) for every item assigned to
    it. Items are dispatched in submission order; results are returned in
    the same order regardless of completion order.

    Args:
      init_fn: Builds the per-worker context. Must be picklable. Called
        once per worker as ``init_fn(rank: int, device: torch.device)``.
      work_fn: Computes one result. Must be picklable. Called as
        ``work_fn(ctx, device, item)``.
      items: Work units to process. Must be picklable when more than one
        worker is used.
      devices: Devices to use. Defaults to :func:`available_devices`.
      num_workers: Override worker count. Defaults to
        :func:`recommend_num_workers`.
      progress: If True, logs simple progress lines as items complete.
      progress_desc: Tag used in progress lines.
      on_result: Optional callback invoked as ``on_result(item_id, result)``
        as soon as each result arrives -- useful for streaming progress
        without buffering everything.

    Returns:
      A list of results, one per input item, in the same order.
    """
    items_list = list(items)
    n_items = len(items_list)
    if n_items == 0:
        return []

    devs = list(devices) if devices is not None else available_devices()
    if not devs:
        devs = [torch.device("cpu")]
    n_workers = recommend_num_workers(n_items, devices=devs, requested=num_workers)

    # Single-worker fast path: avoid spawn overhead and let exceptions
    # propagate naturally to the caller for easier debugging.
    if n_workers <= 1:
        device = devs[0]
        ctx = init_fn(0, device)
        results: list[R] = []
        for i, item in enumerate(items_list):
            r = work_fn(ctx, device, item)
            if on_result is not None:
                on_result(i, r)
            results.append(r)
            if progress:
                logger.info("[%s] %d/%d", progress_desc, i + 1, n_items)
        return results

    # Multi-worker path. Use torch.multiprocessing with the spawn context so
    # CUDA initialisation in the parent does not poison the child.
    import torch.multiprocessing as mp_torch

    mp_ctx = mp_torch.get_context("spawn")

    # Map a worker rank to a device, cycling if there are more workers than
    # devices (lets users oversubscribe a GPU when memory allows).
    worker_devices = [devs[r % len(devs)] for r in range(n_workers)]

    # A single shared work queue gives dynamic load balancing: whenever a
    # worker finishes an item it immediately pulls the next pending one, so
    # a GPU never sits idle while another grinds through expensive items.
    in_queue = mp_ctx.Queue()
    out_queue = mp_ctx.Queue()
    procs = []
    for r, dev in enumerate(worker_devices):
        spec = _WorkerSpec(rank=r, device_str=str(dev), init_fn=init_fn, work_fn=work_fn)
        p = mp_ctx.Process(
            target=_gpu_worker_loop,
            args=(spec, in_queue, out_queue),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Wait for all workers to finish their init step. If any worker fails
    # to initialise we tear the rest down before raising so we do not leave
    # orphaned subprocesses behind.
    ready = 0
    init_errors: list[str] = []
    while ready < n_workers:
        item_id, kind, payload = out_queue.get()
        if kind == "ready":
            ready += 1
        elif kind == "init_error":
            init_errors.append(str(payload))
            ready += 1  # so we don't deadlock
        else:  # pragma: no cover -- should not happen before any work submitted
            raise RuntimeError(f"unexpected worker message {kind!r} during init")
    if init_errors:
        _shutdown_workers(in_queue, procs)
        raise RuntimeError(f"{len(init_errors)} worker(s) failed to initialise:\n" + "\n---\n".join(init_errors))

    # Submit every item to the shared queue; workers pull dynamically so the
    # load stays balanced regardless of per-item cost heterogeneity.
    for idx, item in enumerate(items_list):
        in_queue.put((idx, item))

    results_buf: list[Any] = [None] * n_items
    received = 0
    started = time.time()
    while received < n_items:
        item_id, kind, payload = out_queue.get()
        if kind == "result":
            results_buf[item_id] = payload
            received += 1
            if on_result is not None:
                on_result(item_id, payload)  # type: ignore[arg-type]
            if progress:
                elapsed = time.time() - started
                rate = received / max(1e-6, elapsed)
                logger.info("[%s] %d/%d  (%.2f items/s)", progress_desc, received, n_items, rate)
        elif kind == "error":
            _shutdown_workers(in_queue, procs)
            raise RuntimeError(f"worker raised on item {item_id}:\n{payload}")
        # ready/done messages from a previous phase can be safely ignored.

    for _ in procs:
        in_queue.put(None)
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    return results_buf  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Joblib-style parallel for CPU-bound work
# ---------------------------------------------------------------------------


def parallel_apply_threads(
    work_fn: Callable[[T], R],
    items: Iterable[T],
    *,
    n_jobs: int | None = None,
) -> list[R]:
    """Run ``work_fn`` over ``items`` using a thread pool.

    Useful for CPU-bound NumPy work that releases the GIL (e.g. BLAS calls
    inside ISTA). Falls back to a sequential map when ``n_jobs <= 1`` or
    only a single item is supplied so the call site has identical
    semantics in both modes.
    """
    items_list = list(items)
    if not items_list:
        return []
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1))
    if n_jobs <= 1 or len(items_list) == 1:
        return [work_fn(it) for it in items_list]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
        return list(ex.map(work_fn, items_list))


__all__ = [
    "available_devices",
    "chunk_indices",
    "parallel_apply_threads",
    "parallel_map_with_init",
    "parse_devices_arg",
    "recommend_num_workers",
    "split_for_workers",
]
