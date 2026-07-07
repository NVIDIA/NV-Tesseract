# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sdk
from sdk import inference_worker


class FakeModel:
    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def install_worker_stubs(monkeypatch, *, fake_evaluate):
    fake_torch = types.SimpleNamespace(
        load=lambda *args, **kwargs: {"model": {"weights": 1}},
        device=lambda value: value,
        use_deterministic_algorithms=lambda *args, **kwargs: None,
        manual_seed=lambda seed: None,
        backends=types.SimpleNamespace(
            cudnn=types.SimpleNamespace(
                deterministic=False,
                benchmark=False,
            )
        ),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            manual_seed=lambda seed: None,
        ),
    )

    fake_models_pkg = types.ModuleType("models")
    fake_main_model = types.ModuleType("models.main_model")
    fake_main_model.TSDiffuser_Generic = lambda *args, **kwargs: FakeModel()
    fake_models_pkg.main_model = fake_main_model

    fake_inference_ad = types.ModuleType("sdk.inference_ad")
    fake_inference_ad.evaluate_ad_tesseract2 = fake_evaluate
    fake_inference_ad.get_dataloader = lambda *args, **kwargs: (["loader1"], ["loader2"])
    fake_inference_ad.get_dataloader_from_windows = None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "models.main_model", fake_main_model)
    monkeypatch.setitem(sys.modules, "sdk.inference_ad", fake_inference_ad)
    monkeypatch.setattr(sdk, "inference_ad", fake_inference_ad, raising=False)

    return fake_inference_ad


def build_worker_args(tmp_path: Path) -> tuple[Path, Path]:
    args_path = tmp_path / "args.json"
    result_path = tmp_path / "result.json"
    args = {
        "gpu_id": 0,
        "data_chunk": {
            "window_shm": True,
            "shm_name": "fake-shm",
            "shape": [2, 3, 4],
            "dtype": "float32",
            "window_indices": [0, 1],
            "split": 4,
        },
        "model_path": "model.pth",
        "config": {"model": {"target_dim": 4}},
        "target_dim": 4,
        "scale_factor": 1.0,
        "nsample": 5,
        "seed": 7,
        "deterministic": False,
        "preprocess_model_dir": None,
        "use_dpm_solver": False,
        "dpm_steps": 20,
    }
    args_path.write_text(json.dumps(args))
    return args_path, result_path


def test_worker_keeps_shared_memory_open_until_after_evaluate(monkeypatch, tmp_path):
    events = []

    class FakeSharedMemory:
        def __init__(self, name):
            assert name == "fake-shm"
            self.buf = bytearray(np.zeros((2, 3, 4), dtype=np.float32).nbytes)
            self.closed = False

        def close(self):
            self.closed = True
            events.append("close")

    fake_shm = FakeSharedMemory("fake-shm")

    def fake_evaluate(model, loader1, loader2, nsample, use_dpm_solver, dpm_steps):
        events.append(("evaluate", fake_shm.closed, loader1, loader2, nsample, use_dpm_solver, dpm_steps))
        return {
            "residual": np.array([1.0]),
            "residual_l2": np.array([2.0]),
            "target": np.array([[3.0]]),
            "recon": np.array([[4.0]]),
        }

    fake_inference_ad = install_worker_stubs(monkeypatch, fake_evaluate=fake_evaluate)

    def fake_get_dataloader_from_windows(windows, *, split, window_indices):
        events.append(("loaders", fake_shm.closed, windows.shape, split, tuple(window_indices)))
        return ["loader1"], ["loader2"]

    fake_inference_ad.get_dataloader_from_windows = fake_get_dataloader_from_windows
    monkeypatch.setattr(inference_worker.shared_memory, "SharedMemory", lambda name: fake_shm)

    args_path, result_path = build_worker_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["inference_worker.py", str(args_path), str(result_path)])

    inference_worker.main()

    saved = json.loads(result_path.read_text())
    assert saved["gpu_id"] == 0
    assert saved["results"]["residual"] == [1.0]
    assert events == [
        ("loaders", False, (2, 3, 4), 4, (0, 1)),
        ("evaluate", False, ["loader1"], ["loader2"], 5, False, 20),
        "close",
    ]
    assert fake_shm.closed is True


def test_worker_closes_shared_memory_when_evaluate_fails(monkeypatch, tmp_path):
    events = []

    class FakeSharedMemory:
        def __init__(self, name):
            assert name == "fake-shm"
            self.buf = bytearray(np.zeros((2, 3, 4), dtype=np.float32).nbytes)
            self.closed = False

        def close(self):
            self.closed = True
            events.append("close")

    fake_shm = FakeSharedMemory("fake-shm")

    def fake_evaluate(model, loader1, loader2, nsample, use_dpm_solver, dpm_steps):
        events.append(("evaluate", fake_shm.closed))
        raise RuntimeError("boom")

    fake_inference_ad = install_worker_stubs(monkeypatch, fake_evaluate=fake_evaluate)
    fake_inference_ad.get_dataloader_from_windows = lambda windows, *, split, window_indices: (["loader1"], ["loader2"])
    monkeypatch.setattr(inference_worker.shared_memory, "SharedMemory", lambda name: fake_shm)

    args_path, result_path = build_worker_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["inference_worker.py", str(args_path), str(result_path)])

    with pytest.raises(SystemExit, match="1"):
        inference_worker.main()

    saved = json.loads(result_path.read_text())
    assert saved["gpu_id"] == 0
    assert saved["error"] == "boom"
    assert events == [("evaluate", False), "close"]
    assert fake_shm.closed is True
