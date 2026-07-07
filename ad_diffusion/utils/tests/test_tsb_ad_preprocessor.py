# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import tsb_ad_preprocessor


def test_preprocess_simple_uses_feature_only_target_dim(monkeypatch):
    captured = {}
    stub_output = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    raw_data = np.array([[10.0], [20.0]], dtype=np.float32)

    def fake_preprocess_for_inference(*, data, domain, model_dir, target_dim, add_metadata):
        captured["data"] = data
        captured["domain"] = domain
        captured["model_dir"] = model_dir
        captured["target_dim"] = target_dim
        captured["add_metadata"] = add_metadata
        return stub_output.copy()

    monkeypatch.setattr(tsb_ad_preprocessor, "preprocess_for_inference", fake_preprocess_for_inference)

    result = tsb_ad_preprocessor.preprocess_simple(
        raw_data,
        model_dir="models",
        domain="Sensor",
        scale_factor=2.5,
    )

    assert captured["data"] is raw_data
    assert captured["domain"] == "Sensor"
    assert captured["model_dir"] == "models"
    assert captured["target_dim"] == 38
    assert captured["add_metadata"] is False
    np.testing.assert_allclose(result, stub_output * 2.5)
