# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from examples import finetune_example


def make_args(**overrides) -> SimpleNamespace:
    args = {
        "ckpt_init": "auto",
        "use_cross_channel": False,
        "standardizer_init": "standardizer.pkl",
        "repo_id": "nvidia/nv-tesseract-forecasting",
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_resolve_checkpoint_init_can_be_disabled():
    assert finetune_example.resolve_checkpoint_init(make_args(ckpt_init="none")) is None
    assert finetune_example.resolve_checkpoint_init(make_args(ckpt_init="false")) is None
    assert finetune_example.resolve_checkpoint_init(make_args(ckpt_init="0")) is None


def test_resolve_checkpoint_init_uses_explicit_checkpoint_path():
    assert finetune_example.resolve_checkpoint_init(make_args(ckpt_init="custom.pt")) == "custom.pt"


def test_resolve_checkpoint_init_downloads_standard_checkpoint(monkeypatch):
    calls = []

    def fake_download_model_weights(*, standardizer_pkl, ckpt, repo_id):
        calls.append({"standardizer_pkl": standardizer_pkl, "ckpt": ckpt, "repo_id": repo_id})
        return standardizer_pkl, f"downloaded/{ckpt}"

    monkeypatch.setattr(finetune_example, "download_model_weights", fake_download_model_weights)

    ckpt_path = finetune_example.resolve_checkpoint_init(make_args())

    assert ckpt_path == f"downloaded/{finetune_example.CHECKPOINT_BASE}"
    assert calls == [
        {
            "standardizer_pkl": "standardizer.pkl",
            "ckpt": finetune_example.CHECKPOINT_BASE,
            "repo_id": "nvidia/nv-tesseract-forecasting",
        }
    ]


def test_resolve_checkpoint_init_downloads_cross_channel_checkpoint(monkeypatch):
    calls = []

    def fake_download_model_weights(*, standardizer_pkl, ckpt, repo_id):
        calls.append({"standardizer_pkl": standardizer_pkl, "ckpt": ckpt, "repo_id": repo_id})
        return standardizer_pkl, f"downloaded/{ckpt}"

    monkeypatch.setattr(finetune_example, "download_model_weights", fake_download_model_weights)

    ckpt_path = finetune_example.resolve_checkpoint_init(make_args(use_cross_channel=True))

    assert ckpt_path == f"downloaded/{finetune_example.CHECKPOINT_CROSS_CHANNEL}"
    assert calls[0]["ckpt"] == finetune_example.CHECKPOINT_CROSS_CHANNEL
