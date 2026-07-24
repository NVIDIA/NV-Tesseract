# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

try:
    from backbone import BackbonePipeline
except ImportError:
    from forecasting.backbone import BackbonePipeline

DEFAULT_MODEL_NAME = os.environ.get(
    "TESSERACT_BACKBONE_MODEL",
    "AutonLab/MOMENT-1-large",
)


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    forecast_horizon: int = 96,
    seq_len: int = 2048,
    head_dropout: float = 0.1,
    weight_decay: float = 0.0,
    freeze_encoder: bool = True,
    freeze_embedder: bool = True,
    freeze_head: bool = False,
    use_cross_channel: bool = False,
    cross_channel_heads: int = 8,
    cross_channel_dropout: float = 0.1,
    local_files_only: bool = False,
    device: str | None = None,
) -> torch.nn.Module:
    """
    Constructs and initializes the forecasting backbone with a trainable head.
    """
    pipe = BackbonePipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "seq_len": seq_len,
            "forecast_horizon": forecast_horizon,
            "head_dropout": head_dropout,
            "weight_decay": weight_decay,
            "freeze_encoder": freeze_encoder,
            "freeze_embedder": freeze_embedder,
            "freeze_head": freeze_head,
            "use_cross_channel": use_cross_channel,
            "cross_channel_heads": cross_channel_heads,
            "cross_channel_dropout": cross_channel_dropout,
        },
        local_files_only=local_files_only,
    )
    pipe.init()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = pipe.to(device)
    return pipe


def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
