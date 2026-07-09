# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-file backbone model with optional Cross-Channel Attention.

New parameter: use_cross_channel (bool, default False)
  - False → identical base architecture, loads old weights with strict=True
  - True  → adds CrossChannelAttention after T5 encoder, loads old weights with strict=False
             (cross_channel_attn weights initialise randomly; all other weights load normally)

Exposes:
    from backbone import BackbonePipeline
    from backbone.utils.utils import control_randomness

Requires: torch, transformers, huggingface_hub, numpy
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import types
import warnings
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from transformers import T5Config, T5EncoderModel, T5Model

# ---------------------------------------------------------------------------
# common — task name constants
# ---------------------------------------------------------------------------


@dataclass
class TASKS:
    RECONSTRUCTION: str = "reconstruction"
    FORECASTING: str = "forecasting"
    CLASSIFICATION: str = "classification"
    EMBED: str = "embedding"


# ---------------------------------------------------------------------------
# data — output container
# ---------------------------------------------------------------------------


@dataclass
class TimeseriesOutputs:
    forecast: npt.NDArray = None
    anomaly_scores: npt.NDArray = None
    logits: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


class NamespaceWithDefaults(Namespace):
    @classmethod
    def from_namespace(cls, namespace):
        new_instance = cls()
        for attr in dir(namespace):
            if not attr.startswith("__"):
                setattr(new_instance, attr, getattr(namespace, attr))
        return new_instance

    def getattr(self, key, default=None):
        return getattr(self, key, default)


def control_randomness(seed: int = 13):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_anomaly_criterion(anomaly_criterion: str = "mse"):
    if anomaly_criterion == "mse":
        return nn.MSELoss(reduction="none")
    if anomaly_criterion == "mae":
        return nn.L1Loss(reduction="none")
    raise ValueError(f"Anomaly criterion {anomaly_criterion} not supported.")


def _reduce(metric, reduction="mean", axis=None):
    if reduction == "mean":
        return np.nanmean(metric, axis=axis)
    if reduction == "sum":
        return np.nansum(metric, axis=axis)
    if reduction == "none":
        return metric


class EarlyStopping:
    def __init__(self, patience: int = 3, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info("EarlyStopping counter: %s out of %s", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class Masking:
    def __init__(self, mask_ratio: float = 0.3, patch_len: int = 8, stride: int | None = None):
        self.mask_ratio = mask_ratio
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride

    @staticmethod
    def convert_seq_to_patch_view(mask: torch.Tensor, patch_len: int = 8, stride: int | None = None):
        stride = patch_len if stride is None else stride
        mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
        return (mask.sum(dim=-1) == patch_len).long()

    @staticmethod
    def convert_patch_to_seq_view(mask: torch.Tensor, patch_len: int = 8):
        return mask.repeat_interleave(patch_len, dim=-1)

    def generate_mask(self, x: torch.Tensor, input_mask: torch.Tensor | None = None):
        if x.ndim == 4:
            return self._mask_patch_view(x, input_mask=input_mask)
        if x.ndim == 3:
            return self._mask_seq_view(x, input_mask=input_mask)

    def _mask_patch_view(self, x, input_mask=None):
        input_mask = self.convert_seq_to_patch_view(input_mask, self.patch_len, self.stride)
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True)
        batch_size, _, n_patches, _ = x.shape
        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(batch_size, n_patches, device=x.device)
        noise = torch.where(input_mask == 1, noise, torch.ones_like(noise))
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.zeros([batch_size, n_patches], device=x.device)
        for i in range(batch_size):
            mask[i, : len_keep[i]] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask.long()

    def _mask_seq_view(self, x, input_mask=None):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        mask = self._mask_patch_view(x, input_mask=input_mask)
        return self.convert_patch_to_seq_view(mask, self.patch_len).long()


# ---------------------------------------------------------------------------
# RevIN
# ---------------------------------------------------------------------------


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    return (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)


def nanstd(tensor, dim=None, keepdim=False):
    return nanvar(tensor, dim=dim, keepdim=keepdim).sqrt()


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str = "norm", mask: torch.Tensor = None):
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x, mask=None):
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]))
        n_channels = x.shape[1]
        mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
        masked_x = torch.where(mask, x, torch.nan)
        self.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
        self.stdev = nanstd(masked_x, dim=-1, keepdim=True).detach() + self.eps

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, model_name="foundation"):
        super().__init__()
        self.model_name = model_name
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.model_name in ("foundation", "TimesNet", "GPT4TS"):
            return self.pe[:, : x.size(2)]
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        seq_len: int = 512,
        patch_len: int = 8,
        stride: int = 8,
        patch_dropout: float = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding
        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        if orth_gain is not None:
            nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(patch_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mask = Masking.convert_seq_to_patch_view(mask, patch_len=self.patch_len).unsqueeze(-1)
        n_channels = x.shape[1]
        mask = mask.repeat_interleave(self.d_model, dim=-1).unsqueeze(1).repeat(1, n_channels, 1, 1)
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x)
        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if self.stride != self.patch_len:
            warnings.warn("Stride and patch length are not equal.")

    def forward(self, x):
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)


# ---------------------------------------------------------------------------
# CrossChannelAttention  ← NEW
#
# Applied after T5 encoder output [B, C, n_patches, d_model].
# For every patch position, all C channels attend to each other.
#
# Key properties:
#   • Residual + LayerNorm → stable training
#   • Weights are named "cross_channel_attn.*" so old checkpoints
#     (which have no such keys) load cleanly with strict=False
# ---------------------------------------------------------------------------


class CrossChannelAttention(nn.Module):
    """
    Multi-head self-attention across the channel dimension.

    Input / output shape:  [B, C, n_patches, d_model]

    For each patch position the C channel embeddings attend to each other,
    letting the model learn inter-channel correlations without touching the
    T5 encoder weights.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, P, D]
        B, C, P, D = x.shape
        # Treat every (batch, patch) pair as an independent sequence of C tokens
        x_r = x.permute(0, 2, 1, 3).reshape(B * P, C, D)  # [B*P, C, D]
        attn_out, _ = self.attn(x_r, x_r, x_r)
        x_r = self.norm(x_r + self.dropout(attn_out))  # residual + norm
        return x_r.reshape(B, P, C, D).permute(0, 2, 1, 3)  # [B, C, P, D]


# ---------------------------------------------------------------------------
# Forecasting / classification heads
# ---------------------------------------------------------------------------


class PretrainHead(nn.Module):
    def __init__(self, d_model: int = 768, patch_len: int = 8, head_dropout: float = 0.1, orth_gain: float = 1.41):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)
        if orth_gain is not None:
            nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(self.dropout(x))
        return x.flatten(start_dim=2, end_dim=3)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: float = 0.1,
        reduction: str = "concat",
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        if reduction == "mean":
            self.linear = nn.Linear(d_model, n_classes)
        elif reduction == "concat":
            self.linear = nn.Linear(n_channels * d_model, n_classes)
        else:
            raise ValueError(f"Reduction {reduction} not implemented.")

    def forward(self, x, input_mask: torch.Tensor = None):
        x = torch.mean(x, dim=1)
        return self.linear(self.dropout(x))


class ForecastingHead(nn.Module):
    def __init__(self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: float = 0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# BackboneModel  — channel-independent base + optional cross-channel attention
# ---------------------------------------------------------------------------

SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class BackboneModel(nn.Module):
    """
    Backbone model with an optional CrossChannelAttention layer.

    Config flags (passed via model_kwargs):
        use_cross_channel   (bool, default False)
        cross_channel_heads (int,  default 8)
        cross_channel_dropout (float, default 0.1)

    Backward compatibility:
        use_cross_channel=False  →  architecture == original base model,
                                    loads old weights with strict=True
        use_cross_channel=True   →  extra cross_channel_attn.* parameters,
                                    loads old weights with strict=False
    """

    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len

        # ── standard backbone layers (identical to original base) ──────────
        self.normalizer = RevIN(num_features=1, affine=config.getattr("revin_affine", False))
        self.tokenizer = Patching(patch_len=config.patch_len, stride=config.patch_stride_len)
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            patch_dropout=config.getattr("patch_dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
        self.encoder = self._get_transformer_backbone(config)
        self.head = self._get_head(self.task_name)

        # ── cross-channel attention (NEW, optional) ─────────────────────────
        self.use_cross_channel = config.getattr("use_cross_channel", False)
        if self.use_cross_channel:
            self.cross_channel_attn = CrossChannelAttention(
                d_model=config.d_model,
                n_heads=config.getattr("cross_channel_heads", 8),
                dropout=config.getattr("cross_channel_dropout", 0.1),
            )

        # ── freeze flags ────────────────────────────────────────────────────
        if config.getattr("freeze_embedder", True):
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if config.getattr("freeze_encoder", True):
            self.encoder = freeze_parameters(self.encoder)
        if config.getattr("freeze_head", False):
            self.head = freeze_parameters(self.head)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _update_inputs(self, config, **kwargs) -> NamespaceWithDefaults:
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config) -> NamespaceWithDefaults:
        if config.d_model is None and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS:
            config.d_model = config.t5_config["d_model"]
        elif config.d_model is None:
            raise ValueError("d_model must be specified.")
        if config.transformer_type not in ("encoder_only", "decoder_only", "encoder_decoder"):
            raise ValueError("transformer_type must be one of ['encoder_only', 'decoder_only', 'encoder_decoder']")
        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length != patch length.")
        return config

    def _get_head(self, task_name: str) -> nn.Module:
        _t = TASKS()
        if task_name != _t.RECONSTRUCTION:
            warnings.warn("Only reconstruction head is pre-trained; others need fine-tuning.")
        if task_name == _t.RECONSTRUCTION:
            return PretrainHead(
                self.config.d_model,
                self.config.patch_len,
                self.config.getattr("head_dropout", 0.1),
                self.config.getattr("orth_gain", 1.41),
            )
        if task_name == _t.CLASSIFICATION:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("head_dropout", 0.1),
                reduction=self.config.getattr("reduction", "concat"),
            )
        if task_name == _t.FORECASTING:
            num_patches = (
                max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
            ) // self.config.patch_stride_len + 1
            self.head_nf = self.config.d_model * num_patches
            return ForecastingHead(self.head_nf, self.config.forecast_horizon, self.config.getattr("head_dropout", 0.1))
        if task_name == _t.EMBED:
            return nn.Identity()
        raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        model_config = T5Config.from_dict(config.t5_config)
        if config.getattr("randomly_initialize_backbone", False):
            backbone = T5Model(model_config)
        else:
            backbone = T5EncoderModel(model_config)
        backbone = backbone.get_encoder()
        if config.getattr("enable_gradient_checkpointing", True):
            backbone.gradient_checkpointing_enable()
        return backbone

    def _apply_cross_channel(self, enc_out: torch.Tensor) -> torch.Tensor:
        """Apply cross-channel attention if enabled. enc_out: [B, C, P, D]"""
        if self.use_cross_channel:
            enc_out = self.cross_channel_attn(enc_out)
        return enc_out

    # ── forward methods ──────────────────────────────────────────────────────

    def __call__(self, *args: Any, **kwargs: Any) -> TimeseriesOutputs:
        return self.forward(*args, **kwargs)

    def embed(
        self, *, x_enc: torch.Tensor, input_mask: torch.Tensor = None, reduction: str = "mean", **kwargs
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape
        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model))

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        # ── cross-channel attention ─────────────────────────────────────────
        enc_out = self._apply_cross_channel(enc_out)

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(1, 1, self.config.d_model)
            enc_out = (input_mask_patch_view * enc_out).sum(dim=1) / input_mask_patch_view.sum(dim=1)
        elif reduction == "none":
            pass
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        return TimeseriesOutputs(embeddings=enc_out, input_mask=input_mask, metadata=reduction)

    def reconstruction(
        self, *, x_enc: torch.Tensor, input_mask: torch.Tensor = None, mask: torch.Tensor = None, **kwargs
    ) -> TimeseriesOutputs:
        batch_size, n_channels, _ = x_enc.shape
        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model))

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(inputs_embeds=enc_in, decoder_inputs_embeds=enc_in, attention_mask=attention_mask)
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        # ── cross-channel attention ─────────────────────────────────────────
        enc_out = self._apply_cross_channel(enc_out)

        dec_out = self.head(enc_out)
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out, pretrain_mask=mask, illegal_output=None)

    def forecast(self, *, x_enc: torch.Tensor, input_mask: torch.Tensor = None, **kwargs) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model))

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        # ── cross-channel attention ─────────────────────────────────────────
        enc_out = self._apply_cross_channel(enc_out)

        dec_out = self.head(enc_out)
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)

    def classify(
        self, *, x_enc: torch.Tensor, input_mask: torch.Tensor = None, reduction: str = "concat", **kwargs
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape
        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model))

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        # ── cross-channel attention ─────────────────────────────────────────
        enc_out = self._apply_cross_channel(enc_out)

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)
        elif reduction == "concat":
            enc_out = enc_out.permute(0, 2, 3, 1).reshape(batch_size, n_patches, self.config.d_model * n_channels)
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        logits = self.head(enc_out, input_mask=input_mask)
        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata=reduction)

    def forward(
        self, *, x_enc: torch.Tensor, input_mask: torch.Tensor = None, mask: torch.Tensor = None, **kwargs
    ) -> TimeseriesOutputs:
        _t = TASKS()
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if self.task_name == _t.RECONSTRUCTION:
            return self.reconstruction(x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs)
        if self.task_name == _t.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
        if self.task_name == _t.FORECASTING:
            return self.forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
        if self.task_name == _t.CLASSIFICATION:
            return self.classify(x_enc=x_enc, input_mask=input_mask, **kwargs)
        raise NotImplementedError(f"Task {self.task_name} not implemented.")


class BackbonePipeline(BackboneModel, PyTorchModelHubMixin):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        self.new_task_name = kwargs.get("model_kwargs", {}).pop("task_name", TASKS.RECONSTRUCTION)
        super().__init__(config, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("model_kwargs", {"task_name": TASKS.RECONSTRUCTION})
        kwargs["model_kwargs"].setdefault("task_name", TASKS.RECONSTRUCTION)
        config = Namespace(**kwargs["model_kwargs"])

        if config.task_name == TASKS.FORECASTING:
            if not hasattr(config, "forecast_horizon"):
                raise ValueError("forecast_horizon must be specified for forecasting.")
        if config.task_name == TASKS.CLASSIFICATION:
            if not hasattr(config, "n_channels"):
                raise ValueError("n_channels required for classification.")
            if not hasattr(config, "num_class"):
                raise ValueError("num_class required for classification.")

    def init(self) -> None:
        if self.new_task_name != TASKS.RECONSTRUCTION:
            self.task_name = self.new_task_name
            self.head = self._get_head(self.new_task_name)


# ---------------------------------------------------------------------------
# Forecasting metrics
# ---------------------------------------------------------------------------


@dataclass
class ForecastingMetrics:
    mae: object = None
    mse: object = None
    mape: object = None
    smape: object = None
    rmse: object = None


class sMAPELoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def _abs(self, input):
        return F.l1_loss(input, torch.zeros_like(input), reduction="none")

    def _divide_no_nan(self, a, b):
        div = a / b
        div[div != div] = 0.0
        div[div == float("inf")] = 0.0
        return div

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        delta_y = self._abs(input - target)
        scale = self._abs(target) + self._abs(input)
        error = self._divide_no_nan(delta_y, scale)
        return 200 * torch.nanmean(error)


def _divide_no_nan(a, b):
    div = a / b
    div[div != div] = 0.0
    div[div == float("inf")] = 0.0
    return div


def _fm_mae(y, y_hat, reduction="mean", axis=None):
    return _reduce(np.abs(y - y_hat), reduction=reduction, axis=axis)


def _fm_mse(y, y_hat, reduction="mean", axis=None):
    return _reduce(np.square(y - y_hat), reduction=reduction, axis=axis)


def _fm_rmse(y, y_hat, reduction="mean", axis=None):
    return np.sqrt(_fm_mse(y, y_hat, reduction, axis))


def _fm_mape(y, y_hat, reduction="mean", axis=None):
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    error = _divide_no_nan(delta_y, scale)
    return 100 * _reduce(error, reduction=reduction, axis=axis)


def _fm_smape(y, y_hat, reduction="mean", axis=None):
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    error = _divide_no_nan(delta_y, scale)
    return 200 * _reduce(error, reduction=reduction, axis=axis)


def get_forecasting_metrics(y, y_hat, reduction="mean", axis=None):
    return ForecastingMetrics(
        mae=_fm_mae(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
        mse=_fm_mse(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
        mape=_fm_mape(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
        smape=_fm_smape(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
        rmse=_fm_rmse(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
    )


# ---------------------------------------------------------------------------
# Register submodule paths under "backbone.*" for local utility imports.
# ---------------------------------------------------------------------------


def _register_submodules():
    _utils = types.ModuleType("backbone.utils")
    _utils_utils = types.ModuleType("backbone.utils.utils")
    _utils_utils.control_randomness = control_randomness
    _utils_utils.NamespaceWithDefaults = NamespaceWithDefaults
    _utils_utils.get_anomaly_criterion = get_anomaly_criterion
    _utils_utils.EarlyStopping = EarlyStopping
    _utils_utils._reduce = _reduce

    _utils_masking = types.ModuleType("backbone.utils.masking")
    _utils_masking.Masking = Masking

    _utils_metrics = types.ModuleType("backbone.utils.forecasting_metrics")
    _utils_metrics.get_forecasting_metrics = get_forecasting_metrics
    _utils_metrics.ForecastingMetrics = ForecastingMetrics
    _utils_metrics.sMAPELoss = sMAPELoss

    _utils.utils = _utils_utils
    _utils.masking = _utils_masking
    _utils.forecasting_metrics = _utils_metrics

    _data = types.ModuleType("backbone.data")
    _data_base = types.ModuleType("backbone.data.base")
    _data_base.TimeseriesOutputs = TimeseriesOutputs
    _data.base = _data_base

    _common = types.ModuleType("backbone.common")
    _common.TASKS = TASKS

    _models = types.ModuleType("backbone.models")
    _backbone_mod = types.ModuleType("backbone.models.backbone")
    _backbone_mod.BackboneModel = BackboneModel
    _backbone_mod.BackbonePipeline = BackbonePipeline
    _backbone_mod.CrossChannelAttention = CrossChannelAttention
    _backbone_mod.freeze_parameters = freeze_parameters
    _models.backbone = _backbone_mod

    _layers = types.ModuleType("backbone.models.layers")
    _embed = types.ModuleType("backbone.models.layers.embed")
    _embed.PatchEmbedding = PatchEmbedding
    _embed.Patching = Patching
    _embed.PositionalEmbedding = PositionalEmbedding
    _revin = types.ModuleType("backbone.models.layers.revin")
    _revin.RevIN = RevIN
    _layers.embed = _embed
    _layers.revin = _revin
    _models.layers = _layers

    sys.modules.update(
        {
            "backbone.utils": _utils,
            "backbone.utils.utils": _utils_utils,
            "backbone.utils.masking": _utils_masking,
            "backbone.utils.forecasting_metrics": _utils_metrics,
            "backbone.data": _data,
            "backbone.data.base": _data_base,
            "backbone.common": _common,
            "backbone.models": _models,
            "backbone.models.backbone": _backbone_mod,
            "backbone.models.layers": _layers,
            "backbone.models.layers.embed": _embed,
            "backbone.models.layers.revin": _revin,
        }
    )


_register_submodules()
