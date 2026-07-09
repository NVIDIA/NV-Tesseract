# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
diff_models.py
--------------------------------------------------------------------------

This module defines the core diffusion model architecture

Key Components:
---------------
1. DiffusionEmbedding: Encodes timestep information using sinusoidal embeddings.
2. diff_TSDiffuser: base layers.
3. ResidualBlock: Defines the residual layers with attention-based transformations.
4. get_torch_trans: Creates transformer encoders for time and feature processing.
5. Conv1d_with_init: Initializes 1D convolutional layers with Kaiming normalization.

Usage:
------
- A subset model is used within the general model to refine the imputation.

"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # Use smaller initialization for better stability
    # Scale down the standard Kaiming init
    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    with torch.no_grad():
        layer.weight *= 0.5  # Scale down weights for stability
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

        # Initialize with smaller weights for stability
        nn.init.xavier_uniform_(self.projection1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.projection2.weight, gain=0.5)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_TSDiffuser(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.strategy_embedding = nn.Embedding(2, config["diffusion_embedding_dim"])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.xavier_uniform_(self.output_projection2.weight, gain=0.01)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

        # Add LayerNorm for stability after skip aggregation and projections
        self.norm_after_skip = nn.LayerNorm(self.channels)
        self.norm_after_proj1 = nn.LayerNorm(self.channels)

    def forward(self, x, cond_info, diffusion_step, strategy_type):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        strategy_emb = self.strategy_embedding(strategy_type)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, strategy_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)

        # Apply LayerNorm after skip aggregation
        x = x.transpose(1, 2)  # (B, K*L, channels)
        x = self.norm_after_skip(x)
        x = x.transpose(1, 2)  # (B, channels, K*L)

        x = self.output_projection1(x)  # (B,channel,K*L)

        # Apply LayerNorm after first projection
        x = x.transpose(1, 2)  # (B, K*L, channels)
        x = self.norm_after_proj1(x)
        x = x.transpose(1, 2)  # (B, channels, K*L)

        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.strategy_projection = nn.Linear(diffusion_embedding_dim, channels)

        # Initialize projections with smaller weights
        nn.init.xavier_uniform_(self.diffusion_projection.weight, gain=0.5)
        nn.init.xavier_uniform_(self.strategy_projection.weight, gain=0.5)

        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        # Add LayerNorm for stability
        self.norm_after_time = nn.LayerNorm(channels)
        self.norm_after_feature = nn.LayerNorm(channels)
        self.norm_after_gate = nn.LayerNorm(channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (B*K, channel, L)
        # Apply LayerNorm - need to transpose to put channel dimension last
        y = y.permute(0, 2, 1)  # (B*K, L, channel)
        y = self.norm_after_time(y)
        y = y.permute(0, 2, 1)  # (B*K, channel, L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (B*L, channel, K)
        # Apply LayerNorm - need to transpose to put channel dimension last
        y = y.permute(0, 2, 1)  # (B*L, K, channel)
        y = self.norm_after_feature(y)
        y = y.permute(0, 2, 1)  # (B*L, channel, K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb, strategy_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        strategy_emb = self.strategy_projection(strategy_emb).unsqueeze(-1)

        y = x + diffusion_emb + strategy_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)

        # Apply LayerNorm after gated activation for stability
        # Need to transpose for LayerNorm (expects last dimension to be normalized)
        y = y.transpose(1, 2)  # (B, K*L, channel)
        y = self.norm_after_gate(y)
        y = y.transpose(1, 2)  # (B, channel, K*L)

        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
