# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
main_model.py
----------------------------------------------------------------------------

This module defines the core architecture for the models.
It includes the base class `TSDiffuser_base` and its specialized versions.
More specialized versions can be added as needed

Key Components:
---------------
1. TSDiffuser (Base Class)
   - Implements diffusion-based imputation for missing time-series data.
   - Supports both conditional and unconditional imputation strategies.
   - Uses a diffusion model to predict missing values iteratively.
   - Defines different masking strategies (random, historical) to simulate missing data.
   - **NEW: DPM-Solver support for 50-100x faster inference**

2. Diffusion Process
   - Forward diffusion: Adds noise to the observed data based on a predefined schedule.
   - Reverse process: Uses the trained diffusion model to reconstruct missing values.
   - Implements different noise schedules: Quadratic ("quad") and Linear ("linear").
   - **NEW: DPM-Solver++ for efficient sampling with 10-50 steps instead of 1000**

3. Imputation Methods
   - impute(): Standard reverse diffusion for missing value imputation.
   - ddim_impute(): Implements the Deterministic DDIM process for faster inference.
   - **NEW: dpm_solver_impute(): Fast inference using DPM-Solver++ (10-50 steps)**
   - get_middle_impute_value(): Captures intermediate imputation steps for analysis.

4. Model Variants
   - TSDiffuser_Generic (Flexible class for any dataset)

5. Evaluation
   - evaluate(): Generates imputed samples and computes errors against the ground truth.
   - **NEW: evaluate_with_dpm(): Fast evaluation using DPM-Solver**

Usage:
---------------
This module is used in conjunction with `diff_models.py` (which contains the diffusion model),
`utils.py` (which provides evaluation utilities), and `dataset.py` (which loads time-series data).

For fast inference, use dpm_solver_impute() with num_steps=20 for ~50x speedup.
"""

import logging
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diff_models import diff_TSDiffuser

logger = logging.getLogger(__name__)


class TSDiffuser_base(nn.Module):
    def __init__(self, target_dim, config, device, ratio=0.7):
        super().__init__()
        self.device = device
        self.ratio = ratio
        self.target_dim = target_dim

        self.ddim_eta = 1
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_aux_loss = config["model"].get("use_aux_loss", False)

        if self.use_aux_loss:
            self.aux_weight_order1 = config["model"].get("aux_weight_order1", 0.4)
            self.aux_weight_order2 = config["model"].get("aux_weight_order2", 0.4)
            self.aux_loss_normalize = config["model"].get("aux_loss_normalize", False)
            self.aux_loss_max_value = config["model"].get("aux_loss_max_value", None)
            logger.info(
                "Using aux loss with weight: Order 1: %s and Order 2: %s",
                self.aux_weight_order1,
                self.aux_weight_order2,
            )
            if self.aux_loss_normalize:
                logger.info("Auxiliary loss normalization is enabled")
            if self.aux_loss_max_value is not None:
                logger.info("Auxiliary loss clipping enabled at max value: %s", self.aux_loss_max_value)
        else:
            self.aux_weight_order1 = 0.0
            self.aux_weight_order2 = 0.0
            self.aux_loss_normalize = False

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional:
            self.emb_total_dim += 1
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)

        # Add LayerNorm for embedding stability
        self.embed_norm = nn.LayerNorm(self.emb_feature_dim)
        self.time_embed_norm = nn.LayerNorm(self.emb_time_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_TSDiffuser(config_diff, input_dim)

        # parameters for models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)

        # Clamp alpha values to prevent numerical issues
        self.alpha = np.clip(self.alpha, a_min=1e-6, a_max=1.0 - 1e-6)

        # Log warning if alpha approaches dangerous values
        min_alpha = np.min(self.alpha)
        if min_alpha < 1e-5:
            logger.warning("Minimum alpha value is very small: %s", min_alpha)

        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def process_data(self, batch):
        """
        Abstract method to process batch data. Must be implemented by derived classes.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Tuple of processed data: (observed_data, observed_mask, observed_tp, gt_mask,
                                    for_pattern_mask, cut_length, strategy_type)
        """
        raise NotImplementedError("process_data method must be implemented by derived classes")

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask, ratio=0.7):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)  # (b, *)
        for i in range(len(observed_mask)):
            # sample_ratio = np.random.rand()  # missing ratio
            sample_ratio = ratio  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask, ratio=self.ratio)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        # Apply LayerNorm to time embeddings
        time_embed = self.time_embed_norm(time_embed)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)

        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))  # (K,emb)
        # Apply LayerNorm to feature embeddings
        feature_embed = self.embed_norm(feature_embed)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self,
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        is_train,
        strategy_type,
    ):
        loss_sum = 0
        main_loss_sum = 0
        aux_loss_order1_sum = 0
        aux_loss_order2_sum = 0

        for t in range(self.num_steps):  # calculate loss for all t
            loss_result = self.calc_loss(
                observed_data,
                cond_mask,
                observed_mask,
                side_info,
                is_train,
                strategy_type=strategy_type,
                set_t=t,
            )

            # Handle both dict and scalar returns
            if isinstance(loss_result, dict):
                loss_sum += loss_result["total_loss"].detach()
                main_loss_sum += loss_result["main_loss"].detach()
                aux_loss_order1_sum += loss_result["aux_loss_order1"].detach()
                aux_loss_order2_sum += loss_result["aux_loss_order2"].detach()
            else:
                loss_sum += loss_result.detach()

        if self.use_aux_loss:
            return {
                "total_loss": loss_sum / self.num_steps,
                "main_loss": main_loss_sum / self.num_steps,
                "aux_loss_order1": aux_loss_order1_sum / self.num_steps,
                "aux_loss_order2": aux_loss_order2_sum / self.num_steps,
            }
        return loss_sum / self.num_steps

    def temporal_consistency_loss(self, predicted, target, mask=None, order=1):
        """
        Compute temporal consistency loss

        Args:
            predicted: Model predictions [B, K, L] (batch, features, time)
            target: Ground truth [B, K, L]
            mask: Valid positions mask [B, K, L]
            order: 1 for first-order (velocity), 2 for second-order (acceleration)
        """
        if order == 1:
            # First-order difference (velocity consistency)
            pred_diff = predicted[:, :, 1:] - predicted[:, :, :-1]
            target_diff = target[:, :, 1:] - target[:, :, :-1]

            if mask is not None:
                # Only compute loss where both consecutive points are valid
                mask_diff = mask[:, :, 1:] * mask[:, :, :-1]
                # Use larger epsilon for float16 compatibility
                epsilon = 1e-4 if pred_diff.dtype == torch.float16 else 1e-8
                mask_sum = mask_diff.sum() + epsilon
                # Use mean-like computation for better scaling
                loss = torch.sum((pred_diff - target_diff) ** 2 * mask_diff) / mask_sum
            else:
                loss = F.mse_loss(pred_diff, target_diff)

        elif order == 2:
            # Second-order difference (acceleration consistency)
            pred_acc = predicted[:, :, 2:] - 2 * predicted[:, :, 1:-1] + predicted[:, :, :-2]
            target_acc = target[:, :, 2:] - 2 * target[:, :, 1:-1] + target[:, :, :-2]

            if mask is not None:
                mask_acc = mask[:, :, 2:] * mask[:, :, 1:-1] * mask[:, :, :-2]
                # Use larger epsilon for float16 compatibility
                epsilon = 1e-4 if pred_acc.dtype == torch.float16 else 1e-8
                mask_sum = mask_acc.sum() + epsilon
                # Use mean-like computation for better scaling
                loss = torch.sum((pred_acc - target_acc) ** 2 * mask_acc) / mask_sum
            else:
                loss = F.mse_loss(pred_acc, target_acc)

        return loss

    def calc_loss(
        self,
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        is_train,
        strategy_type,
        set_t=-1,
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        # Add stability to alpha calculations
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        # Clamp alpha to prevent numerical issues
        current_alpha = torch.clamp(current_alpha, min=1e-6, max=1.0 - 1e-6)

        observed_data, _, _ = self._normalize_observed_data(observed_data)

        # Generate noise with slight reduction to prevent overflow
        noise = torch.randn_like(observed_data) * 0.999

        # Use more numerically stable computation
        sqrt_alpha = torch.sqrt(current_alpha + 1e-8)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - current_alpha + 1e-8)

        noisy_data = sqrt_alpha * observed_data + sqrt_one_minus_alpha * noise

        # Check for extreme values in noisy_data
        if torch.isnan(noisy_data).any() or torch.isinf(noisy_data).any():
            logger.warning("NaN/Inf detected in noisy_data at timesteps %s", t.cpu().numpy())
            noisy_data = torch.nan_to_num(noisy_data, nan=0.0, posinf=50.0, neginf=-50.0)
            # Additional clamping for safety
            noisy_data = torch.clamp(noisy_data, min=-200.0, max=200.0)

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t, strategy_type)  # (B,K,L)

        # Check for NaN in predictions
        if torch.isnan(predicted).any():
            t_values = t.cpu().numpy()
            logger.warning("NaN detected in model predictions at timesteps %s", t_values)
            # Replace NaN with zeros to prevent propagation
            predicted = torch.nan_to_num(predicted, nan=0.0)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask

        # More stable loss calculation using mean
        # Count valid elements for proper mean computation
        num_eval = target_mask.sum()
        num_eval = torch.clamp(num_eval, min=1.0)  # Ensure never divide by zero

        # Use larger epsilon for float16 compatibility
        epsilon = 1e-4 if residual.dtype == torch.float16 else 1e-8

        # Compute MSE loss only on valid (unmasked) positions
        # This is equivalent to mean but more numerically stable
        loss = (residual**2).sum() / (num_eval + epsilon)

        if self.use_aux_loss:
            # Compute the denoised data from the noise prediction
            # Using the DDPM formula: x_0 = (x_t - sqrt(1-alpha) * predicted_noise) / sqrt(alpha)
            sqrt_alpha = current_alpha**0.5
            sqrt_one_minus_alpha = (1.0 - current_alpha) ** 0.5

            # Add stability check for very small alpha values
            # Clamp sqrt_alpha to prevent division by very small numbers
            sqrt_alpha_stable = torch.clamp(sqrt_alpha, min=1e-3)

            # Reconstruct the denoised data (x_0 prediction) with stability
            denoised_data = (noisy_data - sqrt_one_minus_alpha * predicted) / sqrt_alpha_stable

            # Clamp denoised data to prevent extreme values
            denoised_data = torch.clamp(denoised_data, min=-100.0, max=100.0)

            # Apply temporal consistency loss on the denoised data, not the noise
            aux_loss_order1 = self.temporal_consistency_loss(denoised_data, observed_data, target_mask, order=1)
            aux_loss_order2 = self.temporal_consistency_loss(denoised_data, observed_data, target_mask, order=2)

            # Optionally clip auxiliary losses to prevent explosion
            if self.aux_loss_max_value is not None:
                aux_loss_order1 = torch.clamp(aux_loss_order1, max=self.aux_loss_max_value)
                aux_loss_order2 = torch.clamp(aux_loss_order2, max=self.aux_loss_max_value)

            # Optionally normalize auxiliary losses to main loss scale
            if self.aux_loss_normalize:
                # Normalize aux losses to have similar magnitude as main loss
                # Avoid .item() call which causes GPU-CPU sync
                loss_detached = loss.detach()
                # Use larger epsilon for float16
                norm_epsilon = 1e-4 if aux_loss_order1.dtype == torch.float16 else 1e-8
                aux_loss_order1_normalized = aux_loss_order1 * (
                    loss_detached / (aux_loss_order1.detach() + norm_epsilon)
                )
                aux_loss_order2_normalized = aux_loss_order2 * (
                    loss_detached / (aux_loss_order2.detach() + norm_epsilon)
                )
                total_loss = (
                    loss
                    + self.aux_weight_order1 * aux_loss_order1_normalized
                    + self.aux_weight_order2 * aux_loss_order2_normalized
                )
            else:
                total_loss = loss + self.aux_weight_order1 * aux_loss_order1 + self.aux_weight_order2 * aux_loss_order2

            # Return a dict with loss components when aux loss is used
            return {
                "total_loss": total_loss,
                "main_loss": loss,
                "aux_loss_order1": aux_loss_order1,
                "aux_loss_order2": aux_loss_order2,
            }
        total_loss = loss
        return total_loss

    @staticmethod
    def _normalize_observed_data(observed_data):
        """Apply the per-window scale used by diffusion training."""
        if torch.isnan(observed_data).any():
            logger.warning("NaN detected in observed_data!")
            observed_data = torch.nan_to_num(observed_data, nan=0.0)

        data_mean = observed_data.mean(dim=(1, 2), keepdim=True)
        data_std = observed_data.std(dim=(1, 2), keepdim=True) + 1e-5
        normalized_data = (observed_data - data_mean) / data_std
        normalized_data = torch.clamp(normalized_data, min=-10.0, max=10.0)
        return normalized_data, data_mean, data_std

    @staticmethod
    def _denormalize_samples(samples, data_mean, data_std):
        """Return generated samples to the input scale used by SDK residuals."""
        return samples * data_std.unsqueeze(1) + data_mean.unsqueeze(1)

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, strategy_type):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(
                    diff_input,
                    side_info,
                    torch.tensor([t]).to(self.device),
                    strategy_type,
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def ddim_impute(self, observed_data, cond_mask, side_info, n_samples, strategy_type, ddim_eta=1, ddim_steps=10):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            ddim_timesteps = ddim_steps
            c = self.num_steps // ddim_timesteps
            ddim_timesteps_sequence = np.asarray(list(range(0, self.num_steps, c)))
            ddim_timesteps_previous_sequence = np.append(np.array([0]), ddim_timesteps_sequence[:-1])

            for step_number in range(ddim_timesteps - 1, -1, -1):
                t = ddim_timesteps_sequence[step_number]
                previous_t = ddim_timesteps_previous_sequence[step_number]

                at = torch.tensor(self.alpha[t]).to(self.device)
                at_next = torch.tensor(self.alpha[previous_t]).to(self.device)

                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                xt = diff_input
                et = self.diffmodel(xt, side_info, torch.tensor([t]).to(self.device), strategy_type)
                x0_t = (current_sample - et * (1 - at).sqrt()) / at.sqrt()

                c1 = ddim_eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1**2).sqrt()
                current_sample = at_next.sqrt() * x0_t + c1 * torch.randn_like(current_sample) + c2 * et

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def dpm_solver_impute(self, observed_data, cond_mask, side_info, n_samples, strategy_type, num_steps=20):
        """
        Fast inference using DPM-Solver++ (10-20 steps instead of 1000).
        No retraining needed - works with existing trained model.

        Args:
            observed_data: Observed time series data (B, K, L)
            cond_mask: Conditioning mask (B, K, L)
            side_info: Side information embeddings (B, emb_dim, K, L)
            n_samples: Number of samples to generate
            strategy_type: Strategy type for evaluation
            num_steps: Number of DPM-Solver steps (default: 20, recommended: 10-50)

        Returns:
            imputed_samples: Generated samples (B, n_samples, K, L)

        Note:
            This method provides 50-100x speedup over standard diffusion with minimal quality loss.
            Recommended num_steps: 20 for best quality/speed tradeoff, 10 for maximum speed.
        """
        try:
            from utils.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper
        except ImportError as e:
            raise ImportError("DPM-Solver not found. Use the local copy (ad_diffusion_oss.dpm_solver_pytorch).") from e

        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        # Create noise schedule from existing betas
        noise_schedule = NoiseScheduleVP(
            schedule="discrete", betas=torch.tensor(self.beta, dtype=torch.float32).to(self.device)
        )

        # Precompute timesteps we'll use for unconditional model
        if self.is_unconditional:
            # Calculate which timesteps DPM-Solver will use
            timesteps_to_use = torch.linspace(0, self.num_steps - 1, num_steps).long()

        for i in range(n_samples):
            # Generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_cond_history = {}

                # Precompute noisy conditions for timesteps we'll actually use
                for t in timesteps_to_use:
                    t_idx = t.item()
                    noise = torch.randn_like(observed_data)
                    alpha_t = self.alpha_hat[t_idx]
                    beta_t = self.beta[t_idx]
                    noisy_obs_t = (alpha_t**0.5) * observed_data + (beta_t**0.5) * noise
                    noisy_cond_history[t_idx] = noisy_obs_t * cond_mask

            # Initialize from noise
            current_sample = torch.randn_like(observed_data)

            # Define model wrapper for DPM-Solver
            def model_fn(x, t_continuous):
                """
                Wrapper for diffusion model that DPM-Solver can call.

                Args:
                    x: Current sample (B, K, L)
                    t_continuous: Continuous time in [0, 1]

                Returns:
                    Predicted noise
                """
                # Convert continuous time [0, 1] to discrete timestep
                t_discrete = (t_continuous * (self.num_steps - 1)).long()
                t_discrete = torch.clamp(t_discrete, 0, self.num_steps - 1)

                # Handle batch dimension properly
                if t_discrete.dim() == 0:
                    t_discrete = t_discrete.unsqueeze(0)

                # Use first timestep if batched (DPM-Solver uses same t for whole batch)
                t_idx = t_discrete[0].item()

                # Prepare input based on conditioning strategy
                if self.is_unconditional:
                    # Use precomputed noisy condition if available
                    if t_idx in noisy_cond_history:
                        noisy_cond = noisy_cond_history[t_idx]
                    else:
                        # Fallback: compute on the fly (shouldn't happen with precompute)
                        noise = torch.randn_like(observed_data)
                        alpha_t = self.alpha_hat[t_idx]
                        beta_t = self.beta[t_idx]
                        noisy_cond = (alpha_t**0.5) * observed_data + (beta_t**0.5) * noise
                        noisy_cond = noisy_cond * cond_mask

                    diff_input = cond_mask * noisy_cond + (1.0 - cond_mask) * x
                    diff_input = diff_input.unsqueeze(1)  # (B, 1, K, L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * x).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)

                # Get noise prediction from diffusion model
                t_tensor = torch.full((B,), t_idx, dtype=torch.long, device=self.device)
                noise_pred = self.diffmodel(diff_input, side_info, t_tensor, strategy_type)

                return noise_pred

            # Wrap model for DPM-Solver
            model_wrapped = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",  # Our model predicts noise
                model_kwargs={},
                guidance_type="uncond",
            )

            # Create DPM-Solver
            dpm_solver = DPM_Solver(
                model_wrapped,
                noise_schedule,
                algorithm_type="dpmsolver++",  # Best variant
                correcting_x0_fn=None,  # No correction needed
            )

            # Sample with DPM-Solver (much faster than 1000 steps!)
            current_sample = dpm_solver.sample(
                current_sample,
                steps=num_steps,  # 10-50 steps instead of 1000!
                order=2,  # Second-order solver (more accurate)
                skip_type="time_uniform",  # Uniform time steps
                method="singlestep",  # Single-step method
            )

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def get_middle_impute_value(self, observed_data, cond_mask, side_info, n_samples, strategy_type):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        imputed_middle_samples = torch.zeros(B, self.num_steps, K, L)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(
                    diff_input,
                    side_info,
                    torch.tensor([t]).to(self.device),
                    strategy_type,
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

                imputed_middle_samples[:, t] = current_sample.detach()

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples, imputed_middle_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            strategy_type,
        ) = self.process_data(batch)

        self.target_strategy = "random"
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask, ratio=self.ratio)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(
            observed_data,
            cond_mask,
            observed_mask,
            side_info,
            is_train,
            strategy_type=strategy_type,
        )

    def evaluate(self, batch, n_samples):
        """
        Standard evaluation using full diffusion process.

        Args:
            batch: Batch of data
            n_samples: Number of samples to generate

        Returns:
            Tuple of (samples, observed_data, target_mask, observed_mask, observed_tp)
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            strategy_type,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            normalized_data, data_mean, data_std = self._normalize_observed_data(observed_data)
            samples = self.impute(normalized_data, cond_mask, side_info, n_samples, strategy_type)
            samples = self._denormalize_samples(samples, data_mean, data_std)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

    def evaluate_with_dpm(self, batch, n_samples, dpm_steps=20):
        """
        Fast evaluation using DPM-Solver for 50-100x speedup.

        Args:
            batch: Batch of data
            n_samples: Number of samples to generate
            dpm_steps: Number of DPM-Solver steps (10-50, default: 20)

        Returns:
            Tuple of (samples, observed_data, target_mask, observed_mask, observed_tp)

        Note:
            This method uses DPM-Solver++ instead of standard diffusion, providing
            massive speedup with minimal quality loss. Recommended for production inference.
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            strategy_type,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            normalized_data, data_mean, data_std = self._normalize_observed_data(observed_data)
            # Use DPM-Solver for fast sampling
            samples = self.dpm_solver_impute(
                normalized_data, cond_mask, side_info, n_samples, strategy_type, num_steps=dpm_steps
            )
            samples = self._denormalize_samples(samples, data_mean, data_std)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp


class TSDiffuser_Generic(TSDiffuser_base):
    def __init__(self, config, device, target_dim=None, ratio=0.7, cut_length_strategy="none", default_cut_length=0):
        """
        Generic TSDiffuser class that works with any dataset.

        Args:
            config: Configuration dictionary
            device: Device to run on
            target_dim: Number of features (if None, will be inferred from data)
            ratio: Missing data ratio for random masking
            cut_length_strategy: Strategy for cut_length ('none', 'fixed', 'batch', 'dynamic')
            default_cut_length: Default cut length value when strategy is 'fixed'
        """
        # If target_dim is not provided, try to get it from config
        if target_dim is None:
            if "model" in config and "target_dim" in config["model"]:
                target_dim = config["model"]["target_dim"]
            else:
                raise ValueError("target_dim must be provided either as parameter or in config['model']['target_dim']")

        super(TSDiffuser_Generic, self).__init__(target_dim, config, device, ratio)
        self.cut_length_strategy = cut_length_strategy
        self.default_cut_length = default_cut_length

    def process_data(self, batch):
        """
        Process batch data with flexible handling for different dataset formats.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Tuple of processed data: (observed_data, observed_mask, observed_tp, gt_mask,
                                    for_pattern_mask, cut_length, strategy_type)
        """
        # Extract basic data
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        # Permute dimensions to match expected format (B, K, L)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        # Handle for_pattern_mask (historical mask)
        if "hist_mask" in batch:
            for_pattern_mask = batch["hist_mask"].to(self.device).float()
            for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        else:
            # Default to observed_mask if hist_mask not available
            for_pattern_mask = observed_mask

        # Handle cut_length based on strategy
        cut_length = self._get_cut_length(batch, observed_data.shape[0])

        # Handle strategy_type
        if "strategy_type" in batch:
            strategy_type = batch["strategy_type"].to(self.device).long()
        else:
            # Default strategy type if not provided
            strategy_type = torch.zeros(observed_data.shape[0], dtype=torch.long, device=self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            strategy_type,
        )

    def _get_cut_length(self, batch, batch_size):
        """
        Get cut_length based on the configured strategy.

        Args:
            batch: Batch dictionary
            batch_size: Size of the batch

        Returns:
            Tensor of cut_length values
        """
        if self.cut_length_strategy == "none":
            # No cutting - all time steps are evaluated
            return torch.zeros(batch_size, dtype=torch.long, device=self.device)

        if self.cut_length_strategy == "fixed":
            # Use a fixed cut length for all samples
            return torch.full((batch_size,), self.default_cut_length, dtype=torch.long, device=self.device)

        if self.cut_length_strategy == "batch":
            # Use cut_length from batch if available, otherwise use default
            if "cut_length" in batch:
                return batch["cut_length"].to(self.device).long()
            return torch.full((batch_size,), self.default_cut_length, dtype=torch.long, device=self.device)

        if self.cut_length_strategy == "dynamic":
            # Dynamic calculation based on data characteristics
            # This could be based on sequence length, missing data patterns, etc.
            if "cut_length" in batch:
                return batch["cut_length"].to(self.device).long()
            # Default dynamic strategy: cut first 10% of sequence
            seq_length = batch["observed_data"].shape[1]
            dynamic_cut = max(1, int(seq_length * 0.1))
            return torch.full((batch_size,), dynamic_cut, dtype=torch.long, device=self.device)

        raise ValueError(f"Unknown cut_length_strategy: {self.cut_length_strategy}")
