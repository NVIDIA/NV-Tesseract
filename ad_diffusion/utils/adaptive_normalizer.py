# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable adaptive normalization for diffusion preprocessing."""

import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


class AdaptiveNormalizer:
    """Select and fit a distribution-aware transform without hard clipping."""

    def __init__(self, target_range=100):
        self.target_range = target_range
        self.method = None
        self.params = {}

    def fit(self, data):
        """Fit the normalizer based on the observed distribution."""
        if data.ndim > 1:
            data = data.ravel()

        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            self.method = "zeros"
            return self

        if np.all(valid_data == valid_data[0]):
            self.method = "constant"
            self.params["constant_value"] = valid_data[0]
            return self

        skewness = stats.skew(valid_data)
        kurtosis = stats.kurtosis(valid_data)
        self.params["median"] = np.median(valid_data)
        self.params["mad"] = np.median(np.abs(valid_data - self.params["median"]))
        if self.params["mad"] == 0:
            self.params["mad"] = np.std(valid_data) / 1.4826
            if self.params["mad"] == 0:
                self.params["mad"] = 1.0
        self.params["q1"] = np.percentile(valid_data, 25)
        self.params["q3"] = np.percentile(valid_data, 75)
        self.params["iqr"] = self.params["q3"] - self.params["q1"]

        if abs(skewness) > 2 or kurtosis > 7:
            n_unique = len(np.unique(valid_data))
            if n_unique > 10:
                self.method = "quantile"
                transformer = QuantileTransformer(
                    n_quantiles=min(n_unique, len(valid_data), 10000),
                    output_distribution="normal",
                    subsample=100000,
                )
                transformer.fit(valid_data.reshape(-1, 1))
                self.params["quantile_transformer"] = transformer
            else:
                self.method = "robust-zscore"
        elif abs(skewness) > 1:
            self.method = "yeo-johnson"
            transformer = PowerTransformer(method="yeo-johnson")
            transformer.fit(valid_data.reshape(-1, 1))
            self.params["power_transformer"] = transformer
        else:
            self.method = "robust-zscore"

        return self

    def transform(self, data):
        """Transform data using the fitted parameters."""
        original_shape = data.shape
        data = data.ravel()

        if self.method in {"zeros", "constant"}:
            return np.zeros_like(data).reshape(original_shape)

        nan_mask = np.isnan(data)
        result = np.zeros_like(data)
        if np.all(nan_mask):
            return result.reshape(original_shape)

        valid_data = data[~nan_mask]
        if self.method == "quantile":
            transformed = self.params["quantile_transformer"].transform(valid_data.reshape(-1, 1)).ravel()
            transformed *= self.target_range / 3
        elif self.method == "yeo-johnson":
            transformed = self.params["power_transformer"].transform(valid_data.reshape(-1, 1)).ravel()
            std_transformed = np.std(transformed)
            if std_transformed > 0:
                transformed *= min(self.target_range / (3 * std_transformed), 1.0)
            else:
                transformed = np.zeros_like(transformed)
        else:
            median = self.params["median"]
            mad = self.params["mad"] + 1e-8
            extreme_low = valid_data < (self.params["q1"] - 3 * self.params["iqr"])
            extreme_high = valid_data > (self.params["q3"] + 3 * self.params["iqr"])
            extreme_mask = extreme_low | extreme_high
            transformed = (valid_data - median) / (1.4826 * mad)

            if np.any(extreme_mask):
                extreme_values = valid_data[extreme_mask]
                signs = np.sign(extreme_values - median)
                log_values = np.log1p(np.abs(extreme_values - median) / mad)
                transformed[extreme_mask] = signs * log_values * 3

            percentile = np.percentile(np.abs(transformed), 99.9)
            if percentile > 0:
                transformed *= min(self.target_range / percentile, 1.0)
            else:
                transformed = np.zeros_like(transformed)

        scale_factor = self.target_range * (2 / np.pi)
        transformed = np.arctan(transformed / (self.target_range * 0.5)) * scale_factor
        result[~nan_mask] = transformed
        return result.reshape(original_shape)

    def fit_transform(self, data):
        """Fit the normalizer and transform the same data."""
        return self.fit(data).transform(data)
