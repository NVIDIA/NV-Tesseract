# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TSB-AD Preprocessing for Inference

This module provides preprocessing functions for new datasets to match
the format of the merged TSB-AD training data, using the original
feature engineering methods for complete consistency.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities from utils directory
from utils.json_utils import load_preprocessing_models

logger = logging.getLogger(__name__)

ORIGINAL_NORMALIZER_AVAILABLE = True

# Constants from original merger
TARGET_FEATURES = 38
DOMAIN_MAP = {
    "WebService": 0,
    "Medical": 1,
    "Facility": 2,
    "Synthetic": 3,
    "HumanActivity": 4,
    "Sensor": 5,
    "Environment": 6,
    "Finance": 7,
    "Traffic": 8,
}

# Domain categories
HEAVY_TAILED_DOMAINS = ["WebService", "Traffic", "Finance"]
SENSOR_DOMAINS = ["Sensor", "Medical", "Facility"]
BOUNDED_DOMAINS = ["Synthetic", "HumanActivity", "Environment"]


class AdaptiveNormalizerCompat:
    """Compatibility class for deserialized AdaptiveNormalizer objects"""

    def __init__(self):
        self.target_range = 100
        self.method = None
        self.params = {}

    def transform(self, data):
        """Use SimpleAdaptiveNormalizer for transform"""
        normalizer = SimpleAdaptiveNormalizer({"method": self.method, "target_range": self.target_range, **self.params})
        return normalizer.transform(data)


def _load_normalizers(filepath):
    """Load normalizers from JSON file using the inference-only compat class."""
    return load_preprocessing_models(filepath, normalizer_class=AdaptiveNormalizerCompat)


def detect_domain(data):
    """
    Automatically detect the most appropriate domain based on data characteristics.

    Args:
        data: numpy array of shape (n_samples, n_features)

    Returns:
        str: Detected domain name
    """
    # Flatten data for analysis
    flat_data = data.ravel()
    flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaN values

    # Calculate statistical properties
    # Basic statistics
    std_val = np.std(flat_data)
    min_val = np.min(flat_data)
    max_val = np.max(flat_data)

    # Distribution characteristics
    skewness = stats.skew(flat_data)
    kurtosis = stats.kurtosis(flat_data)

    # Robust statistics
    q1 = np.percentile(flat_data, 25)
    q3 = np.percentile(flat_data, 75)
    iqr = q3 - q1

    # Check for outliers
    outlier_low = np.sum(flat_data < (q1 - 3 * iqr))
    outlier_high = np.sum(flat_data > (q3 + 3 * iqr))
    outlier_ratio = (outlier_low + outlier_high) / len(flat_data)

    # Check data range characteristics
    range_val = max_val - min_val
    normalized_range = range_val / (std_val + 1e-8)

    domain = "Sensor"

    # Decision logic based on domain characteristics

    # Heavy-tailed domains: high skewness, high kurtosis, many outliers
    if abs(skewness) > 3 or kurtosis > 10 or outlier_ratio > 0.1:
        # Further distinguish between heavy-tailed domains
        if min_val < 0 and max_val > 0 and abs(skewness) > 5:
            domain = "Finance"  # Financial data often has extreme positive/negative values
        if min_val >= 0 and skewness > 4:
            domain = "WebService"  # Web service metrics are often positive with long tail
        domain = "Traffic"  # Traffic data can have various patterns

    # Bounded domains: limited range, low outliers, more symmetric
    if normalized_range < 10 and outlier_ratio < 0.02 and abs(skewness) < 1:
        # Further distinguish between bounded domains
        if 0 <= min_val <= 1 and 0 <= max_val <= 1:
            domain = "Synthetic"  # Often normalized to [0,1]
        if min_val >= -1 and max_val <= 1:
            domain = "HumanActivity"  # Activity data often in [-1, 1] range
        domain = "Environment"  # Environmental data with bounded ranges

    # Sensor domains: moderate characteristics, some outliers
    # Further distinguish between sensor domains
    if outlier_ratio > 0.05:
        domain = "Medical"  # Medical sensors can have anomalies
    if normalized_range > 20:
        domain = "Facility"  # Facility sensors might have wider ranges

    return domain


def bounded_univariate_features(x, history, target_range=100, window_size=20, target_dim=38):
    """
    Create bounded features for univariate normalized data.
    Adapted from the original TSBADMergerNoClip class.
    """
    features = np.zeros(target_dim)

    # Original value (already bounded)
    features[0] = x

    # Get historical context (if available)
    if len(history) > 0:
        history_array = np.array(history)
        window = history_array[-window_size:]

        # Basic statistics (bounded by input range)
        features[1] = np.mean(window)
        std_val = np.std(window) if len(window) > 1 else 0
        features[2] = np.tanh(std_val / 10) * 10  # Bounded std
        features[3] = np.median(window)
        features[4] = np.percentile(window, 25) if len(window) > 1 else x
        features[5] = np.percentile(window, 75) if len(window) > 1 else x

        # Bounded differences
        if len(window) >= 2:
            features[6] = np.tanh((x - window[-2]) / 20) * 20
            diff_mean = np.mean(np.diff(window))
            features[7] = np.tanh(diff_mean / 20) * 20

        # Simple lags (already bounded)
        lags = [1, 2, 5, 10]
        for i, lag in enumerate(lags):
            if lag < len(history):
                features[8 + i] = history[-lag - 1]
            else:
                features[8 + i] = x

        # Moving averages (bounded by input range)
        ma_windows = [3, 5, 10]
        for i, w in enumerate(ma_windows):
            if w <= len(window):
                features[12 + i] = np.mean(window[-w:])
            else:
                features[12 + i] = np.mean(window)

        # Bounded trend indicator
        if len(window) >= 5:
            recent = np.mean(window[-2:])
            older = np.mean(window[-5:-3]) if len(window) >= 5 else window[0]
            trend = recent - older
            features[15] = np.tanh(trend / 30) * 30

        # Additional bounded features
        if len(window) >= 10:
            # Bounded variance ratio
            var_recent = np.var(window[-5:]) if len(window[-5:]) > 1 else 0
            var_older = np.var(window[-10:-5]) if len(window[-10:-5]) > 1 else 1e-8
            ratio = var_recent / (var_older + 1e-8)
            features[16] = np.arctan(ratio) * (target_range * 2 / np.pi)

            # Bounded autocorrelation
            if len(window) >= 20:
                if np.std(window) > 0:
                    autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
                    if not np.isnan(autocorr):
                        features[17] = autocorr * 50  # correlation is already [-1, 1]
                    else:
                        features[17] = 0
                else:
                    features[17] = 0
    else:
        # If no history, fill with current value
        features[1:] = x * 0.5  # Slightly damped

    # Remaining features stay as zeros (padding)
    return features


def bounded_multivariate_features(data, n_original, target_range=100):
    """
    Create bounded features for multivariate normalized data.
    Adapted from the original TSBADMergerNoClip class.
    """
    n_samples = len(data)
    expanded = np.zeros((n_samples, TARGET_FEATURES))

    # Keep original normalized features
    expanded[:, :n_original] = data[:, :n_original]

    # Add bounded interaction features
    idx = n_original
    if n_original >= 2 and idx < TARGET_FEATURES:
        # Bounded pairwise products
        for i in range(min(n_original, 3)):
            for j in range(i + 1, min(n_original, 4)):
                if idx < TARGET_FEATURES:
                    # Use tanh to bound products
                    expanded[:, idx] = np.tanh(data[:, i] * data[:, j] / target_range) * target_range
                    idx += 1

        # Bounded differences (already bounded if inputs are)
        for i in range(min(n_original, 3)):
            for j in range(i + 1, min(n_original, 4)):
                if idx < TARGET_FEATURES:
                    expanded[:, idx] = data[:, i] - data[:, j]
                    idx += 1

        # Bounded ratios
        for i in range(min(n_original, 2)):
            for j in range(min(n_original, 2)):
                if i != j and idx < TARGET_FEATURES:
                    ratio = data[:, i] / (data[:, j] + 1e-8)
                    expanded[:, idx] = np.arctan(ratio / 10) * (target_range * 2 / np.pi)
                    idx += 1

        # Bounded sums (ensure they stay in range)
        for i in range(min(n_original, 3)):
            for j in range(i + 1, min(n_original, 4)):
                if idx < TARGET_FEATURES:
                    # Use tanh to keep sums bounded
                    expanded[:, idx] = np.tanh((data[:, i] + data[:, j]) / (target_range * 2)) * target_range
                    idx += 1

    # Apply additional clipping for safety
    expanded = np.clip(expanded, -target_range * 1.5, target_range * 1.5)

    return expanded


class SimpleAdaptiveNormalizer:
    """Simplified version of AdaptiveNormalizer for inference"""

    def __init__(self, params):
        self.method = params["method"]
        self.params = params
        self.target_range = params.get("target_range", 100)

    def transform(self, data):
        """Transform data using fitted parameters"""
        original_shape = data.shape
        data = data.ravel()

        if self.method == "zeros":
            return np.zeros_like(data).reshape(original_shape)

        if self.method == "constant":
            return np.zeros_like(data).reshape(original_shape)

        # Handle NaN values
        nan_mask = np.isnan(data)
        result = np.zeros_like(data)

        if not np.all(nan_mask):
            valid_data = data[~nan_mask]

            # Stage 1: Apply main transformation
            if self.method == "quantile":
                transformed = self.params["quantile_transformer"].transform(valid_data.reshape(-1, 1)).ravel()
                transformed *= self.target_range / 3

            elif self.method == "yeo-johnson":
                transformed = self.params["power_transformer"].transform(valid_data.reshape(-1, 1)).ravel()
                std_transformed = np.std(transformed)
                if std_transformed > 0:
                    scale = self.target_range / (3 * std_transformed)
                    transformed *= min(scale, 1.0)
                else:
                    transformed = np.zeros_like(transformed)

            else:  # robust-zscore
                median = self.params["median"]
                mad = self.params.get("mad", 1.0) + 1e-8
                q1 = self.params.get("q1", median - 1.0)
                q3 = self.params.get("q3", median + 1.0)
                iqr = self.params.get("iqr", 2.0)

                # Identify extreme values
                extreme_low = valid_data < (q1 - 3 * iqr)
                extreme_high = valid_data > (q3 + 3 * iqr)
                extreme_mask = extreme_low | extreme_high

                # Apply robust z-score
                transformed = (valid_data - median) / (1.4826 * mad)

                # Soft transformation for extreme values
                if np.any(extreme_mask):
                    extreme_vals = valid_data[extreme_mask]
                    signs = np.sign(extreme_vals - median)
                    log_vals = np.log1p(np.abs(extreme_vals - median) / mad)
                    transformed[extreme_mask] = signs * log_vals * 3

                # Scale to target range
                percentile_val = np.percentile(np.abs(transformed), 99.9)
                if percentile_val > 0:
                    scale = self.target_range / percentile_val
                    transformed *= min(scale, 1.0)
                else:
                    transformed = np.zeros_like(transformed)

                    # Stage 2: Apply soft bounding with arctan
            scale_factor = self.target_range * (2 / np.pi)
            # Clip before arctan to prevent extreme values
            transformed = np.clip(transformed, -1000, 1000)
            transformed = np.arctan(transformed / (self.target_range * 0.5)) * scale_factor

            # Final safety check
            transformed = np.nan_to_num(transformed, nan=0.0, posinf=self.target_range, neginf=-self.target_range)

            result[~nan_mask] = transformed

        return result.reshape(original_shape)


def preprocess_for_inference(data, domain=None, model_dir=None, target_dim=40, add_metadata=True):
    """
    Preprocess new data for inference using saved TSB-AD preprocessing models.

    Args:
        data: numpy array or pandas DataFrame with shape (n_samples, n_features)
        domain: str or None, domain name (e.g., "WebService", "Medical", "Sensor", etc.)
                If None, will auto-detect based on data characteristics
        model_dir: str or Path, directory containing saved preprocessing models
        target_dim: int, total target dimensions including metadata (default: 40)
                   If add_metadata=True, reserves 2 columns for domain and padding_mask
                   So actual features = target_dim - 2
        add_metadata: bool, whether to add domain/padding_mask columns

    Returns:
        np.ndarray: Preprocessed data ready for model inference
    """
    # Convert to Path
    model_dir = Path(model_dir)

    # Calculate actual feature dimensions
    if add_metadata:
        # Reserve 2 columns for domain and padding_mask
        target_features = target_dim - 2
        logger.info("Target dimensions: %s (features: %s + domain: 1 + padding: 1)", target_dim, target_features)
    else:
        target_features = target_dim
        logger.info("Target dimensions: %s (features only, no metadata)", target_dim)

    # Convert DataFrame to numpy if needed
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Auto-detect domain if not provided
    if domain is None:
        logger.info("Domain not specified. Auto-detecting based on data characteristics...")
        domain = detect_domain(data)
        logger.info("Detected domain: %s", domain)

    # Load preprocessing models from JSON
    domain_normalizers = _load_normalizers(model_dir / "domain_normalizers.json")
    domain_preprocessors = load_preprocessing_models(model_dir / "domain_preprocessors.json")

    # Load PCA models if they exist
    pca_models = {}
    pca_path = model_dir / "pca_models.json"
    if pca_path.exists():
        pca_models = load_preprocessing_models(pca_path)

    n_samples, n_features = data.shape

    # Step 1: Apply domain preprocessing
    if domain in domain_preprocessors:
        preprocess_type, preprocess_param = domain_preprocessors[domain]

        if preprocess_type == "log":
            # Log transformation
            if preprocess_param != 0:  # Data was shifted
                data = np.log1p(data - preprocess_param + 1)
            else:
                data = np.log1p(data)

        elif preprocess_type == "sinh":
            # Sinh transformation
            scale = preprocess_param
            # Clip to prevent overflow (sinh(710) is approximately the max float64)
            data_scaled = np.clip(data / scale, -700, 700)
            data = np.sinh(data_scaled)
            # Additional clipping after sinh to ensure reasonable values
            data = np.clip(data, -1e10, 1e10)

    # Step 2: Apply adaptive normalization
    normalizer_key = f"features_{n_features}"

    if domain in domain_normalizers and normalizer_key in domain_normalizers[domain]:
        normalizers = domain_normalizers[domain][normalizer_key]

        if n_features == 1:
            # Univariate data
            if hasattr(normalizers, "transform"):
                # Direct normalizer object
                data = normalizers.transform(data.ravel()).reshape(-1, 1)
            else:
                # Wrapped normalizer parameters
                normalizer = SimpleAdaptiveNormalizer(normalizers)
                data = normalizer.transform(data.ravel()).reshape(-1, 1)
        else:
            # Multivariate data
            normalized_data = np.zeros_like(data)
            for i in range(min(n_features, len(normalizers))):
                if hasattr(normalizers[i], "transform"):
                    normalized_data[:, i] = normalizers[i].transform(data[:, i])
                else:
                    normalizer = SimpleAdaptiveNormalizer(normalizers[i])
                    normalized_data[:, i] = normalizer.transform(data[:, i])
            data = normalized_data

    # Step 3: Handle dimensionality
    if n_features > target_features:
        # Apply PCA
        pca_key = f"{domain}_features_{n_features}"
        if pca_key in pca_models:
            data = pca_models[pca_key].transform(data[:, :target_features])
        else:
            # Fallback: Apply normal PCA to the data
            pca = PCA(n_components=target_features)
            data = pca.fit_transform(data)
            logger.info("Applied PCA to %s features to %s features", n_features, target_features)

    elif n_features < target_features:
        # Expand features using the original feature engineering methods
        if n_features == 1:
            # For univariate, create temporal features
            logger.info("Creating temporal features for univariate data...")
            expanded = np.zeros((n_samples, target_features))

            # Process each sample with history
            history = []
            for i in range(n_samples):
                # Update history (maintain window of past values)
                if len(history) >= 20:  # window_size
                    history.pop(0)
                history.append(data[i, 0])

                # Create features for this sample
                expanded[i, :] = bounded_univariate_features(
                    data[i, 0], history, target_range=100, window_size=20, target_dim=target_features
                )
            data = expanded
        else:
            # For multivariate, create interaction features
            logger.info("Creating interaction features for %s-dimensional data...", n_features)
            data = bounded_multivariate_features(data, n_features, target_range=100)

    # Step 4: Add metadata columns if requested
    if add_metadata:
        # Domain mapping
        domain_map = {
            "WebService": 0,
            "Medical": 1,
            "Facility": 2,
            "Synthetic": 3,
            "HumanActivity": 4,
            "Sensor": 5,
            "Environment": 6,
            "Finance": 7,
            "Traffic": 8,
        }

        domain_id = domain_map.get(domain, 0)

        # Create metadata columns (only domain and padding_mask, no label)
        domain_col = np.full((n_samples, 1), domain_id)
        padding_mask = np.ones((n_samples, 1))  # All features are valid

        # Combine all columns (features + domain + padding_mask)
        data = np.hstack([data, domain_col, padding_mask])

        # Verify final dimensions
        assert data.shape[1] == target_dim, f"Expected {target_dim} columns but got {data.shape[1]}"

    return data


def preprocess_simple(data, model_dir, domain=None, scale_factor=1.0):
    """
    Simplified preprocessing function for quick integration.

    Args:
        data: Input data (numpy array or DataFrame)
        model_dir: Directory containing saved preprocessing models
        domain: Domain name or None for auto-detection (default: None)
        scale_factor: Additional scaling factor (default: 1.0)

    Returns:
        Preprocessed data ready for model inference
    """
    # Apply full preprocessing
    preprocessed = preprocess_for_inference(
        data=data,
        domain=domain,
        model_dir=model_dir,
        target_features=38,
        add_metadata=False,  # Don't add metadata for simple version
    )

    # Apply additional scaling if needed
    if scale_factor != 1.0:
        preprocessed *= scale_factor

    return preprocessed


# Example usage
if __name__ == "__main__":
    # Example: Load and preprocess new data
    # data = pd.read_csv("new_dataset.csv")
    # model_dir = "path/to/merged_final_models"
    #
    # preprocessed = preprocess_for_inference(
    #     data=data.values,
    #     domain="Sensor",
    #     model_dir=model_dir
    # )
    pass
