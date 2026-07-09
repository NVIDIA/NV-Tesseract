# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Adaptive Thresholding Methods for Time Series Anomaly Detection

This module implements two novel adaptive thresholding strategies for time series
anomaly detection based on confidence sequences and multi-scale analysis.

Methods:
-----------------------------------------------------------------------
1. Segmented Confidence Sequences (SCS):
   - Applies confidence sequences within adaptive segments identified via APCA
   - Maintains separate confidence bounds per regime/segment
   - Dynamically assigns incoming points to segments for local online learning

2. Multi-Scale Adaptive Confidence Segments (MACS):
   - Combines rolling-window segmentation at multiple temporal scales
   - Uses confidence sequence-based thresholding across time and resolution
   - Maintains CST-based bounds at short, medium, and long windows

References:
-----------------------------------------------------------------------
- Amazon Science: "Online Adaptive Anomaly Thresholding with Confidence Sequences"
- Springer: "Adaptive Methods for Time Series Analysis"

Usage:
-----------------------------------------------------------------------
# Import removed - these classes are defined in this file

# Initialize thresholders
scs_thresholder = SCSThreshold(window_size=100, confidence_level=0.95)
macs_thresholder = MACSThreshold(short_window=10, medium_window=50, long_window=250)

# Apply thresholding
anomalies_scs = scs_thresholder.detect_anomalies(reconstruction_errors, time_series_data)
anomalies_macs = macs_thresholder.detect_anomalies(reconstruction_errors, time_series_data)

See the paper at: https://dl.acm.org/doi/full/10.1145/3787120.3787130
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseAdaptiveThreshold(ABC):
    """Base class for adaptive thresholding methods."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize base adaptive threshold.

        Args:
            confidence_level: Confidence level for threshold bounds (0.95 = 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level  # Significance level

    @abstractmethod
    def detect_anomalies(self, scores: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using adaptive thresholding.

        Args:
            scores: Anomaly scores (e.g., reconstruction errors)
            data: Original time series data

        Returns:
            Binary anomaly predictions (0: normal, 1: anomaly)
        """

    def _calculate_confidence_sequence_bounds(
        self, scores: np.ndarray, window_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence sequence bounds for anomaly detection.

        Based on the confidence sequence methodology from the Amazon Science paper.

        Args:
            scores: Anomaly scores
            window_size: Rolling window size for local statistics

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        n = len(scores)
        if n == 0:
            return np.array([]), np.array([])

        # O(n) expanding-window mean/std using cumulative sums
        scores = scores.astype(np.float64, copy=False)
        cumsum = np.cumsum(scores)
        cumsum_sq = np.cumsum(scores**2)
        counts = np.arange(1, n + 1, dtype=np.float64)

        mean_scores = cumsum / counts
        # Var = E[x^2] - (E[x])^2
        var_scores = (cumsum_sq / counts) - (mean_scores**2)
        var_scores = np.maximum(var_scores, 0.0)
        std_scores = np.sqrt(var_scores)

        # Bound width based on std
        bound_width = np.where(std_scores > 0, 1.5 * std_scores, 0.1)
        if self.confidence_level > 0.95:
            bound_width *= 1.2
        elif self.confidence_level < 0.90:
            bound_width *= 0.8

        lower_bounds = mean_scores - bound_width
        upper_bounds = mean_scores + bound_width

        # Stabilize first point
        lower_bounds[0] = scores[0] - 0.1
        upper_bounds[0] = scores[0] + 0.1

        return lower_bounds, upper_bounds


class SCSThreshold(BaseAdaptiveThreshold):
    """
    Segmented Confidence Sequences (SCS) Thresholding.

    This method applies confidence sequences within adaptive segments identified
    via APCA or clustering-based segmentation. It maintains separate confidence
    bounds per regime/segment for local online learning with guarantees.
    """

    def __init__(
        self,
        window_size: int = 100,
        confidence_level: float = 0.95,
        n_segments: int = 5,
        segmentation_method: str = "apca",
        min_segment_size: int = 50,
        percentile_filter: float = 95.0,  # Configurable percentile filter
        verbose: bool = True,  # Control logging output
    ):
        """
        Initialize SCS thresholding.

        Args:
            window_size: Size of rolling window for local statistics
            confidence_level: Confidence level for bounds
            n_segments: Number of segments for clustering-based segmentation
            segmentation_method: "apca" or "kmeans"
            min_segment_size: Minimum size for segments
            percentile_filter: Percentile threshold for additional filtering
            verbose: Whether to output logging information
        """
        super().__init__(confidence_level)
        self.window_size = window_size
        self.n_segments = n_segments
        self.segmentation_method = segmentation_method
        self.min_segment_size = min_segment_size
        self.percentile_filter = percentile_filter
        self.verbose = verbose
        self.segments = None
        self.segment_bounds = {}

    def _segment_time_series(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Segment time series using APCA or K-means clustering.

        Args:
            data: Time series data

        Returns:
            List of (start_idx, end_idx) tuples for each segment
        """
        if self.segmentation_method == "apca":
            return self._apca_segmentation(data)
        if self.segmentation_method == "kmeans":
            return self._kmeans_segmentation(data)
        raise ValueError(f"Unknown segmentation method: {self.segmentation_method}")

    def _apca_segmentation(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Adaptive Piecewise Constant Approximation (APCA) segmentation.

        Args:
            data: Time series data

        Returns:
            List of segment boundaries
        """
        n = len(data)
        segments = []

        # Handle multi-dimensional data by flattening or using first dimension
        if data.ndim > 1:
            # Use the first dimension for segmentation (or take mean across features)
            data_1d = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]
        else:
            data_1d = data

        # Calculate data characteristics
        data_std = np.std(data_1d)
        coefficient_of_variation = data_std / (np.mean(data_1d) + 1e-8)

        # For flat data, use fixed-size segments
        if coefficient_of_variation < 0.1:
            segment_size = max(200, n // 15)  # Create ~15 segments
            for i in range(0, n, segment_size):
                start_idx = i
                end_idx = min(i + segment_size - 1, n - 1)
                segments.append((start_idx, end_idx))
            return segments

        # For variable data, use APCA with more aggressive splitting
        current_start = 0
        current_end = n - 1

        while current_start < current_end:
            # Find the best split point in current segment
            segment_data = data_1d[current_start : current_end + 1]

            if len(segment_data) < self.min_segment_size * 2:
                # Segment is too small, keep as is
                segments.append((current_start, current_end))
                break

            # Calculate error for each possible split point
            best_split = current_start
            min_error = float("inf")

            # Reduce search space for speed on long sequences
            max_candidates = 200
            split_start = current_start + self.min_segment_size
            split_end = current_end - self.min_segment_size + 1
            candidate_count = max(1, split_end - split_start)
            step = max(1, candidate_count // max_candidates)

            for split_point in range(split_start, split_end, step):
                # Calculate mean of left and right segments
                left_mean = np.mean(data_1d[current_start:split_point])
                right_mean = np.mean(data_1d[split_point : current_end + 1])

                # Calculate reconstruction error
                left_error = np.sum((data_1d[current_start:split_point] - left_mean) ** 2)
                right_error = np.sum((data_1d[split_point : current_end + 1] - right_mean) ** 2)
                total_error = left_error + right_error

                if total_error < min_error:
                    min_error = total_error
                    best_split = split_point

            # Check if splitting improves the approximation
            no_split_mean = np.mean(segment_data)
            no_split_error = np.sum((segment_data - no_split_mean) ** 2)

            # More aggressive splitting for better segmentation
            improvement_threshold = 0.7 if coefficient_of_variation > 0.3 else 0.5

            if min_error < no_split_error * improvement_threshold:
                # Split the segment
                segments.append((current_start, best_split - 1))
                current_start = best_split
            else:
                # Keep current segment as is
                segments.append((current_start, current_end))
                break

        return segments

    def _kmeans_segmentation(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """
        K-means based segmentation using sliding window features.

        Args:
            data: Time series data

        Returns:
            List of segment boundaries
        """
        n = len(data)
        window_size = min(self.window_size, n // 4)

        # Handle multi-dimensional data by flattening or using first dimension
        if data.ndim > 1:
            # Use the first dimension for segmentation (or take mean across features)
            data_1d = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]
        else:
            data_1d = data

        # Extract features from sliding windows
        features = []
        for i in range(0, n - window_size + 1, window_size // 2):
            window = data_1d[i : i + window_size]
            features.append(
                [np.mean(window), np.std(window), np.median(window), stats.skew(window) if len(window) > 2 else 0]
            )

        if len(features) < self.n_segments:
            # Not enough features, return single segment
            return [(0, n - 1)]

        # Apply K-means clustering
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=min(self.n_segments, len(features)), random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Convert cluster assignments to segment boundaries
        segments = []
        current_cluster = cluster_labels[0]
        segment_start = 0

        for i, cluster in enumerate(cluster_labels):
            if cluster != current_cluster:
                # End current segment
                segment_end = min(segment_start + window_size - 1, n - 1)
                segments.append((segment_start, segment_end))

                # Start new segment
                segment_start = i * window_size // 2
                current_cluster = cluster

        # Add final segment
        segments.append((segment_start, n - 1))

        return segments

    def _calculate_segment_bounds(
        self, scores: np.ndarray, segments: List[Tuple[int, int]]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate confidence bounds for each segment.

        Args:
            scores: Anomaly scores
            segments: List of segment boundaries

        Returns:
            Dictionary mapping segment index to (lower_bounds, upper_bounds)
        """
        segment_bounds = {}

        for seg_idx, (start_idx, end_idx) in enumerate(segments):
            segment_scores = scores[start_idx : end_idx + 1]

            if len(segment_scores) < 2:
                # Handle small segments
                mean_score = np.mean(segment_scores) if len(segment_scores) > 0 else 0
                segment_bounds[seg_idx] = (
                    np.full(len(segment_scores), mean_score - 0.1),
                    np.full(len(segment_scores), mean_score + 0.1),
                )
                continue

            # Calculate confidence bounds for this segment
            lower_bounds, upper_bounds = self._calculate_confidence_sequence_bounds(segment_scores, self.window_size)

            segment_bounds[seg_idx] = (lower_bounds, upper_bounds)

        return segment_bounds

    def detect_anomalies(self, scores: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using Segmented Confidence Sequences.

        Args:
            scores: Anomaly scores (reconstruction errors)
            data: Original time series data

        Returns:
            Binary anomaly predictions
        """
        if self.verbose:
            logger.info("Starting SCS anomaly detection...")

        # Step 1: Segment the time series
        if self.verbose:
            logger.info("Segmenting time series...")
        self.segments = self._segment_time_series(data)
        if self.verbose:
            logger.info(f"Created {len(self.segments)} segments")

        # Step 2: Calculate confidence bounds for each segment
        if self.verbose:
            logger.info("Calculating segment-specific confidence bounds...")
        self.segment_bounds = self._calculate_segment_bounds(scores, self.segments)

        # Step 3: Detect anomalies using segment-specific thresholds
        if self.verbose:
            logger.info("Detecting anomalies with segment-specific thresholds...")
        anomalies = np.zeros(len(scores), dtype=int)

        for seg_idx, (start_idx, end_idx) in enumerate(self.segments):
            if seg_idx not in self.segment_bounds:
                continue

            segment_scores = scores[start_idx : end_idx + 1]
            lower_bounds, upper_bounds = self.segment_bounds[seg_idx]

            # Detect anomalies in this segment
            segment_anomalies = ((segment_scores < lower_bounds) | (segment_scores > upper_bounds)).astype(int)

            anomalies[start_idx : end_idx + 1] = segment_anomalies

        # Step 4: Apply additional percentile-based filtering for conservativeness
        # Only keep anomalies that are also above a configurable percentile threshold
        percentile_threshold = np.percentile(scores, self.percentile_filter)
        percentile_filter = scores > percentile_threshold

        # Combine confidence sequence detection with percentile filter
        final_anomalies = anomalies & percentile_filter

        if self.verbose:
            logger.info(f"SCS detected {np.sum(anomalies)} initial anomalies")
            logger.info(f"After percentile filtering: {np.sum(final_anomalies)} anomalies out of {len(scores)} points")
        return final_anomalies


class MACSThreshold(BaseAdaptiveThreshold):
    """
    Multi-Scale Adaptive Confidence Segments (MACS) Thresholding.

    This method combines rolling-window segmentation at multiple temporal scales
    with confidence sequence-based thresholding to generate thresholds that adapt
    across both time and resolution.
    """

    def __init__(
        self,
        short_window: int = 10,
        medium_window: int = 50,
        long_window: int = 250,
        confidence_level: float = 0.95,
        attention_weights: List[float] | None = None,
        percentile_filter: float = 97.0,  # Configurable percentile filter
        verbose: bool = True,  # Control logging output
    ):
        """
        Initialize MACS thresholding.

        Args:
            short_window: Window size for capturing bursts
            medium_window: Window size for short-term drifts
            long_window: Window size for regime shifts
            confidence_level: Confidence level for bounds
            attention_weights: Weights for combining multi-scale results [short, medium, long]
            percentile_filter: Percentile threshold for additional filtering
            verbose: Whether to output logging information
        """
        super().__init__(confidence_level)
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.attention_weights = attention_weights or [0.4, 0.35, 0.25]  # Default weights
        self.percentile_filter = percentile_filter
        self.verbose = verbose

        # Validate weights
        if len(self.attention_weights) != 3:
            raise ValueError("attention_weights must have exactly 3 elements")
        if not np.isclose(sum(self.attention_weights), 1.0):
            raise ValueError("attention_weights must sum to 1.0")

    def _calculate_multi_scale_bounds(self, scores: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate confidence bounds at multiple temporal scales.

        Args:
            scores: Anomaly scores

        Returns:
            Dictionary with bounds for each scale
        """
        scales = {"short": self.short_window, "medium": self.medium_window, "long": self.long_window}

        multi_scale_bounds = {}

        for scale_name, window_size in scales.items():
            if self.verbose:
                logger.info(f"Calculating {scale_name}-scale bounds (window={window_size})...")

            lower_bounds, upper_bounds = self._calculate_confidence_sequence_bounds(scores, window_size)

            multi_scale_bounds[scale_name] = (lower_bounds, upper_bounds)

        return multi_scale_bounds

    def _apply_attention_mechanism(
        self, multi_scale_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]], scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply attention mechanism to combine multi-scale bounds.

        Args:
            multi_scale_bounds: Bounds for each temporal scale
            scores: Anomaly scores

        Returns:
            Combined lower and upper bounds
        """
        n = len(scores)
        combined_lower = np.zeros(n)
        combined_upper = np.zeros(n)

        # Calculate attention scores based on local variance
        roll_window = max(2, min(self.short_window, n // 10))
        local_variance = pd.Series(scores).rolling(window=roll_window, min_periods=1).var().fillna(0).values

        # Normalize variance for attention weights
        max_var = np.max(local_variance)
        normalized_variance = local_variance / max_var if max_var > 0 else np.zeros_like(local_variance)

        # Build weight vectors with vectorized masks
        high_var = normalized_variance > 0.7
        med_var = (normalized_variance > 0.3) & ~high_var
        low_var = ~high_var & ~med_var

        weights = np.zeros((n, 3), dtype=np.float64)
        weights[high_var] = np.array([0.6, 0.3, 0.1])
        weights[med_var] = np.array([0.2, 0.6, 0.2])
        weights[low_var] = np.array([0.1, 0.3, 0.6])

        lower_short, upper_short = multi_scale_bounds["short"]
        lower_med, upper_med = multi_scale_bounds["medium"]
        lower_long, upper_long = multi_scale_bounds["long"]

        combined_lower = weights[:, 0] * lower_short + weights[:, 1] * lower_med + weights[:, 2] * lower_long
        combined_upper = weights[:, 0] * upper_short + weights[:, 1] * upper_med + weights[:, 2] * upper_long

        return combined_lower, combined_upper

    def _detect_regime_changes(self, scores: np.ndarray) -> np.ndarray:
        """
        Detect regime changes using statistical change detection.

        Args:
            scores: Anomaly scores

        Returns:
            Binary array indicating regime change points
        """
        # Use CUSUM-like change detection (vectorized)
        n = len(scores)
        if n == 0:
            return np.array([], dtype=int)

        scores_series = pd.Series(scores)
        rolling_mean = scores_series.rolling(window=self.long_window, min_periods=1).mean().fillna(0).values
        rolling_std = scores_series.rolling(window=self.long_window, min_periods=1).std().fillna(0).values

        # Historical window stats (shifted rolling)
        hist_mean = scores_series.rolling(window=self.long_window, min_periods=1).mean().shift(1).fillna(0).values
        hist_std = scores_series.rolling(window=self.long_window, min_periods=1).std().shift(1).fillna(0).values

        denom = hist_std + 1e-8
        mean_change = np.abs(rolling_mean - hist_mean) / denom
        std_change = np.abs(rolling_std - hist_std) / denom

        regime_changes = (mean_change > 2.0) | (std_change > 1.5)
        return regime_changes.astype(int)

    def detect_anomalies(self, scores: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using Multi-Scale Adaptive Confidence Segments.

        Args:
            scores: Anomaly scores (reconstruction errors)
            data: Original time series data

        Returns:
            Binary anomaly predictions
        """
        if self.verbose:
            logger.info("Starting MACS anomaly detection...")

        # Step 1: Calculate multi-scale confidence bounds
        if self.verbose:
            logger.info("Calculating multi-scale confidence bounds...")
        multi_scale_bounds = self._calculate_multi_scale_bounds(scores)

        # Step 2: Apply attention mechanism to combine scales
        if self.verbose:
            logger.info("Applying attention mechanism...")
        combined_lower, combined_upper = self._apply_attention_mechanism(multi_scale_bounds, scores)

        # Step 3: Detect regime changes
        if self.verbose:
            logger.info("Detecting regime changes...")
        regime_changes = self._detect_regime_changes(scores)

        # Step 4: Generate final anomaly predictions
        if self.verbose:
            logger.info("Generating final anomaly predictions...")
        anomalies = np.zeros(len(scores), dtype=int)

        # Method 1: Flag points that exceed all thresholds
        all_scale_anomalies = np.zeros(len(scores), dtype=int)

        for scale_name in ["short", "medium", "long"]:
            lower_bounds, upper_bounds = multi_scale_bounds[scale_name]
            scale_anomalies = ((scores < lower_bounds) | (scores > upper_bounds)).astype(int)
            all_scale_anomalies += scale_anomalies

        # Points are anomalous if they exceed at least 2 out of 3 thresholds (more selective)
        threshold_violations = all_scale_anomalies >= 2

        # Method 2: Use attention-weighted combined bounds
        attention_anomalies = ((scores < combined_lower) | (scores > combined_upper)).astype(int)

        # Combine both methods with regime change awareness
        for i in range(len(scores)):
            if regime_changes[i]:
                # During regime changes, be more conservative
                anomalies[i] = threshold_violations[i] and attention_anomalies[i]
            else:
                # Normal operation - use attention-weighted bounds
                anomalies[i] = attention_anomalies[i]

        # Step 5: Apply additional percentile-based filtering for conservativeness
        # Only keep anomalies that are also above a configurable percentile threshold
        percentile_threshold = np.percentile(scores, self.percentile_filter)
        percentile_filter = scores > percentile_threshold

        # Combine multi-scale detection with percentile filter
        final_anomalies = anomalies & percentile_filter

        if self.verbose:
            logger.info(f"MACS detected {np.sum(anomalies)} initial anomalies")
            logger.info(f"After percentile filtering: {np.sum(final_anomalies)} anomalies out of {len(scores)} points")
            logger.info(f"Regime changes detected: {np.sum(regime_changes)}")

        return final_anomalies


def compare_adaptive_methods(
    scores: np.ndarray, data: np.ndarray, ground_truth: np.ndarray | None = None, verbose: bool = True
) -> Dict[str, Dict]:
    """
    Compare different adaptive thresholding methods.

    Args:
        scores: Anomaly scores
        data: Original time series data
        ground_truth: Ground truth labels (optional)
        verbose: Whether to output logging information

    Returns:
        Dictionary with results for each method
    """
    results = {}

    # Initialize thresholders
    scs_thresholder = SCSThreshold(
        window_size=100, confidence_level=0.95, n_segments=5, segmentation_method="apca", verbose=verbose
    )

    macs_thresholder = MACSThreshold(
        short_window=10, medium_window=50, long_window=250, confidence_level=0.95, verbose=verbose
    )

    # Apply SCS method
    if verbose:
        logger.info("Applying SCS method...")
    scs_anomalies = scs_thresholder.detect_anomalies(scores, data)
    results["scs"] = {
        "anomalies": scs_anomalies,
        "num_anomalies": np.sum(scs_anomalies),
        "anomaly_rate": np.mean(scs_anomalies),
    }

    # Apply MACS method
    if verbose:
        logger.info("Applying MACS method...")
    macs_anomalies = macs_thresholder.detect_anomalies(scores, data)
    results["macs"] = {
        "anomalies": macs_anomalies,
        "num_anomalies": np.sum(macs_anomalies),
        "anomaly_rate": np.mean(macs_anomalies),
    }

    # Calculate metrics if ground truth is available
    if ground_truth is not None:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        for method_results in results.values():
            anomalies = method_results["anomalies"]

            # Calculate classification metrics
            accuracy = accuracy_score(ground_truth, anomalies)
            precision = precision_score(ground_truth, anomalies, zero_division=0)
            recall = recall_score(ground_truth, anomalies, zero_division=0)
            f1 = f1_score(ground_truth, anomalies, zero_division=0)

            method_results.update({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})

    return results


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)
    n_points = 1000

    # Create synthetic time series with known anomalies
    time_series = np.sin(np.linspace(0, 4 * np.pi, n_points)) + 0.1 * np.random.randn(n_points)

    # Add some anomalies
    anomaly_indices = [200, 400, 600, 800]
    for idx in anomaly_indices:
        time_series[idx] += 3.0  # Large spike

    # Generate reconstruction errors (scores)
    scores = np.abs(np.diff(time_series, prepend=time_series[0])) + 0.1 * np.random.randn(n_points)

    # Create ground truth
    ground_truth = np.zeros(n_points)
    for idx in anomaly_indices:
        ground_truth[idx] = 1

    # Test both methods
    results = compare_adaptive_methods(scores, time_series, ground_truth)

    logger.info("=== Adaptive Thresholding Results ===")
    for method_name, method_results in results.items():
        logger.info("%s:", method_name.upper())
        logger.info("  Anomalies detected: %s", method_results["num_anomalies"])
        logger.info("  Anomaly rate: %.3f", method_results["anomaly_rate"])

        if "accuracy" in method_results:
            logger.info("  Accuracy: %.3f", method_results["accuracy"])
            logger.info("  Precision: %.3f", method_results["precision"])
            logger.info("  Recall: %.3f", method_results["recall"])
            logger.info("  F1-Score: %.3f", method_results["f1_score"])
