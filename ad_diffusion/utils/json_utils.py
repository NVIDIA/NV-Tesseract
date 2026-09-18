# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON serialization utilities for numpy arrays, DataFrames, and sklearn models.

Replaces pickle for data serialization with JSON, preserving dtype,
shape, and handling NaN/Inf values. Includes sklearn model serialization
by extracting fitted parameters.
"""

import json
import math

import numpy as np
import pandas as pd


def _serialize_numpy_for_json(obj):
    """Convert numpy arrays, dtypes, and scalars to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return {
            "__numpy_array__": True,
            "data": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, np.dtype):
        return {"__numpy_dtype__": True, "dtype": str(obj)}
    if isinstance(obj, np.integer):
        return obj.item()
    if isinstance(obj, np.floating):
        val = obj.item()
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, dict):
        return {k: _serialize_numpy_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_serialize_numpy_for_json(item) for item in obj]
    if isinstance(obj, pd.DataFrame):
        return {
            "__dataframe__": True,
            "columns": list(obj.columns),
            "data": _serialize_numpy_for_json(obj.values),
            "index": obj.index.tolist(),
        }
    return obj


def _deserialize_numpy_from_json(obj):
    """Convert JSON-serialized numpy arrays back to numpy objects."""
    if isinstance(obj, dict):
        if obj.get("__numpy_array__"):
            return np.array(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        if obj.get("__numpy_dtype__"):
            return np.dtype(obj["dtype"])
        if obj.get("__dataframe__"):
            arr = _deserialize_numpy_from_json(obj["data"])
            return pd.DataFrame(arr, columns=obj["columns"])
        return {k: _deserialize_numpy_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_numpy_from_json(item) for item in obj]
    return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types, NaN, and Inf."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return _serialize_numpy_for_json(obj)
        if isinstance(obj, np.dtype):
            return str(obj)
        if isinstance(obj, np.integer):
            return obj.item()
        if isinstance(obj, np.floating):
            val = obj.item()
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        return super().default(obj)


def save_data(data, path):
    """Save data (numpy array, DataFrame, or any serializable object) to JSON."""
    serialized = _serialize_numpy_for_json(data)
    with open(path, "w") as f:
        json.dump(serialized, f)


def load_data(path):
    """Load data from JSON, converting numpy arrays back."""
    with open(path) as f:
        raw = json.load(f)
    return _deserialize_numpy_from_json(raw)


# ---------------------------------------------------------------------------
# Sklearn model serialization
# ---------------------------------------------------------------------------


def _serialize_sklearn_model(model):
    """Serialize a fitted sklearn model to a JSON-compatible dict."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import (
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    if model is None:
        return None

    if isinstance(model, RobustScaler):
        data = {
            "__sklearn__": "RobustScaler",
            "center_": model.center_,
            "scale_": model.scale_,
            "n_features_in_": getattr(model, "n_features_in_", None),
        }
        # Custom attributes added by AdaptiveNormalizer
        if hasattr(model, "lower_bound"):
            data["lower_bound"] = model.lower_bound
        if hasattr(model, "upper_bound"):
            data["upper_bound"] = model.upper_bound
        return _serialize_numpy_for_json(data)

    if isinstance(model, StandardScaler):
        return _serialize_numpy_for_json(
            {
                "__sklearn__": "StandardScaler",
                "mean_": model.mean_,
                "scale_": model.scale_,
                "var_": model.var_,
                "n_features_in_": getattr(model, "n_features_in_", None),
                "n_samples_seen_": getattr(model, "n_samples_seen_", None),
            }
        )

    if isinstance(model, QuantileTransformer):
        return _serialize_numpy_for_json(
            {
                "__sklearn__": "QuantileTransformer",
                "quantiles_": model.quantiles_,
                "references_": model.references_,
                "n_quantiles_": model.n_quantiles_,
                "output_distribution": model.output_distribution,
                "subsample": model.subsample,
                "n_features_in_": getattr(model, "n_features_in_", None),
            }
        )

    if isinstance(model, PowerTransformer):
        return _serialize_numpy_for_json(
            {
                "__sklearn__": "PowerTransformer",
                "lambdas_": model.lambdas_,
                "method": model.method,
                "_scaler": _serialize_sklearn_model(model._scaler),
                "n_features_in_": getattr(model, "n_features_in_", None),
            }
        )

    if isinstance(model, PCA):
        return _serialize_numpy_for_json(
            {
                "__sklearn__": "PCA",
                "components_": model.components_,
                "mean_": model.mean_,
                "explained_variance_": model.explained_variance_,
                "explained_variance_ratio_": model.explained_variance_ratio_,
                "singular_values_": model.singular_values_,
                "n_components": model.n_components,
                "n_features_in_": getattr(model, "n_features_in_", None),
                "n_samples_seen_": getattr(model, "n_samples_seen_", None),
            }
        )

    msg = f"Unsupported sklearn model type: {type(model)}"
    raise TypeError(msg)


def _deserialize_sklearn_model(data):
    """Reconstruct a fitted sklearn model from a JSON dict."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import (
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    if data is None:
        return None

    # Deserialize numpy arrays first
    data = _deserialize_numpy_from_json(data)

    model_type = data.get("__sklearn__")

    if model_type == "RobustScaler":
        scaler = RobustScaler()
        scaler.center_ = data["center_"]
        scaler.scale_ = data["scale_"]
        if data.get("n_features_in_") is not None:
            scaler.n_features_in_ = data["n_features_in_"]
        if "lower_bound" in data:
            scaler.lower_bound = data["lower_bound"]
        if "upper_bound" in data:
            scaler.upper_bound = data["upper_bound"]
        return scaler

    if model_type == "StandardScaler":
        scaler = StandardScaler()
        scaler.mean_ = data["mean_"]
        scaler.scale_ = data["scale_"]
        scaler.var_ = data["var_"]
        if data.get("n_features_in_") is not None:
            scaler.n_features_in_ = data["n_features_in_"]
        if data.get("n_samples_seen_") is not None:
            scaler.n_samples_seen_ = data["n_samples_seen_"]
        return scaler

    if model_type == "QuantileTransformer":
        qt = QuantileTransformer(
            n_quantiles=data["n_quantiles_"],
            output_distribution=data["output_distribution"],
            subsample=data["subsample"],
        )
        qt.quantiles_ = data["quantiles_"]
        qt.references_ = data["references_"]
        qt.n_quantiles_ = data["n_quantiles_"]
        if data.get("n_features_in_") is not None:
            qt.n_features_in_ = data["n_features_in_"]
        return qt

    if model_type == "PowerTransformer":
        pt = PowerTransformer(method=data["method"])
        pt.lambdas_ = data["lambdas_"]
        pt._scaler = _deserialize_sklearn_model(data["_scaler"])
        if data.get("n_features_in_") is not None:
            pt.n_features_in_ = data["n_features_in_"]
        return pt

    if model_type == "PCA":
        pca = PCA(n_components=data["n_components"])
        pca.components_ = data["components_"]
        pca.mean_ = data["mean_"]
        pca.explained_variance_ = data["explained_variance_"]
        pca.explained_variance_ratio_ = data["explained_variance_ratio_"]
        pca.singular_values_ = data["singular_values_"]
        if data.get("n_features_in_") is not None:
            pca.n_features_in_ = data["n_features_in_"]
        if data.get("n_samples_seen_") is not None:
            pca.n_samples_seen_ = data["n_samples_seen_"]
        return pca

    msg = f"Unknown sklearn model type: {model_type}"
    raise ValueError(msg)


def serialize_adaptive_normalizer(normalizer):
    """Serialize an AdaptiveNormalizer to a JSON-compatible dict."""
    data = {
        "__adaptive_normalizer__": True,
        "target_range": normalizer.target_range,
        "method": normalizer.method,
    }

    # Serialize params, handling sklearn objects inside
    serialized_params = {}
    for k, v in normalizer.params.items():
        if hasattr(v, "fit"):
            # It's a sklearn model
            serialized_params[k] = _serialize_sklearn_model(v)
        else:
            serialized_params[k] = _serialize_numpy_for_json(v)
    data["params"] = serialized_params

    # Preserve optional extended normalizer state for backward compatibility.
    if hasattr(normalizer, "scalers"):
        data["scalers"] = [_serialize_sklearn_model(s) for s in normalizer.scalers]
    if hasattr(normalizer, "n_features"):
        data["n_features"] = normalizer.n_features
    if hasattr(normalizer, "spike_threshold"):
        data["spike_threshold"] = normalizer.spike_threshold
    if hasattr(normalizer, "fitted"):
        data["fitted"] = normalizer.fitted

    return data


def deserialize_adaptive_normalizer(data, normalizer_class):
    """Reconstruct an AdaptiveNormalizer from a JSON dict."""
    normalizer = normalizer_class.__new__(normalizer_class)
    normalizer.target_range = data["target_range"]
    normalizer.method = data["method"]

    # Deserialize params, reconstructing sklearn objects
    params = {}
    for k, v in data["params"].items():
        if isinstance(v, dict) and "__sklearn__" in v:
            params[k] = _deserialize_sklearn_model(v)
        else:
            params[k] = _deserialize_numpy_from_json(v)
    normalizer.params = params

    # Handle scalers list
    if "scalers" in data:
        normalizer.scalers = [_deserialize_sklearn_model(s) for s in data["scalers"]]
    if "n_features" in data:
        normalizer.n_features = data["n_features"]
    if "spike_threshold" in data:
        normalizer.spike_threshold = data["spike_threshold"]
    if "fitted" in data:
        normalizer.fitted = data["fitted"]

    return normalizer


def save_preprocessing_models(models_dict, path):
    """Save preprocessing models (normalizers, preprocessors, PCA) to JSON.

    Handles nested dicts of AdaptiveNormalizer and sklearn objects.
    """
    serialized = {}
    for key, value in models_dict.items():
        if isinstance(value, dict):
            # Could be nested: domain -> {feature_count -> normalizer}
            serialized[key] = _serialize_model_dict(value)
        elif hasattr(value, "fit") and hasattr(value, "transform"):
            serialized[key] = _serialize_sklearn_model(value)
        else:
            serialized[key] = _serialize_numpy_for_json(value)

    with open(path, "w") as f:
        json.dump(serialized, f)


def load_preprocessing_models(path, normalizer_class=None):
    """Load preprocessing models from JSON.

    Args:
        path: Path to JSON file.
        normalizer_class: The AdaptiveNormalizer class to use for reconstruction.
    """
    with open(path) as f:
        raw = json.load(f)

    return _deserialize_model_dict(raw, normalizer_class)


def _serialize_model_dict(d):
    """Recursively serialize a dict that may contain sklearn models or normalizers."""
    if not isinstance(d, dict):
        return _serialize_numpy_for_json(d)

    # Check if it's an AdaptiveNormalizer (has method + params attributes serialized)
    if hasattr(d, "method") and hasattr(d, "params"):
        return serialize_adaptive_normalizer(d)

    result = {}
    for k, v in d.items():
        if hasattr(v, "method") and hasattr(v, "params") and hasattr(v, "target_range"):
            result[k] = serialize_adaptive_normalizer(v)
        elif hasattr(v, "fit") and hasattr(v, "transform"):
            result[k] = _serialize_sklearn_model(v)
        elif isinstance(v, dict):
            result[k] = _serialize_model_dict(v)
        elif isinstance(v, list | tuple):
            result[k] = [
                serialize_adaptive_normalizer(item)
                if hasattr(item, "method") and hasattr(item, "params")
                else _serialize_numpy_for_json(item)
                for item in v
            ]
        else:
            result[k] = _serialize_numpy_for_json(v)
    return result


def _deserialize_model_dict(d, normalizer_class=None):
    """Recursively deserialize a dict that may contain sklearn models or normalizers."""
    if not isinstance(d, dict):
        return _deserialize_numpy_from_json(d)

    if d.get("__adaptive_normalizer__") and normalizer_class is not None:
        return deserialize_adaptive_normalizer(d, normalizer_class)

    if "__sklearn__" in d:
        return _deserialize_sklearn_model(d)

    if "__numpy_array__" in d:
        return _deserialize_numpy_from_json(d)

    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deserialize_model_dict(v, normalizer_class)
        elif isinstance(v, list):
            result[k] = [
                _deserialize_model_dict(item, normalizer_class) if isinstance(item, dict) else item for item in v
            ]
        else:
            result[k] = v
    return result
