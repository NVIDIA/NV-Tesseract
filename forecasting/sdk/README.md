# Tesseract Forecasting SDK

Programmatic entry point for running `perform_forecasting()` on pandas DataFrames. Supports multivariate forecasting, context-enhanced (DARR) predictions, and model-agnostic interpretability with horizon-specific lag (temporal-axis) and channel (feature-axis) attributions.

## Features

- **DataFrame-first API**: Works with pandas DataFrames; automatically detects numeric columns as features while keeping the provided timestamp + target.
- **Autoregressive horizon extension**: Automatically repeats the model prediction window when `forecast_horizon` exceeds the model's native horizon.
- **DARR mode**: Blends direct model output with kNN-based context memory, with configurable `alpha`, `k`, and `temperature`.
- **Robust preprocessing**: Converts timestamps, fills numeric NULLs with zeros, enforces minimum sequence length (`seq_len`), and standardizes input using saved standardizer metadata.
- **Column alignment**: Automatically handles datasets with different feature sets by aligning to common columns, preventing broadcasting errors.
- **Diverse output**: Produces hybrid, direct, and kNN forecasts when context is provided; otherwise returns only the direct forecast column.
- **Built-in interpretability**: Opt-in `interpretability=True` flag produces a horizon-resolved lag attribution matrix, supporting CSVs / heatmap PNG, a `semantic_flow.csv` of per-transition latent flow magnitudes labeled by history/forecast segment, an `explanation.json` (lag×horizon, latent trajectory, semantic-flow magnitudes, forecast-vs-history diagnostics, a `trajectory_stability` block of temporal-smoothness metrics, and — for multivariate inputs — a `feature_axis` block), and a self-contained PDF report whose final pages cover semantic-flow magnitudes (line chart over the history/forecast split with per-segment summary statistics and the four forecast-vs-history diagnostic ratios), latent-trajectory stability, and the feature-axis attribution. Output format is selectable via `interpretability_output` (`"json"`, `"pdf"`, or `None` for both).
- **Feature-axis attribution**: For multivariate inputs (more than one numeric column), the interpretability run additionally decomposes the forecast along the *feature axis*, reporting a channel × horizon attribution matrix (which input channel drives each forecast step) as `channel_horizon_attributions.csv`, a `channel_horizon_heatmap.png`, and a dedicated PDF page. Computed with a fast first-order (Jacobian-flow) estimator that carries an analytic trust ratio.

## Installation

The SDK is shipped with `tesseract_forecasting`. Install (or update) the package and dependencies using:

```bash
uv sync
```

## Quick start

### Standard inference

```python
from sdk.forecasting import perform_forecasting
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=600, freq="H"),
    "target": np.sin(np.linspace(0, 4 * np.pi, 600)),
    "feature_a": np.random.randn(600),
    "feature_b": np.random.randn(600),
})

forecasts = perform_forecasting(
    df=df,
    seq_len=512,
    forecast_horizon=72,
)

# Result contains a `target_forecast` column with `forecast_horizon` rows
```

### DARR (context-enhanced) inference

```python
context = pd.DataFrame({
    "timestamp": pd.date_range("2022-01-01", periods=2500, freq="H"),
    "target": historical_target_series,
    "feature_a": historical_feature_a,
})

darr_result = perform_forecasting(
    df=df,
    context_df=context,
    seq_len=512,
    forecast_horizon=72,
    alpha=0.2,  # 20% direct, 80% kNN
    k=64,
    temperature=0.05,
)

# DARR output includes hybrid, direct, and kNN forecast columns
```

#### Context data expectations

The `context_df` you supply should contain the same `timestamp` and `target_column` as your input data, but **does not need to have identical feature columns**. The SDK automatically handles column mismatches by:

1. **Detecting differences**: Identifies when input and context datasets have different feature sets
2. **Finding common features**: Determines the intersection of feature columns between datasets  
3. **Automatic alignment**: Uses only common features for consistent predictions
4. **Clear reporting**: Provides warnings about column mismatches and alignment decisions

```python
# Input dataset with many features
df = pd.DataFrame({
    'timestamp': timestamps,
    'target': values,
    'feature_A': data_A,
    'feature_B': data_B, 
    'feature_C': data_C,
})

# Context dataset with fewer features - this is now supported!
context_df = pd.DataFrame({
    'timestamp': historical_timestamps,
    'target': historical_values,
    'feature_A': historical_A,  # Only this feature in common
})

# SDK automatically aligns to use only feature_A
result = perform_forecasting(df=df, context_df=context_df, ...)
```

**Requirements**:
- Both datasets must have the same `timestamp_column` and `target_column`
- At least one common feature column must exist (besides the target)
- Context dataset must have at least `seq_len + model_horizon` rows

For this release, the blending weight `alpha` defaults to `0.01` and can be adjusted via the `alpha` parameter—the hybrid forecast uses that value to combine direct predictions with context-derived neighbors (`alpha * direct + (1 - alpha) * kNN`).

### Interpretability inference

Setting `interpretability=True` runs the loaded model on the trailing `seq_len` window and produces a horizon-resolved explanation alongside the forecast. The artifacts are written to a UTC-stamped subdirectory under `interpretability_out_dir`.

```python
from pathlib import Path

interp_result = perform_forecasting(
    df=df,
    seq_len=512,
    forecast_horizon=100,
    interpretability=True,
    interpretability_output=None,                 # "json", "pdf", or None for both
    interpretability_out_dir=Path("interpretability_output"),
    interpretability_dataset_name="my_dataset.csv",
    n_lags=128,
    softmax_tau=1.0,
)
```

`interpretability_output` selects which artifacts are written:

| Value | Files written under `<interpretability_out_dir>/run_<UTC>/` |
|-------|-------------------------------------------------------------|
| `"json"` | `forecast.csv`, `explanation.json` |
| `"pdf"` | `forecast.csv`, `lag_horizon_attributions.csv`, `lag_horizon_long.csv`, `lag_horizon_heatmap.png`, `semantic_flow.csv`, `explanation_report.pdf` (plus `channel_horizon_attributions.csv` and `channel_horizon_heatmap.png` for multivariate inputs) |
| `None`  | All of the above |

The returned DataFrame is the explanation-aligned forecast (single forward pass, so it lines up 1:1 with the attribution matrix). PDF / heatmap require `matplotlib`; if it's missing, those steps are skipped with a warning while JSON output continues to work.

If the input DataFrame has more than one numeric column, the feature-axis decomposition runs automatically (no extra flag): the report gains a channel × horizon attribution page and `channel_horizon_*` artifacts alongside the temporal lag × horizon outputs. By default this is measured in latent space (output-agnostic); set `channel_output_aware=True` to instead attribute the target channel's forecast directly (see the parameter table below).

## Function signature

```python
perform_forecasting(
    # Input data
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    target_column: str = "target",
    context_df: Optional[pd.DataFrame] = None,
    
    # Model configuration
    standardizer_pkl: str = "standardizer.pkl",
    ckpt: str = "moment_head_512_6hr.pt",
    seq_len: int = 512,
    forecast_horizon: int = 72,
    model_horizon: int = 72,
    
    # Output configuration
    save_preds: Optional[str] = None,
    
    # DARR mode configuration
    alpha: float = 0.01,
    k: int = 64,
    temperature: float = 0.05,
    
    # Additional parameters
    model_name: str = "configured pretrained backbone identifier",
    batch_size: int = 8,
    num_workers: int = 2,
    stride: Optional[int] = None,
    context_stride: Optional[int] = None,
    seed: int = 13,
    device: Optional[str] = None,
    local_files_only: bool = False,

    # Interpretability
    interpretability: bool = False,
    interpretability_output: Optional[str] = None,            # "json" | "pdf" | None
    interpretability_out_dir: Union[str, Path] = "interpretability_output",
    interpretability_run_name: Optional[str] = None,
    interpretability_top_k: int = 5,
    interpretability_dataset_name: Optional[str] = None,
    n_lags: int = 128,
    softmax_tau: float = 1.0,
    channel_output_aware: bool = False,
) -> pd.DataFrame
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `df` | Input DataFrame containing time series data |
| `timestamp_column` / `target_column` | Alias your columns when they differ from the defaults |
| `standardizer_pkl`, `ckpt` | Paths to the training artifacts |
| `seq_len` | Number of rows consumed for a single inference window (must exist in the provided `df`) |
| `forecast_horizon` | How many future steps to return (max: 512). If this exceeds `model_horizon`, the SDK repeats predictions autoregressively |
| `model_horizon` | Native horizon the model was trained on (default: 72). Override if using different weights |
| `context_df` | Enables DARR mode when supplied |
| `alpha` | Blending factor for DARR hybrid output (`alpha * direct + (1 - alpha) * kNN`) |
| `k` / `temperature` | Configure kNN retrieval (number of neighbors and softmax temperature) |
| `stride` / `context_stride` | Stride for windowing; defaults to `model_horizon` |
| `save_preds` | Optional CSV export location |
| `device` | Override device selection (defaults to auto-detected CUDA/MPS/CPU) |
| `local_files_only` | Load model from local cache without network access |
| `interpretability` | Master switch. When `True`, the SDK skips the standard / DARR inference path and runs the interpretability explanation pipeline against the loaded model |
| `interpretability_output` | `"json"` for explanation JSON only, `"pdf"` for the heatmap + PDF bundle, `None` for both. Invalid values raise `ValueError` |
| `interpretability_out_dir` | Parent directory for the run subfolder; created if missing |
| `interpretability_run_name` | Override the auto-stamped `run_<UTC>` folder name |
| `interpretability_top_k` | How many top lag steps per horizon to render in the PDF table (default `5`) |
| `interpretability_dataset_name` | Free-form label embedded in the JSON metadata and the PDF cover page |
| `n_lags` | Number of past steps the lag-attribution matrix resolves (default `128`) |
| `softmax_tau` | Temperature applied when softmaxing scores into per-horizon attribution |
| `channel_output_aware` | Feature-axis only (multivariate inputs). When `False` (default), channel attributions are measured in latent space (output-agnostic "semantic flow"). When `True`, they are measured on the target channel's forecast, so the channel × horizon page answers "which input channel drives the *target's* prediction". More interpretable for a single target, but costs extra forward passes (each probe window is forecast rather than embedded). Requires channel coupling (`use_cross_channel=True`, the default) to attribute non-target channels; otherwise only the target contributes |

## Preprocessing expectations

- Timestamp column must be parseable by pandas and free of NULLs.
- Target column must be numeric; NULLs are filled with zeros.
- All numeric features are automatically included; NULLs become zeros.
- Input length must be at least `seq_len`; otherwise `ValueError` is raised.

## Outputs

- **Standard mode**: `{target_column}_forecast` containing the requested `forecast_horizon` predictions with timestamps inferred from the input frequency.
- **DARR mode**: Returns hybrid predictions in `{target_column}_forecast` (direct and kNN components are computed internally).
- **Interpretability mode**: Returns the explanation-aligned forecast in `{target_column}_forecast` and writes the artifact bundle to `<interpretability_out_dir>/run_<UTC>/`.

### Output DataFrame structure

| Column | Description |
|--------|-------------|
| `{timestamp_column}` | Forecasted timestamps starting after the last input timestamp |
| `{target_column}_forecast` | Predicted values for the forecast horizon |

### Interpretability artifact bundle

When `interpretability=True`, the run directory contains the following files (selected by `interpretability_output`):

| File | Written when | Contents |
|------|--------------|----------|
| `forecast.csv` | json, pdf, both | The returned forecast DataFrame |
| `explanation.json` | json, both | Forecast + full explanation payload (baseline forecast, lag×horizon scores and attributions, latent trajectory, semantic-flow magnitudes, `diagnostics` block including `latent_trajectory_shape` and the forecast-vs-history ratios, `trajectory_stability` block with temporal-smoothness metrics over the context window, a `feature_axis` block for multivariate inputs (estimator method, channel×horizon attributions, Jacobian trust ratios, and the per-channel flow), and dataset metadata) |
| `lag_horizon_attributions.csv` | pdf, both | Wide K×H attribution matrix |
| `lag_horizon_long.csv` | pdf, both | Tidy `(lag, horizon, attribution[, score])` table |
| `lag_horizon_heatmap.png` | pdf, both *(needs matplotlib)* | Visual heatmap, viridis cmap |
| `semantic_flow.csv` | pdf, both | Tidy `(transition_index, segment, flow_magnitude)` table, where `segment` is `history` for transitions fully inside the input window, `forecast` for transitions whose window extends into the model-generated future, and `tail` for any trailing transitions outside both segments |
| `channel_horizon_attributions.csv` | pdf, both *(multivariate only)* | Wide `[C × H]` channel × horizon attribution matrix (first column is the channel label); each horizon column sums to 1 over channels |
| `channel_horizon_heatmap.png` | pdf, both *(multivariate, needs matplotlib)* | Feature-axis heatmap: rows are input channels, columns are forecast steps, brighter = that channel matters more there |
| `explanation_report.pdf` | pdf, both *(needs matplotlib)* | Multi-page report: (1) cover with metadata, (2) forecast preview, (3) lag×horizon heatmap, (4) top-`interpretability_top_k` lag-step tables, (5) semantic-flow magnitudes page with the per-transition flow time series (history/forecast split annotated), a per-segment summary (mean / median / p95 / max / variance / transition count) and the four forecast-vs-history diagnostic ratios (`flow_ratio`, `flow_variance_ratio`, `curvature_ratio`, `latent_diag_mahalanobis_ratio`), (6) latent-trajectory stability table with per-dimension zero-crossing / direction-flip / relative-jitter (mean & p95) and occupancy metrics, and (7) feature-axis attribution page (multivariate inputs only) with the channel × horizon heatmap, a per-channel importance bar, and the Jacobian trust ratio |

The run directory path is printed on stdout when the call completes.

## Error handling

### Column Mismatch Handling

The SDK automatically detects and handles column mismatches between input and context datasets:

```python
# Example: Input has 7 features, context has 3 features
# Before fix: "operands could not be broadcast together with shapes (1,7,24) (1,3,24)"
# After fix: Automatic alignment using common features

Warning: Column mismatch detected between input and context datasets
  Input dataset columns: ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'OT']
  Context dataset columns: ['HULL', 'MULL', 'LUFL']
  Common columns: ['HULL', 'MULL', 'LUFL']
  Using only common columns for consistent predictions: ['HULL', 'MULL', 'LUFL']
```

**Behavior**:
- **Automatic detection**: Identifies when datasets have different feature sets
- **Intelligent alignment**: Uses intersection of feature columns for consistent shapes
- **Clear warnings**: Reports what columns are being used and why
- **Graceful fallback**: Prevents cryptic NumPy broadcasting errors

### Common Error Scenarios

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: No common numeric columns found between input and context datasets` | Datasets share no feature columns (only target) | Ensure context dataset has at least one feature column in common with input |
| `ValueError: Shape mismatch between direct and kNN predictions` | Column alignment failed internally | Check that both datasets have valid numeric columns |
| `ValueError: DataFrame has X rows but seq_len requires at least Y rows` | Insufficient data points | Provide more data or reduce `seq_len` |
| `ValueError: Context DataFrame has X rows but requires at least Y rows` | Context dataset too small | Context needs `seq_len + model_horizon` rows minimum |
| `ValueError: forecast_horizon must be <= 512` | Forecast horizon too large | Reduce `forecast_horizon` or make multiple calls |
| `ValueError: Target column 'X' not found in DataFrame` | Missing target column | Verify column name or use `target_column` parameter |
| `ValueError: interpretability_output must be one of None, 'json', 'pdf'` | Bad value for the artifact selector | Use `None`, `"json"`, or `"pdf"` |
| `Interpretability PDF report skipped: matplotlib is not installed.` | PDF/heatmap path attempted without matplotlib | `uv add matplotlib`, or set `interpretability_output="json"` |

### Data Validation

Common safeguards raised as `ValueError` include:

- Missing or NULL timestamp column
- Unparseable timestamps  
- Missing target column
- Non-numeric target data
- Too few rows for the configured `seq_len`
- Invalid `model_horizon` or `forecast_horizon` (must be positive)
- `forecast_horizon` exceeds maximum limit of 512
- **NEW**: No common feature columns between input and context datasets

## Examples and tests

See `tesseract_forecasting/sdk/tests/test_forecasting.py` for unit test coverage with mockers and `sdk/quick_example.py` for an end-to-end script.
