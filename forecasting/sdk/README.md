# Tesseract Forecasting SDK

Programmatic entry point for running `perform_forecasting()` on pandas DataFrames. Supports multivariate forecasting, context-enhanced (DARR) predictions, and model-agnostic interpretability with horizon-specific lag attributions.

## Features

- **DataFrame-first API**: Works with pandas DataFrames; automatically detects numeric columns as features while keeping the provided timestamp + target.
- **Autoregressive horizon extension**: Automatically repeats the model prediction window when `forecast_horizon` exceeds the model's native horizon.
- **DARR mode**: Blends direct model output with kNN-based context memory, with configurable `alpha`, `k`, and `temperature`.
- **Robust preprocessing**: Converts timestamps, fills numeric NULLs with zeros, enforces minimum sequence length (`seq_len`), and standardizes input using saved standardizer metadata.
- **Column alignment**: Automatically handles datasets with different feature sets by aligning to common columns, preventing broadcasting errors.
- **Diverse output**: Produces hybrid, direct, and kNN forecasts when context is provided; otherwise returns only the direct forecast column.
- **Built-in interpretability**: Opt-in `interpretability=True` produces horizon-resolved lag attribution, semantic-flow diagnostics, latent-trajectory stability, JSON/CSV/PNG artifacts, and a self-contained PDF report. Output format is selectable via `interpretability_output` (`"json"`, `"pdf"`, or `None` for both).
- **Feature-axis interpretability**: For multivariate inputs, the SDK automatically adds batched Jacobian channel-by-horizon attribution to the PDF and artifact bundle.

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
- Context dataset must have at least `seq_len + max(model_horizon, forecast_horizon)` rows so DARR can retrieve the full requested continuation

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
    integrated_gradients=True,
)
```

`interpretability_output` selects which artifacts are written:

| Value | Files written under `<interpretability_out_dir>/run_<UTC>/` |
|-------|-------------------------------------------------------------|
| `"json"` | `forecast.csv`, `explanation.json` |
| `"pdf"` | `forecast.csv`, lag/semantic-flow artifacts, feature-axis artifacts for multivariate input, optional integrated-gradients artifacts, `explanation_report.pdf` |
| `None`  | All of the above |

The returned DataFrame is the explanation-aligned forecast (single forward pass, so it lines up 1:1 with the attribution matrix). PDF / heatmap require `matplotlib`; if it's missing, those steps are skipped with a warning while JSON output continues to work.

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
    integrated_gradients: bool = False,
    integrated_gradients_baseline: Any = "noise",
    integrated_gradients_steps: int = 64,
    integrated_gradients_n_baselines: int = 1,
    integrated_gradients_internal_batch_size: Optional[int] = None,
    integrated_gradients_reduce: str = "l2",
    integrated_gradients_grad_through_norm: bool = True,

    # Multichannel output
    return_all_channels: bool = False,
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
| `channel_output_aware` | Use target-specific directional input-Jacobian effects for feature-axis attribution (default `False`) |
| `integrated_gradients` | Add embedding integrated-gradients attribution to the JSON/PDF report and export its CSV/PNG artifacts |
| `integrated_gradients_baseline` | IG reference input; `"noise"` is the default and avoids the constant-baseline degeneracy of instance normalization |
| `integrated_gradients_steps` | Number of midpoint integration steps (default `64`) |
| `integrated_gradients_n_baselines` | Number of noise baselines averaged as Expected Gradients (default `1`) |
| `integrated_gradients_internal_batch_size` | Maximum interpolation points embedded per batch; `None` processes all steps together |
| `integrated_gradients_reduce` | Embedding scalar objective: `"l2"`, `"sum"`, or `"mean"` |
| `integrated_gradients_grad_through_norm` | Include RevIN statistics in the attribution gradient while leaving forward outputs unchanged |
| `return_all_channels` | When `True`, the result contains one `{column}_forecast` column per processed channel (target column first, then the remaining numeric features) instead of only `{target_column}_forecast`, including in interpretability mode |

## Preprocessing expectations

- Timestamp column must be parseable by pandas and free of NULLs.
- Target column must be numeric; NULLs are filled with zeros.
- All numeric features are automatically included; NULLs become zeros.
- Numeric feature columns should match the set and order the checkpoint/standardizer was trained with; a different order silently misattributes forecasts, and a different column count fails at standardization with a broadcasting error.
- Input length must be at least `seq_len`; otherwise `ValueError` is raised.

## Outputs

- **Standard mode**: `{target_column}_forecast` containing the requested `forecast_horizon` predictions with timestamps inferred from the input frequency.
- **DARR mode**: Returns hybrid predictions in `{target_column}_forecast` (direct and kNN components are computed internally).
- **Interpretability mode**: Returns the explanation-aligned forecast in `{target_column}_forecast` and writes the artifact bundle to `<interpretability_out_dir>/run_<UTC>/`.

With `return_all_channels=True`, the single `{target_column}_forecast` column is replaced by one `{column}_forecast` column per processed channel (target column first, then the remaining numeric features in input-column order). Each forward pass already predicts every channel, so this avoids re-running the SDK once per column when downstream code needs multivariate forecasts. Autoregressive extension (`forecast_horizon > model_horizon`) and DARR blending apply to every channel the same way they apply to the target column; in DARR mode with mismatched input/context columns, only the aligned common columns are forecast (see Column Mismatch Handling).

> **Channel order matters.** The checkpoint and standardizer are trained against a specific channel layout. At inference time the numeric columns of the input DataFrame should match the training-time set and order. With the same columns in a different order the model still runs, but the forecasts can be semantically wrong — especially visible with `return_all_channels=True`, where every channel becomes an output column. With extra or missing numeric columns the channel count no longer matches the standardizer, and the call fails at standardization with a broadcasting error before inference runs.

### Output DataFrame structure

| Column | Description |
|--------|-------------|
| `{timestamp_column}` | Forecasted timestamps starting after the last input timestamp |
| `{target_column}_forecast` | Predicted values for the forecast horizon |
| `{column}_forecast` | (Only with `return_all_channels=True`) Predicted values for each additional numeric feature channel |

### Interpretability artifact bundle

When `interpretability=True`, the run directory contains the following files (selected by `interpretability_output`):

| File | Written when | Contents |
|------|--------------|----------|
| `forecast.csv` | json, pdf, both | The returned forecast DataFrame |
| `explanation.json` | json, both | Forecast + full explanation payload, including lag×horizon values, semantic-flow diagnostics, trajectory stability, feature-axis results for multivariate input, optional Integrated Gradients, and dataset metadata |
| `lag_horizon_attributions.csv` | pdf, both | Wide K×H attribution matrix |
| `lag_horizon_long.csv` | pdf, both | Tidy `(lag, horizon, attribution[, score])` table |
| `lag_horizon_heatmap.png` | pdf, both *(needs matplotlib)* | Visual heatmap, viridis cmap |
| `semantic_flow.csv` | pdf, both | Tidy `(transition_index, segment, flow_magnitude)` table, where `segment` is `history` for transitions fully inside the input window, `forecast` for transitions whose window extends into the model-generated future, and `tail` for any trailing transitions outside both segments |
| `channel_horizon_attributions.csv` | multivariate pdf, both | Feature-axis `C×H` attribution matrix |
| `channel_horizon_heatmap.png` | multivariate pdf, both *(needs matplotlib)* | Feature-axis channel-by-horizon heatmap |
| `integrated_gradients_attributions.csv` | `integrated_gradients=True` with pdf or both | Signed embedding IG value for every channel and context position |
| `integrated_gradients_channel_summary.csv` | `integrated_gradients=True` with pdf or both | Signed effect, absolute effect, and absolute share by channel |
| `integrated_gradients_heatmap.png` | `integrated_gradients=True` with pdf or both *(needs matplotlib)* | Signed channel-by-context IG heatmap |
| `explanation_report.pdf` | pdf, both *(needs matplotlib)* | Multi-page report covering forecast, lag×horizon attribution, semantic flow, latent stability, feature-axis attribution for multivariate input, and optional embedding Integrated Gradients |

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
  Common columns (input order): ['HULL', 'MULL', 'LUFL']
  Using only common columns for consistent predictions: ['HULL', 'MULL', 'LUFL']
```

**Behavior**:
- **Automatic detection**: Identifies when datasets have different feature sets
- **Order-sensitive alignment**: Realignment also triggers when both datasets contain the same columns in a different order — otherwise the direct and kNN predictions would blend mismatched channels
- **One canonical order**: Both temporary datasets are rewritten with the common feature columns in input-DataFrame order (target column first), so the caller-facing channel layout is preserved for the channels that remain
- **Clear warnings**: Reports what columns are being used and why
- **Graceful fallback**: Prevents cryptic NumPy broadcasting errors

### Common Error Scenarios

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: No common numeric columns found between input and context datasets` | Datasets share no feature columns (only target) | Ensure context dataset has at least one feature column in common with input |
| `ValueError: Shape mismatch between direct and kNN predictions` | Input and context columns could not be aligned | Check that both datasets have valid numeric columns |
| `ValueError: DataFrame has X rows but seq_len requires at least Y rows` | Insufficient data points | Provide more data or reduce `seq_len` |
| `ValueError: Context DataFrame has X rows but requires at least Y rows` | Context dataset too small | Context needs `seq_len + max(model_horizon, forecast_horizon)` rows minimum |
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

See `sdk/tests/test_forecasting.py` for unit test coverage with mockers and `sdk/quick_example.py` for an end-to-end script.
