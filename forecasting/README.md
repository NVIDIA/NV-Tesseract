# NV-Tesseract Forecasting

Tesseract Forecasting Model that learns universal representations from diverse temporal data using self-supervised pretraining for forecasting.

## Quick Start with UV

### 1. Install UV (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

### 2. Set Up the Project

```bash
cd forecasting/
uv sync --group dev
uv pip install -e .  # Install package in editable mode for clean imports
```

### 3. Run Forecasting

The model weights are published on Hugging Face (`nvidia/nv-tesseract-forecasting`) and download automatically on first use — no authentication required:

```python
import pandas as pd
from sdk.forecasting import perform_forecasting, DEVICE

# Check device being used
print(f"Using device: {DEVICE}")

# Load your data
df = pd.read_csv("your_data.csv")  # Must have 'timestamp' and 'target' columns

# Perform forecasting - weights auto-downloaded if needed
results = perform_forecasting(
    df=df,
    timestamp_column="timestamp",
    target_column="target",
    forecast_horizon=72,  # Forecast 72 steps ahead
)

print(results)
```

Or run the example:

```bash
# Full example with model download (requires test data)
uv run python sdk/quick_example.py
```

## Fine-tuning

Use `examples/finetune_example.py` to fine-tune NV-Tesseract on your own forecasting data. The CSV must include a timestamp column and one or more numeric target columns. By default, the script auto-downloads the published NV-Tesseract forecasting checkpoint, freezes the pretrained encoder/embedder, and trains the forecasting head. Pass `--ckpt-init none` if you want to train a fresh head from the base backbone instead.

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/your_timeseries.csv \
  --timestamp-col timestamp \
  --target-cols target \
  --seq-len 512 \
  --forecast-horizon 72 \
  --epochs 5 \
  --output-dir artifacts/finetune_my_data
```

For multivariate channel-mixing fine-tuning, enable the cross-channel layer. With `--ckpt-init auto` (the default), the script warm-starts from the published cross-channel checkpoint:

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/multivariate_timeseries.csv \
  --timestamp-col timestamp \
  --target-cols sensor_1,sensor_2,sensor_3 \
  --seq-len 512 \
  --forecast-horizon 72 \
  --use-cross-channel \
  --cross-channel-heads 8 \
  --cross-channel-dropout 0.1 \
  --epochs 5 \
  --output-dir artifacts/finetune_channel_mixing
```

The output directory contains:

- `best_model.pt` - fine-tuned checkpoint
- `standardizer.pkl` - training normalization statistics
- `finetune_metadata.json` - model, data, and training metadata
- `metrics.json` - training and validation metrics for all epochs

Use the fine-tuned artifacts with the SDK:

```python
results = perform_forecasting(
    df=df,
    timestamp_column="timestamp",
    target_column="target",
    seq_len=512,
    forecast_horizon=72,
    model_horizon=72,
    standardizer_pkl="artifacts/finetune_my_data/standardizer.pkl",
    ckpt="artifacts/finetune_my_data/best_model.pt",
    use_cross_channel=False,  # set True if trained with --use-cross-channel
)
```

## UV Commands Reference

### Package Management
```bash
uv add <package>           # Add dependency
uv add --dev <package>     # Add dev dependency
uv remove <package>        # Remove dependency
uv sync                    # Install all dependencies
uv sync --group dev        # Install with dev dependencies
```

### Running Code
```bash
uv run python script.py    # Run Python with project environment
uv run pytest             # Run tests
uv run ruff check         # Run linting
```

### Environment Management
```bash
uv venv                    # Create virtual environment
source .venv/bin/activate  # Activate environment (Unix)
.venv\Scripts\activate     # Activate environment (Windows)
```

## Project Structure

```
forecasting/
├── pyproject.toml         # Project configuration and dependencies
├── README.md             # This file
├── examples/
│   └── finetune_example.py # CSV fine-tuning example
├── sdk/
│   ├── forecasting.py    # Main forecasting module (with auto-download)
│   ├── quick_example.py  # Example usage script
│   └── tests/            # Test files and datasets
├── dataset_longhorizon.py # Dataset utilities
├── interpretability.py   # Model-agnostic explanation engine (lag x horizon + feature axis)
├── model.py              # Model building utilities
├── standardizer.pkl      # Normalization params (auto-downloaded on first use)
└── moment_head_512_6hr.pt  # Head checkpoint (auto-downloaded on first use)
```

## Dependencies

### Core Dependencies
- `datasetsforecast>=1.0.0` - Dataset utilities for forecasting
- `joblib>=1.5.2` - Serialization for model artifacts
- vendored `backbone.py` - forecasting backbone implementation bundled in this package
- `pandas>=2.1.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `torch>=2.7.0` - Deep learning framework
- `tqdm>=4.65.0` - Progress bars
- `huggingface_hub>=0.17.0` - For downloading model weights

### Development Dependencies
- `pytest>=9.0.2` - Testing framework
- `ruff>=0.8.0` - Linting and formatting

### Optional Dependencies
- `matplotlib` - Required only for the interpretability PDF report and heatmap PNG; JSON-only interpretability output works without it

### Note on Dependencies
- `backbone.py` is vendored directly in this package, so forecasting no longer depends on the external backbone package

## Configuration

The project uses `pyproject.toml` for configuration:

- **Build system**: Hatchling
- **Linting**: Ruff with comprehensive rules
- **Python version**: >=3.10
- **Package management**: UV

## Model Weights

The forecasting model requires pre-trained weights from the Hugging Face repository:
- Repository: [`nvidia/nv-tesseract-forecasting`](https://huggingface.co/nvidia/nv-tesseract-forecasting) (auto-downloaded on first use; no authentication required)
- Required files (auto-downloaded to current directory):
  - `standardizer.pkl` - Data normalization parameters
  - `moment_head_512_6hr.pt` - Model checkpoint for standard 6-hour forecasting (standard forecasting only)
  - `run8_best_model_cr.pt` - Model checkpoint for cross-channel forecasting (cross-channel forecasting only)

## Interpretability

Modern deep-learning forecasters predict but don't explain — they can't answer questions like *"Why did the model predict a spike tomorrow at 3pm?"* or *"Is it relying on a real pattern, or on noise?"*. Traditional tools like LIME and SHAP fall short for time-series because they destroy temporal continuity, lack forecast-horizon resolution, and cannot operate inside the latent space modern forecasters use.

NV-Tesseract ships a **Model Agnostic Interpretability Framework** that produces localized, horizon-specific, time-aware explanations without modifying the underlying forecaster. It targets real-world deployments — finance risk, energy grids, manufacturing — where a black-box prediction is not enough.

### The Lag–Horizon Attribution Engine

The framework's core component is the **Lag–Horizon Attribution Engine**, which turns the computed semantic flow (the step-by-step transformation of the model's internal state in latent space) into influence scores. Its output is the **Lag–Horizon Attribution Matrix** `F`:

- **Rows** are the lag `j` — how far back in the past an input occurred (e.g. 24 steps ago).
- **Columns** are the forecast horizon `h` — how far ahead the model is predicting (e.g. step +1, step +48).
- **Value** `F(j, h)` quantifies *how much the input at time `t − j` influenced the forecast at time `t + h`* — the horizon-specific, lag-specific attribution that flat-aggregate tools cannot provide.

Internally `F` is computed by composing the model's consecutive flow operators along the latent path from each past input to each future prediction; per horizon, the scores are softmax-normalized into attributions.

### The Feature Axis (channel × horizon)

For multivariate inputs the same latent semantic flow is decomposed along a second axis — the **feature axis**. The scalar flow magnitude of each latent transition (`|| Z_{t+1} − Z_t ||`) is split into per-channel contributions using a fast, model-agnostic first-order (Jacobian-flow) estimator. Aggregated across lags per horizon, this yields a **channel × horizon attribution matrix** answering *which input channel drives each forecast step* — the feature-axis complement to the lag × horizon (temporal-axis) matrix. 

By default the flow is measured in the model's latent space (output-agnostic "semantic flow"). Passing `channel_output_aware=True` to `perform_forecasting` instead measures it on the **target channel's forecast**, so the attribution answers *which input channel drives the target's prediction* directly — using only the black-box forecast interface (no embedding access). This is more interpretable for a single target at the cost of extra forward passes.

### What the SDK gives you

When you call `perform_forecasting(..., interpretability=True)` the SDK runs the same loaded model on the trailing window and writes a self-contained explanation bundle: the K×H matrix as wide and long CSVs, a heatmap PNG, a per-transition `semantic_flow.csv` with a `history`/`forecast` segment label, a full `explanation.json` (baseline forecast, lag×horizon scores and attributions, latent trajectory, semantic-flow magnitudes, diagnostic ratios that flag whether the forecast segment is volatile relative to history, and a `trajectory_stability` block with temporal-smoothness metrics over the context window), and a multi-page PDF report whose final pages surface (a) the semantic-flow time series with a history/forecast split chart, per-segment summary statistics, and the forecast-vs-history diagnostic ratios, and (b) the latent-trajectory stability metrics. For **multivariate inputs** the bundle additionally includes the feature-axis outputs: a `channel_horizon_attributions.csv` (and `channel_horizon_heatmap.png`), a `feature_axis` block in `explanation.json`, and a dedicated channel × horizon page in the PDF report. See the **Interpretability (Lag x Horizon Explanations)** example below for the call shape and `sdk/README.md` for the full parameter reference and on-disk artifact catalogue.

## Usage Examples

### Basic Forecasting
```python
from sdk.forecasting import perform_forecasting, DEVICE
import pandas as pd
import numpy as np

# Your data should have timestamp and target columns
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'target': np.random.randn(1000).cumsum()
})

# Perform forecasting - model weights auto-downloaded on first run
results = perform_forecasting(
    df=df, 
    forecast_horizon=24,
    timestamp_column="timestamp",
    target_column="target"
)
```

### Context-Enhanced Forecasting (DARR Mode)
```python
from sdk.forecasting import perform_forecasting

# With additional context data for improved accuracy
context_df = pd.read_csv("historical_data.csv")

results = perform_forecasting(
    df=df,
    context_df=context_df,  # Additional context for better predictions
    forecast_horizon=72,
    alpha=0.01,  # Blending parameter for DARR mode
    timestamp_column="timestamp",
    target_column="target"
)
```

### Interpretability (Lag x Horizon Explanations)
```python
from sdk.forecasting import perform_forecasting

# Set interpretability=True to also produce an explanation bundle on disk.
# `interpretability_output` selects what to materialize:
#   - "json" -> forecast.csv + explanation.json
#   - "pdf"  -> forecast.csv + lag_horizon_*.csv/png + explanation_report.pdf
#   - None   -> both bundles
results = perform_forecasting(
    df=df,
    forecast_horizon=72,
    timestamp_column="timestamp",
    target_column="target",
    interpretability=True,
    interpretability_output=None,                 # write both JSON and PDF
    interpretability_out_dir="interpretability_output",
    interpretability_dataset_name="my_dataset.csv",
    n_lags=128,
    softmax_tau=1.0,
)
# Bundle is written under <interpretability_out_dir>/run_<UTC-timestamp>/
```

### Manual Weight Paths (Optional)
```python
from sdk.forecasting import perform_forecasting

# If you want to specify custom paths for model weights
results = perform_forecasting(
    df=df,
    forecast_horizon=72,
    standardizer_pkl="custom/path/standardizer.pkl",  # Default: "standardizer.pkl"
    ckpt="custom/path/model_checkpoint.pt"            # Default: packaged forecast checkpoint path
)
```

## Development

### Running Tests
```bash
uv run pytest
```

### Linting
```bash
uv run ruff check
uv run ruff format
```

### Adding Dependencies
```bash
# Add runtime dependency
uv add new-package

# Add development dependency
uv add --dev pytest-cov
```

## Troubleshooting

### UV Installation Issues
If UV installation fails:
1. Try manual installation: https://docs.astral.sh/uv/getting-started/installation/
2. Use pip: `pip install uv`
3. Use pipx: `pipx install uv`

### Dependency Resolution Issues
If you see stale environment errors referring to an old backbone package:
- remove the old lockfile environment and reinstall; forecasting now vendors the backbone implementation locally
- The `pyproject.toml` is configured to handle this automatically
- Make sure `tool.hatch.metadata.allow-direct-references = true` is set

### Model Weight Download Issues
1. Verify the repository is reachable: `nvidia/nv-tesseract-forecasting`
2. Check network connectivity
3. If you see a `401`/`403` error, accept the model license on the Hugging Face repo page or authenticate: `huggingface-cli login`

### Import Errors
Make sure to run Python through UV to use the project environment:
```bash
uv run python your_script.py
```

### Interpretability PDF / Heatmap Skipped
If you see `Interpretability PDF report skipped: matplotlib is not installed.`, install matplotlib (or pick `interpretability_output="json"`):
```bash
uv add matplotlib
```
