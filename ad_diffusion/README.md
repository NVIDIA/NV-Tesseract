# AD Diffusion

A package for anomaly detection using NV-Tesseract diffusion models.

## Features

- **Diffusion-based Anomaly Detection**: Uses advanced diffusion models for robust anomaly detection
- **Adaptive Thresholding**: Implements SCS (Segmented Confidence Sequences) and MACS (Multi-Scale Adaptive Confidence Segments) methods
- **Multi-GPU Support**: Automatic multi-GPU inference with shared memory optimization
- **Fast Inference**: Supports DPM-Solver for 50-100x speedup over standard diffusion
- **Preprocessing Pipeline**: Complete TSB-AD compatible preprocessing with domain adaptation
- **Auto-download from Hugging Face**: Pretrained weights are fetched automatically from [`nvidia/nv-tesseract-ad-diffusion`](https://huggingface.co/nvidia/nv-tesseract-ad-diffusion) on first use
- **Simple Structure**: Organized as modules without Python package complexity - easy to use and modify

## Quick Start

**Requirements**: Python 3.12+ and uv package manager

First, set up the environment:
```bash
# Verify Python version
python --version  # Should be 3.12.0+

# Install and sync dependencies
uv sync

# Or run directly without activation
uv run python your_script.py
```

Then use the main function:
```python
import pandas as pd
import sys, os
sys.path.append('/path/to/ad_diffusion')  # Adjust to your installation path
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

# Load your data
df = pd.read_csv("your_data.csv")

# Perform anomaly detection — omit model_path/config_path to auto-download
# the pretrained weights from Hugging Face (nvidia/nv-tesseract-ad-diffusion).
results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",  # or "macs"
    # model_path="path/to/your/model.pth",    # optional; defaults to final_model.pth
    # config_path="path/to/config.yaml",      # optional; defaults to curriculum_medium.yaml
    nsample=15,
    preprocess_model_dir="path/to/preprocessing/models",  # optional
)

# Results contain original data plus anomaly detection results
print(f"Detected {results['Anomaly'].sum()} anomalies")
```

## Pretrained Weights

The pretrained Tesseract AD Diffusion model is hosted on Hugging Face:

- **Repository**: [`nvidia/nv-tesseract-ad-diffusion`](https://huggingface.co/nvidia/nv-tesseract-ad-diffusion)
- **Checkpoint**: `final_model.pth`
- **Config**: `curriculum_medium.yaml`

### Auto-download (recommended)

The SDK downloads these two files into the current working directory on first use,
so the simplest path is to just call `perform_anomaly_analysis_with_diffusion`
(or `inference_ad_tesseract2` / `inference_ad_tesseract2_mp`) **without** passing
`model_path`/`config_path`. Subsequent runs will reuse the local copies.

You can also pre-download them explicitly:

```python
from sdk.inference_ad import download_model_weights

model_path, config_path = download_model_weights(
    model_path="weights/final_model.pth",          # optional custom location
    config_path="weights/curriculum_medium.yaml",  # optional custom location
)
```

Or via the example CLI:

```bash
uv run python examples/quick_example.py --download-weights
```

### Hugging Face authentication

Weights are published on the public [`nvidia/nv-tesseract-ad-diffusion`](https://huggingface.co/nvidia/nv-tesseract-ad-diffusion) repository and download without authentication. If a download fails with `401`/`403`, accept the model license on the repo page or authenticate before retrying:

```bash
# Option A: interactive login (writes ~/.cache/huggingface/token)
uv add "huggingface_hub[cli]"
huggingface-cli login

# Option B: non-interactive via env var (great for CI and containers)
export HUGGINGFACE_HUB_TOKEN="hf_xxx_your_token_here"
```

You can create a token at <https://huggingface.co/settings/tokens>. A read-only
token is sufficient for downloading weights.

If you see a `401`/`403` error from `download_model_weights`, accept the model
license on the repo page and, if needed, log in with a Hugging Face token:

## Installation

### Prerequisites
- Python 3.12 or higher
- `uv` package manager (install from https://docs.astral.sh/uv/)

### From Source
```bash
# Clone or download the ad_diffusion directory
cd ad_diffusion

# Install dependencies
uv sync

# Run example
uv run python examples/quick_example.py
```

### Development Setup
```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .
```

## Package Structure

```
ad_diffusion/
├── sdk/                        # Main inference functions
│   ├── anomaly_analysis.py     # Main API function
│   ├── inference_ad.py         # Core inference engine
│   ├── inference_worker.py     # Multi-GPU worker
│   └── thresholds.py          # Threshold strategies (SCS, MACS)
├── models/                     # Diffusion model implementations
│   ├── main_model.py          # TSDiffuser_Generic model
│   ├── diff_models.py         # Diffusion model components
│   └── utils.py               # Model evaluation utilities
├── utils/                      # Utility functions and tools
│   ├── tsb_ad_preprocessor.py # Data preprocessing
│   ├── json_utils.py          # Model loading/saving
│   ├── adaptive_threshold.py  # SCS and MACS implementations
│   └── dpm_solver_pytorch.py  # DPM-Solver for fast inference
├── examples/                   # Examples and sample data
│   ├── quick_example.py       # Complete usage example
│   ├── finetune_example.py    # CSV fine-tuning example
│   └── datasets/              # Sample datasets for testing
├── README.md                  # This file
├── pyproject.toml            # Project configuration
└── uv.lock                   # Dependency lock file
```

## Dataset Usage

Supports both synthetic datasets (auto-generated with ground truth) and custom CSV files. Use `--dataset-path` to specify your own data.

📂 **See [examples/datasets/README.md](examples/datasets/README.md) for detailed dataset usage, format requirements, and examples.**

## Quick Start Example

Run the included example to get started:

```bash
cd ad_diffusion
uv run python examples/quick_example.py --help

# Auto-downloads final_model.pth + curriculum_medium.yaml from Hugging Face
# on first run, then performs anomaly detection on synthetic data.
uv run python examples/quick_example.py

# Use your own dataset for anomaly detection:
uv run python examples/quick_example.py --dataset-path /path/to/your/data.csv

# Pre-download the weights only (e.g. to warm the cache):
uv run python examples/quick_example.py --download-weights

# Use a checkpoint you already have locally:
uv run python examples/quick_example.py --model-path /path/to/final_model.pth

# Combine custom dataset with custom model:
uv run python examples/quick_example.py --dataset-path data.csv --model-path model.pth
```

This example demonstrates:
- Auto-downloading pretrained weights from Hugging Face (`nvidia/nv-tesseract-ad-diffusion`)
- Loading time series data (synthetic generation or custom CSV files)
- Running anomaly detection with diffusion models
- Applying adaptive thresholding (SCS/MACS)
- Using both synthetic and real-world datasets
- Evaluating results against ground truth

## Fine-tuning

Use `examples/finetune_example.py` to fine-tune AD Diffusion on normal windows from your own data. The CSV should contain mostly normal behavior. Numeric feature columns are used for training; use `--timestamp-col`, `--label-col`, and `--drop-cols` to remove metadata columns from the feature matrix.

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/normal_training_data.csv \
  --timestamp-col timestamp \
  --label-col is_anomaly \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-5 \
  --output-dir artifacts/finetune_my_data
```

When using the default temporal validation split, the validation partition must contain at least `--window-length` rows. The default `--val-ratio 0.3` works with the bundled 500-row sample dataset and default `--window-length 100`. If you lower `--val-ratio`, either increase the dataset size, pass a separate `--val-csv`, or reduce `--window-length`.

To use a separate validation file instead of a temporal split:

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/train_normal.csv \
  --val-csv /path/to/val_normal.csv \
  --timestamp-col timestamp \
  --epochs 10 \
  --output-dir artifacts/finetune_my_data
```

The output directory contains:

- `best_finetuned_model.pth` - best validation checkpoint, compatible with the SDK inference loader
- `final_finetuned_model.pth` - final epoch checkpoint
- `metrics.json` - training and validation losses
- `finetune_config.yaml` - model configuration used for fine-tuning

Run anomaly detection with the fine-tuned checkpoint. Pass only numeric analysis columns to `df`; drop timestamp, label, and other metadata columns before calling the SDK.

```python
analysis_df = df.select_dtypes(include="number").drop(columns=["is_anomaly"], errors="ignore")

results = perform_anomaly_analysis_with_diffusion(
    df=analysis_df,
    threshold_strategy="scs",
    model_path="artifacts/finetune_my_data/best_finetuned_model.pth",
    config_path="artifacts/finetune_my_data/finetune_config.yaml",
    nsample=15,
)
```

## System Requirements

### Hardware
- **CPU**: Any modern multi-core processor
- **Memory**: 8GB+ RAM recommended (16GB+ for large datasets)
- **GPU**: Optional but recommended (NVIDIA with CUDA support)

### Software
- **Python**: 3.12 or higher (required for modern typing features)
- **Operating System**: Linux, macOS, or Windows

## Python Dependencies

Core dependencies (see `pyproject.toml` for exact versions):
- `torch>=2.7.0` - Deep learning framework (with CUDA support if GPU available)
- NumPy, Pandas, SciPy
- scikit-learn (for PCA and normalization)
- PyYAML (for configuration files)
- tqdm (for progress bars)

Development dependencies:
- pytest (testing)
- ruff (linting and formatting)

All dependencies are managed automatically by `uv`.

## Usage

### Basic Usage

The main entry point is the `perform_anomaly_analysis_with_diffusion` function:

```python
import sys, os
sys.path.append('/path/to/ad_diffusion')
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion
```

### Parameters

- **df** (pd.DataFrame): Input time series data (all columns must be numeric)
- **threshold_strategy** (str): "scs" or "macs" for adaptive thresholding
- **model_path** (str|Path, optional): Path to the NV-Tesseract AD diffusion model checkpoint.
  If omitted or the file doesn't exist, `final_model.pth` is auto-downloaded from
  `nvidia/nv-tesseract-ad-diffusion` on Hugging Face.
- **config_path** (str|Path, optional): Path to the model config YAML. If omitted
  or missing, `curriculum_medium.yaml` is auto-downloaded from the same repo.
- **nsample** (int): Number of diffusion samples (default: 15)
- **preprocess_model_dir** (str|Path): Optional preprocessing model directory

### Advanced Usage

For direct access to inference functions:

```python
import sys, os
sys.path.append('/path/to/ad_diffusion')
from sdk.inference_ad import inference_ad_tesseract2, inference_ad_tesseract2_mp

# Single GPU inference (auto-downloads weights if needed)
results = inference_ad_tesseract2(
    data=df,
    # model_path omitted -> final_model.pth auto-downloaded
    nsample=30,
    use_dpm_solver=True,   # For faster inference
    dpm_steps=20,
)

# Multi-GPU inference (if available)
results = inference_ad_tesseract2_mp(
    data=df,
    model_path="final_model.pth",  # use a specific local checkpoint if desired
    nsample=30,
)
```

## Running

### With uv (Recommended)
```bash
# Run any script with dependencies
uv run python examples/quick_example.py

# Run with custom arguments
uv run python examples/quick_example.py --model-path /path/to/model.pth

# Use your own dataset
uv run python examples/quick_example.py --dataset-path /path/to/your/data.csv
```

### Manual Python
```bash
# Activate virtual environment first
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows

# Then run normally
python examples/quick_example.py
```


## Performance Notes

- **DPM-Solver**: Use `use_dpm_solver=True` for 50-100x inference speedup
- **Multi-GPU**: Automatically detected and used when available
- **Memory**: Large datasets are processed in chunks to manage memory usage
- **Preprocessing**: Models are cached to avoid repeated loading

## Troubleshooting

### Import Issues
If you get import errors, ensure the path is added correctly:
```python
import sys, os
# Adjust this path to where you installed ad_diffusion
sys.path.append('/full/path/to/ad_diffusion')
```

### CUDA Issues
For GPU-related problems:
```bash
# Check PyTorch CUDA installation
uv run python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode if needed
export CUDA_VISIBLE_DEVICES=""
```

### Memory Issues
For large datasets:
- Reduce `nsample` parameter (try 5-10 instead of 15)
- Process data in smaller chunks
- Use DPM-Solver for reduced memory usage

### Hugging Face Download Issues
If `download_model_weights` raises a `401` / `403` error:

```bash
# Accept the model license on the repo page, then log in if needed
uv add "huggingface_hub[cli]"
huggingface-cli login

# Or set the token in the current shell
export HUGGINGFACE_HUB_TOKEN="hf_xxx"
```

Then accept the model license on the repo page if prompted:
<https://huggingface.co/nvidia/nv-tesseract-ad-diffusion>

If downloads consistently fail with network errors, you can pre-fetch the files
manually with `huggingface-cli download nvidia/nv-tesseract-ad-diffusion
final_model.pth curriculum_medium.yaml --local-dir .` and then pass the local
paths explicitly via `model_path=`/`config_path=`.

### Dependencies
If you encounter dependency conflicts:
```bash
# Clear cache and reinstall
uv cache clean
uv sync --reinstall
```
