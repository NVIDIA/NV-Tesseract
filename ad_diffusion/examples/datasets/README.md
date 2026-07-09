# Datasets Directory

This directory contains sample datasets and examples for AD Diffusion.

**Supports both synthetic datasets (auto-generated with ground truth) and custom CSV files.** Use `--dataset-path` to specify your own data.

## Files

- **`sample_timeseries.csv`**: Synthetic time series data for testing (auto-generated)
- **`sample_timeseries_labels.csv`**: Ground truth labels for evaluation (auto-generated)
- **`anomaly_results.csv`**: Results from running anomaly detection (auto-generated)
- **`test-dataset.csv`**: Example dataset for testing

## Usage

### Synthetic Dataset (Default)
Auto-generated datasets with ground truth labels are automatically created when you run `quick_example.py`:

```bash
# Create sample dataset only
uv run python quick_example.py --create-dataset-only

# Run full example; pretrained weights auto-download from Hugging Face
# (nvidia/nv-tesseract-ad-diffusion) if they aren't already in the CWD.
uv run python quick_example.py

# Or point at a checkpoint you already have locally
uv run python quick_example.py --model-path /path/to/final_model.pth
```

### Custom CSV Files
Use your own CSV data for real-world anomaly detection:

```bash
# Use your own dataset
uv run python examples/quick_example.py --dataset-path /path/to/your/data.csv
```

**Dataset Requirements:**
- CSV format with column headers
- Numeric columns for time series data
- Timestamp column (optional, will be auto-generated if missing)
- No specific naming conventions required

**Example CSV format:**
```csv
timestamp,sensor_1,sensor_2,sensor_3
2024-01-01 00:00:00,1.23,4.56,7.89
2024-01-01 01:00:00,1.45,4.32,8.01
...
```

**Note:** Custom datasets won't have ground truth labels, so evaluation metrics (precision, recall, F1-score) will be skipped. Only anomaly detection results and scores will be provided.

**Additional Usage Examples:**
```bash
# Or from the examples directory
cd ..
uv run python quick_example.py --dataset-path /absolute/path/to/your/data.csv

# Combine with custom model
uv run python quick_example.py --dataset-path data.csv --model-path model.pth
```

Weights download automatically from the public Hugging Face repository. If a download fails with `401`/`403`, authenticate first:

```bash
uv add "huggingface_hub[cli]"
huggingface-cli login              # interactive
# or
export HUGGINGFACE_HUB_TOKEN="hf_xxx"
```

## Dataset Structure

The synthetic dataset includes:
- **3-5 sensor readings** (numeric time series)
- **~500-1000 samples** with hourly timestamps
- **5-8% anomaly rate** with various anomaly types:
  - Spikes (sudden increases)
  - Dips (sudden decreases)
  - Level shifts (sustained changes)
  - Noise bursts (high variance periods)

## Using Your Own Data

### Method 1: Command Line (Recommended)
Use the `--dataset-path` argument with the quick example:

```bash
uv run python ../quick_example.py --dataset-path /path/to/your/data.csv
```

**Format requirements**:
- CSV format with column headers
- Numeric columns for time series data  
- Timestamp column (optional, auto-generated if missing)
- Recommended: 100+ samples for reliable detection
- No missing values in numeric columns
- No specific naming conventions required

**Example CSV format**:
```csv
timestamp,sensor_1,sensor_2,sensor_3
2024-01-01 00:00:00,1.23,4.56,7.89
2024-01-01 01:00:00,1.45,4.32,8.01
...
```

## Synthetic Dataset Features

The auto-generated synthetic dataset includes:
- **Multi-sensor time series** with realistic patterns (sine waves, trends, noise)
- **Various anomaly types**:
  - **Spikes**: Sudden sharp increases
  - **Dips**: Sudden sharp decreases  
  - **Level shifts**: Sustained changes in baseline
  - **Noise bursts**: High variance periods
- **Ground truth labels** for evaluation metrics (precision, recall, F1-score)
- **Configurable parameters**: Sample count, sensor count, anomaly rate
- **Automatic saving** to `sample_timeseries.csv` and `sample_timeseries_labels.csv`

### Method 2: Direct API Call
Load and analyze programmatically:

```python
import pandas as pd
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

df = pd.read_csv("your_data.csv")

# Omit model_path to auto-download the pretrained weights from
# Hugging Face (nvidia/nv-tesseract-ad-diffusion) on first run.
results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",
)
```

### Model Compatibility
- Ensure your model is compatible with your data dimensions
- Use `get_model_target_dim()` to check model requirements  
- Consider preprocessing if needed

### Notes for Custom Datasets
- **No ground truth**: Custom datasets don't include anomaly labels
- **Evaluation metrics**: Precision, recall, and F1-score won't be calculated
- **Results**: You'll get anomaly scores (MAE) and binary anomaly flags
- **Output**: Results are saved to `anomaly_results.csv`
