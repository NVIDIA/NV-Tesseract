# NV-Tesseract

NVIDIA Tesseract is an open-source time series analysis library covering forecasting and anomaly detection. The forecasting module builds on a pretrained transformer backbone; anomaly detection uses diffusion-based models powered by NVIDIA's proprietary algorithms.

## Overview

- **Forecasting**: DataFrame-first API for multivariate time series forecasting with DARR (context-enhanced) mode, built on a vendored backbone.
- **Anomaly Detection**: Diffusion-based multivariate anomaly detection using novel proprietary algorithms.

## Getting Started

### Installation

Clone the repo and install the desired package:

#### Forecasting
```bash
git clone https://github.com/NVIDIA/NV-Tesseract.git
cd NV-Tesseract/forecasting
uv sync --python 3.12   # or: pip install -e .
```

#### Anomaly Detection
```bash
git clone https://github.com/NVIDIA/NV-Tesseract.git
cd NV-Tesseract/ad_diffusion
uv sync --python 3.12   # or: pip install -e .
```

Use the same interpreter/venv when you run the examples below.

### Quick Start

#### Forecasting
```python
from sdk.forecasting import perform_forecasting
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=600, freq="h"),
    "target": np.sin(np.linspace(0, 4 * np.pi, 600)),
    "feature_a": np.random.randn(600),
})

forecasts = perform_forecasting(
    df=df,
    seq_len=512,
    forecast_horizon=72,
)
# Returns a DataFrame with `target_forecast` column containing 72 predictions
```

#### DARR (Context-Enhanced) Forecasting

```python
darr_result = perform_forecasting(
    df=df,
    context_df=historical_df,  # Historical data for kNN retrieval
    seq_len=512,
    forecast_horizon=72,
    alpha=0.2,   # 20% direct, 80% kNN
    k=64,
    temperature=0.05,
)
```

#### Interpretability

Forecasting includes a model-agnostic **interpretability** framework that explains *why* a forecast looks the way it does — without modifying the underlying model. Pass `interpretability=True` to write an explanation bundle alongside the forecast: lag and feature attributions per horizon, semantic-flow diagnostics, trajectory stability, and optional embedding Integrated Gradients (JSON, CSVs, and PDF report):

```python
results = perform_forecasting(
    df=df,
    seq_len=512,
    forecast_horizon=72,
    interpretability=True,
    interpretability_output=None,  # "json", "pdf", or None for both
    interpretability_out_dir="interpretability_output",
)
# Bundle written under interpretability_output/run_<UTC-timestamp>/
```

See [`forecasting/README.md`](forecasting/README.md#interpretability) for the full interpretability reference and artifact catalogue.

#### Anomaly Detection
```python
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion
import pandas as pd

df = pd.read_csv("your_timeseries_data.csv")

results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",  # or "macs"
    nsample=15,
)
# Returns DataFrame with anomaly scores and binary anomaly flags
```

## Requirements

- Python 3.12+
- PyTorch 2.7+
- pandas, numpy
- Pretrained model weights (auto-downloaded from Hugging Face)
- GPU recommended (CUDA or Apple MPS); falls back to CPU automatically

## Usage

### Forecasting
- See [`forecasting/README.md`](forecasting/README.md) for full API reference and examples
- Run [`forecasting/sdk/quick_example.py`](forecasting/sdk/quick_example.py) for an end-to-end example
- Fine-tune on your own CSV with [`forecasting/examples/finetune_example.py`](forecasting/examples/finetune_example.py)

### Anomaly Detection
- See [`ad_diffusion/README.md`](ad_diffusion/README.md) for detailed usage and configuration
- Run [`ad_diffusion/examples/quick_example.py`](ad_diffusion/examples/quick_example.py) for an end-to-end example with synthetic or custom datasets
- Fine-tune on normal windows from your own CSV with [`ad_diffusion/examples/finetune_example.py`](ad_diffusion/examples/finetune_example.py)

## Capabilities

| Module | Status | Description |
|--------|--------|-------------|
| `forecasting/` | ✅ Available | Time series forecasting with DARR (context-enhanced) mode |
| `ad_diffusion/` | ✅ Available | Diffusion-based multivariate anomaly detection with adaptive thresholding |

## Repository Structure

```
NV-Tesseract/
├── .github/
│   └── workflows/
│       └── ci.yml               # CI pipeline
├── scripts/
│   └── add_spdx_headers.py     # SPDX license header tooling
├── third_party/                 # Upstream LICENSE files for vendored/in-tree third-party code
│   ├── README.md
│   └── dpm-solver/
├── forecasting/                 # Time series forecasting
│   ├── pyproject.toml           # Project configuration
│   ├── README.md                # Forecasting documentation
│   ├── backbone.py              # Vendored transformer backbone
│   ├── model.py                 # Model construction utilities
│   ├── dataset_longhorizon.py   # Dataset classes for long-horizon forecasting
│   ├── interpretability.py      # Model-agnostic forecast explanation engine
│   ├── channel_flow.py          # Batched feature-axis Jacobian attribution
│   ├── integrated_gradients.py  # Embedding Integrated Gradients
│   ├── examples/
│   │   ├── finetune_example.py  # CSV fine-tuning example
│   │   └── tests/               # Fine-tuning example tests
│   └── sdk/
│       ├── forecasting.py       # Core perform_forecasting() implementation
│       ├── quick_example.py     # End-to-end usage example
│       ├── README.md            # SDK parameter and artifact reference
│       └── tests/               # Test suite and sample datasets
├── ad_diffusion/                # Multivariate anomaly detection
│   ├── pyproject.toml           # Project configuration
│   ├── README.md                # AD diffusion documentation
│   ├── sdk/                     # Main inference functions
│   │   ├── anomaly_analysis.py  # Main API function
│   │   ├── inference_ad.py      # Core diffusion inference
│   │   ├── inference_worker.py  # Multi-GPU worker
│   │   ├── thresholds.py        # SCS/MACS adaptive thresholding
│   │   └── tests/               # SDK tests
│   ├── models/                  # Diffusion model implementations
│   │   ├── main_model.py
│   │   ├── diff_models.py
│   │   └── utils.py
│   ├── utils/                   # Preprocessing and utilities
│   │   ├── tsb_ad_preprocessor.py
│   │   ├── adaptive_threshold.py
│   │   ├── json_utils.py
│   │   └── dpm_solver_pytorch.py
│   └── examples/                # Usage examples and datasets
│       ├── quick_example.py     # Complete example (synthetic + custom data)
│       ├── finetune_example.py  # CSV fine-tuning example
│       ├── datasets/            # Sample datasets and documentation
│       └── tests/               # Example tests
└── Makefile                     # Linting and formatting commands
```

## Contribution Guidelines

- Start here: [`CONTRIBUTING.md`](CONTRIBUTING.md) — includes Developer Certificate of Origin (`Signed-off-by`) and IP-review expectations for NVIDIA contributors.
- Code of Conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## Security

- Vulnerability disclosure: `SECURITY.md`
- Do not file public issues for security reports.

## Support

- How to get help: [GitHub Issues](https://github.com/NVIDIA/NV-Tesseract/issues)

## License

This project is licensed under the Apache License, Version 2.0 — see [`LICENSE`](LICENSE). Third-party attribution required on distribution is summarized in [`NOTICE`](NOTICE); dependency summaries are in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Blogs

- [New NVIDIA NV-Tesseract Time Series Models Advance Dataset Processing and Anomaly Detection](https://developer.nvidia.com/blog/new-nvidia-nv-tesseract-time-series-models-advance-dataset-processing-and-anomaly-detection/)
- [Smarter Anomaly Detection in Semiconductor Manufacturing with NV-Tesseract and NVIDIA Inference Microservices](https://developer.nvidia.com/blog/smarter-anomaly-detection-in-semiconductor-manufacturing-with-nvidia-nv-tesseract-and-nvidia-nim/)
- [NV-Tesseract-AD: Diffusion-Based Anomaly Detection with Curriculum Learning Across Industries](https://developer.nvidia.com/blog/advancing-anomaly-detection-for-industry-applications-with-nvidia-nv-tesseract-ad/)

## References
- M. Ravikiran, A. Gautam, A. Chulani. "Beyond MAE: Measuring Forecast Reliability with Temporal Dependence-Aware Error (TDE)." *2025 IEEE International Conference on Big Data (BigData)*, pp. 7271–7277, 2025.
- A. Gautam, M. Ravikiran, F. S. Ekiz. "Memory-Augmented Forecasting: Scalability and Generalization Across Temporal Domains." *2025 IEEE International Conference on Big Data (BigData)*, pp. 7258–7265, 2025.
- M. A. Li, A. Gautam. "Segmented Confidence Sequences and Multi-Scale Adaptive Confidence Segments for Anomaly Detection in Nonstationary Time Series." *Proceedings of the 2025 5th International Conference on Artificial Intelligence and Application Technologies*, pp. 6–15, 2025.
