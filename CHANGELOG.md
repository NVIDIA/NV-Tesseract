# Changelog

All notable changes to NV-Tesseract are documented in this file.

## v0.1.0 - 2026-07-07

First public release of NV-Tesseract.

### Added

- Forecasting package with DataFrame-first inference, DARR context-enhanced forecasting, cross-channel forecasting support, and Hugging Face weight loading.
- Forecasting interpretability framework with input attributions, semantic-flow diagnostics, forecast-vs-history ratios, trajectory stability metrics, and optional PDF/JSON artifacts.
- AD Diffusion package for multivariate anomaly detection with SCS and MACS adaptive thresholding, DPM-Solver inference, and Hugging Face weight loading.
- Fine-tuning examples for both forecasting and AD Diffusion.
- Lightweight CI covering linting, SPDX checks, forecasting tests, AD Diffusion tests, and example tests.
- Public documentation for installation, examples, dataset expectations, model assets, contribution flow, security reporting, and third-party notices.

### Changed

- Raised the PyTorch dependency floor to `torch>=2.7.0` for Blackwell GPU support.
- Removed unused forecasting audio/vision PyTorch dependencies and the legacy `mac-mps` extra.
- Switched user-facing progress output from direct `print` calls to logging where appropriate.
- Updated documentation to reference public Hugging Face model repositories.

### Fixed

- Fixed DARR retrieval behavior when forecast horizons exceed the model horizon.
- Fixed AD Diffusion complementary mask aggregation so each target mask selects reconstructions from its own strategy.
- Fixed AD thresholding, packaging, and README examples.
- Fixed pandas frequency deprecation warnings in README examples.

### Notes

- Repository release version: `v0.1.0`.
- Package metadata versions at this release: `forecasting` is `0.1.0`; `ad-diffusion-oss` is `1.0.0`.
