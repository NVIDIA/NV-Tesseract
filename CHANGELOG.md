# Changelog

All notable changes to NV-Tesseract are documented in this file.

## v1.0.0 - 2026-09-21

Changes since v0.1.0.

### Added

- Forecasting output for all input channels through `ForecastingConfig(return_all_channels=True)`. ([#26](https://github.com/NVIDIA/NV-Tesseract/pull/26))
- Hugging Face Hub model wrappers for both forecasting and AD Diffusion, with `from_pretrained()` loading. ([#48](https://github.com/NVIDIA/NV-Tesseract/pull/48))
- TAO AutoML run-config support and scalar `metrics.json` outputs in both fine-tuning examples; per-epoch results are available in `epoch_metrics.json`. ([#54](https://github.com/NVIDIA/NV-Tesseract/pull/54), [#59](https://github.com/NVIDIA/NV-Tesseract/pull/59))
- Automatic multi-GPU distributed fine-tuning in both examples, with `--num-gpus` to limit GPU use. ([#61](https://github.com/NVIDIA/NV-Tesseract/pull/61))
- Forecasting feature-axis attribution, embedding Integrated Gradients, and embedding stability reporting, with JSON, CSV, and PDF artifacts. ([#56](https://github.com/NVIDIA/NV-Tesseract/pull/56), [#62](https://github.com/NVIDIA/NV-Tesseract/pull/62))
- Multi-GPU execution for DARR context embedding and channel-flow Jacobian analysis. ([#67](https://github.com/NVIDIA/NV-Tesseract/pull/67))
- Optional AD Diffusion PDF reports with signal plots, anomaly scores, and ground truth, configured through `ADDiffusionConfig` or `sdk_config.yaml`. ([#76](https://github.com/NVIDIA/NV-Tesseract/pull/76))
- Automatic MPS device detection for AD inference on Apple Silicon. ([#55](https://github.com/NVIDIA/NV-Tesseract/pull/55))

### Changed

- **Breaking:** `perform_forecasting()` now accepts `config=ForecastingConfig(...)` or a YAML path in place of flat inference arguments. Keep `df` and optional `context_df` as direct inputs; move settings such as `target_column`, `forecast_horizon`, and `ckpt` into the config. See the [forecasting SDK guide](forecasting/sdk/README.md) and [YAML template](forecasting/sdk/forecasting_inference_config.yaml). ([#74](https://github.com/NVIDIA/NV-Tesseract/pull/74))
- **Breaking:** AD analysis renames `config_path` to `model_config_path`. Reporting options belong in `sdk_config=ADDiffusionConfig(...)` or a YAML file. See the [AD Diffusion guide](ad_diffusion/README.md). ([#76](https://github.com/NVIDIA/NV-Tesseract/pull/76))
- Simplified forecasting interpretability around batched Jacobian analysis and embedding Integrated Gradients; removed Shapley, coupling, and legacy sharded paths, and corrected semantic-flow report layout. ([#60](https://github.com/NVIDIA/NV-Tesseract/pull/60))
- Contributions target `dev`; `main` is reserved for release-ready code. ([#65](https://github.com/NVIDIA/NV-Tesseract/pull/65))

### Fixed

- Isolated temporary forecasting CSVs per call to prevent collisions between concurrent requests. ([#34](https://github.com/NVIDIA/NV-Tesseract/pull/34))
- Aligned AD inference normalization with training and returned reconstructions in the same scale as targets. ([#41](https://github.com/NVIDIA/NV-Tesseract/pull/41))
- Restored AD preprocessing artifacts with the reusable `AdaptiveNormalizer`, preserving fitted transforms through JSON save/load; added round-trip coverage for normalizers, scalers, transformers, and PCA. ([#71](https://github.com/NVIDIA/NV-Tesseract/pull/71))
- Kept shared memory open until AD worker inference completes. ([#47](https://github.com/NVIDIA/NV-Tesseract/pull/47))
- Aligned DARR input and context channel ordering when the same columns arrive in different orders. ([#52](https://github.com/NVIDIA/NV-Tesseract/pull/52))
- Corrected base/cross-channel checkpoint selection and inferred cross-channel mode from checkpoint weights. ([#73](https://github.com/NVIDIA/NV-Tesseract/pull/73))
- Excluded SDK-added padding from AD MAE, L2, and thresholding, while retaining full-width target and reconstruction outputs. Low-level results include `valid_feature_mask` and `score_feature_count`. Scores may differ for padded inputs. ([#79](https://github.com/NVIDIA/NV-Tesseract/pull/79))
- Verify downloaded model weight files before reporting a successful download in both SDKs. ([#82](https://github.com/NVIDIA/NV-Tesseract/pull/82))
- Clarified that AD `Anomaly` output uses int64 values (`0` = normal, `1` = anomaly); detection behavior is unchanged.
- Synchronized forecasting lockfile metadata with its dependency declarations without changing locked package versions.

### Notes

- Repository release version: `v1.0.0`.
- Package metadata versions: `tesseract_forecasting` 1.0.0 and `ad-diffusion-oss` 1.0.0.

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
