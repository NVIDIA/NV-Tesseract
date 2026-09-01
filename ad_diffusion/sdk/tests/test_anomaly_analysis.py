# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdk import anomaly_analysis


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [2.5, 3.5, 4.5, 5.5, 6.5],
            "feature_3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


@pytest.fixture
def inference_results() -> dict[str, np.ndarray]:
    return {
        "residual": np.array([0.1, 0.5, 0.2, 0.8, 0.3]),
        "residual_l2": np.array([0.2, 0.6, 0.3, 0.9, 0.4]),
        "target": np.array(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
                [1.4, 2.4],
            ]
        ),
        "recon": np.array(
            [
                [0.9, 1.9],
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
            ]
        ),
        "target_dim": np.array(2),
    }


@pytest.fixture(autouse=True)
def patch_model_paths(monkeypatch):
    monkeypatch.setattr(
        anomaly_analysis, "_resolve_model_paths", lambda model_path, config_path: ("model.pth", "config.yaml")
    )
    monkeypatch.setattr(anomaly_analysis, "get_model_target_dim", lambda model_path, config_path: 2)


def test_perform_anomaly_analysis_with_scs_strategy(monkeypatch, numeric_df, inference_results):
    mock_inference = Mock(return_value=inference_results)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", mock_inference)

    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([False, True, False, True, False])
    mock_strategy = Mock()
    mock_strategy.scs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", Mock(return_value=mock_strategy))

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        numeric_df,
        threshold_strategy="scs",
        model_path="model.pth",
        model_config_path="config.yaml",
        nsample=7,
    )

    assert list(result.columns) == ["feature_1", "feature_2", "feature_3", "Anomaly", "MAE"]
    assert result["Anomaly"].tolist() == [False, True, False, True, False]
    assert np.array_equal(result["MAE"].to_numpy(), inference_results["residual"])
    _, kwargs = mock_inference.call_args
    assert kwargs["model_path"] == "model.pth"
    assert kwargs["config_path"] == "config.yaml"
    assert kwargs["nsample"] == 7
    detector_scores, detector_target = mock_thresholder.detect_anomalies.call_args.args
    np.testing.assert_array_equal(detector_scores, inference_results["residual"])
    np.testing.assert_array_equal(detector_target, inference_results["target"])


def test_perform_anomaly_analysis_rejects_non_numeric_columns(numeric_df):
    mixed_df = numeric_df.copy()
    mixed_df["machine_id"] = ["a", "b", "c", "d", "e"]

    with pytest.raises(ValueError, match="contain non-numeric values"):
        anomaly_analysis.perform_anomaly_analysis_with_diffusion(
            mixed_df,
            threshold_strategy="scs",
            model_path="model.pth",
            model_config_path="config.yaml",
        )


def test_perform_anomaly_analysis_rejects_insufficient_rows(monkeypatch):
    monkeypatch.setattr(anomaly_analysis, "get_model_target_dim", lambda model_path, config_path: 10)
    short_df = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Insufficient data samples"):
        anomaly_analysis.perform_anomaly_analysis_with_diffusion(
            short_df,
            threshold_strategy="scs",
            model_path="model.pth",
            model_config_path="config.yaml",
        )


def test_perform_anomaly_analysis_rejects_unknown_threshold(monkeypatch, numeric_df, inference_results):
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", Mock(return_value=inference_results))

    with pytest.raises(ValueError, match="Unknown threshold strategy"):
        anomaly_analysis.perform_anomaly_analysis_with_diffusion(
            numeric_df,
            threshold_strategy="not-a-strategy",
            model_path="model.pth",
            model_config_path="config.yaml",
        )


def test_perform_anomaly_analysis_truncates_long_model_outputs(monkeypatch, numeric_df, inference_results):
    long_results = dict(inference_results)
    long_results["residual"] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    long_results["target"] = np.array([[1.0, 2.0]] * 7)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", Mock(return_value=long_results))

    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([True] * 7)
    mock_strategy = Mock()
    mock_strategy.macs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "MACSThresholdStrategy", Mock(return_value=mock_strategy))

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        numeric_df,
        threshold_strategy="macs",
        model_path="model.pth",
        model_config_path="config.yaml",
    )

    assert len(result) == len(numeric_df)
    assert result["MAE"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_args, _ = mock_thresholder.detect_anomalies.call_args
    assert len(threshold_args[0]) == len(numeric_df)
    assert len(threshold_args[1]) == len(numeric_df)


def test_optional_pdf_report_excludes_metadata_from_inference(monkeypatch, inference_results, tmp_path):
    """Report metadata should be preserved in output but excluded from model features."""
    input_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="min"),
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [2.0, 2.5, 3.0, 3.5, 4.0],
            "GT": [0, 1, 0, 0, 1],
        }
    )
    destination = tmp_path / "anomaly_report.pdf"
    mock_inference = Mock(return_value=inference_results)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", mock_inference)

    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([False, True, False, True, False])
    mock_strategy = Mock()
    mock_strategy.scs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", Mock(return_value=mock_strategy))

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        input_df,
        threshold_strategy="scs",
        model_path="model.pth",
        model_config_path="config.yaml",
        sdk_config=anomaly_analysis.ADDiffusionConfig(report_path=destination),
    )

    inference_df = mock_inference.call_args.kwargs["data"]
    assert list(inference_df.columns) == ["feature_1", "feature_2"]
    assert {"timestamp", "GT", "Anomaly", "MAE"}.issubset(result.columns)
    assert destination.read_bytes().startswith(b"%PDF")


def test_explain_toggle_preserves_numeric_feature_names(monkeypatch, tmp_path):
    """The full analysis path should explain anomalies using the input signal names."""
    feature_names = [
        "x",
        "y",
        "z",
        "010-000-024-033",
        "010-000-030-096",
        "020-000-032-221",
        "020-000-033-111",
    ]
    input_df = pd.DataFrame(np.ones((5, len(feature_names))), columns=feature_names)
    input_df["anomaly"] = [0, 1, 0, 0, 1]
    target = np.zeros((5, 40))
    reconstruction = np.zeros_like(target)
    reconstruction[1, :7] = [0, 0, 0, 7, 6, 5, 4]
    reconstruction[1, 27] = 100
    reconstruction[4, :7] = [1, 2, 3, 4, 5, 6, 7]
    inference_results = {
        "residual": np.array([0.0, 122 / 40, 0.0, 0.0, 28 / 40]),
        "target": target,
        "recon": reconstruction,
    }
    mock_inference = Mock(return_value=inference_results)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", mock_inference)

    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([False, True, False, False, True])
    mock_strategy = Mock()
    mock_strategy.scs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", Mock(return_value=mock_strategy))
    monkeypatch.setattr(anomaly_analysis, "generate_anomaly_detection_report", Mock())

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        input_df,
        threshold_strategy="scs",
        model_path="model.pth",
        model_config_path="config.yaml",
        sdk_config=anomaly_analysis.ADDiffusionConfig(
            explain=True,
            explanation_top_k=3,
            report_path=tmp_path / "report.pdf",
        ),
    )

    inference_df = mock_inference.call_args.kwargs["data"]
    assert list(inference_df.columns) == feature_names
    assert result.loc[0, "TopContributors"] == "[]"
    assert result.loc[1, "TopContributors"] == ('["010-000-024-033","010-000-030-096","020-000-032-221"]')
    assert json.loads(result.loc[1, "ContributionShares"]) == pytest.approx([7 / 122, 6 / 122, 5 / 122], abs=1e-6)
    assert result.loc[1, "ExplanationCoverage"] == pytest.approx(18 / 122)
    assert result.loc[1, "ExplanationMethod"] == "reconstruction_error"
    mock_inference.assert_called_once()


@pytest.mark.parametrize("input_feature_count", [1, 7, 39, 40])
def test_explain_toggle_maps_any_supported_input_width(monkeypatch, input_feature_count):
    """Every directly mapped width up to target_dim should retain its input names."""
    sample_count = 40
    feature_names = [f"signal_{index}" for index in range(input_feature_count)]
    input_df = pd.DataFrame(np.ones((sample_count, input_feature_count)), columns=feature_names)
    target = np.zeros((sample_count, 40))
    reconstruction = np.zeros_like(target)
    reconstruction[0, :input_feature_count] = np.arange(1, input_feature_count + 1)
    inference_results = {
        "residual": np.abs(reconstruction).mean(axis=1),
        "target": target,
        "recon": reconstruction,
    }
    monkeypatch.setattr(anomaly_analysis, "get_model_target_dim", lambda model_path, config_path: 40)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", Mock(return_value=inference_results))
    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([True] + [False] * (sample_count - 1))
    mock_strategy = Mock()
    mock_strategy.scs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", Mock(return_value=mock_strategy))

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        input_df,
        threshold_strategy="scs",
        model_path="model.pth",
        model_config_path="config.yaml",
        sdk_config=anomaly_analysis.ADDiffusionConfig(explain=True),
    )

    contributors = json.loads(result.loc[0, "TopContributors"])
    assert contributors == feature_names[-min(3, input_feature_count) :][::-1]


def test_report_metadata_arguments_are_ignored_without_report_path(monkeypatch, inference_results):
    """Report-only arguments must not alter the default inference feature set."""
    input_df = pd.DataFrame(
        {
            "sample_time": [1, 2, 3, 4, 5],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [2.0, 2.5, 3.0, 3.5, 4.0],
            "expected_anomaly": [0, 1, 0, 0, 1],
        }
    )
    mock_inference = Mock(return_value=inference_results)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", mock_inference)

    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([False, True, False, True, False])
    mock_strategy = Mock()
    mock_strategy.scs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", Mock(return_value=mock_strategy))

    anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        input_df,
        threshold_strategy="scs",
        model_path="model.pth",
        model_config_path="config.yaml",
        sdk_config=anomaly_analysis.ADDiffusionConfig(
            timestamp_column="sample_time",
            ground_truth_column="expected_anomaly",
        ),
    )

    inference_df = mock_inference.call_args.kwargs["data"]
    assert list(inference_df.columns) == list(input_df.columns)


def test_load_sdk_config_accepts_nested_sdk_section(tmp_path):
    config_path = tmp_path / "sdk_config.yaml"
    config_path.write_text(
        """
model:
  target_dim: 40
sdk:
  explain: true
  explanation_top_k: 4
  diagnose_reconstruction: true
  report_title: Custom Report
  report_consolidated_top_k: 8
""",
        encoding="utf-8",
    )

    config = anomaly_analysis.load_sdk_config(config_path)

    assert isinstance(config, anomaly_analysis.ADDiffusionConfig)
    assert config.explain is True
    assert config.explanation_top_k == 4
    assert config.diagnose_reconstruction is True
    assert config.report_title == "Custom Report"
    assert config.report_consolidated_top_k == 8
    assert config.report_max_pages == 10


def test_load_sdk_config_accepts_flat_mapping(tmp_path):
    config_path = tmp_path / "sdk_config.yaml"
    config_path.write_text("report_title: Custom Report\nreport_max_pages: 2\n", encoding="utf-8")

    config = anomaly_analysis.load_sdk_config(config_path)

    assert config.report_title == "Custom Report"
    assert config.report_max_pages == 2


def test_load_sdk_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "sdk_config.yaml"
    config_path.write_text(
        """
sdk:
  report_title: Custom Report
  typo_top_k: 5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="typo_top_k"):
        anomaly_analysis.load_sdk_config(config_path)


def test_commented_yaml_template_matches_config_defaults():
    template_path = Path(anomaly_analysis.__file__).with_name("sdk_config.yaml")

    assert asdict(anomaly_analysis.load_sdk_config(template_path)) == asdict(anomaly_analysis.ADDiffusionConfig())

    previous_content = ""
    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if line.startswith("  ") and not line.startswith("    ") and stripped and not stripped.startswith("#"):
            assert previous_content.startswith("#"), f"Missing comment for config value: {stripped}"
        if stripped:
            previous_content = stripped


def test_perform_anomaly_analysis_uses_yaml_sdk_config(monkeypatch, inference_results, tmp_path):
    destination = tmp_path / "anomaly_report.pdf"
    config_path = tmp_path / "sdk_config.yaml"
    config_path.write_text(
        f"""
sdk:
  timestamp_column: timestamp
  ground_truth_column: GT
  report_path: {destination}
""",
        encoding="utf-8",
    )
    input_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="min"),
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [2.0, 2.5, 3.0, 3.5, 4.0],
            "GT": [0, 1, 0, 0, 1],
        }
    )
    mock_inference = Mock(return_value=inference_results)
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", mock_inference)

    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([False, True, False, True, False])
    mock_strategy = Mock()
    mock_strategy.scs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", Mock(return_value=mock_strategy))

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        input_df,
        threshold_strategy="scs",
        model_path="model.pth",
        model_config_path="config.yaml",
        sdk_config=config_path,
    )

    inference_df = mock_inference.call_args.kwargs["data"]
    assert list(inference_df.columns) == ["feature_1", "feature_2"]
    assert {"timestamp", "GT", "Anomaly", "MAE"}.issubset(result.columns)
    assert destination.read_bytes().startswith(b"%PDF")


def test_reconstruction_diagnosis_is_opt_in_and_hides_raw_scores(monkeypatch, numeric_df):
    """Diagnosis should return only simplified metrics after four repair passes."""
    original = {
        "residual": np.array([0.1, 0.5, 0.2, 0.3, 0.1]),
        "target": np.zeros((5, 3)),
        "recon": np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }

    def repaired(score_at_anomaly):
        return {"residual": np.array([0.1, score_at_anomaly, 0.2, 0.3, 0.1])}

    mock_inference = Mock(side_effect=[original, repaired(0.1), repaired(0.3), repaired(0.4), repaired(0.45)])
    monkeypatch.setattr(anomaly_analysis, "inference_ad_tesseract2_mp", mock_inference)
    mock_thresholder = Mock()
    mock_thresholder.detect_anomalies.return_value = np.array([False, True, False, False, False])
    mock_strategy = Mock()
    mock_strategy.scs_thresholder = mock_thresholder
    monkeypatch.setattr(anomaly_analysis, "SCSThresholdStrategy", Mock(return_value=mock_strategy))

    result = anomaly_analysis.perform_anomaly_analysis_with_diffusion(
        numeric_df,
        threshold_strategy="scs",
        model_path="model.pth",
        model_config_path="config.yaml",
        sdk_config=anomaly_analysis.ADDiffusionConfig(explain=True, diagnose_reconstruction=True),
    )

    assert mock_inference.call_count == 5
    assert result.loc[1, "LikelyReconstructionIssue"] == "Level shift"
    assert result.loc[1, "RepairImpact"] == pytest.approx(0.8)
    assert result.loc[1, "DiagnosticConfidence"] == "High"
    assert result.loc[1, "ExplanationConsistency"] == "Not evaluated"
    assert not any(column.startswith(("effect_", "score_after_")) for column in result.columns)


def test_reconstruction_diagnosis_requires_feature_explanations(numeric_df):
    with pytest.raises(ValueError, match="requires explain=True"):
        anomaly_analysis.perform_anomaly_analysis_with_diffusion(
            numeric_df,
            threshold_strategy="scs",
            model_path="model.pth",
            model_config_path="config.yaml",
            sdk_config=anomaly_analysis.ADDiffusionConfig(diagnose_reconstruction=True),
        )
