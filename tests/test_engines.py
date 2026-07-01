"""Tests for the new ModelDoctor engines and CLI."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from modeldoctor.core.context import EvaluationContext
from modeldoctor.core.explainability import ExplainabilityEngine
from modeldoctor.core.comparison import ModelComparisonEngine
from modeldoctor.doctors.calibration import CalibrationDoctor
from modeldoctor.config.settings import ModelDoctorConfig
from modeldoctor.models.enums import ExplainabilityMode, TaskType


@pytest.fixture
def sample_context():
    """Create a sample context for testing engines."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X[:80], y[:80])
    
    config = ModelDoctorConfig()
    config.compare_models = True
    config.explainability = ExplainabilityMode.DISABLED
    
    return EvaluationContext(
        model=model,
        X_train=X[:80],
        y_train=y[:80],
        X_test=X[80:],
        y_test=y[80:],
        config=config
    )


def test_calibration_doctor(sample_context):
    """Test CalibrationDoctor identifies issues and computes ECE."""
    doctor = CalibrationDoctor()
    assert doctor.can_examine(sample_context) is True
    
    diagnosis = doctor.run(sample_context)
    
    assert diagnosis.doctor_name == "calibration_doctor"
    assert "calibration_metrics" in diagnosis.metadata
    assert "expected_calibration_error" in diagnosis.metadata["calibration_metrics"]


def test_comparison_engine(sample_context):
    """Test ModelComparisonEngine benchmarks alternatives."""
    engine = ModelComparisonEngine()
    results = engine.run(sample_context)
    
    # Should include baseline + LogisticRegression, LightGBM, DecisionTree
    assert len(results) > 1
    
    baseline = next(r for r in results if r.is_baseline)
    assert baseline.model_name == "RandomForestClassifier"
    assert baseline.accuracy > 0
    

def test_explainability_engine_disabled(sample_context):
    """Test ExplainabilityEngine respects config."""
    engine = ExplainabilityEngine()
    info = engine.run(sample_context)
    
    assert info.enabled is False
    assert info.skip_reason == "Explainability disabled in config."


def test_explainability_engine_permutation(sample_context):
    """Test ExplainabilityEngine fallback to permutation."""
    sample_context.config.explainability = ExplainabilityMode.AUTO
    
    with patch("modeldoctor.core.explainability.is_shap_available", return_value=False):
        engine = ExplainabilityEngine()
        info = engine.run(sample_context)
        
        assert info.enabled is True
        assert info.shap_available is False
        assert info.method == "permutation"
        assert len(info.top_features) > 0


@patch("modeldoctor.cli._get_console", return_value=None)
def test_cli_diagnose_command(mock_console, tmp_path):
    """Test CLI diagnose command fallback (no rich)."""
    from modeldoctor.cli import diagnose
    from click.testing import CliRunner
    
    runner = CliRunner()
    
    model_path = tmp_path / "model.pkl"
    data_path = tmp_path / "data.csv"
    model_path.touch()
    data_path.touch()
    
    result = runner.invoke(diagnose, [str(model_path), str(data_path), "--out", str(tmp_path / "out.md")])
    assert result.exit_code == 0
    assert "Analysis complete" in result.output
