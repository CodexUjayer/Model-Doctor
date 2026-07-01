"""Tests for new production-grade features."""

import os
from unittest.mock import MagicMock

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from modeldoctor.core.profile import ModelProfile
from modeldoctor.prescription.knowledge_base import KnowledgeBase
from modeldoctor.core.pipeline import DiagnosticPipeline
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.models.metadata import DoctorMetadata
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.enums import TaskType
import numpy as np


def test_model_profile_tree():
    model = RandomForestClassifier()
    profile = ModelProfile.introspect(model)
    assert profile.model_family == "tree"
    assert profile.estimator_type == "classifier"
    assert profile.ensemble is True
    assert profile.supports_shap is True


def test_model_profile_linear():
    model = LogisticRegression()
    profile = ModelProfile.introspect(model)
    assert profile.model_family == "linear"
    assert profile.estimator_type == "classifier"
    assert profile.ensemble is False


def test_knowledge_base_empty_dir(tmp_path):
    kb = KnowledgeBase(kb_dir=tmp_path)
    assert len(kb.entries) == 0


class DummyClassificationDoctor(BaseDoctor):
    name = "dummy_clf"
    dimension = "general"
    metadata = DoctorMetadata(
        name="dummy_clf",
        priority=10,
        supported_tasks=[TaskType.BINARY_CLASSIFICATION.value],
    )
    
    def examine(self, context):
        return self._new_diagnosis()


class DummyRegressionDoctor(BaseDoctor):
    name = "dummy_reg"
    dimension = "general"
    metadata = DoctorMetadata(
        name="dummy_reg",
        priority=20,
        supported_tasks=[TaskType.REGRESSION.value],
    )
    
    def examine(self, context):
        return self._new_diagnosis()


def test_pipeline_capability_filtering():
    from modeldoctor.core.registry import DoctorRegistry
    
    registry = DoctorRegistry()
    registry.register(DummyClassificationDoctor)
    registry.register(DummyRegressionDoctor)
    
    pipeline = DiagnosticPipeline(registry=registry)
    
    # Mock a classification context
    context = MagicMock(spec=EvaluationContext)
    context.task_type = TaskType.BINARY_CLASSIFICATION
    context.profile = ModelProfile(model_family="tree")
    
    diagnoses = pipeline.run(context)
    
    summary = pipeline.last_summary
    assert "dummy_clf" in summary.executed
    assert "dummy_reg" not in summary.executed
    assert "dummy_reg" in summary.skipped
    
    assert summary.skipped["dummy_reg"] == "Unsupported task type: binary_classification"
