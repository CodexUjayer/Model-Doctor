"""PredictionDoctor — analyzes prediction quality for classification and regression using Evidence Engine."""

import numpy as np
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.enums import Severity, Confidence, TaskType
from modeldoctor.diagnosis import EvidenceBuilder, ConfidenceEngine, RiskEngine, RULES


class PredictionDoctor(BaseDoctor):
    """Diagnoses prediction quality across classification and regression tasks."""

    name = "PredictionDoctor"
    dimension = "Prediction Quality"
    priority = 60

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = EvidenceBuilder()

        is_classification = context.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )

        if is_classification:
            self._assess_classification(context, builder)
        else:
            self._assess_regression(context, builder)

        evidence_list = builder.get_all()
        if not evidence_list:
            diagnosis.dimension_score = 100.0
            return diagnosis

        confidence_score, confidence_label = ConfidenceEngine.compute(evidence_list)
        risk_score, risk_level, risk_expl = RiskEngine.compute(evidence_list)

        conf_map = {
            "Very High": Confidence.HIGH,
            "High": Confidence.HIGH,
            "Medium": Confidence.MEDIUM,
            "Low": Confidence.LOW,
        }
        sev_map = {
            "CRITICAL": Severity.CRITICAL,
            "HIGH": Severity.CRITICAL,
            "MEDIUM": Severity.WARNING,
            "LOW": Severity.INFO,
            "INFO": Severity.INFO,
        }

        severity = sev_map.get(risk_level, Severity.INFO)

        if severity in [Severity.CRITICAL, Severity.WARNING]:
            legacy_evidence = {e.name: e.measured_value for e in evidence_list}
            finding = Finding(
                title="Poor Prediction Quality Detected",
                description=risk_expl,
                severity=severity,
                confidence=conf_map.get(confidence_label, Confidence.MEDIUM),
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=legacy_evidence,
                structured_evidence=evidence_list,
                tags=["prediction_quality"],
            )
            diagnosis.add_finding(finding)

        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis

    def _assess_classification(self, context: EvaluationContext, builder: EvidenceBuilder) -> None:
        metrics = context.classification_metrics
        accuracy = metrics.get("accuracy")
        f1 = metrics.get("f1")

        if accuracy is not None:
            if accuracy < RULES.prediction_accuracy_critical:
                builder.add(
                    name="Test Accuracy",
                    description=f"Test accuracy of {accuracy:.3f} is critically low.",
                    measured_value=float(accuracy),
                    expected_range=f">= {RULES.prediction_accuracy_warning}",
                    weight="Very High",
                    normalized_score=1.0,
                )
            elif accuracy < RULES.prediction_accuracy_warning:
                builder.add(
                    name="Test Accuracy",
                    description=f"Test accuracy of {accuracy:.3f} is below the warning threshold.",
                    measured_value=float(accuracy),
                    expected_range=f">= {RULES.prediction_accuracy_warning}",
                    weight="High",
                    normalized_score=0.65,
                )

        if f1 is not None and f1 < RULES.prediction_f1_critical:
            builder.add(
                name="F1 Score",
                description=f"F1 score of {f1:.3f} is critically low, indicating poor precision/recall balance.",
                measured_value=float(f1),
                expected_range=f">= {RULES.prediction_f1_critical}",
                weight="High",
                normalized_score=0.9,
            )

    def _assess_regression(self, context: EvaluationContext, builder: EvidenceBuilder) -> None:
        metrics = context.regression_metrics
        r2 = metrics.get("r2")

        if r2 is not None:
            if r2 < RULES.prediction_r2_critical:
                builder.add(
                    name="R² Score",
                    description=f"R² of {r2:.3f} is critically low — model explains very little variance.",
                    measured_value=float(r2),
                    expected_range=f">= {RULES.prediction_r2_warning}",
                    weight="Very High",
                    normalized_score=1.0,
                )
            elif r2 < RULES.prediction_r2_warning:
                builder.add(
                    name="R² Score",
                    description=f"R² of {r2:.3f} is below the warning threshold.",
                    measured_value=float(r2),
                    expected_range=f">= {RULES.prediction_r2_warning}",
                    weight="High",
                    normalized_score=0.65,
                )
