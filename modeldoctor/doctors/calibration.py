"""CalibrationDoctor — diagnoses probability calibration issues using Evidence Engine."""

import numpy as np
from typing import Dict, Any

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.metadata import DoctorMetadata
from modeldoctor.models.enums import TaskType, Severity, Confidence
from modeldoctor.diagnosis import EvidenceBuilder, ConfidenceEngine, RiskEngine, RULES


class CalibrationDoctor(BaseDoctor):
    """Diagnoses probability calibration for classification models.

    Computes Expected Calibration Error (ECE) and Brier Score, then
    produces actionable findings backed by structured evidence.
    """

    name = "CalibrationDoctor"
    dimension = "Calibration"

    metadata = DoctorMetadata(
        name="CalibrationDoctor",
        priority=65,
        dimension="Calibration",
        supported_tasks=[TaskType.BINARY_CLASSIFICATION.value],
    )

    def can_examine(self, context: EvaluationContext) -> bool:
        """Only runs if the model supports probability output."""
        return (
            context.task_type == TaskType.BINARY_CLASSIFICATION
            and context.test_probabilities is not None
        )

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = EvidenceBuilder()

        y_true = context.y_test
        y_prob = context.test_probabilities[:, 1]

        # Brier Score
        brier_score = context.classification_metrics.get("brier_score")
        if brier_score is None:
            from sklearn.metrics import brier_score_loss
            brier_score = brier_score_loss(y_true, y_prob)

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y_true, y_prob)

        # Store raw metrics in diagnosis metadata for downstream consumers
        diagnosis.metadata["calibration_metrics"] = {
            "brier_score": float(brier_score),
            "expected_calibration_error": float(ece),
        }

        # Signal 1: ECE
        if ece > RULES.calibration_ece_critical:
            builder.add(
                name="Expected Calibration Error",
                description=f"ECE={ece:.3f} is critically high. Predicted probabilities are unreliable.",
                measured_value=float(ece),
                expected_range=f"<= {RULES.calibration_ece_warning}",
                weight="Very High",
                normalized_score=1.0,
            )
        elif ece > RULES.calibration_ece_warning:
            builder.add(
                name="Expected Calibration Error",
                description=f"ECE={ece:.3f} indicates moderate miscalibration.",
                measured_value=float(ece),
                expected_range=f"<= {RULES.calibration_ece_warning}",
                weight="High",
                normalized_score=0.7,
            )

        # Signal 2: Brier Score (supportive signal — never sole trigger for CRITICAL)
        brier = float(brier_score)
        if brier > RULES.calibration_brier_critical:
            builder.add(
                name="Brier Score",
                description=f"Brier Score={brier:.3f} is high (random guessing = 0.25).",
                measured_value=brier,
                expected_range=f"<= {RULES.calibration_brier_warning}",
                weight="Medium",
                normalized_score=0.65,
            )
        elif brier > RULES.calibration_brier_warning:
            builder.add(
                name="Brier Score",
                description=f"Brier Score={brier:.3f} is slightly elevated.",
                measured_value=brier,
                expected_range=f"<= {RULES.calibration_brier_warning}",
                weight="Low",
                normalized_score=0.4,
            )

        evidence_list = builder.get_all()

        if not evidence_list:
            # Good calibration — emit an INFO finding for completeness
            diagnosis.add_finding(
                self._finding(
                    severity=Severity.INFO,
                    title="Good Probability Calibration",
                    description=f"The model is well-calibrated (ECE={ece:.3f}, Brier={brier:.3f}).",
                    evidence={
                        "expected_calibration_error": round(ece, 4),
                        "brier_score": round(brier, 4),
                    },
                )
            )
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
            "HIGH": Severity.WARNING,
            "MEDIUM": Severity.WARNING,
            "LOW": Severity.INFO,
            "INFO": Severity.INFO,
        }

        severity = sev_map.get(risk_level, Severity.INFO)

        if severity in [Severity.CRITICAL, Severity.WARNING]:
            legacy_evidence = {e.name: e.measured_value for e in evidence_list}
            finding = Finding(
                title="Probability Miscalibration Detected",
                description=risk_expl,
                severity=severity,
                confidence=conf_map.get(confidence_label, Confidence.MEDIUM),
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=legacy_evidence,
                structured_evidence=evidence_list,
                tags=["calibration"],
            )
            diagnosis.add_finding(finding)

        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis

    def _compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error using equal-width binning."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binned = np.digitize(y_prob, bins) - 1
        binned = np.clip(binned, 0, n_bins - 1)

        ece = 0.0
        n_samples = len(y_true)

        for i in range(n_bins):
            bin_idx = binned == i
            bin_count = int(np.sum(bin_idx))

            if bin_count > 0:
                bin_acc = float(np.mean(y_true[bin_idx]))
                bin_conf = float(np.mean(y_prob[bin_idx]))
                ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

        return float(ece)
