"""CalibrationDoctor — diagnoses probability calibration issues."""

import numpy as np
from typing import Dict, Any

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.metadata import DoctorMetadata
from modeldoctor.models.enums import TaskType, Severity


class CalibrationDoctor(BaseDoctor):
    """Diagnoses probability calibration for classification models.

    Computes Expected Calibration Error (ECE) and Brier Score, then
    produces actionable findings with prescriptions embedded in the
    finding evidence for the PrescriptionEngine to act on.
    """

    name = "calibration_doctor"
    dimension = "Calibration"

    metadata = DoctorMetadata(
        name="calibration_doctor",
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

        y_true = context.y_test
        y_prob = context.test_probabilities[:, 1]

        # Brier Score (from context metrics if available, else compute)
        brier_score = context.classification_metrics.get("brier_score")
        if brier_score is None:
            from sklearn.metrics import brier_score_loss
            brier_score = brier_score_loss(y_true, y_prob)

        # Compute Expected Calibration Error (ECE)
        ece = self._compute_ece(y_true, y_prob)

        # Store metrics in metadata (Diagnosis doesn't have a metrics field)
        diagnosis.metadata["calibration_metrics"] = {
            "brier_score": float(brier_score),
            "expected_calibration_error": float(ece),
        }

        # Check for poor calibration
        if ece > 0.15:
            diagnosis.add_finding(
                self._finding(
                    severity=Severity.ERROR,
                    title="Severe Probability Miscalibration",
                    description=(
                        f"The model's predicted probabilities are poorly calibrated "
                        f"(ECE={ece:.3f}). Predictions do not reflect true likelihoods."
                    ),
                    evidence={
                        "expected_calibration_error": round(ece, 4),
                        "brier_score": round(float(brier_score), 4),
                        "severity_threshold": 0.15,
                        "prescription": "Platt Scaling or Isotonic Regression",
                    },
                )
            )
        elif ece > 0.05:
            diagnosis.add_finding(
                self._finding(
                    severity=Severity.WARNING,
                    title="Moderate Probability Miscalibration",
                    description=(
                        f"The model is slightly miscalibrated (ECE={ece:.3f}). "
                        "Consider post-hoc calibration."
                    ),
                    evidence={
                        "expected_calibration_error": round(ece, 4),
                        "brier_score": round(float(brier_score), 4),
                        "severity_threshold": 0.05,
                        "prescription": "Isotonic Regression on a holdout set",
                    },
                )
            )
        else:
            diagnosis.add_finding(
                self._finding(
                    severity=Severity.INFO,
                    title="Good Probability Calibration",
                    description=f"The model is well-calibrated (ECE={ece:.3f}).",
                    evidence={
                        "expected_calibration_error": round(ece, 4),
                        "brier_score": round(float(brier_score), 4),
                    },
                )
            )

        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis

    def _compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error using equal-width binning."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binned = np.digitize(y_prob, bins) - 1
        # Clip to valid range [0, n_bins-1]
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
