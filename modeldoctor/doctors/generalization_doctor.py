"""GeneralizationDoctor — analyzes cross-validation variance and train/test gap using Evidence Engine."""

import numpy as np
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.enums import Severity, Confidence
from modeldoctor.diagnosis import EvidenceBuilder, ConfidenceEngine, RiskEngine, RULES


class GeneralizationDoctor(BaseDoctor):
    """Diagnoses variance across CV folds and robustness to distribution shift."""

    name = "GeneralizationDoctor"
    dimension = "Generalization"
    priority = 70

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = EvidenceBuilder()

        # Signal 1: Train/Test Score Gap
        train_score = context.train_score
        test_score = context.test_score
        gap = train_score - test_score

        if gap > RULES.generalization_gap_critical:
            builder.add(
                name="Train/Test Score Gap",
                description=(
                    f"The gap between training ({train_score:.3f}) and test ({test_score:.3f}) "
                    f"score is {gap:.3f}, indicating poor generalization."
                ),
                measured_value=float(gap),
                expected_range=f"<= {RULES.generalization_gap_warning}",
                weight="Very High",
                normalized_score=1.0,
            )
        elif gap > RULES.generalization_gap_warning:
            builder.add(
                name="Train/Test Score Gap",
                description=(
                    f"The gap between training ({train_score:.3f}) and test ({test_score:.3f}) "
                    f"score is {gap:.3f}. Minor generalization concern."
                ),
                measured_value=float(gap),
                expected_range=f"<= {RULES.generalization_gap_warning}",
                weight="High",
                normalized_score=0.65,
            )

        # Signal 2: Cross-Validation Variance
        cv_scores = context.cv_scores
        if cv_scores is not None and len(cv_scores) > 1:
            cv_std = float(np.std(cv_scores))
            cv_mean = float(np.mean(cv_scores))

            if cv_std > RULES.generalization_cv_std_critical:
                builder.add(
                    name="CV Score Variance",
                    description=(
                        f"CV standard deviation of {cv_std:.3f} (mean={cv_mean:.3f}) is critically high. "
                        "The model is highly sensitive to the data split."
                    ),
                    measured_value=cv_std,
                    expected_range=f"<= {RULES.generalization_cv_std_warning}",
                    weight="High",
                    normalized_score=1.0,
                )
            elif cv_std > RULES.generalization_cv_std_warning:
                builder.add(
                    name="CV Score Variance",
                    description=(
                        f"CV standard deviation of {cv_std:.3f} (mean={cv_mean:.3f}) is elevated."
                    ),
                    measured_value=cv_std,
                    expected_range=f"<= {RULES.generalization_cv_std_warning}",
                    weight="Medium",
                    normalized_score=0.6,
                )

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
                title="Poor Generalization Detected",
                description=risk_expl,
                severity=severity,
                confidence=conf_map.get(confidence_label, Confidence.MEDIUM),
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=legacy_evidence,
                structured_evidence=evidence_list,
                tags=["generalization"],
            )
            diagnosis.add_finding(finding)

        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
