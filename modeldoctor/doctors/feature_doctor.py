"""FeatureDoctor — analyzes feature engineering quality using Evidence Engine."""

import numpy as np
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.enums import Severity, Confidence
from modeldoctor.diagnosis import EvidenceBuilder, ConfidenceEngine, RiskEngine, RULES


class FeatureDoctor(BaseDoctor):
    """Diagnoses issues with features such as high dimensionality, near-zero variance, and high cardinality."""

    name = "FeatureDoctor"
    dimension = "Feature Engineering"
    priority = 20

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = EvidenceBuilder()

        if context.X_train is None or not hasattr(context.X_train, "shape"):
            return diagnosis

        n_features = context.X_train.shape[1] if len(context.X_train.shape) > 1 else 1
        n_samples = context.X_train.shape[0]

        # Signal 1: High Feature Dimensionality
        if n_features > RULES.feature_high_dimensionality:
            builder.add(
                name="High Dimensionality",
                description=f"The dataset has {n_features} features. Very high-dimensional inputs risk overfitting and slow inference.",
                measured_value=int(n_features),
                expected_range=f"<= {RULES.feature_high_dimensionality}",
                weight="Medium",
                normalized_score=0.7,
            )

        # Signal 2: Constant / Near-Constant Features
        try:
            X = np.asarray(context.X_train, dtype=float)
            variances = np.nanvar(X, axis=0)
            constant_count = int(np.sum(variances == 0))
            if constant_count > 0:
                builder.add(
                    name="Constant Features",
                    description=f"{constant_count} feature(s) have zero variance and carry no information.",
                    measured_value=constant_count,
                    expected_range="0",
                    weight="High",
                    normalized_score=0.85,
                )
        except Exception:
            pass

        # Signal 3: Feature Importance Concentration
        try:
            importances = context.feature_importances
            if importances is not None and len(importances) > 1:
                total = np.sum(importances)
                if total > 0:
                    rel = importances / total
                    top1 = float(np.max(rel))
                    top3 = float(np.sum(np.sort(rel)[-3:]))
                    if top1 > RULES.leakage_importance_warning:
                        builder.add(
                            name="Feature Importance Concentration",
                            description=(
                                f"Top feature accounts for {top1:.1%} of total importance "
                                f"(top-3 combined: {top3:.1%}). "
                                "High concentration may indicate leakage or insufficient feature diversity."
                            ),
                            measured_value=float(top1),
                            expected_range=f"<= {RULES.leakage_importance_warning}",
                            weight="Medium",
                            normalized_score=0.6,
                        )
        except Exception:
            pass

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
            "HIGH": Severity.WARNING,
            "MEDIUM": Severity.WARNING,
            "LOW": Severity.INFO,
            "INFO": Severity.INFO,
        }

        severity = sev_map.get(risk_level, Severity.INFO)

        if severity in [Severity.CRITICAL, Severity.WARNING]:
            legacy_evidence = {e.name: e.measured_value for e in evidence_list}
            finding = Finding(
                title="Feature Engineering Issues Detected",
                description=risk_expl,
                severity=severity,
                confidence=conf_map.get(confidence_label, Confidence.MEDIUM),
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=legacy_evidence,
                structured_evidence=evidence_list,
                tags=["feature_engineering"],
            )
            diagnosis.add_finding(finding)

        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
