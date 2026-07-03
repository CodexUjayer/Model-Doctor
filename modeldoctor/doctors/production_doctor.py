"""ProductionDoctor — analyzes deployment readiness using Evidence Engine."""

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.enums import Severity, Confidence
from modeldoctor.diagnosis import EvidenceBuilder, ConfidenceEngine, RiskEngine, RULES


class ProductionDoctor(BaseDoctor):
    """Diagnoses model size, serialization, and latency for production."""

    name = "ProductionDoctor"
    dimension = "Production Readiness"
    priority = 80

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = EvidenceBuilder()

        # Signal 1: Inference Latency
        try:
            latency_ms = context.prediction_latency_ms
            if latency_ms > RULES.production_latency_critical_ms:
                builder.add(
                    name="Inference Latency",
                    description=f"Single-sample inference latency of {latency_ms:.1f}ms exceeds the critical threshold of {RULES.production_latency_critical_ms}ms.",
                    measured_value=latency_ms,
                    expected_range=f"<= {RULES.production_latency_warning_ms}ms",
                    weight="High",
                    normalized_score=1.0,
                )
            elif latency_ms > RULES.production_latency_warning_ms:
                builder.add(
                    name="Inference Latency",
                    description=f"Single-sample inference latency of {latency_ms:.1f}ms is elevated.",
                    measured_value=latency_ms,
                    expected_range=f"<= {RULES.production_latency_warning_ms}ms",
                    weight="Medium",
                    normalized_score=0.6,
                )
        except Exception:
            pass

        # Signal 2: Model Size
        try:
            size_bytes = context.model_size
            size_mb = size_bytes / (1024 * 1024)
            if size_mb > RULES.production_size_critical_mb:
                builder.add(
                    name="Model Size",
                    description=f"Serialized model size of {size_mb:.1f}MB exceeds the critical threshold of {RULES.production_size_critical_mb}MB.",
                    measured_value=size_mb,
                    expected_range=f"<= {RULES.production_size_warning_mb}MB",
                    weight="High",
                    normalized_score=1.0,
                )
            elif size_mb > RULES.production_size_warning_mb:
                builder.add(
                    name="Model Size",
                    description=f"Serialized model size of {size_mb:.1f}MB is large.",
                    measured_value=size_mb,
                    expected_range=f"<= {RULES.production_size_warning_mb}MB",
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
                title="Production Readiness Concerns Detected",
                description=risk_expl,
                severity=severity,
                confidence=conf_map.get(confidence_label, Confidence.MEDIUM),
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=legacy_evidence,
                structured_evidence=evidence_list,
                tags=["production", "deployment"],
            )
            diagnosis.add_finding(finding)

        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
