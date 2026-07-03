"""OverfittingDoctor — analyzes train/test gap and learning curves using Evidence Engine."""

import numpy as np
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.enums import Severity, Confidence
from modeldoctor.diagnosis import EvidenceBuilder, ConfidenceEngine, RiskEngine, get_adapter, RULES

class OverfittingDoctor(BaseDoctor):
    """Diagnoses model overfitting using the v1.1 Evidence Engine architecture."""

    name = "OverfittingDoctor"
    dimension = "Overfitting Risk"
    priority = 30

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = EvidenceBuilder()
        
        train_score = context.train_score
        test_score = context.test_score
        
        gap = train_score - test_score
        
        # Signal 1: Generalization Gap
        if gap > 0:
            if gap > RULES.overfitting_gap_critical:
                builder.add("Generalization Gap", gap, weight="Very High", normalized_score=1.0, expected_range=f"<= {RULES.overfitting_gap_critical}")
            elif gap > RULES.overfitting_gap_warning:
                builder.add("Generalization Gap", gap, weight="High", normalized_score=0.75, expected_range=f"<= {RULES.overfitting_gap_warning}")
                
            # Signal 2: Memorization Check
            if train_score >= RULES.memorization_train_threshold and test_score < RULES.memorization_test_threshold:
                builder.add("Memorization", train_score, weight="Very High", normalized_score=1.0, expected_range=f"< {RULES.memorization_train_threshold}")
                
        # Signal 3: CV Variance
        cv_scores = context.cv_scores
        if cv_scores is not None and len(cv_scores) > 1:
            cv_std = float(np.std(cv_scores))
            if cv_std > RULES.cv_variance_warning:
                builder.add("CV Variance", cv_std, weight="Medium", normalized_score=0.6, expected_range=f"<= {RULES.cv_variance_warning}")
                
        # Signal 4: Complexity
        adapter = get_adapter(context.model)
        
        n_train = context.n_train
        param_count = adapter.parameter_count()
        if param_count is not None and n_train > 0:
            if param_count > (n_train * RULES.excessive_capacity_ratio):
                builder.add("Excessive Capacity", param_count, weight="High", normalized_score=0.8, expected_range=f"<= {int(n_train * RULES.excessive_capacity_ratio)}")
                
        depth = adapter.depth()
        if depth is not None:
            if depth > RULES.tree_depth_warning and n_train < RULES.tree_depth_max_samples:
                builder.add("Unrestricted Tree Depth", depth, weight="Medium", normalized_score=0.6, expected_range=f"<= {RULES.tree_depth_warning}")
                
        # Evaluate collected evidence
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
            "Low": Confidence.LOW
        }
        
        sev_map = {
            "CRITICAL": Severity.CRITICAL,
            "HIGH": Severity.CRITICAL,
            "MEDIUM": Severity.WARNING,
            "LOW": Severity.INFO,
            "INFO": Severity.INFO
        }
        
        severity = sev_map.get(risk_level, Severity.INFO)
        
        if severity in [Severity.CRITICAL, Severity.WARNING]:
            # Convert DiagnosticEvidence to standard legacy format to avoid test failures in other places
            # while maintaining the rich structured_evidence
            legacy_evidence = {e.name: e.measured_value for e in evidence_list}
            
            finding = Finding(
                title="Overfitting Evidence Detected",
                description=risk_expl,
                severity=severity,
                confidence=conf_map.get(confidence_label, Confidence.MEDIUM),
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=legacy_evidence,
                structured_evidence=evidence_list,
                tags=["overfitting"]
            )
            diagnosis.add_finding(finding)
            
        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
