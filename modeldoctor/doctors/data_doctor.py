"""DataDoctor — analyzes data quality issues like imbalance, missing values, and outliers using Evidence Engine."""

import numpy as np
import pandas as pd

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.enums import Severity, Confidence, TaskType
from modeldoctor.diagnosis import EvidenceBuilder, ConfidenceEngine, RiskEngine, RULES

class DataDoctor(BaseDoctor):
    """Diagnoses data-level issues before model training."""

    name = "DataDoctor"
    dimension = "Data Quality"
    priority = 10

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = EvidenceBuilder()
        
        # 1. Missing Values
        if context.X_train is not None:
            if hasattr(context.X_train, "isna"):
                missing = context.X_train.isna().sum().sum()
                total = context.X_train.size
            elif hasattr(context.X_train, "isnull"):
                missing = 0
                total = 1
            else:
                missing = np.isnan(np.asarray(context.X_train, dtype=float)).sum()
                total = context.X_train.size

            if total > 0:
                missing_pct = missing / total
                if missing_pct > RULES.data_missing_critical:
                    builder.add("Missing Values", float(missing_pct), weight="High", normalized_score=1.0, expected_range=f"<= {RULES.data_missing_warning}")
                elif missing_pct > RULES.data_missing_warning:
                    builder.add("Missing Values", float(missing_pct), weight="Medium", normalized_score=0.7, expected_range=f"<= {RULES.data_missing_warning}")

            # 2. Constant Features
            try:
                if hasattr(context.X_train, "var"):
                    # For Pandas dataframe
                    variances = context.X_train.var(axis=0)
                    constant_count = (variances == 0).sum()
                else:
                    X_arr = np.asarray(context.X_train, dtype=float)
                    variances = np.nanvar(X_arr, axis=0)
                    constant_count = (variances == 0).sum()
                    
                if constant_count > 0:
                    builder.add("Constant Features", int(constant_count), weight="Low", normalized_score=0.8, expected_range="0")
            except Exception:
                pass

        # 3. Class Imbalance
        if context.y_train is not None and context.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            y = np.asarray(context.y_train)
            classes, counts = np.unique(y, return_counts=True)
            if len(classes) >= 2:
                min_count = counts.min()
                max_count = counts.max()
                ratio = max_count / max(min_count, 1)

                if ratio > RULES.data_imbalance_critical:
                    builder.add("Class Imbalance", float(ratio), weight="High", normalized_score=1.0, expected_range=f"<= {RULES.data_imbalance_warning}")
                elif ratio > RULES.data_imbalance_warning:
                    builder.add("Class Imbalance", float(ratio), weight="Medium", normalized_score=0.7, expected_range=f"<= {RULES.data_imbalance_warning}")

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
            legacy_evidence = {e.name: e.measured_value for e in evidence_list}
            
            finding = Finding(
                title="Data Quality Issues Detected",
                description=risk_expl,
                severity=severity,
                confidence=conf_map.get(confidence_label, Confidence.MEDIUM),
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=legacy_evidence,
                structured_evidence=evidence_list,
                tags=["data_quality"]
            )
            diagnosis.add_finding(finding)
            
        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
