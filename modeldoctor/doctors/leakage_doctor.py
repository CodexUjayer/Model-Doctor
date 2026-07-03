"""LeakageDoctor — analyzes data leakage risk."""

import numpy as np
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.enums import Severity, Confidence
from modeldoctor.diagnosis import EvidenceBuilder, ConfidenceEngine, RiskEngine, RULES

class LeakageDoctor(BaseDoctor):
    """Diagnoses potential data leakage (features containing future info)."""

    name = "LeakageDoctor"
    dimension = "Leakage Risk"
    priority = 40

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = EvidenceBuilder()
        
        # 1. Target Correlation
        # Ensure we can compute correlation
        if context.X_train is not None and context.y_train is not None:
            # We can compute Pearson correlation for numeric targets (or binary)
            try:
                y = context.y_train.astype(float)
                # Compute correlation for each feature
                # Avoid calculating for huge feature sets to save time
                n_features = context.X_train.shape[1]
                if n_features < 1000:
                    X = context.X_train.astype(float)
                    # Ignore constant features and nan
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_centered = y - np.mean(y)
                        X_centered = X - np.mean(X, axis=0)
                        
                        # Handle case where variance is zero
                        var_x = np.sum(X_centered**2, axis=0)
                        var_y = np.sum(y_centered**2)
                        
                        if var_y > 0:
                            cov = np.sum(X_centered * y_centered[:, np.newaxis], axis=0)
                            corrs = cov / np.sqrt(var_x * var_y)
                            
                            # Filter out NaNs (e.g. constant features)
                            valid_corrs = np.where(np.isnan(corrs), 0, corrs)
                            
                            max_corr_idx = np.argmax(np.abs(valid_corrs))
                            max_corr = np.abs(valid_corrs[max_corr_idx])
                            feature_name = context.feature_names[max_corr_idx] if context.feature_names else f"Feature {max_corr_idx}"
                            
                            if max_corr > RULES.leakage_correlation_critical:
                                builder.add(
                                    name="High Target Correlation",
                                    description=f"Feature '{feature_name}' has an exceptionally high correlation ({max_corr:.2f}) with the target. This strongly implies target leakage.",
                                    measured_value=float(max_corr),
                                    expected_range=f"<= {RULES.leakage_correlation_critical}",
                                    weight="Very High",
                                    normalized_score=1.0,
                                    metadata={"feature": feature_name}
                                )
                            elif max_corr > RULES.leakage_correlation_warning:
                                builder.add(
                                    name="Suspicious Target Correlation",
                                    description=f"Feature '{feature_name}' has high correlation ({max_corr:.2f}) with the target. Verify it does not contain future information.",
                                    measured_value=float(max_corr),
                                    expected_range=f"<= {RULES.leakage_correlation_warning}",
                                    weight="High",
                                    normalized_score=0.7,
                                    metadata={"feature": feature_name}
                                )
            except Exception:
                pass
                
        # 2. Perfect Predictors (Feature Importance)
        try:
            importances = context.feature_importances
            if importances is not None and len(importances) > 0:
                total_imp = np.sum(importances)
                if total_imp > 0:
                    rel_importances = importances / total_imp
                    max_imp_idx = np.argmax(rel_importances)
                    max_imp = rel_importances[max_imp_idx]
                    feature_name = context.feature_names[max_imp_idx] if context.feature_names else f"Feature {max_imp_idx}"
                    
                    if max_imp > RULES.leakage_importance_critical:
                        builder.add(
                            name="Single Feature Dominance",
                            description=f"Feature '{feature_name}' accounts for {max_imp:.1%} of total model importance. The model is almost entirely relying on this single feature, which is a classic sign of a perfect predictor (leakage).",
                            measured_value=float(max_imp),
                            expected_range=f"<= {RULES.leakage_importance_critical}",
                            weight="Very High",
                            normalized_score=1.0,
                            metadata={"feature": feature_name}
                        )
                    elif max_imp > RULES.leakage_importance_warning:
                        builder.add(
                            name="Disproportionate Feature Importance",
                            description=f"Feature '{feature_name}' accounts for {max_imp:.1%} of total model importance.",
                            measured_value=float(max_imp),
                            expected_range=f"<= {RULES.leakage_importance_warning}",
                            weight="Medium",
                            normalized_score=0.6,
                            metadata={"feature": feature_name}
                        )
        except Exception:
            pass

        evidence_list = builder.get_all()
        if not evidence_list:
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
                title="Data Leakage Risk Detected",
                description=risk_expl,
                severity=severity,
                confidence=conf_map.get(confidence_label, Confidence.MEDIUM),
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=legacy_evidence,
                structured_evidence=evidence_list,
                tags=["leakage"]
            )
            diagnosis.add_finding(finding)
            
        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
