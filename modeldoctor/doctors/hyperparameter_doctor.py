"""HyperparameterDoctor — analyzes model hyperparameters."""

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.enums import Severity


class HyperparameterDoctor(BaseDoctor):
    """Diagnoses suboptimal hyperparameter configurations."""

    name = "hyperparameter_doctor"
    dimension = "Hyperparameter Tuning"
    priority = 50

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        
        # Placeholder: check for default settings in complex models like RF or XGB
        if hasattr(context.model, "get_params"):
            params = context.model.get_params()
            if "max_depth" in params and params["max_depth"] is None:
                diagnosis.findings.append(
                    self._finding(
                        title="Unconstrained Tree Depth",
                        description="Tree depth is not constrained, which may lead to overfitting.",
                        severity=Severity.WARNING,
                        evidence=[self._evidence("max_depth", "None", expected="Int (e.g. 10)")]
                    )
                )
                
        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
