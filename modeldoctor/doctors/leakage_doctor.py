"""LeakageDoctor — analyzes data leakage risk."""

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.enums import Severity


class LeakageDoctor(BaseDoctor):
    """Diagnoses potential data leakage (features containing future info)."""

    name = "leakage_doctor"
    dimension = "Leakage Risk"
    priority = 40

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        
        # Placeholder: look for features with correlation > 0.99 to target
        
        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
