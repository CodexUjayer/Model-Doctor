"""OverfittingDoctor — analyzes train/test gap and learning curves."""

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.enums import Severity


class OverfittingDoctor(BaseDoctor):
    """Diagnoses model overfitting by comparing training and validation performance."""

    name = "overfitting_doctor"
    dimension = "Overfitting Risk"
    priority = 30

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        
        # In a real implementation we would compute metrics on train vs test
        # Placeholder logic:
        # gap = train_score - test_score
        # if gap > 0.10: ...
        
        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis
