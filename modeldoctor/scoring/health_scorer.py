"""HealthScorer — aggregates dimension scores into an overall health score."""

from typing import List

from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.health import DimensionScore, HealthScore


class HealthScorer:
    """Computes overall model health based on individual diagnoses."""

    DEFAULT_WEIGHTS = {
        "Data Quality": 0.20,
        "Feature Engineering": 0.15,
        "Overfitting Risk": 0.20,
        "Leakage Risk": 0.15,
        "Hyperparameter Tuning": 0.10,
        "Prediction Quality": 0.10,
        "Generalization": 0.05,
        "Production Readiness": 0.05,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS

    def compute_health(self, diagnoses: List[Diagnosis]) -> HealthScore:
        """Aggregates diagnoses into a single HealthScore."""
        dimension_scores = []
        total_weight = 0.0
        weighted_sum = 0.0

        for diag in diagnoses:
            weight = self.weights.get(diag.dimension, 0.10)
            score = diag.dimension_score
            
            dimension_scores.append(
                DimensionScore(
                    dimension=diag.dimension,
                    score=score,
                    weight=weight
                )
            )
            
            weighted_sum += score * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return HealthScore(
            overall=overall_score,
            dimension_scores=dimension_scores
        )
