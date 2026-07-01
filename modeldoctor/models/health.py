"""Health scoring data models.

A HealthScore captures the holistic wellness of a model across multiple
diagnostic dimensions. Each Doctor contributes a DimensionScore; the
HealthScorer aggregates them into an overall score.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from modeldoctor.models.enums import Severity


# ---------------------------------------------------------------------------
# Dimension weights — must sum to 1.0
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: Dict[str, float] = {
    "data_quality": 0.20,
    "feature_engineering": 0.15,
    "overfitting": 0.20,
    "leakage": 0.15,
    "hyperparameters": 0.10,
    "prediction_quality": 0.10,
    "generalization": 0.05,
    "production_readiness": 0.05,
}


class DimensionScore(BaseModel):
    """Health score for a single diagnostic dimension.

    Attributes:
        dimension: The health dimension identifier (must match a key in ``DIMENSION_WEIGHTS``).
        score: Raw score from 0 (worst) to 100 (best).
        weight: The weight applied to this dimension in the overall score.
        max_severity: The worst severity finding in this dimension.
        summary: One-sentence summary of the dimension's health.
    """

    dimension: str
    score: float = Field(ge=0.0, le=100.0)
    weight: float = Field(ge=0.0, le=1.0)
    max_severity: Optional[Severity] = None
    summary: str = ""

    @property
    def weighted_score(self) -> float:
        """Score multiplied by weight for aggregation."""
        return self.score * self.weight

    @property
    def grade(self) -> str:
        """Letter grade for display purposes."""
        if self.score >= 90:
            return "A"
        if self.score >= 75:
            return "B"
        if self.score >= 60:
            return "C"
        if self.score >= 40:
            return "D"
        return "F"

    @property
    def status_emoji(self) -> str:
        """Traffic-light emoji for the score."""
        if self.score >= 80:
            return "✅"
        if self.score >= 60:
            return "⚠️"
        return "❌"


class HealthScore(BaseModel):
    """Overall health score aggregated across all diagnostic dimensions.

    Attributes:
        overall: Weighted average score (0–100).
        dimensions: Per-dimension breakdown.
        grade: Overall letter grade (A–F).
        verdict: Short human-readable verdict.
        total_findings: Total number of findings across all doctors.
        critical_count: Number of CRITICAL findings.
        error_count: Number of ERROR findings.
        warning_count: Number of WARNING findings.
    """

    overall: float = Field(ge=0.0, le=100.0)
    dimensions: List[DimensionScore] = Field(default_factory=list)
    total_findings: int = 0
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0

    @property
    def grade(self) -> str:
        if self.overall >= 90:
            return "A"
        if self.overall >= 75:
            return "B"
        if self.overall >= 60:
            return "C"
        if self.overall >= 40:
            return "D"
        return "F"

    @property
    def verdict(self) -> str:
        """Short human-readable verdict based on overall score."""
        if self.overall >= 90:
            return "Excellent — production-ready"
        if self.overall >= 75:
            return "Good — minor improvements recommended"
        if self.overall >= 60:
            return "Fair — several issues require attention"
        if self.overall >= 40:
            return "Poor — significant problems detected"
        return "Critical — not suitable for production"

    @property
    def status_emoji(self) -> str:
        if self.overall >= 80:
            return "✅"
        if self.overall >= 60:
            return "⚠️"
        return "❌"

    def dimension_by_name(self, name: str) -> Optional[DimensionScore]:
        """Look up a dimension score by name."""
        return next((d for d in self.dimensions if d.dimension == name), None)
