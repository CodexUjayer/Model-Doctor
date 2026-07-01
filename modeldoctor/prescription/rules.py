"""Prescription rules models and types."""

from typing import Any, List

from pydantic import BaseModel, Field

from modeldoctor.models.enums import Confidence, Priority


class RuleCondition(BaseModel):
    """A condition to evaluate for a prescription rule."""

    field: str
    operator: str  # equals, not_equals, gt, lt, ge, le, contains, etc.
    value: Any


class RuleRecommendation(BaseModel):
    """The recommendation to yield when a rule fires."""

    action: str
    rationale: str
    confidence: Confidence = Confidence.MEDIUM
    estimated_impact: str = ""
    priority: Priority = Priority.MEDIUM


class PrescriptionRule(BaseModel):
    """A single rule in the prescription engine."""

    id: str
    name: str
    knowledge_id: str = ""
    conditions: List[RuleCondition] = Field(default_factory=list)
    recommendations: List[RuleRecommendation] = Field(default_factory=list)
    root_cause: str = ""
