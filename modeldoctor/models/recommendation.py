"""Recommendation and prescription data models.

Recommendations are the *prescriptive* counterpart to Findings.
A Recommendation answers: "Given this diagnosis, what should the practitioner do?"
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from modeldoctor.models.enums import Confidence, Priority


class Recommendation(BaseModel):
    """A single, actionable improvement recommendation.

    Attributes:
        id: Unique identifier (auto-generated).
        title: Short, descriptive title.
        description: Imperative description of what to do (replaces `action`).
        rationale: Why this action will help; root cause linkage.
        confidence: How confident the system is this will help.
        estimated_improvement: Free-text description of expected improvement (replaces `estimated_impact`).
        priority: How urgently this should be addressed.
        implementation_difficulty: "Easy", "Medium", or "Hard".
        supporting_evidence: Dictionary of structured evidence points.
        affected_metrics: List of metrics this recommendation will improve.
        code_example: Optional Python snippet demonstrating the fix.
        references: Links to papers, docs, or relevant resources.
        tags: Categorisation tags.
        source_finding_ids: IDs of Finding objects this recommendation addresses.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "Recommendation"
    description: str = Field(alias="action")
    rationale: str
    confidence: Confidence = Confidence.MEDIUM
    estimated_improvement: str = Field(default="", alias="estimated_impact")
    priority: Priority = Priority.MEDIUM
    implementation_difficulty: str = "Medium"
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)
    affected_metrics: List[str] = Field(default_factory=list)
    code_example: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    source_finding_ids: List[str] = Field(default_factory=list)

    model_config = {
        "populate_by_name": True,
    }

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Recommendation description must not be empty.")
        return v.strip()


class PrescriptionRule(BaseModel):
    """A rule that fired in the prescription engine, producing recommendations.

    Attributes:
        rule_id: Identifier of the YAML rule that triggered.
        rule_name: Human-readable rule name.
        root_cause: Explanation of why this pattern is problematic.
        recommendations: Recommendations produced by this rule.
        matched_conditions: The conditions that were satisfied.
        metadata: Rule metadata (source file, version, etc.).
    """

    rule_id: str
    rule_name: str
    root_cause: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    matched_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PrescriptionResult(BaseModel):
    """Aggregated output from the Prescription Engine.

    Attributes:
        fired_rules: Rules that matched the current model context.
        all_recommendations: Deduplicated, priority-sorted recommendations.
        rules_evaluated: Total number of rules tested.
        rules_fired: Number of rules that matched.
    """

    fired_rules: List[PrescriptionRule] = Field(default_factory=list)
    all_recommendations: List[Recommendation] = Field(default_factory=list)
    rules_evaluated: int = 0
    rules_fired: int = 0

    @property
    def critical_recommendations(self) -> List[Recommendation]:
        return [r for r in self.all_recommendations if r.priority == Priority.CRITICAL]

    @property
    def high_recommendations(self) -> List[Recommendation]:
        return [r for r in self.all_recommendations if r.priority == Priority.HIGH]

    def sorted_by_priority(self) -> List[Recommendation]:
        """Return recommendations ordered from CRITICAL → LOW."""
        order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }
        return sorted(self.all_recommendations, key=lambda r: order.get(r.priority, 99))
