"""Core diagnostic data models: Evidence, Finding, and Diagnosis.

These are the primary output types produced by each Doctor module.
They are intentionally separate from recommendations so that diagnosis
(what is wrong) and prescription (what to do) remain decoupled.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from modeldoctor.models.enums import Severity


class Evidence(BaseModel):
    """A single piece of quantitative or qualitative evidence supporting a finding.

    Attributes:
        label: Short human-readable label (e.g., ``"Train Accuracy"``).
        value: The observed value (numeric, string, or dict).
        expected: Optional expected/reference value for comparison.
        unit: Optional unit string (e.g., ``"%"``, ``"ms"``).
        context: Additional context dictionary for structured data.
    """

    label: str
    value: Any
    expected: Optional[Any] = None
    unit: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

    def formatted(self) -> str:
        """Return a human-readable string representation."""
        unit_str = f" {self.unit}" if self.unit else ""
        val_str = f"{self.value:.4f}" if isinstance(self.value, float) else str(self.value)
        result = f"{self.label}: {val_str}{unit_str}"
        if self.expected is not None:
            exp_str = (
                f"{self.expected:.4f}" if isinstance(self.expected, float) else str(self.expected)
            )
            result += f" (expected: {exp_str}{unit_str})"
        return result


class Finding(BaseModel):
    """A single diagnostic observation produced by a Doctor.

    Attributes:
        id: Unique identifier for this finding (auto-generated UUID).
        title: Short, descriptive title (≤ 80 chars).
        explanation: Detailed explanation of the issue (replaces `description`).
        severity: Impact severity of this finding.
        evidence: Dictionary of structured evidence items supporting the finding.
        affected_components: Model components or features implicated.
        tags: Free-form categorisation tags (e.g., ``["overfitting", "regularization"]``).
        doctor_name: Name of the Doctor that produced this finding.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    explanation: str = Field(alias="description")
    severity: Severity
    evidence: Dict[str, Any] = Field(default_factory=dict)
    affected_components: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    doctor_name: str = ""

    model_config = {
        "populate_by_name": True,
    }

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Finding title must not be empty.")
        return v.strip()

    @field_validator("evidence", mode="before")
    @classmethod
    def convert_legacy_evidence(cls, v: Any) -> Dict[str, Any]:
        """Convert legacy List[Evidence] to Dict[str, Any] for backwards compatibility."""
        if isinstance(v, list):
            result = {}
            for item in v:
                if isinstance(item, Evidence):
                    result[item.label] = item.value
                elif hasattr(item, "label") and hasattr(item, "value"):
                    result[item.label] = item.value
                else:
                    result[str(item)] = item
            return result
        if not isinstance(v, dict):
            return {"value": v}
        return v


class Diagnosis(BaseModel):
    """The complete diagnostic result from a single Doctor module.

    Attributes:
        doctor_name: Name of the Doctor that produced this diagnosis.
        dimension: Health dimension this diagnosis covers (e.g., ``"overfitting"``)
        findings: List of individual findings discovered.
        dimension_score: Raw score 0–100 for this health dimension.
        passed: True when no ERROR or CRITICAL findings were found.
        execution_time_ms: Wall-clock time for the Doctor's examination, in milliseconds.
        timestamp: When the examination completed.
        metadata: Arbitrary structured data the Doctor wants to surface.
        error: If the Doctor raised an exception, the error message is stored here.
    """

    doctor_name: str
    dimension: str
    findings: List[Finding] = Field(default_factory=list)
    dimension_score: float = Field(default=100.0, ge=0.0, le=100.0)
    passed: bool = True
    execution_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    @property
    def critical_findings(self) -> List[Finding]:
        """Findings with CRITICAL severity."""
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def error_findings(self) -> List[Finding]:
        """Findings with ERROR severity."""
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warning_findings(self) -> List[Finding]:
        """Findings with WARNING severity."""
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def info_findings(self) -> List[Finding]:
        """Findings with INFO severity."""
        return [f for f in self.findings if f.severity == Severity.INFO]

    @property
    def max_severity(self) -> Optional[Severity]:
        """The highest severity among all findings, or None if there are none."""
        if not self.findings:
            return None
        order = [Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.INFO]
        for s in order:
            if any(f.severity == s for f in self.findings):
                return s
        return None

    def add_finding(self, finding: Finding) -> None:
        """Append a finding and update the ``passed`` flag."""
        finding.doctor_name = self.doctor_name
        self.findings.append(finding)
        if finding.severity in (Severity.ERROR, Severity.CRITICAL):
            self.passed = False
