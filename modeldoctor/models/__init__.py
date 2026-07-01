"""Models package — public re-exports."""

from modeldoctor.models.diagnosis import Diagnosis, Evidence, Finding
from modeldoctor.models.enums import (
    Confidence,
    ExplainabilityMode,
    FrameworkType,
    Priority,
    ReportFormat,
    Severity,
    TaskType,
)
from modeldoctor.models.health import DIMENSION_WEIGHTS, DimensionScore, HealthScore
from modeldoctor.models.recommendation import (
    PrescriptionResult,
    PrescriptionRule,
    Recommendation,
)
from modeldoctor.models.report import (
    ExplainabilityInfo,
    ModelPassport,
    Report,
    SectionReview,
)

__all__ = [
    # Diagnosis
    "Evidence",
    "Finding",
    "Diagnosis",
    # Enums
    "Severity",
    "Priority",
    "Confidence",
    "TaskType",
    "FrameworkType",
    "ExplainabilityMode",
    "ReportFormat",
    # Health
    "DimensionScore",
    "HealthScore",
    "DIMENSION_WEIGHTS",
    # Recommendation
    "Recommendation",
    "PrescriptionRule",
    "PrescriptionResult",
    # Report
    "ModelPassport",
    "ExplainabilityInfo",
    "SectionReview",
    "Report",
]
