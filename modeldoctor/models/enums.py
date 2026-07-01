"""Enumeration types used throughout the ModelDoctor framework.

All enums use ``str`` as a mixin so they serialize cleanly with Pydantic v2
and JSON without any custom serializer.
"""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """The machine-learning task the model is solving."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity level of a diagnostic finding.

    Ordered from least to most severe: INFO < WARNING < ERROR < CRITICAL.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        """Numeric weight used when aggregating severity into health scores."""
        return {
            Severity.INFO: 0.0,
            Severity.WARNING: 5.0,
            Severity.ERROR: 15.0,
            Severity.CRITICAL: 30.0,
        }[self]

    def __lt__(self, other: "Severity") -> bool:  # type: ignore[override]
        order = [Severity.INFO, Severity.WARNING, Severity.ERROR, Severity.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: "Severity") -> bool:  # type: ignore[override]
        return self == other or self < other

    def __gt__(self, other: "Severity") -> bool:  # type: ignore[override]
        return not self <= other

    def __ge__(self, other: "Severity") -> bool:  # type: ignore[override]
        return not self < other


class Priority(str, Enum):
    """Priority level of a recommendation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Confidence(str, Enum):
    """Confidence level of a recommendation or finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExplainabilityMode(str, Enum):
    """Controls SHAP/LIME computation behaviour.

    - ``AUTO``: enable if the ``shap`` package is installed, skip silently otherwise.
    - ``FULL``: always attempt; raise ``ImportError`` if ``shap`` is not installed.
    - ``DISABLED``: never run; fastest option.
    """

    AUTO = "auto"
    FULL = "full"
    DISABLED = "disabled"


class FrameworkType(str, Enum):
    """ML framework / library that produced the model."""

    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    UNKNOWN = "unknown"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
