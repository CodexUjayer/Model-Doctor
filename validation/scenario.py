"""Validation Scenario Base Class."""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class ExpectedResult:
    """Expected diagnostic output for a validation scenario."""
    # List of substrings that must appear in the findings' titles or descriptions
    findings: List[str] = field(default_factory=list)
    # List of finding severities that must be present (e.g., ["critical", "warning"])
    severity: List[str] = field(default_factory=list)
    # Expected confidence level strings ["HIGH", "MEDIUM", "LOW"]
    confidence: List[str] = field(default_factory=list)
    # Expected health score range for this dimension (min, max)
    health_score: Optional[Tuple[float, float]] = None
    # Substrings expected in the recommendations
    recommendations: List[str] = field(default_factory=list)
    # Ensure no CRITICAL/ERROR findings if it's supposed to be healthy
    passed: Optional[bool] = None

class ValidationScenario(abc.ABC):
    """Abstract base class for all Validation Scenarios."""

    name: str = ""
    category: str = ""
    description: str = ""
    random_seed: int = 42

    @abc.abstractmethod
    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        """Build and return (X_train, y_train, X_test, y_test)."""
        ...

    @abc.abstractmethod
    def build_model(self, X_train: Any, y_train: Any) -> Any:
        """Build, train, and return the model."""
        ...

    @abc.abstractmethod
    def expected(self) -> ExpectedResult:
        """Return the ExpectedResult for this scenario."""
        ...
