"""BaseDoctor — abstract base class for all diagnostic modules.

Every Doctor in the system inherits from ``BaseDoctor`` and implements
the ``examine()`` method.  The base class provides:

- Automatic execution timing
- Structured error handling (failed doctors produce an error Diagnosis)
- Pre-examination guard via ``can_examine()``
- Uniform logging
- A set of builder helpers for creating Findings and Recommendations

Example::

    class MyCustomDoctor(BaseDoctor):
        name = "my_custom"
        dimension = "custom_dimension"
        priority = 90

        def examine(self, context: EvaluationContext) -> Diagnosis:
            diagnosis = self._new_diagnosis()
            if some_condition:
                diagnosis.add_finding(
                    self._finding(
                        title="Something is wrong",
                        description="...",
                        severity=Severity.WARNING,
                        evidence=[self._evidence("Metric", 0.42)],
                    )
                )
            diagnosis.dimension_score = self._compute_score(diagnosis)
            return diagnosis
"""

from __future__ import annotations

import abc
import time
import traceback
import uuid
from typing import List, Optional, Union

from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Evidence, Finding
from modeldoctor.models.enums import Confidence, Priority, Severity
from modeldoctor.models.recommendation import Recommendation
from modeldoctor.models.metadata import DoctorMetadata
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


class BaseDoctor(abc.ABC):
    """Abstract base class for all ModelDoctor diagnostic modules.

    Subclasses **must** define:
    - ``name``: unique doctor identifier (snake_case string)
    - ``dimension``: the health dimension this doctor covers
    - ``examine()``: the main diagnostic logic

    Subclasses **may** override:
    - ``priority``: lower numbers run first (default 50)
    - ``can_examine()``: return ``False`` to skip when not applicable
    """

    #: Unique identifier for this doctor (overridden by subclasses).
    name: str = ""
    #: Health dimension this doctor contributes to.
    dimension: str = ""
    #: Lower values run first in the pipeline.
    priority: int = 50

    #: Metadata defining capabilities and requirements.
    metadata: Optional[DoctorMetadata] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            # Fallback for legacy doctors defining name/dimension directly
            if getattr(cls, "metadata", None) is None:
                if not cls.name:
                    raise TypeError(f"{cls.__name__} must define a non-empty 'name' class attribute or 'metadata'.")
                if not cls.dimension:
                    raise TypeError(
                        f"{cls.__name__} must define a non-empty 'dimension' class attribute or 'metadata'."
                    )
                # Auto-generate metadata from legacy fields
                cls.metadata = DoctorMetadata(
                    name=cls.name,
                    priority=getattr(cls, "priority", 50),
                    dimension=cls.dimension
                )
            else:
                # Sync legacy fields to match metadata for backwards compatibility
                cls.name = cls.metadata.name
                cls.priority = cls.metadata.priority
                cls.dimension = cls.metadata.dimension

    @abc.abstractmethod
    def examine(self, context: EvaluationContext) -> Diagnosis:
        """Perform diagnostic examination and return a structured Diagnosis.

        This is the core method that every Doctor must implement.  It should:

        1. Inspect the EvaluationContext for relevant data.
        2. Produce one or more Findings.
        3. Set ``diagnosis.dimension_score`` (0–100).
        4. Return the completed Diagnosis.

        Args:
            context: The shared evaluation context (model + data + cached stats).

        Returns:
            A :class:`Diagnosis` containing all findings for this dimension.
        """
        ...

    def can_examine(self, context: EvaluationContext) -> bool:
        """Return ``True`` if this doctor can run on the given context.

        Override to add task-type or framework guards.  The default
        implementation always returns ``True``.

        Args:
            context: The shared evaluation context.

        Returns:
            ``True`` if examination should proceed.
        """
        return True

    def run(self, context: EvaluationContext) -> Diagnosis:
        """Execute this doctor safely, capturing timing and errors.

        This is called by the pipeline instead of ``examine()`` directly.

        Args:
            context: The shared evaluation context.

        Returns:
            A :class:`Diagnosis`.  On unexpected errors, returns an error
            Diagnosis with ``passed=False`` and the exception message.
        """
        if not self.can_examine(context):
            logger.debug("Doctor '%s' skipped (can_examine=False).", self.name)
            diag = self._new_diagnosis()
            diag.metadata["skipped"] = True
            diag.metadata["skip_reason"] = "Not applicable for this model/task."
            return diag

        logger.info("Running doctor: %s", self.name)
        t0 = time.perf_counter()
        try:
            diagnosis = self.examine(context)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            tb = traceback.format_exc()
            logger.error("Doctor '%s' failed: %s\n%s", self.name, exc, tb)
            diag = self._new_diagnosis()
            diag.passed = False
            diag.error = f"{type(exc).__name__}: {exc}"
            diag.execution_time_ms = elapsed
            diag.metadata["traceback"] = tb
            return diag
        else:
            elapsed = (time.perf_counter() - t0) * 1000
            diagnosis.execution_time_ms = elapsed
            logger.info(
                "Doctor '%s' completed in %.1f ms | score=%.1f | findings=%d",
                self.name,
                elapsed,
                diagnosis.dimension_score,
                len(diagnosis.findings),
            )
            return diagnosis

    # ------------------------------------------------------------------
    # Builder helpers — use these inside examine() for consistency
    # ------------------------------------------------------------------

    def _new_diagnosis(self) -> Diagnosis:
        """Create a fresh Diagnosis pre-populated with this doctor's metadata."""
        return Diagnosis(
            doctor_name=self.name,
            dimension=self.dimension,
            dimension_score=100.0,
            passed=True,
        )

    def _finding(
        self,
        title: str,
        severity: Severity,
        explanation: Optional[str] = None,
        description: Optional[str] = None,
        evidence: Optional[Union[Dict[str, Any], List[Evidence]]] = None,
        affected_components: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Finding:
        """Convenience factory for :class:`Finding` objects."""
        # Support legacy description argument
        expl = explanation if explanation is not None else description
        if expl is None:
            expl = ""
            
        return Finding(
            title=title,
            explanation=expl,
            severity=severity,
            evidence=evidence or {},
            affected_components=affected_components or [],
            tags=tags or [],
            doctor_name=self.name,
        )

    def _evidence(
        self,
        label: str,
        value: object,
        expected: Optional[object] = None,
        unit: Optional[str] = None,
    ) -> Evidence:
        """Convenience factory for :class:`Evidence` objects."""
        return Evidence(label=label, value=value, expected=expected, unit=unit)

    def _recommendation(
        self,
        rationale: str,
        title: str = "Recommendation",
        description: Optional[str] = None,
        action: Optional[str] = None,
        confidence: Confidence = Confidence.MEDIUM,
        priority: Priority = Priority.MEDIUM,
        estimated_improvement: Optional[str] = None,
        estimated_impact: Optional[str] = None,
        implementation_difficulty: str = "Medium",
        supporting_evidence: Optional[Dict[str, Any]] = None,
        affected_metrics: Optional[List[str]] = None,
        code_example: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_finding_ids: Optional[List[str]] = None,
    ) -> Recommendation:
        """Convenience factory for :class:`Recommendation` objects."""
        desc = description if description is not None else action
        if desc is None:
            desc = "Update model configuration"
            
        est_imp = estimated_improvement if estimated_improvement is not None else estimated_impact
        if est_imp is None:
            est_imp = ""
            
        return Recommendation(
            title=title,
            description=desc,
            rationale=rationale,
            confidence=confidence,
            priority=priority,
            estimated_improvement=est_imp,
            implementation_difficulty=implementation_difficulty,
            supporting_evidence=supporting_evidence or {},
            affected_metrics=affected_metrics or [],
            code_example=code_example,
            tags=tags or [],
            source_finding_ids=source_finding_ids or [],
        )

    def _score_from_findings(self, diagnosis: Diagnosis, base: float = 100.0) -> float:
        """Compute a dimension score by deducting penalty weights from findings.

        Args:
            diagnosis: The diagnosis containing findings.
            base: Starting score (default 100).

        Returns:
            Clamped score in [0, 100].
        """
        penalty = sum(f.severity.weight for f in diagnosis.findings)
        return max(0.0, min(100.0, base - penalty))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, priority={self.priority})"
