"""Diagnostic Pipeline — orchestrates the execution of multiple doctors."""

from __future__ import annotations

import importlib.util
import time
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from modeldoctor.core.context import EvaluationContext
from modeldoctor.core.registry import DoctorRegistry
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineSummary(BaseModel):
    """Summary of the pipeline execution."""
    
    executed: List[str] = Field(default_factory=list)
    skipped: Dict[str, str] = Field(default_factory=dict)
    total_time_ms: float = 0.0
    doctor_times_ms: Dict[str, float] = Field(default_factory=dict)


class DiagnosticPipeline:
    """Orchestrates running multiple doctors on an EvaluationContext.

    Args:
        registry: The DoctorRegistry to use. If not provided, uses the default
            registry containing all built-in doctors.
    """

    def __init__(self, registry: Optional[DoctorRegistry] = None) -> None:
        self.registry = registry or DoctorRegistry.default()
        self.last_summary: Optional[PipelineSummary] = None

    def run(
        self, 
        context: EvaluationContext,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Diagnosis]:
        """Run all registered doctors on the given context.

        Doctors are instantiated and executed in the order of their `priority`.

        Args:
            context: The shared evaluation context.
            progress_callback: Optional callback receiving progress messages.

        Returns:
            A list of :class:`Diagnosis` objects, one for each executed doctor.
        """
        logger.info(f"Starting diagnostic pipeline with {len(self.registry)} registered doctors.")
        t0 = time.perf_counter()

        doctors = self.registry.instantiate_all()
        diagnoses = []
        summary = PipelineSummary()

        for doctor in doctors:
            # 1. Check can_examine override
            if not doctor.can_examine(context):
                reason = "can_examine() returned False"
                logger.debug(f"Skipping {doctor.name}: {reason}")
                summary.skipped[doctor.name] = reason
                continue
                
            # 2. Check metadata constraints
            if doctor.metadata:
                md = doctor.metadata
                # Check task
                if md.supported_tasks and context.task_type.value not in md.supported_tasks:
                    reason = f"Unsupported task type: {context.task_type.value}"
                    logger.debug(f"Skipping {doctor.name}: {reason}")
                    summary.skipped[doctor.name] = reason
                    continue
                    
                # Check model family
                if md.supported_model_types and context.profile.model_family not in md.supported_model_types:
                    reason = f"Unsupported model family: {context.profile.model_family}"
                    logger.debug(f"Skipping {doctor.name}: {reason}")
                    summary.skipped[doctor.name] = reason
                    continue
                    
                # Check dependencies
                missing_deps = [dep for dep in md.dependencies if importlib.util.find_spec(dep) is None]
                if missing_deps:
                    reason = f"Missing dependencies: {', '.join(missing_deps)}"
                    logger.info(f"Skipping {doctor.name}: {reason}")
                    summary.skipped[doctor.name] = reason
                    continue
            
            if progress_callback:
                progress_callback(f"Running {doctor.name}...")
                
            d_t0 = time.perf_counter()
            diagnosis = doctor.run(context)
            d_elapsed = (time.perf_counter() - d_t0) * 1000
            
            summary.executed.append(doctor.name)
            summary.doctor_times_ms[doctor.name] = d_elapsed
            diagnoses.append(diagnosis)

        elapsed = (time.perf_counter() - t0) * 1000
        summary.total_time_ms = elapsed
        self.last_summary = summary
        
        logger.info(f"Pipeline completed in {elapsed:.1f} ms. Generated {len(diagnoses)} diagnoses. Skipped {len(summary.skipped)}.")
        
        return diagnoses
