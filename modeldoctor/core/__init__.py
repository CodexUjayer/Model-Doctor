"""Core module containing the base abstractions for ModelDoctor."""

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.core.pipeline import DiagnosticPipeline
from modeldoctor.core.registry import DoctorRegistry

__all__ = [
    "BaseDoctor",
    "EvaluationContext",
    "DiagnosticPipeline",
    "DoctorRegistry",
]
