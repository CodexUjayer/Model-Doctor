"""ModelDoctor — A production-grade model diagnostics library.

Usage::

    import modeldoctor as md

    report = md.diagnose(model, X_train, y_train, X_test, y_test)
    report.dashboard()
"""

__version__ = "0.2.0"

# Public API
from modeldoctor.api import diagnose  # noqa: F401

# Core building blocks (for advanced usage)
from modeldoctor.core.base_doctor import BaseDoctor  # noqa: F401
from modeldoctor.core.context import EvaluationContext  # noqa: F401
from modeldoctor.core.pipeline import DiagnosticPipeline  # noqa: F401
from modeldoctor.core.registry import DoctorRegistry  # noqa: F401
from modeldoctor.core.explainability import ExplainabilityEngine  # noqa: F401
from modeldoctor.core.comparison import ModelComparisonEngine  # noqa: F401
from modeldoctor.config.settings import ModelDoctorConfig  # noqa: F401

# Report model
from modeldoctor.models.report import Report, ModelPassport  # noqa: F401

__all__ = [
    "__version__",
    # Top-level API
    "diagnose",
    # Core
    "BaseDoctor",
    "EvaluationContext",
    "DiagnosticPipeline",
    "DoctorRegistry",
    "ExplainabilityEngine",
    "ModelComparisonEngine",
    "ModelDoctorConfig",
    # Report
    "Report",
    "ModelPassport",
]
