"""ModelDoctor public API — the diagnose() entry-point.

Usage::

    import modeldoctor as md

    report = md.diagnose(
        model, X_train, y_train, X_test, y_test
    )
    report.dashboard()
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from modeldoctor.config.settings import ModelDoctorConfig
from modeldoctor.core.comparison import ModelComparisonEngine
from modeldoctor.core.context import EvaluationContext
from modeldoctor.core.explainability import ExplainabilityEngine
from modeldoctor.core.pipeline import DiagnosticPipeline
from modeldoctor.core.registry import DoctorRegistry
from modeldoctor.models.enums import FrameworkType, TaskType
from modeldoctor.models.health import HealthScore, DimensionScore
from modeldoctor.models.recommendation import PrescriptionResult, Recommendation
from modeldoctor.models.report import ExplainabilityInfo, ModelPassport, Report
from modeldoctor.prescription.engine import PrescriptionEngine
from modeldoctor.reporting.review_generator import ReviewGenerator
from modeldoctor.scoring.health_scorer import HealthScorer
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


def diagnose(
    model: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    *,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    config: Optional[ModelDoctorConfig] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Report:
    """Run the full ModelDoctor diagnostic pipeline and return a Report.

    This is the main entry point for ModelDoctor.  It orchestrates:

    1. Building the :class:`EvaluationContext`.
    2. Running all registered :class:`BaseDoctor` modules.
    3. Running the :class:`PrescriptionEngine`.
    4. Computing the :class:`HealthScore`.
    5. Running the :class:`ExplainabilityEngine`.
    6. Assembling and returning the :class:`Report`.

    Args:
        model: A fitted scikit-learn compatible estimator.
        X_train: Training feature matrix (numpy array or pandas DataFrame).
        y_train: Training target array.
        X_test: Test/hold-out feature matrix.
        y_test: Test/hold-out target array.
        X_val: Optional validation feature matrix.
        y_val: Optional validation target array.
        feature_names: Optional list of feature names.
        config: Optional :class:`ModelDoctorConfig` for tuning behaviour.
        progress_callback: Optional callable for progress reporting.

    Returns:
        A fully populated :class:`Report` object.
    """
    cfg = config or ModelDoctorConfig()

    def _notify(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # --- 1. Build EvaluationContext ---
    _notify("Preparing EvaluationContext...")
    context = EvaluationContext(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        config=cfg,
    )

    # --- 2. Run Diagnostic Pipeline ---
    _notify("Running Doctors...")
    registry = DoctorRegistry.default()
    pipeline = DiagnosticPipeline(registry=registry)
    diagnoses = pipeline.run(context, progress_callback=progress_callback)

    # --- 3. Prescription Engine ---
    _notify("Generating Prescriptions...")
    prescription_engine = PrescriptionEngine()
    prescription_results = prescription_engine.prescribe(context, diagnoses)

    # Flatten all recommendations from prescription results
    all_recs: List[Recommendation] = []
    for pr in prescription_results:
        all_recs.extend(pr.recommendations)

    # Wrap in a single PrescriptionResult for the Report
    prescription = PrescriptionResult(
        fired_rules=[],
        all_recommendations=all_recs,
        rules_evaluated=len(prescription_engine.rules),
        rules_fired=len(prescription_results),
    )

    # --- 4. Health Score ---
    _notify("Computing Health Score...")
    scorer = HealthScorer()
    health_score = _compute_health(diagnoses, scorer)

    # --- 5. Explainability ---
    _notify("Computing Explainability...")
    explainability_engine = ExplainabilityEngine()
    explain_info = explainability_engine.run(context)

    # --- 6. Build Model Passport ---
    from modeldoctor.utils.compat import has_predict_proba, get_feature_importances
    passport = ModelPassport(
        model_class=context.model_class,
        framework=context.framework,
        framework_version=context.framework_ver,
        task_type=context.task_type,
        n_features=context.n_features,
        n_train_samples=context.n_train,
        n_test_samples=context.n_test,
        hyperparameters=context.model_params,
        feature_names=context.feature_names,
        model_size_bytes=context.model_size,
        has_predict_proba=has_predict_proba(model),
        has_feature_importances=get_feature_importances(model) is not None,
    )

    # --- 7. Comparison ---
    comparison_results = None
    if cfg.compare_models:
        _notify("Running Model Comparison...")
        comparison_engine = ModelComparisonEngine()
        comparison_results = comparison_engine.run(context)

    # --- 8. Assemble Report ---
    _notify("Building Report...")
    import modeldoctor
    pipeline_summary_dict = None
    if pipeline.last_summary:
        pipeline_summary_dict = pipeline.last_summary.model_dump()

    report = Report(
        passport=passport,
        health_score=health_score,
        diagnoses=diagnoses,
        prescription=prescription,
        top_recommendations=sorted(
            all_recs,
            key=lambda r: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                r.priority.value if hasattr(r.priority, "value") else str(r.priority).lower(), 99
            ),
        )[:10],
        explainability=explain_info,
        modeldoctor_version=modeldoctor.__version__,
        pipeline_summary=pipeline_summary_dict,
    )

    # --- 9. Executive Summary ---
    try:
        review_gen = ReviewGenerator()
        report.executive_summary = review_gen.generate_executive_summary(report)
    except Exception as exc:
        logger.warning("ReviewGenerator failed: %s", exc)
        report.executive_summary = (
            f"ModelDoctor analysis complete. "
            f"Overall health: {health_score.overall:.1f}/100 ({health_score.grade}). "
            f"{len(diagnoses)} doctors executed, "
            f"{len(all_recs)} recommendations generated."
        )

    # Attach comparison results to report metadata
    if comparison_results:
        report.metadata["comparison"] = [r.model_dump() for r in comparison_results]

    _notify("Analysis complete.")
    return report


def _compute_health(diagnoses, scorer: HealthScorer) -> HealthScore:
    """Delegate to HealthScorer, handling empty diagnoses gracefully."""
    if not diagnoses:
        return HealthScore(overall=0.0, dimensions=[])
    return scorer.compute_health(diagnoses)
