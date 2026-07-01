"""MLflow integration for ModelDoctor."""

from typing import Optional, Dict, Any
import json
import tempfile
from pathlib import Path

from modeldoctor.models.report import Report
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


def log_report_to_mlflow(report: Report, run_id: Optional[str] = None) -> None:
    """Log a ModelDoctor report to the active MLflow run.
    
    Logs:
    - Overall health score as a metric
    - Dimension scores as metrics
    - Passport metadata as tags
    - Full JSON report as an artifact
    - Markdown report as an artifact
    
    Args:
        report: The generated ModelDoctor Report.
        run_id: Optional specific run_id to log to (defaults to active run).
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed. Skipping MLflow logging.")
        return

    client = mlflow.tracking.MlflowClient()
    active_run = mlflow.active_run()
    
    if run_id is None:
        if active_run is None:
            logger.warning("No active MLflow run found. Call mlflow.start_run() first.")
            return
        run_id = active_run.info.run_id

    logger.info(f"Logging ModelDoctor report to MLflow run {run_id}...")

    # Log metrics
    metrics = {"modeldoctor_overall_health": report.health_score.overall}
    for dim in report.health_score.dimensions:
        metrics[f"modeldoctor_{dim.dimension}_score"] = dim.score
        
    for k, v in metrics.items():
        client.log_metric(run_id, k, v)

    # Log tags (passport data)
    tags = {
        "modeldoctor.model_class": report.passport.model_class,
        "modeldoctor.task_type": report.passport.task_type.value if hasattr(report.passport.task_type, "value") else str(report.passport.task_type),
        "modeldoctor.framework": report.passport.framework.value if hasattr(report.passport.framework, "value") else str(report.passport.framework),
        "modeldoctor.health_grade": report.health_score.grade,
    }
    for k, v in tags.items():
        client.log_tag(run_id, k, str(v))

    # Log artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # JSON
        json_path = tmp_path / "modeldoctor_report.json"
        json_path.write_text(report.to_json())
        client.log_artifact(run_id, str(json_path), "modeldoctor")
        
        # Markdown
        try:
            from modeldoctor.reporting.markdown_renderer import MarkdownRenderer
            md_path = tmp_path / "modeldoctor_report.md"
            md_path.write_text(MarkdownRenderer().render(report))
            client.log_artifact(run_id, str(md_path), "modeldoctor")
        except Exception as e:
            logger.warning(f"Failed to log Markdown artifact: {e}")
            
    logger.info("Successfully logged to MLflow.")


# Helper to attach to Report class directly
def attach_mlflow_method():
    """Dynamically attach log_to_mlflow to Report class."""
    if not hasattr(Report, "log_to_mlflow"):
        Report.log_to_mlflow = log_report_to_mlflow

# Attempt to attach it
attach_mlflow_method()
