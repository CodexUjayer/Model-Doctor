"""Top-level report data models: ModelPassport and Report.

A ``Report`` is the final artifact produced by the full diagnostic pipeline.
It bundles all diagnoses, prescriptions, health scores, and generated prose
into a single structured object that can be serialised to Markdown or HTML.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.enums import FrameworkType, ReportFormat, TaskType
from modeldoctor.models.health import HealthScore
from modeldoctor.models.recommendation import PrescriptionResult, Recommendation


class ModelPassport(BaseModel):
    """Metadata about the model being examined.

    Attributes:
        model_class: Python class name (e.g., ``"RandomForestClassifier"``).
        framework: The ML framework/library.
        task_type: The inferred ML task.
        n_features: Number of input features.
        n_train_samples: Training set size.
        n_test_samples: Test set size.
        hyperparameters: Model hyperparameters as a dict.
        feature_names: Optional list of feature names.
        model_size_bytes: Serialised model size in bytes.
        has_predict_proba: Whether the model supports probability output.
        has_feature_importances: Whether the model exposes feature importances.
        framework_version: Version string of the framework (if detectable).
        extra: Any additional model metadata.
    """

    model_config = {"protected_namespaces": ()}

    model_class: str
    framework: FrameworkType = FrameworkType.UNKNOWN
    task_type: TaskType = TaskType.UNKNOWN
    n_features: int = 0
    n_train_samples: int = 0
    n_test_samples: int = 0
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_names: Optional[List[str]] = None
    model_size_bytes: int = 0
    has_predict_proba: bool = False
    has_feature_importances: bool = False
    framework_version: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)


class ExplainabilityInfo(BaseModel):
    """Summary of explainability analysis performed (or skipped).

    Attributes:
        enabled: Whether SHAP/LIME was run.
        mode: The explainability mode used.
        shap_available: Whether the shap package was importable.
        method: Which method was used (``"shap"`` | ``"lime"`` | ``"permutation"`` | ``None``).
        top_features: Top N features by importance (feature_name → mean |SHAP|).
        skip_reason: If skipped, why.
    """

    enabled: bool = False
    mode: str = "auto"
    shap_available: bool = False
    method: Optional[str] = None
    top_features: Dict[str, float] = Field(default_factory=dict)
    skip_reason: Optional[str] = None


class SectionReview(BaseModel):
    """Prose review for a specific report section.

    Attributes:
        section: Section identifier.
        title: Display title.
        summary: Main prose summary.
        key_points: Bullet-point highlights.
        generated_by: ``"deterministic"`` or ``"ai"`` and model name.
    """

    section: str
    title: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    generated_by: str = "deterministic"


class Report(BaseModel):
    """The complete diagnostic report for a model.

    This is the top-level artifact produced by ``modeldoctor.diagnose()``.

    Attributes:
        passport: Model metadata.
        health_score: Aggregated health score.
        diagnoses: Per-doctor diagnosis objects.
        prescription: Output from the prescription engine.
        top_recommendations: Priority-sorted de-duplicated recommendations.
        section_reviews: Prose reviews for each report section.
        explainability: Explainability analysis summary.
        executive_summary: High-level prose summary of the entire review.
        generated_at: Timestamp of report generation.
        modeldoctor_version: Library version that produced this report.
        metadata: Free-form metadata (e.g., run config, environment info).
        pipeline_summary: Summary of the pipeline execution.
    """

    passport: ModelPassport
    health_score: HealthScore
    diagnoses: List[Diagnosis] = Field(default_factory=list)
    prescription: PrescriptionResult = Field(default_factory=PrescriptionResult)
    top_recommendations: List[Recommendation] = Field(default_factory=list)
    section_reviews: List[SectionReview] = Field(default_factory=list)
    explainability: ExplainabilityInfo = Field(default_factory=ExplainabilityInfo)
    executive_summary: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    modeldoctor_version: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pipeline_summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def diagnosis_by_doctor(self, name: str) -> Optional[Diagnosis]:
        """Return the diagnosis produced by a specific doctor."""
        return next((d for d in self.diagnoses if d.doctor_name == name), None)

    def section_review(self, section: str) -> Optional[SectionReview]:
        """Return the prose review for a specific section."""
        return next((s for s in self.section_reviews if s.section == section), None)

    def save(self, path: str | Path, fmt: Optional[ReportFormat] = None) -> Path:
        """Save this report to *path* in the appropriate format.

        The format is inferred from the file extension unless *fmt* is given.

        Args:
            path: Destination file path.
            fmt: Override the output format.

        Returns:
            The resolved :class:`pathlib.Path` to the saved file.

        Raises:
            ValueError: If the format cannot be determined.
            ImportError: If the HTML renderer dependencies are missing.
        """
        from modeldoctor.reporting.markdown_renderer import MarkdownRenderer
        from modeldoctor.reporting.html_renderer import HtmlRenderer

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if fmt is None:
            suffix = dest.suffix.lower()
            fmt_map = {
                ".md": ReportFormat.MARKDOWN,
                ".markdown": ReportFormat.MARKDOWN,
                ".html": ReportFormat.HTML,
                ".htm": ReportFormat.HTML,
                ".json": ReportFormat.JSON,
            }
            fmt = fmt_map.get(suffix)
            if fmt is None:
                raise ValueError(
                    f"Cannot infer report format from extension '{suffix}'. "
                    "Pass fmt= explicitly or use .md / .html / .json."
                )

        if fmt == ReportFormat.MARKDOWN:
            content = MarkdownRenderer().render(self)
            dest.write_text(content, encoding="utf-8")
        elif fmt == ReportFormat.HTML:
            content = HtmlRenderer().render(self)
            dest.write_text(content, encoding="utf-8")
        elif fmt == ReportFormat.JSON:
            dest.write_text(
                self.model_dump_json(indent=2, exclude_none=False),
                encoding="utf-8",
            )

        return dest

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation (JSON-serialisable)."""
        return json.loads(self.model_dump_json())

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return self.model_dump_json(indent=2)
        
    def to_yaml(self) -> str:
        """Return a YAML string representation."""
        try:
            import yaml
            return yaml.dump(self.to_dict(), sort_keys=False)
        except ImportError:
            raise ImportError("pyyaml is required for YAML export. Run: pip install pyyaml")

    def to_dataframe(self) -> Any:
        """Return a pandas DataFrame of the diagnoses and findings."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame export. Run: pip install pandas")
            
        rows = []
        for diag in self.diagnoses:
            for finding in diag.findings:
                rows.append({
                    "dimension": diag.dimension,
                    "doctor": diag.doctor_name,
                    "severity": finding.severity.name,
                    "title": finding.title,
                    "description": finding.description,
                    "evidence": str(finding.evidence)
                })
        return pd.DataFrame(rows)

    def to_pdf(self, path: str | Path) -> Path:
        """Export report as a PDF document."""
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError("weasyprint is required for PDF export. Run: pip install weasyprint")
            
        from modeldoctor.reporting.html_renderer import HtmlRenderer
        html_content = HtmlRenderer().render(self)
        
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        HTML(string=html_content).write_pdf(str(dest))
        return dest

    def export_bundle(self, path: str | Path) -> Path:
        """Export a ZIP bundle containing JSON, Markdown, and HTML reports."""
        dest = Path(path)
        if dest.suffix != ".zip":
            dest = dest.with_suffix(".zip")
            
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("report.json", self.to_json())
            
            from modeldoctor.reporting.markdown_renderer import MarkdownRenderer
            zf.writestr("report.md", MarkdownRenderer().render(self))
            
            from modeldoctor.reporting.html_renderer import HtmlRenderer
            zf.writestr("report.html", HtmlRenderer().render(self))
            
        return dest

    def dashboard(self, port: int = 8080, open_browser: bool = True) -> None:
        """Launch the interactive HTML dashboard."""
        from modeldoctor.reporting.server import launch_dashboard
        launch_dashboard(self.to_dict(), port=port, open_browser=open_browser)

