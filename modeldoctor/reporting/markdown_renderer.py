"""MarkdownRenderer — renders reports as Markdown."""

from pathlib import Path

from modeldoctor.models.report import Report
from modeldoctor.reporting.base_renderer import BaseRenderer
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


class MarkdownRenderer(BaseRenderer):
    """Renders ModelDoctor reports to Markdown format."""

    def render(self, report: Report, output_path: str | Path | None = None) -> str:
        lines = []
        
        # Header
        lines.append(f"# ModelDoctor Report: {report.passport.model_class}")
        lines.append(f"**Task Type**: {report.passport.task_type.name}")
        lines.append("")
        
        # Health Score
        lines.append("## Health Score")
        lines.append(f"**Overall Health**: {report.health_score.overall:.1f}/100")
        lines.append("")
        for ds in report.health_score.dimension_scores:
            lines.append(f"- **{ds.dimension}**: {ds.score:.1f}/100")
        lines.append("")
        
        # Executive Summary
        if report.executive_summary:
            lines.append("## Executive Summary")
            lines.append(report.executive_summary)
            lines.append("")
            
        # Findings
        lines.append("## Key Findings")
        for diag in report.diagnoses:
            if not diag.findings:
                continue
            lines.append(f"### {diag.dimension}")
            for finding in diag.findings:
                lines.append(f"- **[{finding.severity.name}] {finding.title}**: {finding.description}")
        lines.append("")
        
        # Recommendations
        if report.prescriptions:
            lines.append("## Recommendations")
            for result in report.prescriptions:
                lines.append(f"### Cause: {result.root_cause}")
                for rec in result.recommendations:
                    lines.append(f"- **{rec.action}** (Impact: {rec.estimated_impact})")
                    lines.append(f"  *Rationale*: {rec.rationale}")
        
        content = "\n".join(lines)
        
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved Markdown report to {out_path}")
            
        return content
