"""ReviewGenerator — generates AI prose summaries from structured data."""

from modeldoctor.models.report import Report


class ReviewGenerator:
    """Generates prose executive summaries based on the structured report data."""
    
    def generate_executive_summary(self, report: Report) -> str:
        """Generate a basic prose summary from the structured data."""
        
        score = report.health_score.overall
        if score >= 90:
            health_text = "excellent condition"
        elif score >= 70:
            health_text = "good condition but has some minor issues"
        elif score >= 50:
            health_text = "fair condition with several areas for improvement"
        else:
            health_text = "poor condition with critical issues"
            
        total_findings = sum(len(d.findings) for d in report.diagnoses)
        
        summary = (
            f"The model ({report.passport.model_class}) was evaluated on {report.passport.task_type.name} task. "
            f"Overall, the model is in {health_text} with a health score of {score:.1f}/100. "
            f"The diagnostic pipeline identified {total_findings} notable findings."
        )
        
        return summary
