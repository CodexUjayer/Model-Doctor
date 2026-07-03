from typing import List, Tuple
from modeldoctor.diagnosis.evidence import DiagnosticEvidence

class RiskEngine:
    """Computes a risk/severity score independently of confidence."""
    
    @staticmethod
    def compute(evidence_list: List[DiagnosticEvidence]) -> Tuple[float, str, str]:
        """Returns (score 0-100, level, explanation)."""
        if not evidence_list:
            return 0.0, "INFO", "No evidence provided."
            
        # Simply use the highest normalized score for the primary risk
        max_score = max((ev.normalized_score for ev in evidence_list), default=0.0)
        
        # Scale 0-1 to 0-100
        score = max_score * 100
        
        if score >= 90:
            level = "CRITICAL"
            explanation = "Severe risk detected threatening model integrity."
        elif score >= 70:
            level = "HIGH"
            explanation = "High risk detected requiring immediate attention."
        elif score >= 40:
            level = "MEDIUM"
            explanation = "Moderate risk detected."
        elif score >= 10:
            level = "LOW"
            explanation = "Minor risk detected."
        else:
            level = "INFO"
            explanation = "No significant risk detected."
            
        return score, level, explanation
