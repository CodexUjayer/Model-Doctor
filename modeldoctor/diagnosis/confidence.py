from typing import List, Tuple
from modeldoctor.diagnosis.evidence import DiagnosticEvidence

class ConfidenceEngine:
    """Derives a mathematically sound confidence score based on evidence consensus."""
    
    @staticmethod
    def compute(evidence_list: List[DiagnosticEvidence]) -> Tuple[float, str]:
        """Returns (score 0-100, label)."""
        if not evidence_list:
            return 0.0, "Low"
            
        weight_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Very High": 4.0}
        
        total_score = 0.0
        max_possible = 0.0
        
        # In a real engine, we'd look at signal agreement.
        # For simplicity, we just aggregate the strength of signals.
        for ev in evidence_list:
            w = weight_map.get(ev.contribution_weight, 2.0)
            total_score += w
            max_possible += 4.0 # Max weight
            
        # Confidence scales with number of independent signals too
        num_signals = len(evidence_list)
        base_confidence = (total_score / max_possible) * 100 if max_possible > 0 else 0
        
        # Boost confidence for having multiple signals
        boost = min(30.0, num_signals * 5.0)
        final_score = min(100.0, base_confidence + boost)
        
        if final_score >= 80:
            label = "Very High"
        elif final_score >= 60:
            label = "High"
        elif final_score >= 40:
            label = "Medium"
        else:
            label = "Low"
            
        return final_score, label
