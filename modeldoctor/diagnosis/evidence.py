import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class DiagnosticEvidence(BaseModel):
    """A rich, structured piece of evidence supporting a finding."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    measured_value: Any
    expected_range: Optional[str] = None
    normalized_score: float = 0.0
    contribution_weight: str = "Medium" # Low, Medium, High, Very High
    confidence_contribution: float = 0.0
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EvidenceBuilder:
    """Utility for collecting signals and building Evidence."""
    def __init__(self):
        self.evidence: List[DiagnosticEvidence] = []
        
    def add(self, name: str, measured_value: Any, weight: str = "Medium", normalized_score: float = 0.0, **kwargs) -> None:
        self.evidence.append(DiagnosticEvidence(
            name=name, measured_value=measured_value, contribution_weight=weight, normalized_score=normalized_score, **kwargs
        ))
        
    def get_all(self) -> List[DiagnosticEvidence]:
        return self.evidence
