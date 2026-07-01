"""PrescriptionEngine — evaluates rules against the EvaluationContext."""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.recommendation import Recommendation, PrescriptionResult
from modeldoctor.prescription.rules import PrescriptionRule
from modeldoctor.prescription.knowledge_base import KnowledgeBase
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


class PrescriptionEngine:
    """Evaluates YAML rules to generate prescriptive recommendations."""

    def __init__(self, rules_dir: str | Path | None = None, kb_dir: str | Path | None = None) -> None:
        self.rules: List[PrescriptionRule] = []
        self.knowledge_base = KnowledgeBase(kb_dir)
        if rules_dir is None:
            rules_dir = Path(__file__).parent / "rules"
        self._load_rules(Path(rules_dir))

    def _load_rules(self, rules_dir: Path) -> None:
        if not rules_dir.exists():
            logger.warning(f"Rules directory not found: {rules_dir}")
            return
            
        for file in rules_dir.glob("*.yaml"):
            try:
                with open(file, "r") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, list):
                        for rule_data in data:
                            self.rules.append(PrescriptionRule.parse_obj(rule_data))
            except Exception as e:
                logger.error(f"Failed to load rules from {file}: {e}")
                
        logger.info(f"Loaded {len(self.rules)} prescription rules.")

    def prescribe(self, context: EvaluationContext, diagnoses: List[Diagnosis]) -> List[PrescriptionResult]:
        """Evaluate rules and return prescriptions."""
        # Simple proof-of-concept evaluator
        
        # Build evaluation context dictionary
        eval_dict: Dict[str, Any] = {
            "model_type": type(context.model).__name__ if context.model else "Unknown",
        }
        
        # Extract features from diagnoses if needed
        # (For now, we'll just evaluate rules based on model type)
        
        results = []
        for rule in self.rules:
            if self._evaluate_conditions(rule, eval_dict):
                recs = []
                # Check for knowledge base entry
                kb_entry = None
                if rule.knowledge_id:
                    kb_entry = self.knowledge_base.get_entry(rule.knowledge_id)
                
                for i, r in enumerate(rule.recommendations):
                    rec = Recommendation(
                        id=f"{rule.id}_{i}",
                        title=kb_entry.prescription if kb_entry else "Recommendation",
                        description=r.action,
                        rationale=kb_entry.rationale if kb_entry else r.rationale,
                        confidence=kb_entry.confidence_level if kb_entry else r.confidence,
                        priority=r.priority,
                        estimated_improvement=r.estimated_impact,
                        references=kb_entry.references if kb_entry else [],
                    )
                    recs.append(rec)
                    
                results.append(
                    PrescriptionResult(
                        rule_id=rule.id,
                        root_cause=kb_entry.diagnosis if kb_entry else rule.root_cause,
                        recommendations=recs
                    )
                )
        
        return results

    def _evaluate_conditions(self, rule: PrescriptionRule, eval_dict: Dict[str, Any]) -> bool:
        for condition in rule.conditions:
            val = eval_dict.get(condition.field)
            if condition.operator == "equals" and val != condition.value:
                return False
            # (Add other operators as needed)
        return True
