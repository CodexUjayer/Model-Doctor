"""Benchmark runner for validating doctors."""

from typing import Any, Dict, List

from modeldoctor.core.context import EvaluationContext
from modeldoctor.core.pipeline import DiagnosticPipeline
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """Runs a model and dataset through the pipeline to validate against expected findings."""

    def __init__(self, pipeline: DiagnosticPipeline = None):
        self.pipeline = pipeline or DiagnosticPipeline()

    def run_benchmark(self, model: Any, X: Any, y: Any, expected: Dict[str, bool]) -> Dict[str, bool]:
        """Run benchmark and compare findings to expected flags.
        
        Args:
            model: Trained model.
            X: Data features.
            y: Data labels.
            expected: Dict mapping dimension names to boolean (e.g. {"overfitting": True}).
            
        Returns:
            Dict of evaluation results (dimension -> True if matched expected).
        """
        logger.info(f"Running benchmark with expected findings: {expected}")
        
        context = EvaluationContext(
            model=model,
            X_train=X,
            y_train=y,
        )
        
        diagnoses = self.pipeline.run(context)
        results = {}
        
        # Check diagnoses
        for dimension, should_trigger in expected.items():
            diag = next((d for d in diagnoses if d.dimension.lower() == dimension.lower()), None)
            
            if diag is None:
                results[dimension] = not should_trigger
                continue
                
            has_findings = len(diag.findings) > 0
            # Also consider failure or severe score
            is_triggered = has_findings or diag.dimension_score < 80.0
            
            results[dimension] = (is_triggered == should_trigger)
            if results[dimension]:
                logger.info(f"PASS: {dimension} correctly {'triggered' if should_trigger else 'ignored'}")
            else:
                logger.error(f"FAIL: {dimension} should be {should_trigger} but was {is_triggered}")
                
        return results
