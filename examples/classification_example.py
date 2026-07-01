"""Example usage of ModelDoctor for a classification task."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from modeldoctor.core.context import EvaluationContext
from modeldoctor.core.pipeline import DiagnosticPipeline
from modeldoctor.models.report import Report, ModelPassport
from modeldoctor.models.enums import TaskType
from modeldoctor.prescription.engine import PrescriptionEngine
from modeldoctor.scoring.health_scorer import HealthScorer
from modeldoctor.reporting.markdown_renderer import MarkdownRenderer
from modeldoctor.reporting.review_generator import ReviewGenerator

def main():
    print("1. Generating synthetic data and training model...")
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # create some imbalance
    y_train[:80] = 0
    y_train[80:] = 1
    
    # Intentionally overfit deep tree
    model = RandomForestClassifier(max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    
    print("2. Initializing ModelDoctor EvaluationContext...")
    context = EvaluationContext(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    
    print("3. Running Diagnostic Pipeline...")
    pipeline = DiagnosticPipeline()
    diagnoses = pipeline.run(context)
    
    print("4. Evaluating Prescriptions...")
    engine = PrescriptionEngine()
    prescriptions = engine.prescribe(context, diagnoses)
    
    print("5. Computing Health Score...")
    scorer = HealthScorer()
    health_score = scorer.compute_health(diagnoses)
    
    print("6. Generating Report...")
    passport = ModelPassport(
        model_class="RandomForestClassifier",
        task_type=TaskType.BINARY_CLASSIFICATION,
        extra={"dataset_name": "Synthetic Classification"},
    )
    
    report = Report(
        passport=passport,
        health_score=health_score,
        diagnoses=diagnoses,
        prescriptions=prescriptions,
    )
    
    review_gen = ReviewGenerator()
    report.executive_summary = review_gen.generate_executive_summary(report)
    
    renderer = MarkdownRenderer()
    out_path = "model_report.md"
    renderer.render(report, output_path=out_path)
    
    print(f"\nAnalysis complete! Report saved to {out_path}")
    print(f"Overall Health Score: {health_score.overall:.1f}/100")

if __name__ == "__main__":
    main()
