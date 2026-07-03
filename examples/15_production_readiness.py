"""
15 - Production Readiness

Evaluates whether a model is physically suited for a production deployment
environment by examining its serialized size, inference latency, and
memory footprint.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md
from utils import get_classification_data


def main():
    print("Generating data and training model...")
    X_train, X_test, y_train, y_test = get_classification_data(random_state=42)

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Inject a large array to bloat the serialized model size.
    # This simulates a common production mistake where training data or
    # intermediate results are accidentally stored on the model object.
    model.dummy_large_array_ = np.ones((50_000, 1_000))

    print("Running ModelDoctor diagnostics...")
    report = md.diagnose(model, X_train, y_train, X_test, y_test)

    print(f"\nOverall Health Score: {report.health_score.overall:.1f} / 100")

    # Collect production-related findings from the ProductionDoctor
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    prod_findings = [f for f in all_findings if f.doctor_name == "ProductionDoctor"]

    if prod_findings:
        print("\nProduction Readiness Findings:")
        for f in prod_findings:
            print(f"  [{f.severity.value.upper()}] {f.title}")
            print(f"  {f.explanation}")
            if f.structured_evidence:
                print("  Evidence:")
                for ev in f.structured_evidence:
                    # structured_evidence items are DiagnosticEvidence Pydantic objects
                    name = getattr(ev, 'name', getattr(ev, 'signal_name', '?'))
                    measured = getattr(ev, 'measured_value', getattr(ev, 'value', '?'))
                    expected = getattr(ev, 'expected_range', getattr(ev, 'threshold', '?'))
                    print(f"    - {name}: {measured} (Expected: {expected})")
    else:
        print("\nNo production readiness issues detected.")

    print(f"\nModel Passport:")
    print(f"  Serialized Size: {report.passport.model_size_bytes:,} bytes")
    print(f"  Framework:       {report.passport.framework}")
    print(f"  Task Type:       {report.passport.task_type}")


if __name__ == "__main__":
    main()
