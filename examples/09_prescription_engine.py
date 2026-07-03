"""
09 - Prescription Engine

Demonstrates how ModelDoctor generates actionable prescriptions for
identified issues, providing the exact engineering steps required to fix them.
"""

from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md
from utils import get_overfit_data


def main():
    print("Generating an overfitted model dataset...")
    # get_overfit_data generates a complex, noisy dataset that makes
    # unconstrained models prone to memorizing training noise.
    X_train, X_test, y_train, y_test = get_overfit_data(random_state=42)

    # Unrestricted max_depth combined with high feature noise causes heavy overfitting
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    print("Running ModelDoctor diagnostics...")
    report = md.diagnose(model, X_train, y_train, X_test, y_test)

    print(f"\nOverall Health Score: {report.health_score.overall:.1f} / 100")

    # Collect high-severity findings from all Doctors
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    critical_warnings = [f for f in all_findings if f.severity.value in ("warning", "critical")]

    print(f"\nFindings: {len(critical_warnings)} issues detected")
    for f in critical_warnings:
        print(f"  [{f.severity.value.upper()}] {f.title}")

    print("\n--- Generated Prescriptions ---")
    recs = report.prescription.all_recommendations
    if not recs:
        print("No prescriptions generated (model passed all checks).")
        return

    for i, rec in enumerate(recs, 1):
        print(f"\nPrescription {i}: {rec.description}")
        print(f"  Rationale: {rec.rationale}")
        print(f"  Estimated Gain: {rec.estimated_improvement}")
        print(f"  Confidence: {rec.confidence.value}")


if __name__ == "__main__":
    main()
