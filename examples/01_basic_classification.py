"""
01 - Basic Classification

The "Hello World" of ModelDoctor. Trains a scikit-learn Random Forest
classifier, runs the full diagnostic pipeline, and prints the results.
"""

from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md
from utils import get_classification_data


def print_report_summary(report):
    """Print the key sections of a ModelDoctor report."""
    print(f"\nOverall Health Score: {report.health_score.overall:.1f} / 100")
    print(f"Grade: {report.health_score.grade}")

    # Collect all findings across every Doctor's diagnosis
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    
    flagged = [f for f in all_findings if f.severity.value in ("warning", "critical")]
    if flagged:
        print(f"\nFindings ({len(flagged)} issues flagged):")
        for f in flagged:
            print(f"  [{f.severity.value.upper()}] {f.title}")
            print(f"    {f.explanation}")
    else:
        print("\nNo high-severity issues found.")

    # Prescriptions live in report.prescription.all_recommendations
    recs = report.prescription.all_recommendations
    if recs:
        print(f"\nRecommendations ({len(recs)}):")
        for rec in recs:
            print(f"  - {rec.description}")
            print(f"    Estimated gain: {rec.estimated_improvement}")
    else:
        print("\nNo prescriptions generated.")


def main():
    print("Generating data and training model...")
    X_train, X_test, y_train, y_test = get_classification_data(random_state=42)

    # Train a standard Random Forest classifier
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    print("Running ModelDoctor diagnostics...")
    report = md.diagnose(model, X_train, y_train, X_test, y_test)

    print_report_summary(report)
    print("\nRun complete.")


if __name__ == "__main__":
    main()
