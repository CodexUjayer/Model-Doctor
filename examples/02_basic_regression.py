"""
02 - Basic Regression

Demonstrates ModelDoctor on a regression task. Trains a scikit-learn
Random Forest regressor, runs diagnostics, and prints the findings.
"""

from sklearn.ensemble import RandomForestRegressor
import modeldoctor as md
from utils import get_regression_data


def main():
    print("Generating data and training model...")
    X_train, X_test, y_train, y_test = get_regression_data(random_state=100)

    # Train a standard regressor
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=100)
    model.fit(X_train, y_train)

    print("Running ModelDoctor diagnostics...")
    report = md.diagnose(model, X_train, y_train, X_test, y_test)

    print(f"\nOverall Health Score: {report.health_score.overall:.1f} / 100")
    print(f"Grade: {report.health_score.grade}")

    # Collect all findings across every Doctor's diagnosis
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    flagged = [f for f in all_findings if f.severity.value in ("warning", "critical")]

    if flagged:
        print(f"\nFindings ({len(flagged)} issues):")
        for f in flagged:
            print(f"  [{f.severity.value.upper()}] {f.title}: {f.explanation}")
    else:
        print("\nNo high-severity issues found. Model appears healthy.")

    recs = report.prescription.all_recommendations
    if recs:
        print(f"\nRecommendations ({len(recs)}):")
        for rec in recs:
            print(f"  - {rec.description}")
    
    print("\nRun complete.")


if __name__ == "__main__":
    main()
