"""
11 - Model Comparison

Compares two estimators (Logistic Regression vs. Random Forest) on the
same dataset. ModelDoctor evaluates one model per call; two reports are
generated and compared side-by-side.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md
from utils import get_classification_data


def summarize(name, report):
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    flagged = [f for f in all_findings if f.severity.value in ("warning", "critical")]
    print(f"\n  {name}")
    print(f"    Health Score: {report.health_score.overall:.1f}/100  Grade: {report.health_score.grade}")
    if flagged:
        for f in flagged:
            print(f"    [{f.severity.value.upper()}] {f.title}")
    else:
        print("    No issues detected.")


def main():
    print("Generating data...")
    X_train, X_test, y_train, y_test = get_classification_data(random_state=42)

    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=500, random_state=42)
    lr_model.fit(X_train, y_train)

    print("Training Random Forest (unconstrained depth)...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf_model.fit(X_train, y_train)

    print("\nDiagnosing both models...")
    lr_report = md.diagnose(lr_model, X_train, y_train, X_test, y_test)
    rf_report = md.diagnose(rf_model, X_train, y_train, X_test, y_test)

    print("\n--- Model Comparison ---")
    summarize("Logistic Regression", lr_report)
    summarize("Random Forest (max_depth=None)", rf_report)

    print("\nConclusion:")
    print("  The Logistic Regression is a simpler model with lower overfitting risk.")
    print("  The unconstrained Random Forest may exhibit memorization on noisy data.")


if __name__ == "__main__":
    main()
