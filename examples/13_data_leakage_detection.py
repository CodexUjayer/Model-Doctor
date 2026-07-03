"""
13 - Data Leakage Detection

Demonstrates how ModelDoctor detects subtle proxy target leakage —
a common training error where a feature contains future information
that is unavailable at inference time.
"""

from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md
from utils import get_leaky_data


def main():
    print("Generating a dataset with intentional target leakage...")
    # The 'customer_id_leak' feature in this dataset is strongly correlated
    # with the target, simulating a common leak (e.g., a proxy ID or timestamp).
    X_train, X_test, y_train, y_test = get_leaky_data(random_state=42)

    print(f"Features: {list(X_train.columns)}")
    print("Note: 'customer_id_leak' is constructed from the target label.")

    print("\nTraining a model on the leaky dataset...")
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    print("Running ModelDoctor diagnostics...")
    report = md.diagnose(model, X_train, y_train, X_test, y_test)

    print(f"\nOverall Health Score: {report.health_score.overall:.1f} / 100")

    # Look for findings from the LeakageDoctor
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    leakage_findings = [f for f in all_findings if "Leakage" in f.title or f.doctor_name == "LeakageDoctor"]

    if leakage_findings:
        print("\nLeakage findings detected:")
        for f in leakage_findings:
            print(f"  [{f.severity.value.upper()}] {f.title}")
            print(f"  Explanation: {f.explanation}")
    else:
        print("\nNo leakage findings detected.")

    print("\nWhy this matters:")
    print("  A model trained on a leaky feature produces inflated accuracy in testing")
    print("  but will fail catastrophically when deployed (the leak is unavailable at inference).")


if __name__ == "__main__":
    main()
