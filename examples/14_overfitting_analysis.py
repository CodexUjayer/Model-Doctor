"""
14 - Overfitting Analysis

Demonstrates how ModelDoctor detects complex models that have memorized
training noise and fail to generalize to unseen data.
"""

from sklearn.tree import DecisionTreeClassifier
import modeldoctor as md
from utils import get_overfit_data


def main():
    print("Generating a noisy dataset...")
    # get_overfit_data creates a high-dimensional dataset with label noise,
    # making it difficult for unconstrained models to generalize.
    X_train, X_test, y_train, y_test = get_overfit_data(random_state=42)

    print("Training an unconstrained Decision Tree (max_depth=None)...")
    model = DecisionTreeClassifier(max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy:  {test_acc:.3f}")
    print(f"  Generalization Gap: {train_acc - test_acc:.3f}")

    print("\nRunning ModelDoctor diagnostics...")
    report = md.diagnose(model, X_train, y_train, X_test, y_test)

    print(f"\nOverall Health Score: {report.health_score.overall:.1f} / 100")

    # Gather overfitting-related findings
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    overfit_findings = [f for f in all_findings if f.doctor_name in ("OverfittingDoctor", "hyperparameter_doctor")]

    print("\nOverfitting Findings:")
    if overfit_findings:
        for f in overfit_findings:
            print(f"  [{f.severity.value.upper()}] {f.title}")
            print(f"  {f.explanation}")
    else:
        print("  No overfitting findings for this scenario.")

    print("\nModelDoctor examines:")
    print("  - The generalization gap (train accuracy vs. test accuracy).")
    print("  - Memorization rate on noisy label data.")
    print("  - Unconstrained hyperparameters (e.g., max_depth=None).")


if __name__ == "__main__":
    main()
