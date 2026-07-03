"""
12 - Probability Calibration

Demonstrates how ModelDoctor detects poorly calibrated probability
estimates using Expected Calibration Error (ECE). Also shows how
Platt Scaling improves calibration scores.
"""

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import modeldoctor as md
from utils import get_classification_data


def calibration_summary(name, report):
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    cal_findings = [f for f in all_findings if "Calibration" in f.title]
    print(f"\n  {name} — Health: {report.health_score.overall:.1f}/100")
    if cal_findings:
        for f in cal_findings:
            print(f"    [{f.severity.value.upper()}] {f.title}: {f.explanation}")
    else:
        print("    No calibration issues detected.")


def main():
    print("Generating data...")
    X_train, X_test, y_train, y_test = get_classification_data(random_state=42)

    print("Training uncalibrated SVM (probability=True)...")
    # SVMs with Platt probability can produce poorly calibrated scores
    raw_svm = SVC(probability=True, random_state=42)
    raw_svm.fit(X_train, y_train)

    print("Training Platt-scaled (calibrated) SVM...")
    calibrated_svm = CalibratedClassifierCV(SVC(random_state=42), method="sigmoid", cv=5)
    calibrated_svm.fit(X_train, y_train)

    print("\nDiagnosing both models...")
    raw_report = md.diagnose(raw_svm, X_train, y_train, X_test, y_test)
    cal_report = md.diagnose(calibrated_svm, X_train, y_train, X_test, y_test)

    print("\n--- Calibration Comparison ---")
    calibration_summary("Uncalibrated SVM", raw_report)
    calibration_summary("Platt-Scaled SVM", cal_report)

    print("\nNote: Calibration doctor checks Expected Calibration Error (ECE).")
    print("Lower ECE means predicted probabilities better reflect real-world likelihoods.")


if __name__ == "__main__":
    main()
