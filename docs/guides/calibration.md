# Calibration Guide

A well-calibrated classifier produces predicted probabilities that reflect actual likelihoods. For example, if a model assigns a 70% probability to the positive class, roughly 70% of those predictions should actually be positive. Poor calibration means the model is systematically overconfident or underconfident, which is particularly dangerous in applications like medical diagnosis, credit scoring, or risk management.

## What is Expected Calibration Error?

**Expected Calibration Error (ECE)** is ModelDoctor's primary calibration metric. It measures the average discrepancy between predicted confidence and observed accuracy across all probability bins.

- ECE of `0.0` = perfect calibration.
- ECE above `0.10` typically indicates poor calibration.
- ECE above `0.20` triggers a `CRITICAL` finding.

## Running a Calibration Diagnosis

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import modeldoctor as md

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVMs with probability=True can produce poorly calibrated scores
model = SVC(probability=True, random_state=42)
model.fit(X_train, y_train)

report = md.diagnose(model, X_train, y_train, X_test, y_test)

# Review calibration findings
all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
cal_findings = [f for f in all_findings if "Calibration" in f.title]
for f in cal_findings:
    print(f"[{f.severity.value.upper()}] {f.title}: {f.explanation}")
```

## Applying Platt Scaling

Platt Scaling (sigmoid calibration) is one of the simplest and most effective post-hoc calibration methods. Wrap any estimator with `CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

raw_svm = SVC(random_state=42)
calibrated_svm = CalibratedClassifierCV(raw_svm, method="sigmoid", cv=5)
calibrated_svm.fit(X_train, y_train)

cal_report = md.diagnose(calibrated_svm, X_train, y_train, X_test, y_test)
print(f"Calibrated Health Score: {cal_report.health_score.overall:.1f}/100")
```

## Calibration Methods Compared

| Method | Best For | Limitation |
|---|---|---|
| **Platt Scaling** (`method="sigmoid"`) | SVMs and boosted models | Assumes sigmoid-shaped miscalibration. |
| **Isotonic Regression** (`method="isotonic"`) | Large datasets | Can overfit on small samples. |
| **Temperature Scaling** | Neural networks | Requires custom implementation; not in sklearn. |

## Which Models Need Calibration?

Some models produce inherently well-calibrated probabilities, while others are systematically overconfident:

| Model | Calibration | Notes |
|---|---|---|
| `LogisticRegression` | ✅ Good | Produces reliable probabilities by design. |
| `RandomForestClassifier` | ⚠️ Moderate | Probabilities can be pushed toward 0 and 1. |
| `GradientBoostingClassifier` | ⚠️ Moderate | Can be overconfident on certain splits. |
| `SVC (probability=True)` | ❌ Poor | Uses Platt scaling internally, often miscalibrated. |
| `DecisionTreeClassifier` | ❌ Poor | Extreme overconfidence at leaf nodes. |

## Notes on the CalibrationDoctor

The `CalibrationDoctor` is only active for classification tasks where the model implements `predict_proba`. It is automatically skipped for:

- Regression tasks.
- Models that do not support probability output.

## See Also

- [Classification Guide](classification.md)
- [Diagnostic Doctors Reference](../concepts/diagnostic-doctors.md)
