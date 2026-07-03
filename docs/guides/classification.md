# Classification Guide

This guide covers diagnosing binary and multiclass classification models with ModelDoctor from start to finish.

## Setup

Install ModelDoctor with all optional dependencies:

```bash
pip install "modeldoctor[all]"
```

## Step 1: Train a Classifier

ModelDoctor evaluates models *after* they have been trained. The model must implement `fit`, `predict`, and ideally `predict_proba`.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)
```

## Step 2: Run Diagnostics

```python
import modeldoctor as md

report = md.diagnose(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
```

ModelDoctor automatically infers the task as `BINARY_CLASSIFICATION` or `MULTICLASS_CLASSIFICATION` based on the number of unique values in `y_train`.

## Step 3: Review Findings

```python
print(f"Health Score: {report.health_score.overall:.1f} / 100")
print(f"Grade:        {report.health_score.grade}")

all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
for finding in all_findings:
    if finding.severity.value in ("warning", "critical"):
        print(f"[{finding.severity.value.upper()}] {finding.title}")
        print(f"  {finding.explanation}")
```

## Doctors Active for Classification

All eight built-in Doctors run on classification tasks:

| Doctor | What It Checks |
|---|---|
| `OverfittingDoctor` | Train/test accuracy gap and memorization. |
| `LeakageDoctor` | Feature correlation with target (proxy leakage). |
| `DataDoctor` | Class imbalance, missing values, duplicate rows/columns. |
| `FeatureDoctor` | High dimensionality, zero-variance features. |
| `PredictionDoctor` | Raw accuracy, F1, and precision sanity checks. |
| `CalibrationDoctor` | Expected Calibration Error (ECE) and probability reliability. |
| `ProductionDoctor` | Serialized model size and inference latency. |
| `GeneralizationDoctor` | Cross-validation fold stability. |

## Handling Class Imbalance

If `DataDoctor` raises a `Severe Class Imbalance` finding, consider:

```python
from sklearn.ensemble import RandomForestClassifier

# Set class weights to balance the minority class
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
```

Or use SMOTE to oversample the minority class before passing data to `md.diagnose()`.

## Multiclass Classification

ModelDoctor fully supports multiclass classification. The `CalibrationDoctor` uses a one-vs-rest extension of ECE, and the `PredictionDoctor` reports macro-averaged F1.

## See Also

- [Regression Guide](regression.md)
- [Calibration Guide](calibration.md)
- [Diagnostic Doctors Reference](../concepts/diagnostic-doctors.md)
