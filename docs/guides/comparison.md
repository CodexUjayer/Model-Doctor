# Model Comparison Guide

ModelDoctor evaluates one model per `diagnose()` call. To compare two models, simply call `diagnose()` twice and compare the resulting reports.

## Basic Comparison

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import modeldoctor as md

# Shared dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train two models
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)

# Diagnose both
lr_report = md.diagnose(lr_model, X_train, y_train, X_test, y_test)
rf_report = md.diagnose(rf_model, X_train, y_train, X_test, y_test)
```

## Reading Both Reports

```python
def summarize(name, report):
    all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
    flagged = [f for f in all_findings if f.severity.value in ("warning", "critical")]
    print(f"\n{name}")
    print(f"  Health Score: {report.health_score.overall:.1f}/100  Grade: {report.health_score.grade}")
    if flagged:
        for f in flagged:
            print(f"  [{f.severity.value.upper()}] {f.title}")
    else:
        print("  No issues detected.")

summarize("Logistic Regression", lr_report)
summarize("Random Forest (max_depth=None)", rf_report)
```

## Comparing Health Scores

```python
lr_score = lr_report.health_score.overall
rf_score = rf_report.health_score.overall

print(f"\nLogistic Regression: {lr_score:.1f}/100")
print(f"Random Forest:       {rf_score:.1f}/100")

if lr_score > rf_score:
    print("Logistic Regression is the healthier model for production.")
else:
    print("Random Forest is the healthier model for production.")
```

## Viewing Dashboards Side by Side

Open both dashboards in separate browser tabs for a visual comparison:

```python
lr_report.save_html("lr_report.html")
rf_report.save_html("rf_report.html")
```

## What to Look For

When comparing models, pay particular attention to:

| Dimension | Why It Matters |
|---|---|
| **Health Score** | The single highest-level signal — higher is better. |
| **OverfittingDoctor** | Unconstrained trees often overfit significantly. |
| **CalibrationDoctor** | Probability reliability varies widely across model families. |
| **ProductionDoctor** | Ensemble models can be 10–100x larger than linear models. |
| **GeneralizationDoctor** | CV stability indicates robustness across different data splits. |

## See Also

- [Classification Guide](classification.md)
- [Calibration Guide](calibration.md)
- [Diagnostic Doctors Reference](../concepts/diagnostic-doctors.md)
