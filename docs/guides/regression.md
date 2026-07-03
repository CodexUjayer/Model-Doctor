# Regression Guide

This guide covers diagnosing regression models with ModelDoctor.

## Setup

```bash
pip install "modeldoctor[all]"
```

## Step 1: Train a Regressor

ModelDoctor supports any scikit-learn compatible regressor that implements `fit` and `predict`.

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    noise=0.1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
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

print(f"Health Score: {report.health_score.overall:.1f} / 100")
print(f"Grade:        {report.health_score.grade}")
```

ModelDoctor infers the task as `REGRESSION` when `y_train` contains continuous float values.

## Step 3: Review Regression-Specific Findings

```python
all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
for finding in all_findings:
    if finding.severity.value in ("warning", "critical"):
        print(f"[{finding.severity.value.upper()}] {finding.title}: {finding.explanation}")
```

## Doctors Active for Regression

| Doctor | What It Checks |
|---|---|
| `OverfittingDoctor` | Train/test R² gap and memorization. |
| `LeakageDoctor` | Feature correlation with continuous target. |
| `DataDoctor` | Missing values, duplicate rows/columns. |
| `FeatureDoctor` | High dimensionality, zero-variance features. |
| `PredictionDoctor` | R², MSE, and MAE sanity checks. |
| `ProductionDoctor` | Serialized model size and inference latency. |
| `GeneralizationDoctor` | Cross-validation fold stability. |

!!! note
    `CalibrationDoctor` is **not active** for regression tasks, as it is specifically designed to evaluate probability estimates from classifiers.

## Regression Metrics

The `PredictionDoctor` evaluates the following regression metrics:

- **R² Score**: Proportion of variance explained by the model.
- **MSE (Mean Squared Error)**: Average squared residuals.
- **MAE (Mean Absolute Error)**: Average absolute residuals.

A very low R² (below ~0.1) may trigger a `Poor Prediction Quality` finding.

## Overfitting in Regression

An unconstrained `RandomForestRegressor` with `max_depth=None` often memorizes training data, producing a near-perfect training R² but a much lower test R². The `OverfittingDoctor` will flag this as a `Generalization Gap`.

To fix:
```python
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
```

## See Also

- [Classification Guide](classification.md)
- [Diagnostic Doctors Reference](../concepts/diagnostic-doctors.md)
