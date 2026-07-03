# Quick Start

This guide will get you up and running with ModelDoctor in under five minutes.

## 1. Train a Simple Model

ModelDoctor is designed to evaluate models *after* they have been trained. First, let's create a synthetic dataset and train a standard scikit-learn model.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Create a dataset (with intentional noise and redundancy)
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=5, 
    n_redundant=10, 
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)
```

## 2. Run Diagnostics

Import ModelDoctor and pass the model and datasets into `md.diagnose()`. ModelDoctor will automatically infer the task type (classification vs. regression) and run the appropriate diagnostic checks.

```python
import modeldoctor as md

# 3. Diagnose the model
report = md.diagnose(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
```

## 3. Read the Report

You can immediately view the health score and diagnostic findings programmatically:

```python
# Print the overall health score (0-100)
print(f"Health Score: {report.health_score.overall:.1f} / 100")
print(f"Grade: {report.health_score.grade}")

# Iterate through high-risk findings across all doctors
all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
for finding in all_findings:
    if finding.severity.value in ("warning", "critical"):
        print(f"[{finding.severity.value.upper()}] {finding.title}: {finding.explanation}")
```

## 4. Launch the Dashboard

The easiest way to understand your model's health is through the interactive HTML dashboard.

```python
# Opens the interactive dashboard in your default web browser
report.dashboard()
```

## 5. Export Reports

You can export the report to various formats for CI/CD pipelines or compliance documentation.

```python
# Save as a self-contained HTML file
report.save_html("model_health_report.html")

# Save as machine-readable JSON
report.save_json("model_health_report.json")

# Save as a Markdown summary
report.save_markdown("model_health_report.md")
```

Next, dive deeper into the [Diagnose API](../guides/classification.md) or learn how the [Architecture](../concepts/health-score.md) works under the hood.
