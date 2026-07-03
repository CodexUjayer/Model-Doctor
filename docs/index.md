# ModelDoctor

**Diagnose your machine learning models like a senior ML engineer.**

<div align="center">
  <img src="images/dashboard.png" alt="ModelDoctor Dashboard" width="100%" />
</div>

## What is ModelDoctor?

ModelDoctor is an automated clinical diagnostics engine for machine learning models. It runs a comprehensive suite of rigorous checks against your trained model to identify hidden problems, explain why they occur, and prescribe actionable fixes before deployment.

## Why does it exist?

Traditional machine learning evaluation focuses almost entirely on aggregate metrics like accuracy, precision, and recall. While these numbers indicate how well a model performs on a specific dataset, they rarely explain *why* it behaves that way or whether the model will actually survive in a production environment. 

ModelDoctor bridges this gap by evaluating models holistically. Instead of leaving you to interpret raw numbers, ModelDoctor explains what issues exist, why they are problematic, and exactly how to fix them.

## How does it work?

1. **Contextualization:** ModelDoctor wraps your scikit-learn model, training data, and test data into an `EvaluationContext`.
2. **Diagnosis:** Specialized `Doctors` (like `LeakageDoctor` and `OverfittingDoctor`) independently analyze the context.
3. **Evidence Synthesis:** The `EvidenceEngine` collects findings, which are scored by the `ConfidenceEngine` and `RiskEngine`.
4. **Prescription:** The `PrescriptionEngine` generates actionable recommendations.
5. **Reporting:** Results are compiled into an interactive HTML dashboard, JSON, or Markdown report.

## How do I get started?

Install the library:
```bash
pip install modeldoctor[all]
```

Run a diagnosis:
```python
import modeldoctor as md

# diagnose() takes any scikit-learn compatible estimator
report = md.diagnose(model, X_train, y_train, X_test, y_test)

# Open the interactive HTML dashboard
report.dashboard()
```

---

## Documentation Navigation

- **Getting Started**
  - [Installation](getting-started/installation.md)
  - [Quick Start](getting-started/quickstart.md)
  - [Dashboard](getting-started/dashboard.md)
  - [CLI Reference](getting-started/cli.md)
- **Core Concepts**
  - [Health Score](concepts/health-score.md)
  - [Diagnostic Doctors](concepts/diagnostic-doctors.md)
  - [Evaluation Context](concepts/evaluation-context.md)
  - [Findings](concepts/findings.md)
  - [Prescriptions](concepts/prescriptions.md)
  - [Reports](concepts/reports.md)
- **User Guides**
  - [Classification](guides/classification.md)
  - [Regression](guides/regression.md)
  - [Model Comparison](guides/comparison.md)
  - [Explainability](guides/explainability.md)
  - [Calibration](guides/calibration.md)
  - [MLflow Integration](guides/mlflow.md)
  - [Plugins & Custom Doctors](guides/plugins.md)
  - [Dashboard Customization](guides/dashboard-customization.md)
- **Examples**
  - [Basic Usage](examples/basic.md)
  - [Advanced Usage](examples/advanced.md)
- **Reference**
  - [API Reference](api/reference.md)
  - [Roadmap](roadmap.md)
- **Support**
  - [FAQ](faq.md)
  - [Troubleshooting](troubleshooting.md)
  - [Contributing](contributing.md)
  - [Changelog](changelog.md)
  - [Validation Lab](validation.md)
