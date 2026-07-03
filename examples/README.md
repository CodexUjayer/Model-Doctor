# ModelDoctor Examples

This directory contains a suite of standalone scripts demonstrating the full capabilities of ModelDoctor. They are designed to be read, executed, and modified by new users.

## Prerequisites

To run these examples, ensure you have the core library installed. Some examples (like Dashboard and MLflow) require optional dependencies.

```bash
pip install modeldoctor[all]
```

All examples use synthetic data generation (via `scikit-learn`) so no external datasets need to be downloaded.

## Example Index

We recommend running the examples in this order:

### Getting Started (Basic Workflows)
- `01_basic_classification.py`: The "Hello World". Evaluates a Random Forest classifier.
- `02_basic_regression.py`: Evaluates a regression task.

### Reporting & Exports
- `03_dashboard.py`: Renders and launches the interactive HTML dashboard.
- `04_html_report.py`: Saves a self-contained HTML report to disk.
- `05_json_export.py`: Serializes the report to JSON for programmatic ingestion.
- `06_pdf_export.py`: Demonstrates upcoming PDF export capabilities (planned for v1.1).
- `07_mlflow_logging.py`: Logs reports and metrics directly to a local MLflow run.

### Advanced Functionality
- `08_custom_doctor.py`: Extends ModelDoctor by writing a custom domain-specific rule.
- `09_prescription_engine.py`: Demonstrates the generation of actionable engineering fixes.
- `10_validation_runner.py`: Programmatically runs the internal Validation Laboratory benchmarks.

### Specific Diagnostics (Deep Dives)
- `11_model_comparison.py`: Diagnoses and compares two distinct estimators on the same dataset.
- `12_probability_calibration.py`: Analyzes Expected Calibration Error (ECE) via Platt Scaling.
- `13_data_leakage_detection.py`: Detects a subtle proxy feature that leaks target information.
- `14_overfitting_analysis.py`: Diagnoses a complex model that memorized training noise.
- `15_production_readiness.py`: Evaluates inference latency and serialization footprint.

## Shared Utilities
- `utils.py`: Contains common functions for generating synthetic datasets (classification, regression, leaky data, etc.) used across the scripts.

## Running an Example

Execute any script directly via Python. They will all complete in under one minute.

```bash
python 01_basic_classification.py
```
