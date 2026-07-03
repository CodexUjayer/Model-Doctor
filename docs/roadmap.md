# Roadmap

This page documents the planned improvements and upcoming releases for ModelDoctor.

## v1.0 — Current Release

The initial stable release of ModelDoctor includes:

- Core evaluation engine (`EvaluationContext`) with lazy metric computation.
- Eight built-in Doctors: `OverfittingDoctor`, `LeakageDoctor`, `PredictionDoctor`, `DataDoctor`, `FeatureDoctor`, `CalibrationDoctor`, `ProductionDoctor`, `GeneralizationDoctor`.
- Evidence, Confidence, Risk, and Prescription engines.
- Interactive HTML dashboard with embedded JavaScript and CSS.
- JSON, Markdown, and CSV export formats.
- MLflow integration (`log_report`).
- Validation Laboratory with 54 benchmark scenarios achieving 98.1% diagnostic accuracy.
- Custom Doctor API (`BaseDoctor`, `DoctorRegistry`, `EvidenceBuilder`).

---

## v1.1 — Planned

### PDF Export

Render ModelDoctor reports directly to a styled PDF. Intended for compliance documentation, audit trails, and sharing with non-technical stakeholders.

```python
# Planned API
report.save_pdf("model_health_report.pdf")
```

### ZIP Archive

Export a ZIP archive containing all report formats simultaneously:

```python
# Planned API
report.save_zip("report_bundle.zip")
# Contains: dashboard.html, report.json, summary.md, findings.csv
```

### Threshold Configuration via YAML

Load custom diagnostic thresholds from a YAML config file without writing Python:

```yaml
# config.yaml
overfitting:
  generalization_gap_warning: 0.10
  generalization_gap_critical: 0.20
calibration:
  ece_warning: 0.10
  ece_critical: 0.20
```

```bash
modeldoctor diagnose --model model.pkl --data data.csv --target y --config config.yaml
```

---

## v1.2 — Exploratory

### PyTorch / TensorFlow Support

Neural networks present unique diagnostic failure modes that scikit-learn models do not encounter:

- Vanishing and exploding gradients.
- Catastrophic forgetting in continual learning scenarios.
- Weight initialization sensitivity.

Dedicated Doctors for deep learning models are under active research.

### XGBoost / LightGBM Native Diagnostics

While XGBoost and LightGBM already work via their scikit-learn wrappers, native integration would expose additional signals such as:

- Feature split gain histograms.
- Leaf value distributions.
- Boosting round convergence patterns.

### Advanced Multiclass Calibration

The current `CalibrationDoctor` supports multiclass classification via one-vs-rest ECE decomposition. A dedicated multiclass calibration analysis covering macro/micro ECE variants and per-class confidence profiles is planned.

### Dashboard Themes

Support for customizable dashboard themes (light, dark, high-contrast, branded).

---

## Feature Requests & Contributions

To suggest a feature or report a bug, open an issue on GitHub.

To contribute a new Doctor or fix, see the [Contributing Guide](contributing.md).
