# MLflow Integration

ModelDoctor seamlessly integrates with MLflow to track diagnostic health scores, metadata, and artifacts alongside your standard training metrics.

## Installation

Ensure `mlflow` is installed in your environment. It is not a direct dependency of the ModelDoctor core library:

```bash
pip install mlflow
```

## Usage

Use the `log_report` function to attach a ModelDoctor report to your active MLflow run.

```python
import mlflow
import modeldoctor as md
from modeldoctor.integrations.mlflow import log_report

# Train your model and run diagnostics
report = md.diagnose(model, X_train, y_train, X_test, y_test)

with mlflow.start_run():
    # Log standard training metrics
    mlflow.log_metric("accuracy", 0.95)
    
    # Log the ModelDoctor report alongside your standard metrics
    log_report(report, artifact_path="modeldoctor")
```

## What Gets Logged?

### Metrics

The following scalar metrics are tracked in your MLflow run:

| Metric | Description |
|---|---|
| `md_health_score` | The overall 0–100 health score. |
| `md_critical_findings` | Count of `CRITICAL` severity findings. |
| `md_warning_findings` | Count of `WARNING` severity findings. |

### Artifacts

The integration automatically renders and uploads several files to the specified MLflow `artifact_path` (defaulting to `modeldoctor/`):

| File | Description |
|---|---|
| `dashboard.html` | The interactive HTML dashboard. |
| `report.json` | The raw JSON payload for downstream programmatic consumption. |
| `summary.md` | A human-readable Markdown summary of the findings. |

## Viewing Results in the MLflow UI

After logging, start the MLflow tracking UI to view results:

```bash
mlflow ui
```

Navigate to your experiment and click the run to see the `md_health_score`, `md_critical_findings`, and `md_warning_findings` metrics alongside your standard training metrics.

## Limitations

- The integration assumes an MLflow run is already active (`mlflow.start_run()`). If one is not active, it will create a new run, which may fragment your tracking history.
- Ensure the `[dashboard]` extra is installed for HTML dashboard generation within the MLflow artifact upload.

## See Also

- [Reports Reference](../concepts/reports.md)
- [Roadmap](../roadmap.md)
