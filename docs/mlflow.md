# MLflow Integration

ModelDoctor seamlessly integrates with MLflow to track diagnostic health scores, metadata, and artifacts alongside your standard training metrics.

## Usage

Use the `log_report` function to attach a ModelDoctor report to your active MLflow run.

```python
import mlflow
import modeldoctor as md
from modeldoctor.integrations.mlflow import log_report

# Train your model and run diagnostics
report = md.diagnose(model, X_train, y_train, X_test, y_test)

with mlflow.start_run():
    # Log standard metrics
    mlflow.log_metric("accuracy", 0.95)
    
    # Log the ModelDoctor report
    log_report(report, artifact_path="modeldoctor")
```

## What gets logged?

### Metrics
- `md_health_score`: The overall 0-100 health score.
- `md_critical_findings`: The count of `CRITICAL` severity findings.
- `md_warning_findings`: The count of `WARNING` severity findings.

### Artifacts
The integration automatically renders and uploads several files to the specified MLflow `artifact_path` (defaulting to `modeldoctor/`):
- `dashboard.html`: The interactive HTML dashboard.
- `report.json`: The raw JSON payload for programmatic downstream consumption.
- `summary.md`: A human-readable Markdown summary of the findings.

## Limitations

- The integration assumes an MLflow run is already active (`mlflow.start_run()`). If one is not active, it will create a new run, which may fragment your tracking history.
- Ensure you have `mlflow` installed in your environment, as it is not a direct dependency of the `modeldoctor` core library.
