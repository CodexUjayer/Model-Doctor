# Reports

The `Report` object is the final output of `md.diagnose()`. It encapsulates all diagnostic results and provides export utilities for sharing findings across teams and systems.

## Report Properties

| Property | Type | Description |
|---|---|---|
| `health_score` | `HealthScore` | The overall 0–100 diagnostic score and letter grade. |
| `diagnoses` | `List[Diagnosis]` | Per-doctor diagnosis objects containing findings and dimension scores. |
| `prescription` | `Prescription` | The `PrescriptionEngine` output with actionable recommendations. |
| `passport` | `ModelPassport` | Metadata about the model footprint, serialized size, and inference latency. |

## Export Methods

### `dashboard()`
Opens the interactive HTML dashboard in the default system web browser. Requires the `[dashboard]` extra.

### `save_html(path: str)`
Renders and saves a self-contained HTML dashboard to the specified path. The file requires no server and can be shared as an email attachment.

### `save_json(path: str)`
Serializes the entire report to a structured JSON file. Ideal for programmatic ingestion into monitoring systems, CI pipelines, or dashboards.

### `save_markdown(path: str)`
Exports a human-readable Markdown summary of the findings and prescriptions. Useful for appending to pull requests or CI/CD logs.

### `save_csv(path: str)`
Exports a flat CSV file containing the list of findings (dimension, title, severity). Suitable for downstream analysis in spreadsheets.

### `to_dataframe() -> pandas.DataFrame`
Returns the findings as a pandas `DataFrame` for direct analysis in a notebook.

## ModelPassport

The `ModelPassport` is a metadata snapshot attached to every report:

```python
class ModelPassport(BaseModel):
    framework: str
    model_family: str
    training_samples: int
    feature_count: int
    model_size_bytes: int
    inference_latency_ms: float
```

## Example: Full Export Pipeline

```python
import modeldoctor as md

report = md.diagnose(model, X_train, y_train, X_test, y_test)

# Save all formats
report.save_html("report.html")
report.save_json("report.json")
report.save_markdown("report.md")
report.save_csv("findings.csv")

# Inspect the model passport
print(f"Model size: {report.passport.model_size_bytes:,} bytes")
print(f"Framework: {report.passport.framework}")
```

For a full list of supported export formats, see the [Reporting](../guides/mlflow.md) reference.
