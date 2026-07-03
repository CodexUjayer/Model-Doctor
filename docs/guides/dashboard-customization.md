# Dashboard Customization

The ModelDoctor HTML dashboard is a self-contained, static HTML file. This guide covers how to configure and customize the dashboard output.

## Launching the Dashboard

```python
import modeldoctor as md

report = md.diagnose(model, X_train, y_train, X_test, y_test)

# Open in browser (writes to a temp file)
report.dashboard()

# Or save permanently
report.save_html("my_report.html")
```

## Dashboard Sections

The dashboard is organized into four tabs:

| Tab | Description |
|---|---|
| **Summary** | Health score gauge, grade, and a high-level findings overview. |
| **Diagnostics** | Per-doctor dimension scores, detailed findings, and evidence tables. |
| **Prescriptions** | Actionable fix recommendations with implementation steps. |
| **Model Passport** | Model family, framework, training dataset size, serialized model size, and inference latency. |

## Filtering Findings

The findings table in the **Diagnostics** tab includes built-in search and severity filtering. You can:

- Filter by severity (`INFO`, `WARNING`, `CRITICAL`) using the dropdown.
- Search for a specific diagnostic dimension or finding title using the search box.

## Saving for Sharing

The dashboard is a completely self-contained `.html` file. All JavaScript, CSS, and chart data are embedded inline. You can:

- Email it as an attachment.
- Upload it to cloud storage (S3, GCS, Azure Blob).
- Attach it to a Jira or Linear ticket.
- Host it as a GitHub Pages artifact from a CI pipeline.

```python
# CI/CD pipeline example
import modeldoctor as md

report = md.diagnose(model, X_train, y_train, X_test, y_test)
report.save_html("artifacts/model_health_report.html")
```

## Customizing the Report Path

```python
# Specify an output path
report.save_html("reports/v1.0/classification_health_report.html")
```

## Exporting to Other Formats

The dashboard's data can also be exported in machine-readable formats:

```python
# JSON — full structured data for programmatic use
report.save_json("report.json")

# Markdown — human-readable summary for PRs and logs
report.save_markdown("report.md")

# CSV — flat findings table for spreadsheet analysis
report.save_csv("findings.csv")

# pandas DataFrame — for notebook analysis
df = report.to_dataframe()
print(df[["title", "severity", "dimension"]].head())
```

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| Blank screen / missing CSS | `[dashboard]` extra not installed | `pip install "modeldoctor[dashboard]"` |
| Browser doesn't open | OS restricts temp file access | Use `save_html()` and open manually |
| `TemplateNotFound` error | Jinja2 templates missing | Reinstall: `pip install -U "modeldoctor[dashboard]"` |

## See Also

- [Dashboard Overview](../getting-started/dashboard.md)
- [Reports Reference](../concepts/reports.md)
- [MLflow Integration](mlflow.md)
