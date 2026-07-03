# Reporting

The `Report` object returned by `md.diagnose()` contains the final synthesized output of the diagnostic pipeline. ModelDoctor supports exporting this report into several standard formats.

## Supported Exports

### HTML
Generates a fully self-contained, interactive HTML dashboard. Ideal for sharing with stakeholders or attaching to tickets.

```python
report.save_html("report.html")
```

### JSON
Exports the raw data schema, including all numerical evidence, thresholds, and configuration settings. Ideal for programmatic ingestion into monitoring systems.

```python
report.save_json("report.json")
```

### Markdown
Generates a human-readable markdown file summarizing the findings and prescriptions. Ideal for appending to pull requests or commit messages.

```python
report.save_markdown("report.md")
```

### CSV
Exports the flat list of findings (dimension, severity, title).

```python
report.save_csv("findings.csv")
```

### DataFrame
Returns a pandas `DataFrame` of the findings for downstream analysis in a notebook.

```python
df = report.to_dataframe()
print(df.head())
```

### PDF & ZIP
*Note: PDF export and ZIP archiving are currently tracked on the roadmap for v1.1 and are not supported in the current release.*
