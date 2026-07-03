# Dashboard

ModelDoctor provides a fully self-contained, interactive HTML dashboard for exploring diagnostic reports.

<div align="center">
  <img src="../images/dashboard.png" alt="ModelDoctor Interactive Dashboard" width="100%" />
</div>

## Launching

If you are working in a Jupyter Notebook or local Python script, the easiest way to view the dashboard is by calling `report.dashboard()`.

```python
import modeldoctor as md

report = md.diagnose(...)
report.dashboard()
```

This command will automatically write the HTML file to a temporary directory and open it in your default web browser.

## Features

### Navigation
The dashboard is split into distinct tabs, allowing you to easily navigate between the high-level **Summary**, detailed **Diagnostic Charts**, **Prescriptions**, and the **Model Passport**.

### Embedded Server
The dashboard is a static HTML file. It does not require a backend Python server, Node.js, or any active compute. You can email it, upload it to an S3 bucket, or attach it to a Jira ticket. All charts, JavaScript, and CSS are embedded directly within the file.

### Filtering and Searching
The findings table includes built-in search and severity filtering. You can instantly filter down to `CRITICAL` issues or search for specific dimensions like `Leakage`.

## HTML Reports

To save the dashboard permanently without opening it:

```python
report.save_html("path/to/report.html")
```

## Troubleshooting

- **Blank Screen / Missing CSS**: Ensure you have installed the `[dashboard]` extra (`pip install modeldoctor[dashboard]`). Without Jinja2, the HTML generation will fail or produce incomplete files.
- **File Not Found**: If `report.dashboard()` fails to open your browser, it may be due to restrictive OS settings regarding temporary files. Use `save_html()` and open the file manually.

For advanced dashboard configuration, see the [Dashboard Customization](../guides/dashboard-customization.md) guide.
