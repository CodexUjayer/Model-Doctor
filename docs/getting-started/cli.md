# CLI

ModelDoctor includes a command-line interface (CLI) for running diagnostics without writing custom Python scripts. This is especially useful for integrating ModelDoctor into CI/CD pipelines.

## Usage

```bash
modeldoctor [OPTIONS] COMMAND [ARGS]...
```

## Commands

### `diagnose`

Runs the diagnostic pipeline against a serialized model and dataset.

```bash
modeldoctor diagnose --model model.pkl --data dataset.csv --target y_col --out report.json
```

**Arguments & Flags:**

- `--model` (Required): Path to the serialized scikit-learn model (`.pkl`, `.joblib`).
- `--data` (Required): Path to the CSV file containing the features and target.
- `--target` (Required): The column name representing the target variable.
- `--out`: Path to save the resulting report. The format is inferred from the extension (`.json`, `.html`, `.md`).
- `--config`: Path to a custom `config.yaml` file to override default thresholds.

### `serve`

Starts a lightweight local web server to host and view previously generated HTML dashboards.

```bash
modeldoctor serve --dir ./reports --port 8080
```

**Arguments & Flags:**

- `--dir`: The directory containing ModelDoctor `.html` reports.
- `--port`: The port to bind the server to (default: 8080).

## Exit Codes

The `diagnose` command returns exit codes based on the diagnostic severity, making it ideal for CI/CD gating:

- `0`: Success. The model is healthy or only contains `INFO` level findings.
- `1`: Execution error (e.g., file not found, unpickling error).
- `2`: Warning. The model generated `WARNING` level findings.
- `3`: Critical. The model generated `CRITICAL` level findings.
