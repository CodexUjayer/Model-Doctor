# Troubleshooting

## Import Errors

### `ModuleNotFoundError: No module named 'pydantic'`
ModelDoctor strictly relies on Pydantic v2 for data serialization. Ensure it is installed:
```bash
pip install pydantic>=2.0
```

### `ImportError: cannot import name 'log_report'`
Ensure you are importing from the correct namespace:
```python
# Correct
from modeldoctor.integrations.mlflow import log_report
```

## Execution Errors

### `AttributeError: 'EvaluationContext' object has no attribute 'metrics'`
This occurs if the internal evaluation context fails to build properly, usually because the provided `model` has not been fitted, or it lacks `predict()` methods. Ensure you have called `model.fit(X, y)` before passing it to `md.diagnose()`.

### `ValueError: Number of features of the model must match the input.`
`X_train` and `X_test` must have the exact same shape (number of columns). The model must also have been trained on an identical schema.

## Dashboard Issues

### `TemplateNotFound` Error
You are attempting to render the HTML dashboard but the Jinja2 templates are missing. This happens if the library was installed without dashboard support. 
```bash
pip install "modeldoctor[dashboard]"
```

## Windows/Linux/macOS Notes

ModelDoctor is OS-agnostic and relies entirely on pure Python paths. However, when using `report.dashboard()` on Windows Subsystem for Linux (WSL), the auto-open command may fail to find a valid web browser. Use `report.save_html()` instead.
