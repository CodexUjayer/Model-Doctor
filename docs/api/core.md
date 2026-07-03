# Core API

## `md.diagnose()`

The primary entry point to ModelDoctor.

```python
def diagnose(
    model: Any, 
    X_train: Any, 
    y_train: Any, 
    X_test: Any, 
    y_test: Any, 
    *, 
    X_val: Optional[Any] = None, 
    y_val: Optional[Any] = None, 
    feature_names: Optional[List[str]] = None, 
    config: Optional[ModelDoctorConfig] = None, 
    progress_callback: Optional[Callable[[str], None]] = None
) -> Report
```

### Parameters

- **`model`**: A fitted scikit-learn compatible estimator. Must implement `fit` and `predict` (and optionally `predict_proba`).
- **`X_train`**: Training feature matrix (`numpy.ndarray` or `pandas.DataFrame`).
- **`y_train`**: Training target array (`numpy.ndarray` or `pandas.Series`).
- **`X_test`**: Test/hold-out feature matrix.
- **`y_test`**: Test/hold-out target array.
- **`X_val`**: Optional validation feature matrix.
- **`y_val`**: Optional validation target array.
- **`feature_names`**: Optional list of feature names. If omitted and data is a DataFrame, column names are extracted automatically.
- **`config`**: Optional `ModelDoctorConfig` for tuning threshold values.
- **`progress_callback`**: Optional callable for receiving string updates during pipeline execution.

### Returns

- **`Report`**: A populated report object containing all findings, prescriptions, and export methods.

### Exceptions

- **`ValueError`**: Raised if shapes between `X_train` and `X_test` mismatch, or if the model is entirely unsupported/unfitted.
