# Context API

## `EvaluationContext`

The `EvaluationContext` represents the state of the model and data throughout the diagnostic pipeline. It heavily utilizes lazy evaluation (via `@cached_property` or similar internal mechanisms) to prevent unnecessary computation.

```python
class EvaluationContext:
    def __init__(self, model, X_train, y_train, X_test, y_test, ...)
```

### Properties

- **`model`**: The original fitted model.
- **`X_train`, `y_train`**: The training datasets.
- **`X_test`, `y_test`**: The testing datasets.
- **`task_type`**: The inferred `TaskType` (e.g., `TaskType.BINARY_CLASSIFICATION`).
- **`feature_names`**: A list of strings representing feature names.
- **`feature_importances`**: A lazy-evaluated numpy array of feature importance scores (uses SHAP or permutation importance).
- **`train_score`**: The model's score on the training data.
- **`test_score`**: The model's score on the testing data.
- **`cv_scores`**: Array of cross-validation scores (lazy evaluated).
- **`classification_metrics`**: A dictionary containing accuracy, f1, precision, recall (classification only).
- **`regression_metrics`**: A dictionary containing mse, mae, r2 (regression only).

### Notes
Doctors should prefer accessing these properties rather than computing metrics manually to ensure values are cached and reused across the pipeline.
