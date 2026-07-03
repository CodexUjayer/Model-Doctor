# Evaluation Context

The `EvaluationContext` is the central state container passed through the entire ModelDoctor pipeline. It wraps your model, training data, and test data, and exposes a set of lazily-evaluated properties that Doctors can query without triggering redundant computation.

## Construction

`EvaluationContext` is constructed internally by `md.diagnose()`. You do not normally need to instantiate it directly.

```python
class EvaluationContext:
    def __init__(self, model, X_train, y_train, X_test, y_test, ...)
```

## Properties

| Property | Type | Description |
|---|---|---|
| `model` | `BaseEstimator` | The original fitted model. |
| `X_train` | `ndarray` / `DataFrame` | The training feature matrix. |
| `y_train` | `ndarray` / `Series` | The training target array. |
| `X_test` | `ndarray` / `DataFrame` | The test feature matrix. |
| `y_test` | `ndarray` / `Series` | The test target array. |
| `task_type` | `TaskType` | The inferred task (`BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, `REGRESSION`). |
| `feature_names` | `List[str]` | Feature names extracted from DataFrame columns or passed explicitly. |
| `feature_importances` | `ndarray` | Lazy-evaluated feature importance scores (SHAP or permutation importance). |
| `train_score` | `float` | The model's score on the training data. |
| `test_score` | `float` | The model's score on the testing data. |
| `cv_scores` | `ndarray` | Cross-validation scores (lazy evaluated). |
| `classification_metrics` | `dict` | Accuracy, F1, precision, recall (classification only). |
| `regression_metrics` | `dict` | MSE, MAE, R² (regression only). |

## Lazy Evaluation

Properties like `cv_scores` and `feature_importances` are expensive to compute. The `EvaluationContext` uses internal caching so each value is computed at most once, even if multiple Doctors request it. This prevents redundant computation across the pipeline.

## Notes for Custom Doctor Authors

Doctors should always access data through `EvaluationContext` properties rather than computing metrics manually. This ensures:

1. Computed values are cached and shared across Doctors.
2. The correct metric variant (classification vs. regression) is used automatically.
3. Task-type inference is consistent across all Doctors.

See the [Plugins & Custom Doctors](../guides/plugins.md) guide for a complete custom Doctor implementation.
