# Explainability Guide

ModelDoctor integrates feature importance analysis into the diagnostic pipeline to help identify leakage, irrelevant features, and dimensionality problems.

## How Feature Importance Works

When a Doctor requests feature importances, the `EvaluationContext` computes them using the best available method:

1. **SHAP values** — Used if the `[shap]` extra is installed. Provides accurate, model-agnostic Shapley values.
2. **Permutation Importance** — Used as a fallback when SHAP is not installed. Measures the drop in model score when each feature is randomly shuffled.
3. **Native importances** — For tree-based models, scikit-learn's built-in `feature_importances_` attribute is used as a fast secondary signal.

## Installing SHAP Support

```bash
pip install "modeldoctor[shap]"
```

Without the `[shap]` extra, ModelDoctor falls back to permutation importance. The diagnostic results will still be correct, but SHAP visualizations will not appear in the HTML dashboard.

## Which Doctors Use Feature Importance

| Doctor | How It Uses Feature Importance |
|---|---|
| `LeakageDoctor` | Detects if a single feature dominates importance (proxy leakage). |
| `FeatureDoctor` | Identifies zero-variance or constant features. |
| `OverfittingDoctor` | Checks for over-reliance on a small subset of features. |

## Running an Explainability-Focused Diagnosis

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import modeldoctor as md

X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Provide explicit feature names for readable results
feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

report = md.diagnose(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names
)
```

## Passing Named Features with DataFrames

If you pass pandas DataFrames instead of numpy arrays, ModelDoctor automatically extracts column names:

```python
import pandas as pd

X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])

report = md.diagnose(model, X_train_df, y_train, X_test_df, y_test)
```

## Viewing Feature Importance in the Dashboard

The HTML dashboard includes a **Feature Importance** tab when feature names are available. SHAP bar charts are rendered when SHAP is installed; otherwise a permutation importance bar chart is displayed.

```python
report.dashboard()  # Opens in browser
```

## Performance Considerations

Permutation importance requires shuffling each feature and re-scoring the model, which can be slow for large datasets. For datasets with more than 100k rows, consider sampling before calling `md.diagnose()`:

```python
import numpy as np

# Stratified sample of 10k rows
idx = np.random.default_rng(42).choice(len(X_train), size=10_000, replace=False)
report = md.diagnose(model, X_train[idx], y_train[idx], X_test, y_test)
```

## See Also

- [Calibration Guide](calibration.md)
- [Evaluation Context](../concepts/evaluation-context.md)
- [Findings Reference](../concepts/findings.md)
