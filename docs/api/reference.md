# API Reference

Complete API reference for ModelDoctor v1.0.

---

## Core API

### `md.diagnose()`

The primary entry point to ModelDoctor. Executes the entire diagnostic pipeline synchronously and returns a populated `Report` object.

```python
import modeldoctor as md

report = md.diagnose(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    X_val=None,
    y_val=None,
    feature_names=None,
    config=None,
    progress_callback=None
)
```

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `BaseEstimator` | A fitted scikit-learn compatible estimator. Must implement `fit` and `predict`. |
| `X_train` | `ndarray` / `DataFrame` | The training feature matrix. |
| `y_train` | `ndarray` / `Series` | The training target array. |
| `X_test` | `ndarray` / `DataFrame` | The test (hold-out) feature matrix. |
| `y_test` | `ndarray` / `Series` | The test (hold-out) target array. |
| `X_val` | `ndarray` / `DataFrame` | Optional secondary validation feature matrix. |
| `y_val` | `ndarray` / `Series` | Optional secondary validation target array. |
| `feature_names` | `List[str]` | Optional explicit feature names. If not provided and inputs are DataFrames, column names are used automatically. |
| `config` | `ModelDoctorConfig` | Optional configuration object to override default thresholds and active Doctors. |
| `progress_callback` | `Callable[[str], None]` | Optional function that receives string progress updates. |

#### Returns

- **`Report`**: A populated report object containing all findings, prescriptions, and export methods.

#### Exceptions

- **`ValueError`**: Raised if shapes between `X_train` and `X_test` mismatch, or if the model is unfitted or entirely unsupported.

---

## Report Object

The `Report` object encapsulates the final diagnostic results and provides export utilities.

### Properties

| Property | Type | Description |
|---|---|---|
| `health_score` | `HealthScore` | The overall 0–100 diagnostic score and letter grade. |
| `diagnoses` | `List[Diagnosis]` | Per-doctor diagnosis objects containing findings and dimension scores. |
| `prescription` | `PrescriptionResult` | Generated actionable recommendations. |
| `passport` | `ModelPassport` | Metadata about the model footprint and datasets. |

### Methods

#### `dashboard()`
Opens the interactive HTML dashboard in the default system web browser.

#### `save_html(path: str)`
Renders and saves the self-contained HTML dashboard to the specified path.

#### `save_json(path: str)`
Serializes the entire report to a JSON file.

#### `save_markdown(path: str)`
Exports a human-readable Markdown summary of the findings and prescriptions.

#### `save_csv(path: str)`
Exports a flat CSV file of the findings (dimension, title, severity).

#### `to_dataframe() -> pandas.DataFrame`
Returns the findings as a pandas `DataFrame`.

---

## EvaluationContext

The central state container passed through the pipeline. Constructed internally by `md.diagnose()`.

```python
class EvaluationContext:
    def __init__(self, model, X_train, y_train, X_test, y_test, ...)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `model` | `BaseEstimator` | The original fitted model. |
| `X_train`, `y_train` | `ndarray` / `DataFrame` | Training datasets. |
| `X_test`, `y_test` | `ndarray` / `DataFrame` | Testing datasets. |
| `task_type` | `TaskType` | Inferred task (`BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, `REGRESSION`). |
| `feature_names` | `List[str]` | Feature names extracted from DataFrame columns or passed explicitly. |
| `feature_importances` | `ndarray` | Lazy-evaluated feature importance scores (SHAP or permutation importance). |
| `train_score` | `float` | Model score on training data. |
| `test_score` | `float` | Model score on testing data. |
| `cv_scores` | `ndarray` | Cross-validation scores (lazy evaluated). |
| `classification_metrics` | `dict` | Accuracy, F1, precision, recall (classification only). |
| `regression_metrics` | `dict` | MSE, MAE, R² (regression only). |

---

## BaseDoctor

The abstract base class that all Doctors must implement.

```python
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis

class BaseDoctor(ABC):
    name: str
    dimension: str
    priority: int

    @abstractmethod
    def examine(self, context: EvaluationContext) -> Diagnosis:
        pass
```

### Built-in Doctors

| Doctor | Dimension | Task Support |
|---|---|---|
| `OverfittingDoctor` | Overfitting Risk | Classification + Regression |
| `LeakageDoctor` | Data Leakage | Classification + Regression |
| `PredictionDoctor` | Prediction Quality | Classification + Regression |
| `DataDoctor` | Data Quality | Classification + Regression |
| `FeatureDoctor` | Feature Quality | Classification + Regression |
| `CalibrationDoctor` | Calibration | Classification only |
| `ProductionDoctor` | Production Readiness | Classification + Regression |
| `GeneralizationDoctor` | Generalization | Classification + Regression |

---

## Data Models

### `Finding`

Represents a diagnosed issue after Confidence and Risk engines have processed evidence.

```python
class Finding(BaseModel):
    title: str
    description: str
    severity: Severity          # INFO | WARNING | CRITICAL
    confidence: Confidence      # LOW | MEDIUM | HIGH
    risk_score: float
    risk_level: str
    evidence: Dict[str, Any]
    structured_evidence: List[DiagnosticEvidence]
    tags: List[str]
```

### `DiagnosticEvidence`

Raw signals collected by Doctors.

```python
class DiagnosticEvidence(BaseModel):
    name: str
    measured_value: float
    expected_range: str
    weight: str                 # Low | Medium | High | Very High
    normalized_score: float     # 0.0 (no violation) to 1.0 (maximum violation)
```

### `ModelPassport`

Snapshot of the model's footprint.

```python
class ModelPassport(BaseModel):
    framework: str
    model_family: str
    training_samples: int
    feature_count: int
    model_size_bytes: int
    inference_latency_ms: float
```

### `Prescription`

Generated by the `PrescriptionEngine` for `WARNING` / `CRITICAL` findings.

```python
class Prescription(BaseModel):
    id: str
    finding_title: str
    dimension: str
    recommendation: str
    implementation_steps: List[str]
    estimated_gains: str
    confidence: str
```

---

## Enums

### `Severity`
`INFO` | `WARNING` | `CRITICAL`

### `Confidence`
`LOW` | `MEDIUM` | `HIGH`

### `TaskType`
`BINARY_CLASSIFICATION` | `MULTICLASS_CLASSIFICATION` | `REGRESSION`

---

## Validation API

The Validation Laboratory is an independent module for contributors testing the diagnostic engine.

### `ValidationScenario`

```python
class ValidationScenario(ABC):
    name: str
    category: str
    description: str
    random_seed: int

    @abstractmethod
    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        """Returns X_train, X_test, y_train, y_test"""
        pass

    @abstractmethod
    def build_model(self, X_train, y_train) -> Any:
        """Returns a fitted model"""
        pass

    @abstractmethod
    def expected(self) -> ExpectedResult:
        """Defines the strict passing conditions"""
        pass
```

### `ExpectedResult`

```python
class ExpectedResult(BaseModel):
    passed: bool
    findings: Optional[List[str]] = None
    severity: Optional[List[str]] = None
```

---

## MLflow Integration

```python
from modeldoctor.integrations.mlflow import log_report

with mlflow.start_run():
    log_report(report, artifact_path="modeldoctor")
```

Logs `md_health_score`, `md_critical_findings`, and `md_warning_findings` as scalar metrics, and uploads `dashboard.html`, `report.json`, and `summary.md` as artifacts.

See the [MLflow Integration Guide](../guides/mlflow.md) for details.
