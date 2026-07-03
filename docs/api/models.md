# Models API

ModelDoctor relies on Pydantic models to guarantee data structure integrity throughout the pipeline.

## `Finding`

Represents a final diagnosed issue after Confidence and Risk engines have processed the raw evidence.

```python
class Finding(BaseModel):
    title: str
    description: str
    severity: Severity
    confidence: Confidence
    risk_score: float
    risk_level: str
    evidence: Dict[str, Any]
    structured_evidence: List[DiagnosticEvidence]
    tags: List[str]
```

## `DiagnosticEvidence`

Represents raw signals collected by Doctors.

```python
class DiagnosticEvidence(BaseModel):
    name: str
    measured_value: float
    expected_range: str
    weight: str
    normalized_score: float
```

## `ModelPassport`

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

## Enums

### `Severity`
`INFO`, `WARNING`, `CRITICAL`

### `Confidence`
`LOW`, `MEDIUM`, `HIGH`

### `TaskType`
`BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, `REGRESSION`
