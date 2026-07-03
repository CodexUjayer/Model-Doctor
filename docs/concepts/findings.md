# Findings

A **Finding** is the primary output of a Doctor's examination. It represents a diagnosed issue — a structured object capturing exactly what was found, how severe it is, and the raw evidence that supports the conclusion.

## The Finding Object

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

### Fields

| Field | Type | Description |
|---|---|---|
| `title` | `str` | A short, human-readable label for the issue (e.g., `"Generalization Gap"`). |
| `description` | `str` | A detailed explanation of the finding and why it is problematic. |
| `severity` | `Severity` | The final alert level: `INFO`, `WARNING`, or `CRITICAL`. |
| `confidence` | `Confidence` | How certain ModelDoctor is that this issue is real: `LOW`, `MEDIUM`, `HIGH`. |
| `risk_score` | `float` | A continuous risk score from 0.0 to 1.0. |
| `risk_level` | `str` | The categorical risk level (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`). |
| `evidence` | `dict` | A flat dictionary of the raw measured values (e.g., `{"generalization_gap": 0.23}`). |
| `structured_evidence` | `List[DiagnosticEvidence]` | Structured evidence objects with thresholds and weights. |
| `tags` | `List[str]` | Categorical tags used for filtering (e.g., `["overfitting", "generalization"]`). |

## Severity Levels

| Level | Meaning |
|---|---|
| `INFO` | Healthy or expected behavior. Not shown in primary alert dashboard. |
| `WARNING` | Degrades performance or risks edge-case failures. Review recommended. |
| `CRITICAL` | Will likely cause immediate model failure or entirely invalidate evaluation metrics. |

## DiagnosticEvidence

Each finding is backed by one or more `DiagnosticEvidence` objects — the raw signals collected by the Doctor before Confidence and Risk scoring:

```python
class DiagnosticEvidence(BaseModel):
    name: str
    measured_value: float
    expected_range: str
    weight: str
    normalized_score: float
```

| Field | Description |
|---|---|
| `name` | Name of the signal (e.g., `"Generalization Gap"`). |
| `measured_value` | The raw value measured from the model (e.g., `0.23`). |
| `expected_range` | Human-readable threshold (e.g., `"<= 0.15"`). |
| `weight` | Heuristic importance: `Low`, `Medium`, `High`, `Very High`. |
| `normalized_score` | How severely the threshold was violated (0.0 = no violation, 1.0 = maximum violation). |

## Reading Findings Programmatically

```python
import modeldoctor as md

report = md.diagnose(model, X_train, y_train, X_test, y_test)

# Gather all findings across all doctors
all_findings = [f for d in report.diagnoses for f in (d.findings or [])]

for finding in all_findings:
    if finding.severity.value in ("warning", "critical"):
        print(f"[{finding.severity.value.upper()}] {finding.title}")
        print(f"  {finding.explanation}")
        for ev in finding.structured_evidence:
            print(f"  Evidence: {ev.name} = {ev.measured_value} (expected {ev.expected_range})")
```

## Relationship to Prescriptions

Every `WARNING` or `CRITICAL` finding is automatically matched against the [Prescription Engine](prescriptions.md) knowledge base. If a match is found, a `Prescription` is generated and attached to the report with concrete remediation steps.
