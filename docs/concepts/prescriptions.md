# Prescriptions

The **Prescription Engine** transforms abstract diagnostic findings into concrete, actionable engineering tasks. When ModelDoctor identifies a `WARNING` or `CRITICAL` finding, it queries a built-in knowledge base and attaches a structured `Prescription` to the final report.

## The Prescription Object

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

| Field | Description |
|---|---|
| `id` | Unique identifier for the prescription rule (e.g., `OVR-001`). |
| `finding_title` | The exact title of the finding that triggered this prescription. |
| `dimension` | The diagnostic category (e.g., `Overfitting Risk`). |
| `recommendation` | A high-level description of what needs to change. |
| `implementation_steps` | A list of actionable engineering tasks (e.g., "Change `max_depth` from None to 10"). |
| `estimated_gains` | The expected impact of applying the fix (e.g., "Improves generalization by 10–15%"). |
| `confidence` | Heuristic likelihood that applying this fix resolves the symptom. |

## How Prescriptions Are Generated

After all Doctors complete their evaluations, the `PrescriptionEngine` scans the `Report` for any finding with a `WARNING` or `CRITICAL` severity. For each match, it queries its internal knowledge base (`modeldoctor.prescription.knowledge_base.PRESCRIPTIONS`). If a match is found by dimension, finding title, and tags, a `Prescription` is instantiated and attached to the report.

## Reading Prescriptions

```python
import modeldoctor as md

report = md.diagnose(model, X_train, y_train, X_test, y_test)

recs = report.prescription.all_recommendations
if recs:
    for rec in recs:
        print(f"Prescription [{rec.id}]: {rec.description}")
        print(f"  Estimated Gain: {rec.estimated_improvement}")
        for step in rec.steps:
            print(f"  - {step}")
```

## Knowledge Base

The knowledge base is a static dictionary located at `modeldoctor.prescription.knowledge_base.PRESCRIPTIONS`. It maps specific diagnostic signatures to recovery plans covering the eight built-in diagnostic dimensions.

The knowledge base is not user-facing but can be extended if you build [Custom Doctors](../guides/plugins.md) that produce custom finding titles.
