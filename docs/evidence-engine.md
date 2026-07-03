# Evidence Engine

The `EvidenceEngine` (and its localized component, `EvidenceBuilder`) forms the foundation of ModelDoctor's diagnostic pipeline. 

## Why Evidence Exists

In traditional validation libraries, rules often trigger hard `True`/`False` assertions. If a model's accuracy drops below 0.80, it fails. 

ModelDoctor uses an evidence-based approach instead. Doctors do not make final decisions; they simply collect **Signals** (e.g., "The generalization gap is 0.15"). This allows the downstream engines to weigh multiple pieces of evidence together before issuing a final diagnosis.

## EvidenceBuilder

Each Doctor instantiates an `EvidenceBuilder` during the `examine()` phase. When a threshold is crossed, the Doctor adds a piece of structured evidence to the builder.

```python
if gap > RULES.overfitting_gap_critical:
    builder.add(
        name="Generalization Gap", 
        measured_value=gap, 
        weight="Very High", 
        normalized_score=1.0, 
        expected_range="<= 0.15"
    )
```

## Signal Normalization

The `EvidenceBuilder` forces Doctors to standardize their signals:
- **Measured Value**: The raw float or int (e.g., `0.15`).
- **Expected Range**: A human-readable string explaining the threshold (e.g., `<= 0.15`).
- **Normalized Score**: A float between `0.0` and `1.0` representing how severely the threshold was violated.

## Weighted Evidence

Doctors assign a categorical weight (`Low`, `Medium`, `High`, `Very High`) to each piece of evidence. This indicates the heuristic importance of the signal. For example, a 50MB model size might be a `Medium` weight signal for production readiness, whereas a 2000ms inference latency might be `Very High`.

## Structured Evidence

The output of the `EvidenceEngine` is a collection of `DiagnosticEvidence` objects. These are eventually packaged into the final `Finding` objects, ensuring that all UI dashboards and exported JSON reports have a consistent schema, regardless of which Doctor generated the evidence.
