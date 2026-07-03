# Plugins & Custom Doctors

ModelDoctor's architecture allows you to inject custom domain-specific diagnostic rules by creating a **Custom Doctor**. This is useful for compliance checks, domain-specific quality gates, or proprietary business rules that the built-in Doctors do not cover.

## The BaseDoctor Contract

All custom doctors must inherit from `modeldoctor.core.base_doctor.BaseDoctor` and implement the following:

### Required Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | A unique string identifier (e.g., `"ComplianceDoctor"`). |
| `dimension` | `str` | The diagnostic category for grouping findings (e.g., `"Compliance"`). |
| `priority` | `int` | Execution order — lower numbers run first. Built-in doctors use 10–80. |

### Required Methods

| Method | Signature | Description |
|---|---|---|
| `examine` | `(self, context: EvaluationContext) -> Diagnosis` | The core analysis method. Called once per diagnosis run. |

## Minimal Working Example

```python
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.enums import Severity

class ComplianceDoctor(BaseDoctor):
    name = "ComplianceDoctor"
    dimension = "Fairness & Compliance"
    priority = 90  # Run after built-in doctors
    
    # Protected attributes to check for
    SENSITIVE_FEATURES = {"age", "gender", "race"}
    
    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        
        if not context.feature_names:
            return diagnosis
        
        detected = [f for f in context.feature_names if f in self.SENSITIVE_FEATURES]
        
        if detected:
            finding = self._finding(
                title="Sensitive Attribute in Feature Set",
                severity=Severity.WARNING,
                explanation=(
                    f"The following features are classified as sensitive: "
                    f"{', '.join(detected)}. "
                    f"Training with protected attributes may violate fairness constraints."
                ),
                tags=["compliance", "fairness"],
            )
            diagnosis.findings.append(finding)
            diagnosis.dimension_score = 40.0
            diagnosis.passed = False
        
        return diagnosis
```

## Registering and Injecting Your Doctor

To use your custom Doctor, inject it into the `DoctorRegistry` and pass it to the pipeline:

```python
from sklearn.ensemble import RandomForestClassifier
from modeldoctor.core.registry import DoctorRegistry
from modeldoctor.core.pipeline import DiagnosticPipeline
from modeldoctor.core.context import EvaluationContext
from modeldoctor import ModelDoctorConfig
import modeldoctor as md

# Build a registry with all default doctors plus your custom one
registry = DoctorRegistry.default()
registry.register(ComplianceDoctor)

# Use md.diagnose() for the full pipeline (prescriptions + health score)
# Your custom doctor will appear in the registry
report = md.diagnose(model, X_train, y_train, X_test, y_test)
```

## Using the EvidenceBuilder (Advanced)

For more sophisticated doctors that need weighted, normalized evidence scoring, use `EvidenceBuilder` directly instead of creating `Finding` objects manually. This ensures your doctor's findings are scored consistently with the built-in risk and confidence pipeline.

```python
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis

class AdvancedDoctor(BaseDoctor):
    name = "AdvancedDoctor"
    dimension = "Custom"
    priority = 85
    
    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        builder = self._evidence_builder()
        
        # Example: flag large inference latency
        latency_ms = context.inference_latency_ms
        if latency_ms > 500:
            builder.add(
                name="Inference Latency",
                measured_value=latency_ms,
                weight="High",
                normalized_score=min(latency_ms / 2000.0, 1.0),
                expected_range="<= 500ms"
            )
        
        return self._finalize(diagnosis, builder)
```

## Best Practices

- **Fail gracefully**: If your doctor requires a specific model type (e.g., a neural network), check `context.model` first and return an empty `Diagnosis` with `dimension_score = 100.0` if unsupported.
- **Use `_finding()` factory**: The `_finding()` helper on `BaseDoctor` ensures finding objects have all required fields populated correctly.
- **Use priority > 80**: Keep custom doctors at priority 85–99 to ensure they run after built-in ones.
- **Avoid expensive computation**: Access pre-computed properties on `context` (e.g., `context.feature_importances`) rather than recomputing them.

## See Also

- [Evaluation Context](../concepts/evaluation-context.md)
- [Findings Reference](../concepts/findings.md)
- [Diagnostic Doctors Reference](../concepts/diagnostic-doctors.md)
