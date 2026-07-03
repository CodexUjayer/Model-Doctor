# Custom Doctors

ModelDoctor's architecture allows you to inject custom domain-specific rules by creating a Custom Doctor. 

## BaseDoctor

All custom doctors must inherit from `modeldoctor.core.base_doctor.BaseDoctor`.

### Required Attributes
- `name`: A string identifier (e.g., `"DomainSpecificDoctor"`).
- `dimension`: The category the findings belong to (e.g., `"Compliance"`).
- `priority`: An integer determining execution order (lower numbers run first).

### Required Methods
- `examine(self, context: EvaluationContext) -> Diagnosis`: The core method where analysis occurs.

## Minimal Working Example

```python
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis, Finding
from modeldoctor.models.enums import Severity, Confidence

class ComplianceDoctor(BaseDoctor):
    name = "ComplianceDoctor"
    dimension = "Fairness & Compliance"
    priority = 90
    
    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()
        
        # Check if a protected attribute was used in training
        if context.feature_names and "age" in context.feature_names:
            finding = Finding(
                title="Protected Attribute Used",
                description="The feature 'age' is present in the training data.",
                severity=Severity.CRITICAL,
                confidence=Confidence.HIGH,
                tags=["compliance", "fairness"]
            )
            diagnosis.add_finding(finding)
            diagnosis.dimension_score = 0.0
        else:
            diagnosis.dimension_score = 100.0
            
        return diagnosis
```

## Registration

To use your custom doctor, inject it into the `ModelDoctorConfig` and pass it to `diagnose()`.

```python
from modeldoctor import ModelDoctorConfig
import modeldoctor as md

config = ModelDoctorConfig()
# Base doctors are enabled by default. We append our custom doctor.
config.active_doctors.append(ComplianceDoctor)

report = md.diagnose(..., config=config)
```

## Best Practices
- **Use the EvidenceEngine**: While the minimal example creates a `Finding` directly, advanced doctors should use `EvidenceBuilder`, `ConfidenceEngine`, and `RiskEngine` to ensure scoring is consistent with the rest of the library.
- **Fail Gracefully**: If your doctor requires a specific model type (e.g., neural networks), check `context.model` first and return an empty `Diagnosis` with a score of 100 if the model is unsupported.
