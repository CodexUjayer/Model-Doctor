# Advanced Examples

This page covers three advanced ModelDoctor scenarios that go beyond basic diagnosis: writing a custom Doctor, detecting data leakage, and analyzing overfitting in depth.

All runnable scripts are in the `examples/` directory of the repository.

---

## Example 08: Custom Doctor

**Script:** `examples/08_custom_doctor.py`

Extends ModelDoctor by writing a `FairnessDoctor` that flags protected attributes in the training feature set. This demonstrates the full custom Doctor API: inheritance, the `_finding()` factory, and DoctorRegistry injection.

```python
from sklearn.ensemble import RandomForestClassifier
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.core.registry import DoctorRegistry
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.enums import Severity
import modeldoctor as md


class FairnessDoctor(BaseDoctor):
    """Flags the use of sensitive attributes in the training feature set."""

    name = "FairnessDoctor"
    dimension = "Compliance"
    priority = 90

    SENSITIVE_FEATURES = {"age", "gender", "race", "feature_0"}

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


# Inject via DoctorRegistry
registry = DoctorRegistry.default()
registry.register(FairnessDoctor)
print(f"Registry loaded with {len(registry)} doctors.")
```

For a full explanation of the Custom Doctor API, see the [Plugins & Custom Doctors](../guides/plugins.md) guide.

---

## Example 13: Data Leakage Detection

**Script:** `examples/13_data_leakage_detection.py`

Demonstrates how ModelDoctor detects subtle proxy target leakage — a common training error where a feature contains future information unavailable at inference time.

```python
from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md

# Assumes a leaky dataset where 'customer_id_leak' is derived from the target
report = md.diagnose(model, X_train, y_train, X_test, y_test)

print(f"Health Score: {report.health_score.overall:.1f} / 100")

# Inspect LeakageDoctor findings
all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
leakage_findings = [f for f in all_findings if "Leakage" in f.title]

if leakage_findings:
    print("\nLeakage findings detected:")
    for f in leakage_findings:
        print(f"  [{f.severity.value.upper()}] {f.title}")
        print(f"  Explanation: {f.explanation}")
```

### Why Data Leakage Matters

A model trained on a leaky feature produces inflated accuracy during testing but will fail catastrophically when deployed — the leak is unavailable at inference time. ModelDoctor's `LeakageDoctor` detects this by measuring feature correlation and feature importance concentration.

---

## Example 14: Overfitting Analysis

**Script:** `examples/14_overfitting_analysis.py`

Diagnoses an unconstrained Decision Tree that has memorized training noise, demonstrating the `OverfittingDoctor`'s generalization gap analysis.

```python
from sklearn.tree import DecisionTreeClassifier
import modeldoctor as md

model = DecisionTreeClassifier(max_depth=None, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy:      {train_acc:.3f}")
print(f"Test Accuracy:       {test_acc:.3f}")
print(f"Generalization Gap:  {train_acc - test_acc:.3f}")

report = md.diagnose(model, X_train, y_train, X_test, y_test)

# Gather overfitting findings
all_findings = [f for d in report.diagnoses for f in (d.findings or [])]
overfit_findings = [f for f in all_findings if f.doctor_name == "OverfittingDoctor"]

for f in overfit_findings:
    print(f"[{f.severity.value.upper()}] {f.title}: {f.explanation}")
```

### The OverfittingDoctor Examines

- The generalization gap (train accuracy vs. test accuracy).
- Memorization rate on noisy label data.
- Unconstrained hyperparameters (e.g., `max_depth=None`).

**Fix:** Add `max_depth` constraints, apply regularization, or increase the training dataset size.

---

## Running the Examples

```bash
cd examples
python 08_custom_doctor.py
python 13_data_leakage_detection.py
python 14_overfitting_analysis.py
```

## Next Steps

- [Plugins & Custom Doctors](../guides/plugins.md) — complete guide to writing custom Doctors.
- [Calibration Guide](../guides/calibration.md) — probability calibration with Platt Scaling.
- [Model Comparison](../guides/comparison.md) — comparing two models with separate diagnoses.
