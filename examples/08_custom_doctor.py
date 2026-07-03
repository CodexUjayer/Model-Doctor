"""
08 - Custom Doctor

Demonstrates how to extend ModelDoctor by writing a custom domain-specific
Doctor and injecting it into the diagnostic pipeline via the DoctorRegistry.
"""

from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md
from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.core.pipeline import DiagnosticPipeline
from modeldoctor.core.registry import DoctorRegistry
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.enums import Severity
from utils import get_classification_data


# --- 1. Define a custom Doctor ---
class FairnessDoctor(BaseDoctor):
    """Flags the use of sensitive attributes in the training feature set."""

    name = "FairnessDoctor"
    dimension = "Compliance"
    priority = 90  # Run after most built-in doctors

    # List of feature names considered sensitive/protected
    SENSITIVE_FEATURES = {"age", "gender", "race", "feature_0"}

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()

        if not context.feature_names:
            return diagnosis

        detected = [f for f in context.feature_names if f in self.SENSITIVE_FEATURES]

        if detected:
            # Use _finding() — the recommended factory helper from BaseDoctor
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


def main():
    print("Generating data and training model...")
    X_train, X_test, y_train, y_test = get_classification_data(random_state=42)

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # --- 2. Build a custom registry with the extra doctor ---
    registry = DoctorRegistry.default()          # all 9 built-in doctors
    registry.register(FairnessDoctor)            # inject the custom one
    print(f"Registry loaded with {len(registry)} doctors.")

    # --- 3. Build a pipeline using the custom registry ---
    pipeline = DiagnosticPipeline(registry=registry)

    # Run the pipeline directly (diagnoses only — no prescriptions)
    from modeldoctor.core.context import EvaluationContext
    from modeldoctor import ModelDoctorConfig
    context = EvaluationContext(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=ModelDoctorConfig(),
    )
    diagnoses = pipeline.run(context)

    print(f"\nDoctors executed: {[d.doctor_name for d in diagnoses]}")

    # Check whether the custom doctor produced a finding
    fairness_diagnosis = next((d for d in diagnoses if d.doctor_name == "FairnessDoctor"), None)
    if fairness_diagnosis and fairness_diagnosis.findings:
        for f in fairness_diagnosis.findings:
            print(f"\n[{f.severity.value.upper()}] {f.title}")
            print(f"  {f.explanation}")
    else:
        print("\nFairnessDoctor: No sensitive features detected.")


if __name__ == "__main__":
    main()
