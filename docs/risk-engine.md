# Risk Engine

The `RiskEngine` is responsible for determining the practical danger posed by the evidence collected by the Doctors.

## Risk Scoring

After the `ConfidenceEngine` determines *certainty*, the `RiskEngine` calculates a `risk_score` (float 0.0 - 1.0). This score heavily weights the highest-severity signals. If a single piece of evidence has a normalized score of `1.0` (indicating a massive threshold violation), the `RiskEngine` will output a high risk score, even if all other signals are healthy.

## Severity Mapping

The raw `risk_score` is converted into a categorical `risk_level` (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

Finally, this `risk_level` is mapped to the standard `Severity` enum used across the ModelDoctor reporting system:

- `CRITICAL` ➔ `Severity.CRITICAL`
- `HIGH` ➔ `Severity.WARNING`
- `MEDIUM` ➔ `Severity.WARNING`
- `LOW` ➔ `Severity.INFO`
- `INFO` ➔ `Severity.INFO`

## Critical vs. Warning

A **CRITICAL** severity indicates an issue that will likely cause immediate model failure, severe business impact, or entirely invalidate the evaluation metrics (e.g., obvious target leakage).

A **WARNING** severity indicates an issue that degrades performance, wastes resources, or risks edge-case failures, but might be acceptable in a prototype or low-stakes environment (e.g., mild overfitting or high inference latency).

Any finding categorized as `INFO` is generally excluded from the primary alert dashboard, as it represents healthy or expected behavior.
