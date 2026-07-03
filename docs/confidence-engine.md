# Confidence Engine

The `ConfidenceEngine` calculates how statistically certain ModelDoctor is that a specific issue exists within the model.

## Confidence Calculation

When a Doctor finishes examining the `EvaluationContext`, it passes its collected evidence list to the `ConfidenceEngine`. 

The engine calculates a `confidence_score` (float 0.0 - 1.0) based on:
1. **Evidence Volume**: The number of independent signals collected.
2. **Evidence Weight**: The categorical weights (`Low` to `Very High`) assigned to those signals.
3. **Normalized Scores**: The severity of the threshold violations.

This continuous score is then bucketed into a standard categorical `Confidence` enum:
- `Confidence.LOW`
- `Confidence.MEDIUM`
- `Confidence.HIGH`

## Confidence vs. Risk vs. Severity

It is crucial to understand the distinction between these three concepts in ModelDoctor:

- **Confidence**: How *sure* are we that this phenomenon is occurring? (e.g., "I am highly confident the training dataset has missing values.")
- **Risk**: How *dangerous* is this phenomenon to the application? (e.g., "The missing value ratio is 1%, so the risk is Low.")
- **Severity**: The final categorisation used for alerts and UI (`INFO`, `WARNING`, `CRITICAL`), derived from Risk.

### Why separate Confidence from Risk?

Consider a model that exhibits a 0.01 gap between training and testing accuracy. 
- Because we have thousands of cross-validation samples proving this gap exists, our **Confidence** in the gap is `HIGH`.
- However, a 0.01 gap is completely normal and healthy. Therefore, the **Risk** is `LOW`.

By separating these concepts, ModelDoctor avoids throwing false-positive critical alerts for phenomena that are statistically certain but practically harmless.
