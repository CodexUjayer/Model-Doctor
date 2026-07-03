# Validation Laboratory

The ModelDoctor Validation Laboratory is an independent, evidence-based testing framework designed to prove the accuracy of ModelDoctor's diagnostic engine against real-world edge cases.

## Purpose

When evaluating a diagnostic tool, you must be confident that its heuristics and thresholds align with reality. If a library flags everything as "Overfitting", the alerts become noise. If it misses glaring target leakage, it is dangerous. 

The Validation Laboratory exists to continuously prove that ModelDoctor's diagnoses align with what a senior machine learning engineer would conclude when looking at the same dataset.

## Benchmark Philosophy

The laboratory runs 54 distinct, deterministic benchmark scenarios. 

Each scenario:
1. Generates a specific synthetic dataset (e.g., highly imbalanced, or containing a subtle timestamp leak).
2. Trains a specific Scikit-learn model on that data.
3. Passes the model to `md.diagnose()`.
4. Asserts that the resulting `Report` contains exactly the expected findings and severities, and *nothing else*.

## Benchmark Runner and Reports

The laboratory executes via `validation/benchmark_runner.py`. 

The runner orchestrates the creation of all datasets, models, and contexts, executes the diagnostic pipeline, and compares the output to the scenario's expected results. It generates comprehensive HTML, Markdown, and JSON reports summarizing the pipeline's diagnostic accuracy.

## Current Validation Accuracy

As of v1.0, the Validation Laboratory tests ModelDoctor against **54 Scenarios** covering 8 distinct diagnostic dimensions.

**Diagnostic Accuracy: 98.1%**

The framework successfully catches subtle overfitting memorization, proxy data leaks, calibration errors, and production serialization bloat without throwing false positives on perfectly healthy datasets.

## Limitations & Future Improvements

- **Scenario Diversity**: Currently, the laboratory primarily tests Random Forest, Gradient Boosting, Logistic Regression, and Decision Trees. 
- **Deep Learning**: Benchmarks for neural network specific failure modes (e.g., catastrophic forgetting, vanishing gradients) are planned alongside PyTorch integration.
- **Multiclass Nuance**: While supported, there are fewer benchmark scenarios specifically targeting multiclass calibration errors.
