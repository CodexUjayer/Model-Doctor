# ModelDoctor Validation Framework

This directory contains the standalone validation suite for ModelDoctor. It is designed to act as the final quality gate before a production release by scientifically verifying that the library produces accurate diagnoses and recommendations across realistic machine learning scenarios.

## Structure
- `utils/`: Common generators for datasets and models, as well as assertion helpers.
- `classification/`: Tests for classification models (e.g. overfitting, target leakage, class imbalance).
- `regression/`: Tests for regression models.
- `edge_cases/`: Tests for extreme or broken inputs (e.g. tiny datasets, extreme correlations, missing values).
- `production/`: Tests for production constraints (e.g. model size, slow inference).
- `reports/`: Where the validation benchmark outputs are saved.

## Running the Validation Suite

To run all validation scenarios and produce a summary report:

```bash
python validation/validation_summary.py
```

This will output a Rich console summary and generate `validation_summary.json` and `validation_summary.md` inside `validation/reports/`.

## Adding a New Scenario

1. Create a `test_<name>.py` script in one of the category directories.
2. Define a function starting with `run_validation_` (e.g., `run_validation_target_leakage`).
3. Set up the dataset, train a model, and call `md.diagnose()`.
4. Use the helper functions from `validation.utils.assertions` to verify the report findings.
5. Return the report if successful. The runner will automatically discover and execute it.
