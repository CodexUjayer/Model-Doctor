# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - Upcoming

### Added
- Core evaluation engine (`EvaluationContext`).
- Eight built-in Doctors (`OverfittingDoctor`, `LeakageDoctor`, `PredictionDoctor`, `DataDoctor`, `FeatureDoctor`, `CalibrationDoctor`, `ProductionDoctor`, `GeneralizationDoctor`).
- Evidence, Confidence, Risk, and Prescription engines.
- HTML Dashboard rendering.
- MLflow Integration (`log_report`).
- Validation Laboratory framework featuring 54 benchmark scenarios.

### Changed
- Refactored `BaseDoctor` to use the `EvidenceBuilder`.
- Tuned global thresholds in `rules.py` for 98.1% diagnostic accuracy.

### Removed
- Deprecated legacy hardcoded boolean checks in favor of the normalized Evidence Engine.
