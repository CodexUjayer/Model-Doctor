# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Core architecture (`EvaluationContext`, `BaseDoctor`, `DiagnosticPipeline`, `DoctorRegistry`)
- Data models for `Diagnosis`, `Finding`, `Recommendation`, `Report`
- 8 initial diagnostic modules (Data, Feature, Overfitting, Leakage, Hyperparameter, Prediction, Generalization, Production)
- Prescription Engine using YAML rules
- Health Scorer with weighted dimensions
- Reporting system (Markdown, AI Review Generator)
