# Architecture

ModelDoctor is built on a modular, multi-engine pipeline. The pipeline separates the extraction of raw metrics from the diagnosis of those metrics, and separates the diagnosis from the final risk scoring.

<div align="center">
  <img src="images/architecturediagram.png" alt="ModelDoctor Architecture Diagram" width="100%" />
</div>

## The Evaluation Pipeline

### 1. EvaluationContext
The `EvaluationContext` is the state container passed through the entire pipeline. It wraps your model, training data, and test data. Crucially, it evaluates metrics *lazily*. For example, it only calculates permutation importance or cross-validation scores if a Doctor actually requests them, preventing unnecessary computation.

### 2. ModelProfiler
Before doctors examine the model, the `ModelProfiler` scans the `EvaluationContext` to infer the problem domain. It detects whether the task is binary classification, multiclass classification, or regression. It also builds the `ModelPassport`, which records the model's framework, hyperparameter configuration, and dataset footprint.

### 3. DoctorRegistry & Doctors
ModelDoctor ships with a registry of specialized `Doctors`. A Doctor is an isolated module responsible for analyzing a single dimension of model health (e.g., `LeakageDoctor`, `OverfittingDoctor`). The registry loops through all active Doctors, passing them the `EvaluationContext`.

### 4. EvidenceBuilder
Inside each Doctor, the `EvidenceBuilder` is used to collect raw signals. Instead of immediately deciding if a metric is "good" or "bad", the builder standardizes signals into a structured format (measured value, expected range, and raw severity weighting).

### 5. ConfidenceEngine
Once evidence is collected, it passes through the `ConfidenceEngine`. This engine calculates a statistical confidence score (Low, Medium, High, Very High) representing how certain ModelDoctor is that an issue exists, based on the volume and weight of the collected evidence.

### 6. RiskEngine
The `RiskEngine` calculates the actual danger posed to your application. It converts raw evidence scores into standard Severity mappings (`INFO`, `WARNING`, `CRITICAL`). Two findings might have identical Confidence, but vastly different Risk scores.

### 7. PrescriptionEngine
If the RiskEngine flags a `WARNING` or `CRITICAL` issue, the `PrescriptionEngine` queries its internal knowledge base. It maps the specific finding to actionable steps and estimated gains, giving you a clear path to resolution.

### 8. HealthScorer & ReportRenderer
Finally, the `HealthScorer` aggregates the outputs of all Doctors into a unified 0–100 score. The `ReportRenderer` then formats this data into the final `Report` object, which can be exported to HTML, JSON, or Markdown.
