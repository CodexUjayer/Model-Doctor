# Prescription Engine

The `PrescriptionEngine` transforms abstract diagnostic findings into concrete, actionable engineering tasks.

## Knowledge Base

The engine is backed by a static knowledge base (`modeldoctor.prescription.knowledge_base.PRESCRIPTIONS`). This database maps specific diagnostic signatures (combinations of dimension, finding titles, and tags) to predefined recovery plans.

## How Prescriptions Are Generated

After all Doctors have completed their evaluations, the `PrescriptionEngine` scans the resulting `Report` for any `Finding` with a `WARNING` or `CRITICAL` severity.

For each severe finding, the engine queries the knowledge base. If a match is found, it instantiates a `Prescription` object and attaches it to the final report.

## The Prescription Object

A `Prescription` is a highly structured object containing:

- **Rules**: The specific conditions that triggered the prescription.
- **Recommendations**: A high-level description of what needs to be changed.
- **Implementation Steps**: A bulleted list of actionable engineering tasks (e.g., "Change `max_depth` from None to 10").
- **Estimated Gains**: The expected impact of the fix (e.g., "Prevents memorization, improves generalization by 10-15%").
- **Confidence**: The heuristic likelihood that applying this fix will resolve the underlying symptom.

By providing implementation steps and estimated gains, ModelDoctor bridges the gap between evaluation and engineering, saving developers hours of debugging and research time.
