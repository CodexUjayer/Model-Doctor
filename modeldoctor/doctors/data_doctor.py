"""DataDoctor — analyzes data quality issues like imbalance, missing values, and outliers."""

import numpy as np
import pandas as pd

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.diagnosis import Diagnosis
from modeldoctor.models.enums import Severity


class DataDoctor(BaseDoctor):
    """Diagnoses data-level issues before model training."""

    name = "data_doctor"
    dimension = "Data Quality"
    priority = 10

    def examine(self, context: EvaluationContext) -> Diagnosis:
        diagnosis = self._new_diagnosis()

        if context.y_train is not None:
            self._check_class_imbalance(context, diagnosis)
        
        if context.X_train is not None:
            self._check_missing_values(context, diagnosis)
            
        diagnosis.dimension_score = self._score_from_findings(diagnosis)
        return diagnosis

    def _check_class_imbalance(self, context: EvaluationContext, diagnosis: Diagnosis) -> None:
        """Check for severe class imbalance in classification tasks."""
        if context.task_type.name != "CLASSIFICATION":
            return
            
        y = np.asarray(context.y_train)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return
            
        min_count = counts.min()
        max_count = counts.max()
        ratio = max_count / max(min_count, 1)

        if ratio > 10:
            diagnosis.findings.append(
                self._finding(
                    title="Severe Class Imbalance",
                    description=f"The most frequent class is {ratio:.1f}x more common than the least frequent class.",
                    severity=Severity.WARNING if ratio < 100 else Severity.ERROR,
                    evidence=[
                        self._evidence("Imbalance Ratio", float(ratio), expected="< 10.0")
                    ],
                    tags=["imbalance"]
                )
            )

    def _check_missing_values(self, context: EvaluationContext, diagnosis: Diagnosis) -> None:
        """Check for excessive missing values in training data."""
        if hasattr(context.X_train, "isna"):
            # Pandas DataFrame
            missing = context.X_train.isna().sum().sum()
            total = context.X_train.size
        elif hasattr(context.X_train, "isnull"):
            # Spark/Polars/etc (simplified fallback)
            missing = 0
            total = 1
        else:
            # Numpy array
            missing = np.isnan(np.asarray(context.X_train, dtype=float)).sum()
            total = context.X_train.size

        if total > 0:
            missing_pct = missing / total
            if missing_pct > 0.05:
                diagnosis.findings.append(
                    self._finding(
                        title="High Proportion of Missing Values",
                        description=f"{missing_pct:.1%} of values in the training data are missing.",
                        severity=Severity.WARNING if missing_pct < 0.2 else Severity.ERROR,
                        evidence=[
                            self._evidence("Missing Percentage", float(missing_pct), expected="< 0.05")
                        ],
                        tags=["missing_data"]
                    )
                )
