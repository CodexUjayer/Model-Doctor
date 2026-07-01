"""Task type detection for the EvaluationContext.

Given a model and target array, infers whether the task is binary classification,
multiclass classification, regression, or (as a last resort) unknown.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from modeldoctor.models.enums import TaskType
from modeldoctor.utils.compat import get_classes, has_predict_proba


def detect_task_type(model: Any, y: np.ndarray) -> TaskType:
    """Infer the ML task type from the model and target array.

    Detection logic (in priority order):

    1. If the model exposes ``classes_``, it is a classifier.
       - 2 classes → BINARY_CLASSIFICATION
       - ≥ 3 classes → MULTICLASS_CLASSIFICATION
    2. If ``predict_proba`` exists → classifier (same cardinality rules).
    3. If the target is floating-point with many unique values → REGRESSION.
    4. If the target is integer/boolean with ≤ 2 unique values → BINARY.
    5. If the target is integer with 3–20 unique values → MULTICLASS.
    6. Otherwise → REGRESSION.

    Args:
        model: A fitted estimator.
        y: Target array used for training or testing.

    Returns:
        The inferred :class:`TaskType`.
    """
    y_arr = np.asarray(y)

    # 1. Explicit classes_ attribute
    classes = get_classes(model)
    if classes is not None:
        n_classes = len(classes)
        if n_classes == 2:
            return TaskType.BINARY_CLASSIFICATION
        return TaskType.MULTICLASS_CLASSIFICATION

    # 2. predict_proba implies classification
    if has_predict_proba(model):
        unique = np.unique(y_arr)
        if len(unique) == 2:
            return TaskType.BINARY_CLASSIFICATION
        return TaskType.MULTICLASS_CLASSIFICATION

    # 3. Continuous target → regression
    if _is_continuous(y_arr):
        return TaskType.REGRESSION

    # 4. Discrete target
    unique = np.unique(y_arr)
    n_unique = len(unique)
    if n_unique <= 2:
        return TaskType.BINARY_CLASSIFICATION
    if n_unique <= 20:
        return TaskType.MULTICLASS_CLASSIFICATION

    # 5. Many unique integer values → regression
    return TaskType.REGRESSION


def _is_continuous(y: np.ndarray) -> bool:
    """Heuristic: is the target continuous (float with many unique values)?"""
    if y.dtype.kind == "f":
        # Float array — check if the values look like probabilities or real numbers
        n_unique = len(np.unique(y))
        return n_unique > 20
    return False
