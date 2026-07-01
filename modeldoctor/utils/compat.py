"""Framework compatibility utilities.

Provides a unified interface for inspecting models from different ML frameworks
(scikit-learn, XGBoost, LightGBM, CatBoost) without importing them at the top
level. All imports are guarded so that only the installed frameworks are used.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from modeldoctor.models.enums import FrameworkType


def _is_available(module_name: str) -> bool:
    """Check whether *module_name* is importable without raising."""
    return importlib.util.find_spec(module_name) is not None


def detect_framework(model: Any) -> FrameworkType:
    """Detect which ML framework produced *model*.

    Checks the MRO of the model's class against known base classes from each
    supported framework, in order of specificity.

    Args:
        model: A fitted ML model instance.

    Returns:
        The detected :class:`FrameworkType`.
    """
    cls_name = type(model).__module__

    if "catboost" in cls_name:
        return FrameworkType.CATBOOST
    if "lightgbm" in cls_name:
        return FrameworkType.LIGHTGBM
    if "xgboost" in cls_name:
        return FrameworkType.XGBOOST
    if "sklearn" in cls_name:
        return FrameworkType.SKLEARN

    # Fallback: check MRO strings
    mro_modules = {c.__module__ for c in type(model).__mro__}
    for fw, mod in [
        (FrameworkType.CATBOOST, "catboost"),
        (FrameworkType.LIGHTGBM, "lightgbm"),
        (FrameworkType.XGBOOST, "xgboost"),
        (FrameworkType.SKLEARN, "sklearn"),
    ]:
        if any(mod in m for m in mro_modules):
            return fw

    return FrameworkType.UNKNOWN


def get_model_params(model: Any) -> Dict[str, Any]:
    """Extract hyperparameters from a model in a framework-agnostic way.

    For sklearn-compatible models, calls ``get_params(deep=False)``.
    For others, attempts ``get_params()``, then falls back to ``__dict__``.

    Args:
        model: A fitted ML model instance.

    Returns:
        Dictionary of hyperparameter name → value.
    """
    if hasattr(model, "get_params"):
        try:
            return dict(model.get_params(deep=False))  # type: ignore[no-untyped-call]
        except Exception:
            pass

    # CatBoost
    if hasattr(model, "get_all_params"):
        try:
            return dict(model.get_all_params())
        except Exception:
            pass

    # Last resort
    return {k: v for k, v in vars(model).items() if not k.startswith("_")}


def get_feature_importances(model: Any) -> Optional[np.ndarray]:
    """Extract built-in feature importances if available.

    Checks ``feature_importances_`` (sklearn tree models, XGBoost, LightGBM)
    and ``coef_`` (linear models).

    Args:
        model: A fitted ML model.

    Returns:
        1-D numpy array of importance values, or ``None`` if not available.
    """
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        return np.asarray(fi, dtype=float)

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        return np.abs(coef)

    return None


def has_predict_proba(model: Any) -> bool:
    """Return ``True`` if the model can produce class probabilities."""
    return callable(getattr(model, "predict_proba", None))


def has_decision_function(model: Any) -> bool:
    """Return ``True`` if the model exposes a decision function."""
    return callable(getattr(model, "decision_function", None))


def safe_predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Call ``model.predict(X)`` with basic error handling.

    Args:
        model: A fitted estimator.
        X: Feature matrix.

    Returns:
        Prediction array.

    Raises:
        RuntimeError: If prediction fails.
    """
    try:
        return np.asarray(model.predict(X))
    except Exception as exc:
        raise RuntimeError(f"model.predict() failed: {exc}") from exc


def safe_predict_proba(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """Call ``model.predict_proba(X)`` if available.

    Args:
        model: A fitted estimator.
        X: Feature matrix.

    Returns:
        Probability array ``(n_samples, n_classes)``, or ``None``.
    """
    if not has_predict_proba(model):
        return None
    try:
        return np.asarray(model.predict_proba(X))
    except Exception:
        return None


def model_size_bytes(model: Any) -> int:
    """Estimate the serialised size of *model* in bytes via pickle.

    Args:
        model: A fitted ML model.

    Returns:
        Pickle byte size.  Returns 0 on failure.
    """
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0


def framework_version(framework: FrameworkType) -> str:
    """Return the installed version string of the given framework.

    Args:
        framework: The :class:`FrameworkType` to query.

    Returns:
        Version string, or empty string if not installed.
    """
    module_map = {
        FrameworkType.SKLEARN: "sklearn",
        FrameworkType.XGBOOST: "xgboost",
        FrameworkType.LIGHTGBM: "lightgbm",
        FrameworkType.CATBOOST: "catboost",
    }
    mod_name = module_map.get(framework)
    if not mod_name:
        return ""
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "")
    except ImportError:
        return ""


def is_shap_available() -> bool:
    """Return ``True`` if the ``shap`` package is importable."""
    return _is_available("shap")


def is_lime_available() -> bool:
    """Return ``True`` if the ``lime`` package is importable."""
    return _is_available("lime")


def get_classes(model: Any) -> Optional[np.ndarray]:
    """Return the class labels from a fitted classifier, if available.

    Args:
        model: A fitted classifier.

    Returns:
        Array of class labels, or ``None``.
    """
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    return None


def get_n_outputs(model: Any) -> int:
    """Return the number of outputs of the model."""
    return int(getattr(model, "n_outputs_", 1))
