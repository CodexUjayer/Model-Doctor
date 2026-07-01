"""Utils package."""

from modeldoctor.utils.compat import (
    detect_framework,
    get_feature_importances,
    get_model_params,
    has_predict_proba,
    is_shap_available,
    model_size_bytes,
    safe_predict,
    safe_predict_proba,
)
from modeldoctor.utils.logging import get_logger, setup_logging
from modeldoctor.utils.stats import (
    bootstrap_ci,
    correlation_with_target,
    detect_outliers_iqr,
    entropy,
    imbalance_ratio,
    near_zero_variance_mask,
    safe_div,
    vif_scores,
)

__all__ = [
    # compat
    "detect_framework",
    "get_model_params",
    "get_feature_importances",
    "has_predict_proba",
    "is_shap_available",
    "model_size_bytes",
    "safe_predict",
    "safe_predict_proba",
    # logging
    "get_logger",
    "setup_logging",
    # stats
    "safe_div",
    "entropy",
    "imbalance_ratio",
    "bootstrap_ci",
    "vif_scores",
    "correlation_with_target",
    "detect_outliers_iqr",
    "near_zero_variance_mask",
]
