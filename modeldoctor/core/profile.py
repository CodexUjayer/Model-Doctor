"""Model capability profiling."""

from typing import Any, Dict

from pydantic import BaseModel, Field

from modeldoctor.models.enums import FrameworkType, TaskType
from modeldoctor.utils.compat import (
    detect_framework,
    has_predict_proba,
    get_feature_importances,
)


class ModelProfile(BaseModel):
    """Automatically detected characteristics of a model."""

    model_family: str = "unknown"
    estimator_type: str = "unknown"  # classifier or regressor
    is_probabilistic: bool = False
    ensemble: bool = False
    supports_predict_proba: bool = False
    supports_decision_function: bool = False
    supports_feature_importances: bool = False
    supports_coef: bool = False
    supports_shap: bool = False
    serialization_method: str = "pickle"

    @classmethod
    def introspect(cls, model: Any) -> "ModelProfile":
        """Introspect a model instance to determine its capabilities."""
        profile = cls()
        
        profile.supports_predict_proba = has_predict_proba(model)
        profile.supports_decision_function = hasattr(model, "decision_function")
        profile.supports_feature_importances = get_feature_importances(model) is not None or hasattr(model, "feature_importances_")
        profile.supports_coef = hasattr(model, "coef_")
        
        model_name = type(model).__name__.lower()
        from sklearn.base import is_classifier, is_regressor
        if is_classifier(model) or "classifier" in model_name or "logisticregression" in model_name or "svc" in model_name or hasattr(model, "classes_"):
            profile.estimator_type = "classifier"
            profile.is_probabilistic = profile.supports_predict_proba
        elif is_regressor(model) or "regressor" in model_name or "svr" in model_name:
            profile.estimator_type = "regressor"

        if "tree" in model_name or "forest" in model_name or "boost" in model_name:
            profile.model_family = "tree"
            profile.supports_shap = True
            if "forest" in model_name or "boost" in model_name or "ensemble" in model_name:
                profile.ensemble = True
        elif "linear" in model_name or "logistic" in model_name or "ridge" in model_name or "lasso" in model_name or "svm" in model_name or "svc" in model_name or "svr" in model_name:
            profile.model_family = "linear"
            profile.supports_shap = True # Kernel or Linear explainer
        elif "net" in model_name or "mlp" in model_name or "keras" in model_name or "torch" in model_name:
            profile.model_family = "neural_network"
            profile.supports_shap = True # Deep explainer

        framework = detect_framework(model)
        if framework in (FrameworkType.XGBOOST, FrameworkType.LIGHTGBM, FrameworkType.CATBOOST):
            profile.model_family = "tree"
            profile.ensemble = True
            profile.supports_shap = True
            
        return profile
