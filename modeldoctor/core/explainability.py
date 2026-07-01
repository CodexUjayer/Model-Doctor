"""ExplainabilityEngine — computes model explanations."""

from typing import Optional, Dict

import numpy as np

from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.enums import ExplainabilityMode
from modeldoctor.models.report import ExplainabilityInfo
from modeldoctor.utils.logging import get_logger
from modeldoctor.utils.compat import is_shap_available

logger = get_logger(__name__)


class ExplainabilityEngine:
    """Computes explainability metrics like SHAP and Partial Dependence."""

    def run(self, context: EvaluationContext) -> ExplainabilityInfo:
        """Run explainability analysis on the context."""
        mode = context.config.explainability
        
        info = ExplainabilityInfo(
            enabled=mode != ExplainabilityMode.DISABLED,
            mode=mode.value if hasattr(mode, 'value') else str(mode),
            shap_available=is_shap_available(),
        )

        if not info.enabled:
            info.skip_reason = "Explainability disabled in config."
            return info

        if not info.shap_available:
            if mode == ExplainabilityMode.FULL:
                logger.warning("SHAP requested but not installed.")
            info.skip_reason = "shap package not installed."
            # Fall back to permutation importance
            importances = context.feature_importances
            if importances is not None:
                info.method = "permutation"
                info.top_features = self._extract_top_features(context, importances)
            return info

        # SHAP is available
        importances = context.shap_feature_importances
        if importances is not None:
            info.method = "shap"
            info.top_features = self._extract_top_features(context, importances)
        else:
            info.skip_reason = context.shap_skip_reason or "SHAP computation failed."
            
        return info

    def _extract_top_features(self, context: EvaluationContext, importances: np.ndarray, top_n: int = 10) -> Dict[str, float]:
        """Extract top N features by importance."""
        if context.feature_names:
            names = context.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importances))]
            
        # Create dict of name -> importance
        feat_imp = {names[i]: float(importances[i]) for i in range(len(importances))}
        
        # Sort and take top N
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(sorted_feats)
