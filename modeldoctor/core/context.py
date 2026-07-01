"""EvaluationContext — the shared computation hub for all Doctor modules.

The EvaluationContext is instantiated once per ``diagnose()`` call and passed
to every Doctor.  All expensive computations (predictions, CV scores, SHAP
values, etc.) are computed lazily on first access and cached as instance
attributes, so each Doctor only pays for what it actually uses.

Example::

    ctx = EvaluationContext(
        model=clf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=iris.feature_names,
        config=ModelDoctorConfig(explainability=ExplainabilityMode.AUTO),
    )

    # Lazily computed on first access:
    print(ctx.test_score)       # float
    print(ctx.cv_scores)        # np.ndarray
    print(ctx.feature_importances)  # np.ndarray | None
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from modeldoctor.config.settings import ModelDoctorConfig
from modeldoctor.models.enums import ExplainabilityMode, FrameworkType, TaskType
from modeldoctor.core.task_detector import detect_task_type
from modeldoctor.utils.compat import (
    detect_framework,
    framework_version,
    get_classes,
    get_feature_importances,
    get_model_params,
    has_predict_proba,
    is_shap_available,
    model_size_bytes,
    safe_predict,
    safe_predict_proba,
)
from modeldoctor.core.profile import ModelProfile
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationContext:
    """Shared computation context for a ModelDoctor diagnostic run.

    All properties are lazily computed and cached on first access.  This avoids
    redundant computation when multiple Doctors need the same data.

    Args:
        model: A fitted estimator (sklearn-compatible API).
        X_train: Training feature matrix (numpy array or pandas DataFrame).
        y_train: Training target array.
        X_test: Test/hold-out feature matrix.
        y_test: Test/hold-out target array.
        X_val: Optional validation feature matrix.
        y_val: Optional validation target array.
        feature_names: Optional list of feature names.  If *X_train* is a
            DataFrame, column names are used automatically.
        sample_weight: Optional sample weights for the training set.
        config: Diagnostic configuration.  Defaults to ``ModelDoctorConfig()``.
    """

    def __init__(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[Any] = None,
        config: Optional[ModelDoctorConfig] = None,
    ) -> None:
        self.model = model
        self.config: ModelDoctorConfig = config or ModelDoctorConfig()

        # Convert DataFrames to numpy arrays; keep column names
        self.X_train, self._train_feature_names = self._coerce(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test, _ = self._coerce(X_test)
        self.y_test = np.asarray(y_test)
        self.X_val = np.asarray(X_val) if X_val is not None else None
        self.y_val = np.asarray(y_val) if y_val is not None else None
        self.sample_weight = np.asarray(sample_weight) if sample_weight is not None else None

        # Feature names: prefer explicit arg, then DataFrame cols, else None
        self.feature_names: Optional[List[str]] = (
            feature_names or self._train_feature_names
        )

        # Eagerly derived (cheap) values
        self.framework: FrameworkType = detect_framework(model)
        self.framework_ver: str = framework_version(self.framework)
        self.model_class: str = type(model).__name__
        self.model_params: Dict[str, Any] = get_model_params(model)
        self.task_type: TaskType = detect_task_type(model, self.y_train)
        self.classes: Optional[np.ndarray] = get_classes(model)
        self.n_classes: int = len(self.classes) if self.classes is not None else 0
        self.profile: ModelProfile = ModelProfile.introspect(model)

        # Lazy cache storage
        self._cache: Dict[str, Any] = {}

        logger.debug(
            "EvaluationContext created | model=%s | framework=%s | task=%s | "
            "train=%d | test=%d | features=%d",
            self.model_class,
            self.framework.value,
            self.task_type.value,
            len(self.y_train),
            len(self.y_test),
            self.X_train.shape[1],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(X: Any) -> tuple[np.ndarray, Optional[List[str]]]:
        """Convert *X* to a numpy array, extracting column names if it is a DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.values, list(X.columns)
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr, None

    def _cached(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Return ``self._cache[key]``, computing it via ``fn()`` on first access."""
        if key not in self._cache:
            t0 = time.perf_counter()
            self._cache[key] = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug("Computed '%s' in %.1f ms", key, elapsed)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    @property
    def train_predictions(self) -> np.ndarray:
        """Hard predictions on the training set."""
        return self._cached("train_predictions", safe_predict, self.model, self.X_train)

    @property
    def test_predictions(self) -> np.ndarray:
        """Hard predictions on the test set."""
        return self._cached("test_predictions", safe_predict, self.model, self.X_test)

    @property
    def train_probabilities(self) -> Optional[np.ndarray]:
        """Probability outputs on training set (``None`` for non-probabilistic models)."""
        return self._cached("train_probabilities", safe_predict_proba, self.model, self.X_train)

    @property
    def test_probabilities(self) -> Optional[np.ndarray]:
        """Probability outputs on test set (``None`` for non-probabilistic models)."""
        return self._cached("test_probabilities", safe_predict_proba, self.model, self.X_test)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def train_score(self) -> float:
        """Primary metric score on training data."""
        return self._cached("train_score", self._compute_score, "train")

    @property
    def test_score(self) -> float:
        """Primary metric score on test data."""
        return self._cached("test_score", self._compute_score, "test")

    @property
    def train_test_gap(self) -> float:
        """Absolute difference between train and test scores (overfitting proxy)."""
        return abs(self.train_score - self.test_score)

    def _compute_score(self, split: str) -> float:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            r2_score,
            roc_auc_score,
        )

        X = self.X_train if split == "train" else self.X_test
        y_true = self.y_train if split == "train" else self.y_test
        y_pred = self.train_predictions if split == "train" else self.test_predictions

        if self.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            return float(accuracy_score(y_true, y_pred))
        else:
            return float(r2_score(y_true, y_pred))

    @property
    def classification_metrics(self) -> Dict[str, Any]:
        """Full classification metrics for the test set."""
        return self._cached("classification_metrics", self._compute_clf_metrics)

    def _compute_clf_metrics(self) -> Dict[str, Any]:
        if self.task_type not in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            return {}

        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
            average_precision_score,
            brier_score_loss,
        )

        y_true = self.y_test
        y_pred = self.test_predictions

        avg = "binary" if self.task_type == TaskType.BINARY_CLASSIFICATION else "weighted"
        metrics: Dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
            "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        # AUC-ROC and PR-AUC for binary
        if self.task_type == TaskType.BINARY_CLASSIFICATION and self.test_probabilities is not None:
            from sklearn.metrics import roc_auc_score
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    prob_pos = self.test_probabilities[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_true, prob_pos)
                    metrics["pr_auc"] = average_precision_score(y_true, prob_pos)
                    metrics["brier_score"] = brier_score_loss(y_true, prob_pos)
                except Exception:
                    pass

        # Training metrics
        y_pred_train = self.train_predictions
        metrics["train_accuracy"] = accuracy_score(self.y_train, y_pred_train)
        metrics["train_f1"] = f1_score(
            self.y_train, y_pred_train, average=avg, zero_division=0
        )

        return metrics

    @property
    def regression_metrics(self) -> Dict[str, Any]:
        """Full regression metrics for the test set."""
        return self._cached("regression_metrics", self._compute_reg_metrics)

    def _compute_reg_metrics(self) -> Dict[str, Any]:
        if self.task_type != TaskType.REGRESSION:
            return {}

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

        y_true = self.y_test
        y_pred = self.test_predictions
        mse = mean_squared_error(y_true, y_pred)

        return {
            "r2": r2_score(y_true, y_pred),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),
            "train_r2": r2_score(self.y_train, self.train_predictions),
            "train_mse": mean_squared_error(self.y_train, self.train_predictions),
        }

    @property
    def residuals(self) -> np.ndarray:
        """Prediction residuals on the test set (``y_true - y_pred``)."""
        return self._cached(
            "residuals", lambda: self.y_test.astype(float) - self.test_predictions.astype(float)
        )

    # ------------------------------------------------------------------
    # Cross-Validation
    # ------------------------------------------------------------------

    @property
    def cv_scores(self) -> np.ndarray:
        """Cross-validation scores on the training set."""
        return self._cached("cv_scores", self._compute_cv)

    def _compute_cv(self) -> np.ndarray:
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

        scoring = self._infer_scoring()
        n_splits = self.config.cv_folds

        if self.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ):
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                self.model,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
        return np.asarray(scores, dtype=float)

    def _infer_scoring(self) -> str:
        if self.config.cv_scoring:
            return self.config.cv_scoring
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return "roc_auc"
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return "accuracy"
        return "r2"

    # ------------------------------------------------------------------
    # Feature Importances
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Feature importance array (built-in or permutation fallback)."""
        return self._cached("feature_importances", self._compute_importances)

    def _compute_importances(self) -> Optional[np.ndarray]:
        fi = get_feature_importances(self.model)
        if fi is not None and len(fi) == self.X_train.shape[1]:
            return fi

        # Permutation importance fallback (expensive but universal)
        logger.debug("No built-in importances; computing permutation importance.")
        from modeldoctor.utils.stats import permutation_importance
        try:
            return permutation_importance(
                self.model,
                self.X_test,
                self.y_test,
                n_repeats=3,
                random_state=42,
                scoring=self._infer_scoring(),
            )
        except Exception as exc:
            logger.debug("Permutation importance failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # SHAP values (optional explainability)
    # ------------------------------------------------------------------

    @property
    def shap_values(self) -> Optional[np.ndarray]:
        """SHAP values array, or ``None`` if explainability is disabled/unavailable."""
        return self._cached("shap_values", self._compute_shap)

    @property
    def shap_available(self) -> bool:
        """Whether SHAP computation is enabled and the library is installed."""
        mode = self.config.explainability
        if mode == ExplainabilityMode.DISABLED:
            return False
        if mode == ExplainabilityMode.FULL:
            return is_shap_available()
        return is_shap_available()  # AUTO

    @property
    def shap_skip_reason(self) -> Optional[str]:
        """Human-readable reason why SHAP was skipped, or ``None``."""
        mode = self.config.explainability
        if mode == ExplainabilityMode.DISABLED:
            return "Explainability disabled in config."
        if not is_shap_available():
            return (
                "shap package not installed.\n"
                "  Install:  pip install modeldoctor[explainability]  or  pip install shap"
            )
        return None

    def _compute_shap(self) -> Optional[np.ndarray]:
        mode = self.config.explainability

        if mode == ExplainabilityMode.DISABLED:
            logger.info("SHAP Analysis Skipped — Reason: Explainability disabled in config.")
            return None

        if not is_shap_available():
            if mode == ExplainabilityMode.FULL:
                raise ImportError(
                    "explainability='full' requires the shap package.\n"
                    "  Install:  pip install modeldoctor[explainability]  or  pip install shap"
                )
            logger.info(
                "SHAP Analysis Skipped — Reason: Package not installed.\n"
                "  Install:  pip install modeldoctor[explainability]  or  pip install shap"
            )
            return None

        try:
            import shap

            n_samples = min(len(self.X_test), self.config.max_shap_samples)
            X_sample = self.X_test[:n_samples]

            logger.info("✓ SHAP Analysis Enabled — computing on %d samples.", n_samples)

            # Choose the most appropriate explainer
            try:
                explainer = shap.TreeExplainer(self.model)
                values = explainer.shap_values(X_sample)
            except Exception:
                try:
                    bg = shap.sample(self.X_train, min(100, len(self.X_train)))
                    explainer = shap.KernelExplainer(self.model.predict, bg)
                    values = explainer.shap_values(X_sample, nsamples=50)
                except Exception as exc2:
                    logger.warning("SHAP computation failed: %s", exc2)
                    return None

            # For multi-class, shap_values is a list; take the mean abs across classes
            if isinstance(values, list):
                values = np.mean(np.abs(values), axis=0)

            return np.asarray(values)

        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return None

    @property
    def shap_feature_importances(self) -> Optional[np.ndarray]:
        """Mean |SHAP| per feature, or ``None``."""
        sv = self.shap_values
        if sv is None:
            return None
        if sv.ndim == 1:
            return np.abs(sv)
        return np.abs(sv).mean(axis=0)

    # ------------------------------------------------------------------
    # Data properties
    # ------------------------------------------------------------------

    @property
    def class_distribution_train(self) -> Dict[Any, int]:
        """Class counts in the training set."""
        return self._cached(
            "class_dist_train",
            lambda: {c: int(np.sum(self.y_train == c)) for c in np.unique(self.y_train)},
        )

    @property
    def class_distribution_test(self) -> Dict[Any, int]:
        """Class counts in the test set."""
        return self._cached(
            "class_dist_test",
            lambda: {c: int(np.sum(self.y_test == c)) for c in np.unique(self.y_test)},
        )

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return int(self.X_train.shape[1])

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return int(self.X_test.shape[0])

    @property
    def model_size(self) -> int:
        """Pickle-serialised model size in bytes."""
        return self._cached("model_size", model_size_bytes, self.model)

    @property
    def parameter_count(self) -> Optional[int]:
        """Estimated number of model parameters, if available."""
        return self._cached("parameter_count", self._compute_parameter_count)

    def _compute_parameter_count(self) -> Optional[int]:
        if hasattr(self.model, "count_params"):
            return self.model.count_params()
        
        count = 0
        has_params = False
        if hasattr(self.model, "coef_") and self.model.coef_ is not None:
            count += np.asarray(self.model.coef_).size
            has_params = True
        if hasattr(self.model, "intercept_") and self.model.intercept_ is not None:
            count += np.asarray(self.model.intercept_).size
            has_params = True
            
        return count if has_params else None

    @property
    def missing_values_count(self) -> int:
        """Number of missing values in the training set."""
        return self._cached("missing_values_count", lambda: int(np.isnan(self.X_train.astype(float)).sum()))

    @property
    def duplicate_rows_count(self) -> int:
        """Number of duplicate rows in the training set."""
        return self._cached("duplicate_rows_count", lambda: int(len(self.X_train) - len(np.unique(self.X_train, axis=0))))

    @property
    def cardinality(self) -> Dict[int, int]:
        """Number of unique values per feature index."""
        return self._cached("cardinality", lambda: {i: len(np.unique(self.X_train[:, i])) for i in range(self.X_train.shape[1])})

    # ------------------------------------------------------------------
    # Prediction latency benchmark
    # ------------------------------------------------------------------

    @property
    def prediction_latency_ms(self) -> float:
        """Median single-sample prediction latency in milliseconds."""
        return self._cached("prediction_latency_ms", self._benchmark_latency)

    def _benchmark_latency(self, n_trials: int = 50) -> float:
        sample = self.X_test[:1]
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = safe_predict(self.model, sample)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EvaluationContext("
            f"model={self.model_class}, "
            f"task={self.task_type.value}, "
            f"train={self.n_train}, "
            f"test={self.n_test}, "
            f"features={self.n_features}"
            f")"
        )
