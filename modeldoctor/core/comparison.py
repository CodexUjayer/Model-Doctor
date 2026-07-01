"""ModelComparisonEngine — benchmarks alternative baseline models."""

import time
import warnings
from typing import Any, Dict, List, Optional, Type

import numpy as np
from pydantic import BaseModel, Field

from modeldoctor.core.context import EvaluationContext
from modeldoctor.models.enums import TaskType
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


class ComparisonResult(BaseModel):
    """Result of benchmarking a single comparison model."""
    model_name: str
    is_baseline: bool = False
    accuracy: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    r2_score: float = 0.0
    training_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    memory_mb: float = 0.0


class ModelComparisonEngine:
    """Benchmarks alternative models and creates a leaderboard."""

    def __init__(self):
        self._clf_models = [
            ("LogisticRegression", "sklearn.linear_model.LogisticRegression", {}),
            ("DecisionTree", "sklearn.tree.DecisionTreeClassifier", {"max_depth": 5}),
            ("RandomForest", "sklearn.ensemble.RandomForestClassifier", {"n_estimators": 100}),
            ("LightGBM", "lightgbm.LGBMClassifier", {}),
        ]
        self._reg_models = [
            ("LinearRegression", "sklearn.linear_model.LinearRegression", {}),
            ("DecisionTree", "sklearn.tree.DecisionTreeRegressor", {"max_depth": 5}),
            ("RandomForest", "sklearn.ensemble.RandomForestRegressor", {"n_estimators": 100}),
            ("LightGBM", "lightgbm.LGBMRegressor", {}),
        ]

    def run(self, context: EvaluationContext) -> List[ComparisonResult]:
        """Run comparison benchmark."""
        results = []

        # Baseline (original model)
        baseline = ComparisonResult(
            model_name=context.model_class,
            is_baseline=True,
            inference_time_ms=context.prediction_latency_ms,
            memory_mb=context.model_size / (1024 * 1024)
        )
        
        if context.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            baseline.accuracy = context.classification_metrics.get("accuracy", 0.0)
            baseline.f1_score = context.classification_metrics.get("f1", 0.0)
            baseline.roc_auc = context.classification_metrics.get("roc_auc", 0.0)
            models_to_run = self._clf_models
        else:
            baseline.r2_score = context.regression_metrics.get("r2", 0.0)
            models_to_run = self._reg_models
            
        results.append(baseline)
        
        if not context.config.compare_models:
            return results

        logger.info("Running model comparison benchmark...")

        for name, cls_path, kwargs in models_to_run:
            if name == context.model_class:
                continue
                
            model_cls = self._import_class(cls_path)
            if not model_cls:
                continue
                
            model = model_cls(**kwargs)
            try:
                res = self._benchmark_model(name, model, context)
                results.append(res)
            except Exception as exc:
                logger.warning(f"Failed to benchmark {name}: {exc}")
                
        # Sort by primary metric
        if context.task_type == TaskType.REGRESSION:
            results.sort(key=lambda r: r.r2_score, reverse=True)
        else:
            results.sort(key=lambda r: r.roc_auc if r.roc_auc > 0 else r.accuracy, reverse=True)
            
        return results

    def _benchmark_model(self, name: str, model: Any, context: EvaluationContext) -> ComparisonResult:
        res = ComparisonResult(model_name=name)
        
        # Training time
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(context.X_train, context.y_train)
        res.training_time_ms = (time.perf_counter() - t0) * 1000
        
        # Inference time
        t0 = time.perf_counter()
        y_pred = model.predict(context.X_test)
        res.inference_time_ms = (time.perf_counter() - t0) * 1000 / max(1, len(context.X_test))
        
        # Metrics
        from sklearn.metrics import accuracy_score, f1_score, r2_score
        
        if context.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            res.accuracy = accuracy_score(context.y_test, y_pred)
            avg = "binary" if context.task_type == TaskType.BINARY_CLASSIFICATION else "weighted"
            res.f1_score = f1_score(context.y_test, y_pred, average=avg, zero_division=0)
            
            if hasattr(model, "predict_proba") and context.task_type == TaskType.BINARY_CLASSIFICATION:
                try:
                    from sklearn.metrics import roc_auc_score
                    y_prob = model.predict_proba(context.X_test)[:, 1]
                    res.roc_auc = roc_auc_score(context.y_test, y_prob)
                except Exception:
                    pass
        else:
            res.r2_score = r2_score(context.y_test, y_pred)
            
        # Model size
        import pickle
        res.memory_mb = len(pickle.dumps(model)) / (1024 * 1024)
        
        return res
        
    def _import_class(self, path: str) -> Optional[Type]:
        try:
            module_name, class_name = path.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_name)
            return getattr(mod, class_name)
        except ImportError:
            return None
