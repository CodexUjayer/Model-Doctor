"""Production Validation Scenarios."""

from typing import Tuple, Any
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from validation.scenario import ValidationScenario, ExpectedResult


class ProductionTinyModel(ValidationScenario):
    name = "Tiny Model"
    category = "Production"
    description = "A logistic regression model with a small footprint."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class ProductionHugeRF(ValidationScenario):
    name = "Huge Random Forest"
    category = "Production"
    description = "A random forest with thousands of deep trees."
    random_seed = 100

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=50, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_seed)
        model.fit(X_train, y_train)
        model.dummy_large_array_ = np.ones(25000000) # Force large size
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Model Size"],
            severity=["warning", "critical"]
        )


class ProductionLargeGB(ValidationScenario):
    name = "Large Gradient Boosting"
    category = "Production"
    description = "A deep gradient boosting model that might have slow inference."
    random_seed = 200

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=50, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=self.random_seed)
        model.fit(X_train, y_train)
        model.dummy_large_array_ = np.ones(25000000)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Model Size"],
            severity=["warning", "critical"]
        )


class ProductionSlowInference(ValidationScenario):
    name = "Slow Inference"
    category = "Production"
    description = "A model wrapped to artificially delay inference."
    random_seed = 300

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=100, n_features=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        class SlowModel(BaseEstimator, ClassifierMixin):
            def __init__(self, base):
                self.base = base
            def fit(self, X, y):
                self.base.fit(X, y)
                self.classes_ = self.base.classes_
                return self
            def predict(self, X):
                time.sleep(0.51) # Just above critical threshold of 500ms
                return self.base.predict(X)
            def predict_proba(self, X):
                return self.base.predict_proba(X)
        
        base = LogisticRegression(random_state=self.random_seed)
        model = SlowModel(base)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Inference Latency"],
            severity=["warning", "critical"]
        )


class ProductionFastInference(ValidationScenario):
    name = "Fast Inference"
    category = "Production"
    description = "A model that predicts extremely quickly."
    random_seed = 400

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class ProductionLargeSerializedModel(ValidationScenario):
    name = "Large Serialized Model"
    category = "Production"
    description = "A model with a massive internal array making serialization huge."
    random_seed = 500

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=100, n_features=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        base = LogisticRegression(random_state=self.random_seed)
        base.fit(X_train, y_train)
        # Add a large dummy attribute to simulate a massive model (e.g., 200MB)
        # 1 float64 is 8 bytes. 25M floats = 200MB
        base.dummy_large_array_ = np.ones(25000000)
        return base

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Model Size"],
            severity=["warning", "critical"]
        )
