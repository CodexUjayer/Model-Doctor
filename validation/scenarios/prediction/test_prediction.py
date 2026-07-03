"""Prediction Validation Scenarios."""

from typing import Tuple, Any
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from validation.scenario import ValidationScenario, ExpectedResult


class PredictionClassifierExcellent(ValidationScenario):
    name = "Excellent Classifier"
    category = "Prediction"
    description = "A classifier with near-perfect accuracy and F1."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, class_sep=3.0, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class PredictionClassifierAverage(ValidationScenario):
    name = "Average Classifier"
    category = "Prediction"
    description = "A classifier with okay but not great performance."
    random_seed = 100

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, flip_y=0.45, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Test Accuracy"],
            severity=["warning", "critical"]
        )


class PredictionClassifierPoor(ValidationScenario):
    name = "Poor Classifier"
    category = "Prediction"
    description = "A classifier that is basically guessing."
    random_seed = 200

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, flip_y=0.5, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = DummyClassifier(strategy="uniform", random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Test Accuracy", "F1 Score"],
            severity=["warning", "critical"]
        )


class PredictionRegressionExcellent(ValidationScenario):
    name = "Excellent Regression"
    category = "Prediction"
    description = "A regression model with very high R-squared."
    random_seed = 300

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestRegressor(n_estimators=50, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class PredictionRegressionModerate(ValidationScenario):
    name = "Moderate Regression"
    category = "Prediction"
    description = "A regression model with an okay R-squared."
    random_seed = 400

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_regression(n_samples=1000, n_features=10, noise=50.0, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["R² Score"],
            severity=["warning", "critical"]
        )


class PredictionRegressionPoor(ValidationScenario):
    name = "Poor Regression"
    category = "Prediction"
    description = "A regression model that predicts the mean."
    random_seed = 500

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_regression(n_samples=1000, n_features=10, noise=100.0, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["R² Score"],
            severity=["warning", "critical"]
        )
