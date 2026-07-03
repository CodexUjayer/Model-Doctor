"""Calibration Validation Scenarios."""

from typing import Tuple, Any
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from validation.scenario import ValidationScenario, ExpectedResult


class CalibrationWellCalibratedLR(ValidationScenario):
    name = "Well-calibrated LR"
    category = "Calibration"
    description = "Logistic regression models are typically well-calibrated by default."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=2000, n_features=20, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class CalibrationOverconfidentRF(ValidationScenario):
    name = "Overconfident RF"
    category = "Calibration"
    description = "Random forests can push probabilities away from 0 and 1, but often have poor ECE."
    random_seed = 100

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        # Hard classification problem
        X, y = make_classification(n_samples=2000, n_features=20, n_informative=5, n_redundant=15, flip_y=0.2, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Expected Calibration Error"],
            severity=["warning", "critical"]
        )


class CalibrationOverconfidentTree(ValidationScenario):
    name = "Overconfident Decision Tree"
    category = "Calibration"
    description = "Decision trees output 100% confidence for leaf nodes, causing terrible ECE."
    random_seed = 200

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, flip_y=0.3, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = DecisionTreeClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Expected Calibration Error"],
            severity=["warning", "critical"]
        )


class CalibrationPlattScaling(ValidationScenario):
    name = "Platt Scaling"
    category = "Calibration"
    description = "A classifier calibrated with Platt scaling (sigmoid)."
    random_seed = 300

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=20, flip_y=0.2, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        base_model = RandomForestClassifier(n_estimators=50, random_state=self.random_seed)
        model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class CalibrationIsotonic(ValidationScenario):
    name = "Isotonic Calibration"
    category = "Calibration"
    description = "A classifier calibrated with isotonic regression."
    random_seed = 400

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=2000, n_features=20, flip_y=0.2, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        base_model = RandomForestClassifier(n_estimators=50, random_state=self.random_seed)
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class CalibrationPoorCalibration(ValidationScenario):
    name = "Poor Calibration"
    category = "Calibration"
    description = "An SVM with probability=True on a noisy dataset typically has bad calibration."
    random_seed = 500

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=20, flip_y=0.3, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = SVC(probability=True, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Expected Calibration Error"],
            severity=["warning", "critical"]
        )
