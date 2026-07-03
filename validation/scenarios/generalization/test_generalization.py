"""Generalization Validation Scenarios."""

from typing import Tuple, Any
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from validation.scenario import ValidationScenario, ExpectedResult


class GeneralizationStrong(ValidationScenario):
    name = "Strong Generalization"
    category = "Generalization"
    description = "Model generalizes well to unseen data."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=2000, n_features=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class GeneralizationWeak(ValidationScenario):
    name = "Weak Generalization"
    category = "Generalization"
    description = "Large gap between train and test performance."
    random_seed = 100

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=500, n_features=20, n_informative=10, n_redundant=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = DecisionTreeClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Train/Test Score Gap"],
            severity=["warning", "critical"]
        )


class GeneralizationHighCVVariance(ValidationScenario):
    name = "High CV Variance"
    category = "Generalization"
    description = "Model performance varies wildly across folds."
    random_seed = 200

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        # Very noisy, small dataset — forces high CV std
        X, y = make_classification(n_samples=80, n_features=5, flip_y=0.45, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = DecisionTreeClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["CV Score Variance"],
            severity=["warning", "critical"]
        )


class GeneralizationStableCV(ValidationScenario):
    name = "Stable CV"
    category = "Generalization"
    description = "Model performance is consistent across folds."
    random_seed = 300

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=2000, n_features=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class GeneralizationPoorValidationSplit(ValidationScenario):
    name = "Poor Validation Split"
    category = "Generalization"
    description = "Validation set has a significantly different distribution from training set."
    random_seed = 400

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        # Large shift to guarantee train/test gap > generalization_gap_critical (0.20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
        X_test = X_test + 8.0
        return X_train, X_test, y_train, y_test

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Train/Test Score Gap"],
            severity=["warning", "critical"]
        )


class GeneralizationSmallValidationSet(ValidationScenario):
    name = "Small Validation Set"
    category = "Generalization"
    description = "The validation set is too small to reliably evaluate generalization."
    random_seed = 500

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        # 10 samples in test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=self.random_seed)
        return X_train, X_test, y_train, y_test

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        # Depending on if there's a rule for small validation sets, we might expect a warning.
        # But for now, we just let it run. Let's not strictly require a failure unless implemented.
        return ExpectedResult(passed=None)
