"""Overfitting Validation Scenarios."""

from typing import Tuple, Any
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from validation.scenario import ValidationScenario, ExpectedResult


class OverfittingHealthyRF(ValidationScenario):
    name = "Healthy Random Forest"
    category = "Overfitting"
    description = "A well-regularized Random Forest that does not overfit."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class OverfittingHealthyLR(ValidationScenario):
    name = "Healthy Logistic Regression"
    category = "Overfitting"
    description = "A linear model on a linear problem, naturally robust to overfitting."
    random_seed = 100

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class OverfittingHealthyGB(ValidationScenario):
    name = "Healthy Gradient Boosting"
    category = "Overfitting"
    description = "A tuned Gradient Boosting model with early stopping equivalent parameters."
    random_seed = 101

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class OverfittingTree(ValidationScenario):
    name = "Overfit Decision Tree"
    category = "Overfitting"
    description = "An unconstrained decision tree that memorizes the training data."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, flip_y=0.20, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = DecisionTreeClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Memorization", "Generalization Gap"],
            severity=["warning", "critical"]
        )


class OverfittingRF(ValidationScenario):
    name = "Overfit Random Forest"
    category = "Overfitting"
    description = "A Random Forest with deep trees that overfits the training data."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=500, n_features=50, n_informative=20, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Generalization Gap"],
            severity=["warning", "critical"]
        )


class OverfittingSmallDataset(ValidationScenario):
    name = "Small Dataset High Variance"
    category = "Overfitting"
    description = "A complex model trained on very few samples."
    random_seed = 123

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        # Very small dataset so param_count >> n_train * ratio
        X, y = make_classification(n_samples=40, n_features=10, n_informative=5, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.25, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Generalization Gap"],
            severity=["warning", "critical"]
        )


class UnderfittingTree(ValidationScenario):
    name = "Underfit Tree"
    category = "Overfitting"
    description = "A decision tree that is too shallow and underfits."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = DecisionTreeClassifier(max_depth=1, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        # Note: OverfittingDoctor might just pass this because there is no overfit.
        # True underfitting might be caught by PredictionDoctor. For Overfitting dimension, it passes.
        return ExpectedResult(passed=True)
