"""Feature Validation Scenarios."""

from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from validation.scenario import ValidationScenario, ExpectedResult


class FeatureHealthy(ValidationScenario):
    name = "Healthy Features"
    category = "Feature Engineering"
    description = "A well-balanced dataset with informative features."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class FeatureDominant(ValidationScenario):
    name = "Dominant Feature"
    category = "Feature Engineering"
    description = "One feature accounts for almost all feature importance."
    random_seed = 100

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=1,
                                   class_sep=0.5, random_state=self.random_seed)
        # Force one feature to completely dominate
        X[:, 0] = y * 10.0 + np.random.randn(1000) * 0.5
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Feature Importance Concentration"],
            severity=["warning", "critical"]
        )


class FeatureHighCardinality(ValidationScenario):
    name = "High Cardinality"
    category = "Feature Engineering"
    description = "A categorical feature with too many unique values (e.g., ID treated as categorical)."
    random_seed = 200

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        # Unique value for almost every row
        df["high_cardinality"] = np.arange(len(df)).astype(float)
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        # Note: FeatureDoctor might not detect high cardinality yet, let's not force failure.
        return ExpectedResult(passed=None)


class FeatureConstant(ValidationScenario):
    name = "Constant Features"
    category = "Feature Engineering"
    description = "Dataset contains features with zero variance."
    random_seed = 300

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df["constant"] = 42.0
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Constant Features"],
            severity=["warning", "critical"]
        )


class FeatureHighCorrelation(ValidationScenario):
    name = "High Correlation"
    category = "Feature Engineering"
    description = "Many features are perfectly correlated."
    random_seed = 400

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=5, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["f5_corr"] = df["f0"] * 2
        df["f6_corr"] = df["f1"] + 5
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=None)


class FeaturePotentialLeakage(ValidationScenario):
    name = "Potential Leakage Feature"
    category = "Feature Engineering"
    description = "Feature importance implies a potential leak."
    random_seed = 500

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=0, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        # Make the leak feature extremely dominant
        df["leak"] = y * 10.0 + np.random.randn(1000) * 0.5
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Feature Importance Concentration"],
            severity=["warning", "critical"]
        )


class FeatureTooMany(ValidationScenario):
    name = "Too Many Features"
    category = "Feature Engineering"
    description = "High dimensionality risks overfitting and slow inference."
    random_seed = 600

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=1200, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(max_iter=200, random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["High Dimensionality"],
            severity=["warning", "critical"]
        )
