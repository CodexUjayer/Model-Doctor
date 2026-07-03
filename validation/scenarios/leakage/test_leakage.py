"""Leakage Validation Scenarios."""

from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from validation.scenario import ValidationScenario, ExpectedResult


class LeakageTargetCopied(ValidationScenario):
    name = "Target Copied Into Feature"
    category = "Leakage"
    description = "A feature is an exact copy of the target."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df["target_leak"] = y
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Correlation", "Target Correlation"],
            severity=["warning", "critical"]
        )


class LeakageTimestamp(ValidationScenario):
    name = "Timestamp Leakage"
    category = "Leakage"
    description = "A monotonically increasing feature perfectly separates classes."
    random_seed = 100

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        # Sort so that y=0 comes first, then y=1
        sorted_idx = np.argsort(y)
        df = df.iloc[sorted_idx].reset_index(drop=True)
        y_sorted = y[sorted_idx]
        # Add a sequential timestamp
        df["timestamp"] = np.arange(len(df))
        # Now shuffle them so train_test_split sees random mix, but 'timestamp' still correlates heavily with y
        df["target"] = y_sorted
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        y_final = df["target"].values
        df = df.drop(columns=["target"])
        return train_test_split(df, y_final, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Correlation", "Leakage"],
            severity=["warning", "critical"]
        )


class LeakageCustomerID(ValidationScenario):
    name = "Customer ID Leakage"
    category = "Leakage"
    description = "An ID feature highly correlates with the target."
    random_seed = 200

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        # Make ID heavily correlate with target
        df["customer_id"] = y * 1000 + np.random.randint(1, 100, size=len(y))
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Correlation", "Leakage"],
            severity=["warning", "critical"]
        )


class LeakageDuplicateTarget(ValidationScenario):
    name = "Duplicate Target"
    category = "Leakage"
    description = "A slight variation of the target is included as a feature."
    random_seed = 300

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        # Flip 5% of target to create duplicate
        flip_idx = np.random.choice(len(y), size=int(0.05 * len(y)), replace=False)
        y_dup = y.copy()
        y_dup[flip_idx] = 1 - y_dup[flip_idx]
        df["target_variation"] = y_dup
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Correlation", "Target Correlation"],
            severity=["warning", "critical"]
        )


class LeakageFutureInfo(ValidationScenario):
    name = "Future Information"
    category = "Leakage"
    description = "Feature contains future info (e.g., 'days_to_churn' in a churn model)."
    random_seed = 400

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        # For churn (1), days_to_churn is small (0-30). For non-churn (0), it's large or missing.
        days = np.where(y == 1, np.random.randint(0, 30, size=len(y)), 999)
        df["days_to_churn"] = days
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = RandomForestClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Correlation", "Target Correlation"],
            severity=["warning", "critical"]
        )


class LeakageHealthy(ValidationScenario):
    name = "Healthy Dataset"
    category = "Leakage"
    description = "A dataset with no data leakage."
    random_seed = 500

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, class_sep=0.5, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)


class LeakageWeakCorrelation(ValidationScenario):
    name = "Weak Correlation"
    category = "Leakage"
    description = "A dataset where the best feature has weak correlation, no leakage expected."
    random_seed = 600

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, class_sep=0.5, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)
