"""Data Validation Scenarios."""

from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from validation.scenario import ValidationScenario, ExpectedResult


class DataMissingValues(ValidationScenario):
    name = "Missing Values"
    category = "Data Quality"
    description = "A dataset with a significant amount of missing values."
    random_seed = 42

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        # Introduce 25% missing values
        mask = np.random.rand(*df.shape) < 0.25
        df[mask] = np.nan
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        # Simple mean imputation so model can train
        X_train_imp = X_train.fillna(X_train.mean())
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train_imp, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Missing Values"],
            severity=["warning", "critical"]
        )


class DataDuplicateRows(ValidationScenario):
    name = "Duplicate Rows"
    category = "Data Quality"
    description = "Dataset contains many duplicate rows."
    random_seed = 100

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=500, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df["target"] = y
        # Duplicate half the dataset
        df = pd.concat([df, df.iloc[:500]])
        y_final = df["target"].values
        df = df.drop(columns=["target"])
        return train_test_split(df, y_final, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=None)


class DataDuplicateColumns(ValidationScenario):
    name = "Duplicate Columns"
    category = "Data Quality"
    description = "Dataset contains duplicated columns."
    random_seed = 101

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df["f10_dup"] = df["f0"]  # Exact duplicate with string name
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=None)


class DataConstantFeatures(ValidationScenario):
    name = "Constant Features"
    category = "Data Quality"
    description = "Dataset contains features with zero variance."
    random_seed = 102

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df["constant1"] = 5.0
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


class DataNearConstantFeatures(ValidationScenario):
    name = "Near Constant Features"
    category = "Data Quality"
    description = "Dataset contains features with almost zero variance."
    random_seed = 103

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df["near_const"] = np.random.choice([0, 1], size=1000, p=[0.999, 0.001])
        return train_test_split(df, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=None)



class DataClassImbalance(ValidationScenario):
    name = "Class Imbalance"
    category = "Data Quality"
    description = "Dataset has severe class imbalance."
    random_seed = 104

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, weights=[0.99, 0.01], random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(
            passed=False,
            findings=["Class Imbalance"],
            severity=["warning", "critical"]
        )


class DataSmallDataset(ValidationScenario):
    name = "Small Dataset"
    category = "Data Quality"
    description = "Very few samples in the dataset."
    random_seed = 105

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=20, n_features=5, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=None)


class DataHighlyCorrelated(ValidationScenario):
    name = "Highly Correlated Features"
    category = "Data Quality"
    description = "Features are highly correlated with each other."
    random_seed = 106

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, n_redundant=8, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=None)


class DataHealthy(ValidationScenario):
    name = "Healthy Dataset"
    category = "Data Quality"
    description = "A perfectly clean dataset."
    random_seed = 107

    def build_dataset(self) -> Tuple[Any, Any, Any, Any]:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=self.random_seed)
        return train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

    def build_model(self, X_train: Any, y_train: Any) -> Any:
        model = LogisticRegression(random_state=self.random_seed)
        model.fit(X_train, y_train)
        return model

    def expected(self) -> ExpectedResult:
        return ExpectedResult(passed=True)
