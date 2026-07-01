"""Synthetic benchmark datasets for validation."""

import numpy as np
from sklearn.datasets import make_classification


def make_overfitting_dataset(n_samples: int = 1000):
    """Generate a dataset and a model intentionally overfitted."""
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
    # The runner will use a deep decision tree or random forest
    return X, y


def make_imbalanced_dataset(n_samples: int = 1000):
    """Generate a highly imbalanced dataset."""
    X, y = make_classification(n_samples=n_samples, n_features=20, weights=[0.95, 0.05], random_state=42)
    return X, y


def make_leakage_dataset(n_samples: int = 1000):
    """Generate a dataset with data leakage."""
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
    # Inject leakage (feature perfectly correlated with target)
    X[:, 0] = y + np.random.normal(0, 0.01, size=y.shape)
    return X, y
