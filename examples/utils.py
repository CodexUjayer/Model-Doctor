"""
ModelDoctor Examples Utilities

This module contains shared helper functions for the examples to avoid duplicating
dataset generation and model training code.
"""

from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

def get_classification_data(
    n_samples: int = 1000, 
    n_features: int = 20, 
    n_informative: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generates a standard classification dataset."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative, 
        random_state=random_state
    )
    # Convert to DataFrame for better feature names in reports
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def get_regression_data(
    n_samples: int = 1000, 
    n_features: int = 15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generates a standard regression dataset."""
    X, y = make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        noise=0.1,
        random_state=random_state
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def get_leaky_data(
    n_samples: int = 1000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generates a dataset with intentional target leakage."""
    X, y = make_classification(n_samples=n_samples, n_features=10, random_state=random_state)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    
    # Introduce a leaky feature (e.g., customer_id that accidentally correlates with target)
    np.random.seed(random_state)
    X_df["customer_id_leak"] = y * 100.0 + np.random.randn(n_samples) * 0.1
    y_series = pd.Series(y, name="target")
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def get_overfit_data(
    n_samples: int = 500,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generates a complex dataset prone to overfitting."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=50, 
        n_informative=10, 
        flip_y=0.2, # Add noise to make it hard to generalize
        random_state=random_state
    )
    feature_names = [f"feature_{i}" for i in range(50)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test
