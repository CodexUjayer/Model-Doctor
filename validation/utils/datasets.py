import pandas as pd
from sklearn.datasets import make_classification, make_regression

def get_healthy_classification_dataset(n_samples=1000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=10, 
        n_redundant=2, random_state=random_state
    )
    df = pd.DataFrame(X)
    df.columns = [str(c) for c in df.columns]
    return df, pd.Series(y)

def get_leakage_classification_dataset(n_samples=1000, random_state=42):
    X, y = get_healthy_classification_dataset(n_samples, random_state)
    X['leaky_feature'] = y  # direct leakage
    return X, y

def get_imbalanced_classification_dataset(n_samples=1000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=20, weights=[0.99, 0.01], 
        random_state=random_state
    )
    return pd.DataFrame(X), pd.Series(y)

def get_healthy_regression_dataset(n_samples=1000, random_state=42):
    X, y = make_regression(
        n_samples=n_samples, n_features=20, n_informative=10, 
        noise=0.1, random_state=random_state
    )
    return pd.DataFrame(X), pd.Series(y)
