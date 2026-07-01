"""Shared statistical utility functions.

These are pure functions with no side effects and no framework dependencies.
They are used internally by Doctor modules to avoid duplication.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* when the denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value to return when denominator is 0 (or very close to it).

    Returns:
        The quotient, or *default*.
    """
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy (nats) of a probability distribution.

    Args:
        probabilities: Array of probabilities (must sum to ~1.0).

    Returns:
        Shannon entropy in nats.
    """
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]  # exclude zeros
    return float(-np.sum(p * np.log(p)))


def gini_impurity(class_counts: np.ndarray) -> float:
    """Compute Gini impurity from raw class counts.

    Args:
        class_counts: Array of integer counts per class.

    Returns:
        Gini impurity in [0, 1].
    """
    total = class_counts.sum()
    if total == 0:
        return 0.0
    p = class_counts / total
    return float(1.0 - np.sum(p**2))


def imbalance_ratio(class_counts: np.ndarray) -> float:
    """Compute the ratio of the majority class to the minority class.

    Args:
        class_counts: Array of integer counts per class.

    Returns:
        Ratio ≥ 1.0 (1.0 = perfectly balanced).
    """
    if len(class_counts) < 2:
        return 1.0
    return float(class_counts.max() / max(class_counts.min(), 1))


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable,  # type: ignore[type-arg]
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Estimate a confidence interval for *statistic* via bootstrap.

    Args:
        data: 1-D array of values.
        statistic: A function ``f(arr) -> float``.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., ``0.95`` for 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple ``(lower, upper)`` confidence bounds.
    """
    rng = np.random.default_rng(random_state)
    stats_arr = np.array(
        [statistic(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(stats_arr, 100 * alpha))
    upper = float(np.percentile(stats_arr, 100 * (1.0 - alpha)))
    return lower, upper


def vif_scores(X: np.ndarray) -> np.ndarray:
    """Compute Variance Inflation Factor for each column of *X*.

    A VIF > 10 indicates strong multicollinearity.

    Args:
        X: 2-D numeric array of shape ``(n_samples, n_features)``.

    Returns:
        1-D array of VIF values, shape ``(n_features,)``.
    """
    from numpy.linalg import lstsq

    n_features = X.shape[1]
    vifs = np.zeros(n_features)
    for i in range(n_features):
        y_col = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        # Add intercept
        X_rest_c = np.column_stack([np.ones(len(X_rest)), X_rest])
        coeffs, *_ = lstsq(X_rest_c, y_col, rcond=None)
        y_pred = X_rest_c @ coeffs
        ss_res = np.sum((y_col - y_pred) ** 2)
        ss_tot = np.sum((y_col - y_col.mean()) ** 2)
        r2 = 1.0 - safe_div(ss_res, ss_tot, default=1.0)
        vifs[i] = safe_div(1.0, max(1.0 - r2, 1e-10), default=1e10)
    return vifs


def cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cramér's V — a measure of association between two categorical arrays.

    Args:
        x: 1-D array of category labels.
        y: 1-D array of category labels (same length as *x*).

    Returns:
        Cramér's V in [0, 1].
    """
    from scipy.stats import chi2_contingency

    confusion = np.zeros(
        (len(np.unique(x)), len(np.unique(y))), dtype=int
    )
    x_map = {v: i for i, v in enumerate(np.unique(x))}
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    for xi, yi in zip(x, y):
        confusion[x_map[xi], y_map[yi]] += 1

    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.sum()
    r, k = confusion.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1) + 1e-12)))


def near_zero_variance_mask(X: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Return a boolean mask where ``True`` means the feature has near-zero variance.

    Args:
        X: 2-D feature matrix.
        threshold: Variance below this value is flagged.

    Returns:
        Boolean array of shape ``(n_features,)``.
    """
    variances = np.var(X, axis=0)
    return variances < threshold


def detect_outliers_iqr(X: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Return a boolean mask where ``True`` indicates the row is an outlier.

    Uses the IQR method: a sample is an outlier if any feature lies outside
    ``[Q1 - k*IQR, Q3 + k*IQR]``.

    Args:
        X: 2-D feature matrix ``(n_samples, n_features)``.
        k: IQR multiplier.

    Returns:
        Boolean array of shape ``(n_samples,)``.
    """
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    is_outlier = np.any((X < lower) | (X > upper), axis=1)
    return is_outlier


def correlation_with_target(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pearson correlation of each feature with target *y*.

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        y: Target vector ``(n_samples,)``.

    Returns:
        Tuple of (correlation_array, p_value_array), each of shape ``(n_features,)``.
    """
    n_features = X.shape[1]
    correlations = np.zeros(n_features)
    p_values = np.zeros(n_features)
    for i in range(n_features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = scipy_stats.pearsonr(X[:, i], y)
        correlations[i] = r if np.isfinite(r) else 0.0
        p_values[i] = p if np.isfinite(p) else 1.0
    return correlations, p_values


def permutation_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: Optional[str] = None,
) -> np.ndarray:
    """Compute permutation feature importance using sklearn's implementation.

    Args:
        model: A fitted estimator with a ``score()`` method.
        X: Feature matrix.
        y: True labels.
        n_repeats: Number of permutations per feature.
        random_state: Random seed.
        scoring: Scoring metric to pass to sklearn.

    Returns:
        Array of mean importance values, shape ``(n_features,)``.
    """
    from sklearn.inspection import permutation_importance as sklearn_pi

    result = sklearn_pi(
        model,  # type: ignore[arg-type]
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )
    return result.importances_mean  # type: ignore[no-any-return]
