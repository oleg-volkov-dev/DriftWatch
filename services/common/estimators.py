"""Constrained estimator for the demo's nonnegative risk features."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PositiveLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary L2 logistic regression with nonnegative feature coefficients.

    Constraints apply to engineered risk contributions, not raw inputs: tenure
    reduces new-account risk, time is circular, and small-payment risk may fall
    as amount rises. The intercept is unconstrained. This encodes the demo's
    assumptions while still fitting probabilities to sampled training labels.
    """

    def __init__(self, C=1.0, max_iter=500, tol=1e-8):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if self.C <= 0 or not np.isfinite(self.C):
            raise ValueError("C must be finite and positive")
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("Expected both binary classes 0 and 1")
        self.n_features_in_ = X.shape[1]
        n = len(y)

        def objective(parameters):
            weights, intercept = parameters[:-1], parameters[-1]
            scores = X @ weights + intercept
            residual = expit(scores) - y
            loss = np.mean(np.logaddexp(0, scores) - y * scores)
            loss += np.dot(weights, weights) / (2 * self.C * n)
            gradient = np.r_[X.T @ residual / n + weights / (self.C * n), residual.mean()]
            return loss, gradient

        initial = np.zeros(self.n_features_in_ + 1)
        prevalence = y.mean()
        initial[-1] = np.log(prevalence / (1 - prevalence))
        result = minimize(
            objective, initial, jac=True, method="L-BFGS-B",
            bounds=[(0, None)] * self.n_features_in_ + [(None, None)],
            options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": self.tol},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Risk model did not converge: {result.message}")
        self.coef_ = result.x[:-1].reshape(1, -1)
        self.intercept_ = result.x[-1:]
        self.n_iter_ = np.array([result.nit])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected number of risk features")
        probability = expit(X @ self.coef_[0] + self.intercept_[0])
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
