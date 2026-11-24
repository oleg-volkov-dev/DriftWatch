import joblib
import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from services.common.estimators import PositiveLogisticRegression


def test_constrained_fit_matches_logistic_regression_when_bounds_are_inactive(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(4000, 3))
    y = rng.random(len(X)) < expit(X @ np.array([0.5, 1, 2]) - 2)
    model = PositiveLogisticRegression().fit(X, y)
    reference = LogisticRegression(tol=1e-10, max_iter=1000).fit(X, y)
    assert np.all(reference.coef_ > 0)
    np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X), atol=1e-5)
    path = tmp_path / "positive.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(X), model.predict_proba(X))


def test_opposite_signal_is_removed_and_intercept_is_refitted():
    X = np.arange(100).reshape(-1, 1) / 100
    y = np.arange(100) < 20
    model = PositiveLogisticRegression().fit(X, y)
    assert model.coef_[0, 0] == 0
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], 0.2, atol=1e-6)


def test_nonconvergence_is_not_silently_registered():
    with pytest.raises(RuntimeError, match="converge"):
        PositiveLogisticRegression(max_iter=0).fit([[0], [1], [2]], [0, 0, 1])
