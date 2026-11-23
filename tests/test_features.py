import numpy as np

from services.common.features import (
    compound_payment_risk,
    payment_activity,
    smooth_payment_activity,
)


def test_smooth_attempts_are_bounded_and_continuous_at_ten_and_twenty():
    counts = np.array([0, 1, 9, 10, 11, 19, 20, 1000], dtype=float)
    values = np.column_stack([np.full(len(counts), 3), counts])
    features = smooth_payment_activity(values)
    assert features[0, 0] == 0
    assert features[3, 0] == 0.5
    assert features[-1, 0] == features[-2, 0] == 1
    assert np.all(np.diff(features[:7, 0]) > 0)
    np.testing.assert_array_equal(features[:, 0], features[:, 1])
    for boundary in [10, 20]:
        near = smooth_payment_activity([[3, boundary - 1e-6], [3, boundary + 1e-6]])
        assert abs(near[0, 0] - near[1, 0]) < 1e-6
    values[:, 0] = 50
    assert np.all(smooth_payment_activity(values)[:, 1] == 0)


def test_legacy_transform_keeps_registered_models_unchanged():
    np.testing.assert_array_equal(
        payment_activity([[3, 9], [3, 10], [50, 10]]),
        [[0, 0], [1, 1], [1, 0]],
    )


def test_compound_risk_respects_caps_and_combines_signals():
    # amount, hour, tenure, merchant risk, distance, international
    high = [150, 0, 1, 0.9, 200, 1]
    extreme = [1e6, 0, 1, 0.9, 1e6, 1]
    lower = [
        [20, 0, 1, 0.9, 200, 1],
        [150, 12, 1, 0.9, 200, 1],
        [150, 0, 730, 0.9, 200, 1],
        [150, 0, 1, 0.1, 200, 1],
        [150, 0, 1, 0.9, 10, 1],
        [150, 0, 1, 0.9, 200, 0],
    ]
    risk = compound_payment_risk([high, extreme, *lower])[:, 0]
    assert np.all((risk >= 0) & (risk <= 1))
    assert risk[0] == risk[1]
    assert np.all(risk[2:] < risk[0])
