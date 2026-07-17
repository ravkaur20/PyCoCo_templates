"""Tests for phase-axis helpers used in plotting."""

import numpy as np

from plot_results import norm_x2_from_phase_days, phase_days_from_norm_x2


def test_norm_x2_phase_days_round_trip() -> None:
    gn = {
        "_normalized_only": False,
        "x1_mean": 0.0,
        "x1_std": 1.0,
        "x2_mean": np.log10(0.5),
        "x2_std": 0.1,
        "offset": 0.0,
        "scale_factor": 1.0,
    }
    days = np.linspace(0.1, 3.0, 25, dtype=float)
    u = norm_x2_from_phase_days(days, gn)
    back = phase_days_from_norm_x2(u, gn)
    np.testing.assert_allclose(back, days, rtol=1e-10, atol=1e-12)


def test_norm_x2_normalized_only_is_identity() -> None:
    gn = dict(_normalized_only=True, x1_mean=0.0, x1_std=1.0, x2_mean=0.0, x2_std=1.0, offset=0.0, scale_factor=1.0)
    t = np.array([0.1, 0.55, 2.0])
    np.testing.assert_array_equal(norm_x2_from_phase_days(t, gn), t)
