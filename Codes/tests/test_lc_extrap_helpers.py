"""Unit tests for lc_extrap_helpers (run: python tests/test_lc_extrap_helpers.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lc_extrap_helpers import (
    clip_extrap_uncertainties,
    covariance_is_bad,
)


def test_clip_extrap_abs_cap():
    ee = np.array([0.01, 10.0, np.nan])
    out = clip_extrap_uncertainties(ee, flux_=np.ones(5), fluxerr_=np.full(5, 0.02), abs_cap=0.5, rel_med_max=None)
    assert out[0] == 0.01
    assert out[1] == 0.5
    assert np.isfinite(out[2])


def test_clip_extrap_relative_median_no_abs():
    fluxerr = np.full(10, 0.05)
    ee = np.full(5, 1.0)
    out = clip_extrap_uncertainties(ee, flux_=np.ones(10), fluxerr_=fluxerr, abs_cap=None, rel_med_max=10.0)
    assert np.allclose(out, 10.0 * 0.05)


def test_covariance_is_bad_inf():
    c = np.eye(2)
    assert not covariance_is_bad(c)
    cb = covariance_is_bad(np.array([[1.0, np.inf], [np.inf, 1.0]]))
    assert cb


def test_covariance_large_cond():
    c = np.diag(np.array([1e20, 1e-20]))
    assert covariance_is_bad(c, cond_max=1e12)


def _run():
    test_clip_extrap_abs_cap()
    test_clip_extrap_relative_median_no_abs()
    test_covariance_is_bad_inf()
    test_covariance_large_cond()
    print("ok: all lc_extrap_helpers tests passed")


if __name__ == "__main__":
    _run()
