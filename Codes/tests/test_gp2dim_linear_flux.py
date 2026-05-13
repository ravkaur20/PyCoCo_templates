"""Smoke tests for linear-scaled-flux 2D GP helpers (stdlib unittest; no pytest required)."""
import os
import sys
import unittest

import numpy as np

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
    sys.path.insert(0, CODES)

try:
    import GP2dim_utils_newlog_linear_flux as gl
except ImportError:  # e.g. george not installed
    gl = None

skip_gl = unittest.skipIf(gl is None, "GP2dim_utils_newlog_linear_flux import failed (need george, etc.)")


@skip_gl
class TestLinearFluxTransform(unittest.TestCase):
    def test_transform_finite_and_parametrization(self):
        class Dummy:
            pass

        d = Dummy()
        raw = np.array([[-30.0, -29.9], [-29.8, -30.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-6)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        y, yerr, x1n, x2n = gl.transform2LINEAR_reshape(d, raw, raw_err, off_xa, off_ya)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(np.isfinite(yerr)))
        self.assertTrue(np.all(np.isfinite(x1n)))
        self.assertTrue(np.all(np.isfinite(x2n)))
        self.assertGreater(np.nanmin(yerr), 0.0)
        self.assertEqual(d.grid_norm_info["flux_parametrization"], "linear_scaled")
        self.assertGreater(d.grid_norm_info["scale_factor"], 0)

    def test_affine_inverse_independent_of_mean_scaling(self):
        """Predictive std in physical flux is |scale| * std_scaled (checked in transform_back module)."""
        offset = 1.0e-15
        scale = 2.0e-16
        mu_s = np.array([0.0, 1.0, -0.5])
        phy = gl.scaled_affine_to_physical(mu_s, offset, scale)
        np.testing.assert_allclose(phy, mu_s * scale + offset, rtol=0.0, atol=1e-30)

    def test_floor_off_smaller_than_legacy_spread_floor(self):
        class Dummy:
            pass

        raw = np.array([[-30.0, -29.9], [-29.8, -30.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-9)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        d_off = Dummy()
        d_off.gp_yerr_floor_frac = 0.0
        d_off.gp_yerr_abs_floor = 0.0
        _, yerr_off, _, _ = gl.transform2LINEAR_reshape(d_off, raw, raw_err, off_xa, off_ya)
        d_on = Dummy()
        d_on.gp_yerr_floor_frac = 1e-4
        d_on.gp_yerr_abs_floor = 0.0
        _, yerr_on, _, _ = gl.transform2LINEAR_reshape(d_on, raw, raw_err, off_xa, off_ya)
        self.assertLess(np.min(yerr_off), np.min(yerr_on))


if __name__ == "__main__":
    unittest.main()
