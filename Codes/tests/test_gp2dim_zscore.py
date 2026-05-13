"""Tests for z-score coordinate fork ``GP2dim_utils_newlog_zscore``."""
import os
import sys
import unittest

import numpy as np

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
    sys.path.insert(0, CODES)

try:
    import GP2dim_utils_newlog_zscore as gz
except ImportError:
    gz = None

skip_z = unittest.skipIf(gz is None, "GP2dim_utils_newlog_zscore import failed (need george, pandas, etc.)")


@skip_z
class TestZscoreCoords(unittest.TestCase):
    def test_phase_days_round_trip(self):
        gn = {
            "x2_mean": -1.5,
            "x2_std": 0.4,
            "coord_parametrization": "zscore",
        }
        log_phases = np.array([-2.0, -1.0, -0.3])
        x2n = (log_phases - gn["x2_mean"]) / gn["x2_std"]
        back_log = np.log10(gz.phase_days_from_norm_x2(x2n, gn))
        np.testing.assert_allclose(back_log, log_phases, rtol=0.0, atol=1e-12)

    def test_x2_mask(self):
        gn = {"x2_mean": -1.0, "x2_std": 0.5, "coord_parametrization": "zscore"}
        target = (0.3 - gn["x2_mean"]) / gn["x2_std"]
        x2 = np.array([0.0, target, 2.0])
        m = gz.x2_mask_for_phase(x2, 0.3, gn)
        self.assertEqual(int(m.sum()), 1)
        self.assertTrue(bool(m[1]))

    def test_log10_wl_from_x1_norm_inverse(self):
        gn = {"x1_mean": 3.2, "x1_std": 0.15, "coord_parametrization": "zscore"}
        wl = np.array([3.0, 3.5])
        x1n = (wl - gn["x1_mean"]) / gn["x1_std"]
        out = gz.log10_wavelength_from_x1_norm(x1n, gn)
        np.testing.assert_allclose(out, wl, rtol=0.0, atol=1e-12)

    def test_transform_sets_zscore_grid_norm_info(self):
        class Dummy:
            pass

        d = Dummy()
        raw = np.array([[-30.0, -29.9], [-29.8, -30.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-6)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        y, yerr, x1n, x2n = gz.transform2LOG_reshape(d, raw, raw_err, off_xa, off_ya)
        self.assertEqual(d.grid_norm_info.get("coord_parametrization"), "zscore")
        for k in ("x1_mean", "x1_std", "x2_mean", "x2_std", "x2_train_min"):
            self.assertIn(k, d.grid_norm_info)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(np.isfinite(yerr)))
        self.assertTrue(np.all(np.isfinite(x1n)))
        self.assertTrue(np.all(np.isfinite(x2n)))
        self.assertLess(abs(float(np.mean(x1n))), 1e-9)
        self.assertLess(abs(float(np.mean(x2n))), 1e-9)


if __name__ == "__main__":
    unittest.main()
