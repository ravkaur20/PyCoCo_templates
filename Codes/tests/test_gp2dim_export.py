"""Tests for ``gp2dim_export`` (no George required)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
    sys.path.insert(0, CODES)

import gp2dim_export as ex


class TestGp2dimExport(unittest.TestCase):
    def test_save_and_load_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            X = np.random.randn(40, 2)
            y = np.random.randn(40)
            yerr = np.abs(np.random.randn(40)) * 0.1 + 1e-3
            yc = np.sqrt(yerr**2 + 1e-6**2)
            Xf = np.random.randn(100, 2)
            ex.save_gp_minimal_bundle(
                td,
                X=X,
                y=y,
                yerr=yerr,
                y_compute=yc,
                X_fill=Xf,
                kernel_wls_scale=0.01,
                kernel_time_scale=0.04,
                y_var_scale=float(np.var(y)),
                white_noise_variance=0.01,
                prior=False,
                prior_points=np.zeros((0, 2)),
                prior_values=np.zeros((0,)),
                grid_norm_info={"norm1": 4.0, "offset2": -2.0, "norm2": 1.0},
                gp_module="test",
                mode="extend_spectra",
                snname="TEST",
                kernel_layout="per_axis_Matern32_product",
            )
            d = np.load(os.path.join(td, "gp_minimal_bundle.npz"), allow_pickle=False)
            self.assertEqual(d["X"].shape, (40, 2))
            self.assertEqual(d["y"].shape, (40,))
            self.assertEqual(d["X_fill"].shape, (100, 2))
            np.testing.assert_allclose(d["y_compute"], yc)
            with open(os.path.join(td, "gp_minimal_bundle_meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["snname"], "TEST")
            self.assertIn("grid_norm_info", meta)


if __name__ == "__main__":
    unittest.main()
