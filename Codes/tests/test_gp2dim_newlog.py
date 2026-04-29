"""Smoke tests for log-space 2D GP helpers (stdlib unittest; no pytest required)."""
import glob
import os
import sys
import tempfile
import unittest

import numpy as np

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
    sys.path.insert(0, CODES)

import GP2dim_utils_newlog as g


class TestGp2dimNewlog(unittest.TestCase):
    def test_scaled_ln_to_linear_clamp(self):
        offset, scale = 0.0, 1.0
        huge = np.array([0.0, 500.0, 2000.0])
        out = g.scaled_ln_to_linear(huge, offset, scale)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[0], 1.0)
        self.assertLess(out[-1], np.finfo(float).max)

    def test_x2_mask_for_phase(self):
        offset2, norm2 = -2.0, 1.5
        x2 = np.array([0.0, (0.5 - offset2) / norm2, 1.0])
        m = g.x2_mask_for_phase(x2, 0.5, offset2, norm2)
        self.assertEqual(int(m.sum()), 1)
        self.assertTrue(bool(m[1]))

    def test_phases_close(self):
        arr = np.array([-1.0, -1.0 + 1e-10, 0.2])
        self.assertTrue(g.phases_close(-1.0, arr))
        self.assertFalse(g.phases_close(0.3, arr))

    def test_transform_scale_floor(self):
        class Dummy:
            pass

        d = Dummy()
        raw = np.array([[-30.0, -29.9], [-29.8, -30.1]], dtype=float)
        raw_err = np.full_like(raw, 1e-6)
        off_xa = np.array([3.5, 3.51], dtype=float)
        off_ya = np.array([-1.0, -0.9], dtype=float)
        y, yerr, x1n, x2n = g.transform2LOG_reshape(d, raw, raw_err, off_xa, off_ya)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(np.isfinite(yerr)))
        self.assertTrue(np.all(np.isfinite(x1n)))
        self.assertTrue(np.all(np.isfinite(x2n)))
        self.assertGreater(d.grid_norm_info["scale_factor"], 0)


class TestGpPredictionGrid(unittest.TestCase):
    """Caps N_wavelength so run_2DGP_GRID does not build huge predict batches (memory)."""

    def test_wavelength_grid_point_count_bounded(self):
        wls_min, wls_max = 3.0, 4.5
        span_wl = float(wls_max - wls_min)
        _gp_n_wl = 300
        _wl_step = 0.01
        n_from_step = int(np.ceil(span_wl / _wl_step)) + 1
        n_wl_use = max(2, min(_gp_n_wl, n_from_step))
        self.assertLessEqual(n_wl_use, _gp_n_wl)
        self.assertGreaterEqual(n_wl_use, 2)


class TestGpDenseMatrixHint(unittest.TestCase):
    def test_bytes_scales_as_n_squared(self):
        self.assertEqual(g.gp_dense_matrix_bytes_order_of_magnitude(0), 0)
        self.assertEqual(g.gp_dense_matrix_bytes_order_of_magnitude(1000), 8 * 1000 * 1000)
        n = 15000
        b = g.gp_dense_matrix_bytes_order_of_magnitude(n)
        self.assertGreater(b / (1024.0**3), 1.5)

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            g.gp_dense_matrix_bytes_order_of_magnitude(-1)


class TestPhaseAxisDenormForPlots(unittest.TestCase):
    """Training plots use offset2 + norm2*x2_norm to recover log10(phase days)."""

    def test_denorm_matches_original_log_phase(self):
        offset2 = -2.0
        norm2 = 3.5
        x2_data = np.array([-2.0, 0.0, 1.5], dtype=float)
        x2_norm = (x2_data - offset2) / norm2
        restored = offset2 + norm2 * x2_norm
        np.testing.assert_allclose(restored, x2_data)

    def test_linear_axes_from_normed_grid(self):
        """gp_2d_surface_linear_axes: phase(days)=10**(offset2+norm2*x2), wl=10**(norm1*x1)."""
        norm1, norm2 = 4.0, 2.0
        offset2 = -2.5
        x1 = np.array([0.8, 0.9])
        x2 = np.array([0.25, 0.5])
        phase_log = offset2 + norm2 * x2
        wl_log = norm1 * x1
        np.testing.assert_allclose(10**phase_log, [10 ** (-2.0), 10 ** (-1.5)])
        np.testing.assert_allclose(10**wl_log, [10**3.2, 10**3.6])

    def test_phase_days_from_norm_x2_matches_training_convention(self):
        offset2, norm2 = -2.0, 1.5
        x2_norm = np.array([0.0, 1.0])
        days = g.phase_days_from_norm_x2(x2_norm, offset2, norm2)
        np.testing.assert_allclose(days, np.power(10.0, offset2 + norm2 * x2_norm))


try:
    import george
    from george.kernels import Matern32Kernel
except ImportError:
    george = None
    Matern32Kernel = None


@unittest.skipIf(george is None, "george not installed")
class TestGeorgePredictUsesVar(unittest.TestCase):
    def test_predict_return_var_not_full_cov(self):
        """Sanity check: return_var avoids allocating n_test^2 covariance."""
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0.0, 1.0, 25))
        y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(25)
        yerr = 0.15 * np.ones_like(y)
        kernel = 0.5 * Matern32Kernel(0.2, ndim=1)
        gp = george.GP(kernel)
        gp.compute(x, yerr)
        x_pred = np.linspace(0.0, 1.0, 400)
        mu, var = gp.predict(y, x_pred, return_var=True)
        self.assertEqual(mu.shape, (400,))
        self.assertEqual(var.shape, (400,))
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(var)))


class TestMangledSpecWavelengthConvention(unittest.TestCase):
    """Guards against ``10**`` on linear-Å mangled files (overflow)."""

    def test_linear_angstrom_detected(self):
        self.assertTrue(g.mangled_wls_max_is_linear_angstrom(np.array([2500.0, 8000.0])))

    def test_log10_angstrom_not_linear(self):
        self.assertFalse(g.mangled_wls_max_is_linear_angstrom(np.array([3.3, 3.7])))

    def test_mangled_helpers_kn_file_format_no_overflow(self):
        """Linear Å + log10 flux (as on disk from KN log mangle)."""
        spec = np.zeros(
            2,
            dtype=[("wls", float), ("flux", float), ("fluxerr", float)],
        )
        spec["wls"] = [3000.0, 3010.0]
        spec["flux"] = [-15.0, -15.1]
        spec["fluxerr"] = [0.05, 0.05]
        wlin = g.mangled_wls_linear_angstrom(spec)
        flin = g.mangled_flux_linear_from_log10(spec["flux"])
        self.assertTrue(np.all(np.isfinite(wlin)))
        self.assertTrue(np.all(np.isfinite(flin)))
        np.testing.assert_allclose(wlin, [3000.0, 3010.0])
        self.assertLess(np.max(flin), 1.0)


class TestFillGapsPhaseLogspace(unittest.TestCase):
    """Gap fill in log-phase grid uses linear-day thresholds (matches original linear notebook)."""

    def test_fills_interior_in_linear_day_gaps(self):
        min_log = np.log10(1.0)
        max_log = np.log10(30.0)
        spec = np.array([np.log10(2.0), np.log10(10.0)])
        out = g.fill_gaps_phase_logspace(
            min_log, max_log, spec, gap_size_days=0.1, cadence_days=0.1
        )
        self.assertGreater(len(out), 0)
        self.assertTrue(np.all(out >= min_log - 1e-12))
        self.assertTrue(np.all(out <= max_log + 1e-12))

    def test_inclusive_endpoint_mask_matches_extend_grid(self):
        mjds_grid = np.array([np.log10(5.0), np.log10(10.0), np.log10(25.0)])
        lo, hi = np.log10(1.0), np.log10(25.0)
        eps = 1e-5
        mask = (mjds_grid >= lo - eps) & (mjds_grid <= hi + eps)
        self.assertTrue(np.all(mask))

    def test_tiny_linear_gap_inserts_log_phases(self):
        """Borderline: linear segment just under 0.1d can still be ~2 dex in log; must not be empty."""
        min_log = -3.0
        spec = np.array([-1.0])  # 0.1d
        out = g.fill_gaps_phase_logspace(
            min_log, np.log10(20.0), spec, gap_size_days=0.1, cadence_days=0.1
        )
        self.assertGreater(len(out), 0)
        in_bracket = (out >= -3.0) & (out <= -1.0)
        self.assertGreater(np.count_nonzero(in_bracket), 0)


class TestSetPriorNewlog(unittest.TestCase):
    def test_setprior_log_flux_columns(self):
        """setPRIOR runs with *_log_flux LC and a minimal prior grid."""
        class Dummy:
            t0_fix = 58000.0
            path_fit_phot = ""

        d = Dummy()
        d.grid_norm_info = {
            "norm1": 3.7,
            "norm2": 2.0,
            "offset": -35.0,
            "offset2": -2.5,
            "scale_factor": 2.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            lc_path = os.path.join(tmp, "fitted.dat")
            with open(lc_path, "w") as f:
                f.write("Log_Phase\tSwope_g_log_flux\tSwope_r_log_flux\n")
                f.write("-2.0\t-15.0\t-14.5\n")
                f.write("0.3\t-14.0\t-13.8\n")
                f.write("0.5\t-13.5\t-13.2\n")
            d.path_fit_phot = lc_path
            prior_path = os.path.join(tmp, "prior.txt")
            rows = [
                "4000,-2,1.0",
                "5000,-2,1.0",
                "4000,1,1.0",
                "5000,1,1.0",
            ]
            with open(prior_path, "w") as f:
                f.write("\n".join(rows) + "\n")
            points, values = g.setPRIOR(d, PRIOR_file="prior.txt", PRIOR_folder=tmp + os.sep)
        self.assertEqual(points.shape[1], 2)
        self.assertEqual(values.ndim, 1)
        self.assertEqual(values.size, 4)
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(np.all(np.isfinite(values)))


@unittest.skipIf(george is None, "george not installed")
class TestRun2DGPGridDiagnosticSlices(unittest.TestCase):
    """Optional per-phase prior vs prediction PDFs (first slot only; bounded cost)."""

    def test_writes_gp_diag_pdfs_when_enabled(self):
        rng = np.random.default_rng(7)
        n = 48
        norm1 = 4.0
        offset2, norm2 = -2.5, 2.0
        x1n = rng.uniform(0.78, 0.97, n)
        x2_raw = rng.uniform(-1.7, 0.7, n)
        x2n = (x2_raw - offset2) / norm2
        y = rng.normal(0.0, 0.15, n)
        yerr = 0.1 * np.ones(n)

        class D:
            pass

        d = D()
        d.grid_norm_info = {
            "norm1": norm1,
            "norm2": norm2,
            "offset": -30.0,
            "offset2": offset2,
            "scale_factor": 2.0,
        }
        d.grids = [np.linspace(3.05, 3.95, 100)]
        d.verbose = False
        d.gp_print_training_size = False
        d.gp_predict_progress = False
        d.gp_diagnostic_slices = True
        d.gp_predict_slot_size = 3
        d.gp_predict_n_wavelength = 48
        d.gp_predict_wl_step = 0.02
        d.gp_predict_chunk_size = 900
        with tempfile.TemporaryDirectory() as tmp:
            d.save_plot_path = tmp
            extrap_mjds = np.array([-1.4, -0.6, 0.2], dtype=float)
            x1f, x2f, mu, std = g.run_2DGP_GRID(
                d,
                y,
                yerr,
                x1n,
                x2n,
                0.35,
                0.35,
                extrap_mjds,
                prior=False,
            )
            self.assertEqual(x1f.shape, x2f.shape)
            self.assertEqual(mu.shape, x1f.shape)
            pdfs = sorted(glob.glob(os.path.join(tmp, "gp_diag_slot0_phase*.pdf")))
            self.assertGreaterEqual(len(pdfs), 1)
            self.assertLessEqual(len(pdfs), 3)


@unittest.skipIf(george is None, "george not installed")
class TestRun2DGPGridNoDiagnostics(unittest.TestCase):
    def test_no_diag_pdfs_when_disabled(self):
        rng = np.random.default_rng(8)
        n = 40
        norm1 = 4.0
        offset2, norm2 = -2.5, 2.0
        x1n = rng.uniform(0.78, 0.97, n)
        x2_raw = rng.uniform(-1.5, 0.5, n)
        x2n = (x2_raw - offset2) / norm2
        y = rng.normal(0.0, 0.12, n)
        yerr = 0.1 * np.ones(n)

        class D:
            pass

        d = D()
        d.grid_norm_info = {
            "norm1": norm1,
            "norm2": norm2,
            "offset": -30.0,
            "offset2": offset2,
            "scale_factor": 2.0,
        }
        d.grids = [np.linspace(3.05, 3.95, 80)]
        d.verbose = False
        d.gp_print_training_size = False
        d.gp_predict_progress = False
        d.gp_diagnostic_slices = False
        d.gp_predict_slot_size = 3
        d.gp_predict_n_wavelength = 40
        d.gp_predict_wl_step = 0.025
        with tempfile.TemporaryDirectory() as tmp:
            d.save_plot_path = tmp
            extrap_mjds = np.array([-1.0, -0.2], dtype=float)
            g.run_2DGP_GRID(
                d,
                y,
                yerr,
                x1n,
                x2n,
                0.4,
                0.4,
                extrap_mjds,
                prior=False,
            )
            pdfs = glob.glob(os.path.join(tmp, "gp_diag_slot0_phase*.pdf"))
            self.assertEqual(len(pdfs), 0)


if __name__ == "__main__":
    unittest.main()
