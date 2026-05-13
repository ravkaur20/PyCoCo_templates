"""Tiny smoke tests for vendor collaborator inference (George required)."""

import os
import sys
import tempfile
import unittest

import numpy as np

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
    sys.path.insert(0, CODES)

try:
    from gp_collab_rjf.run_inference import run_gp_from_bundle
except ImportError:
    run_gp_from_bundle = None

skip_inference = unittest.skipIf(
    run_gp_from_bundle is None, "gp_collab_rjf import failed"
)


def _tiny_bundle(mean_none: bool = True):
    rng = np.random.default_rng(42)
    n = 45
    X = rng.uniform(0.12, 0.93, size=(n, 2)).astype(np.float64)
    y = 0.2 * np.sin(4 * np.pi * X[:, 0]) + 0.15 * X[:, 1] + rng.normal(0, 0.04, size=n)
    yerr = np.full(n, 0.06, dtype=np.float64)
    wg = np.linspace(0.2, 0.82, 6)
    ph = np.linspace(0.18, 0.87, 5)
    x1 = []
    x2 = []
    for a in wg:
        for b in ph:
            x1.append(a)
            x2.append(b)
    X_fill = np.column_stack([x1, x2]).astype(np.float64)
    d = dict(
        X=X,
        y=y,
        yerr=yerr,
        X_fill=X_fill,
        kernel_wls_scale=np.float64(1e-2),
        kernel_time_scale=np.float64(1e-2),
        y_var_scale=np.float64(np.var(y)),
        prior_points=np.zeros((0, 2)),
        prior_values=np.zeros(0),
    )
    return d


@skip_inference
class TestGpCollaboratorInference(unittest.TestCase):
    def test_run_mean_none_fixed_hypers(self):
        with tempfile.TemporaryDirectory() as tmp:
            bd = _tiny_bundle()
            out = run_gp_from_bundle(
                bd,
                cache_workdir=tmp,
                mean="none",
                optimize=False,
                predict_chunk=500,
                predict_train=False,
            )
        mu = np.asarray(out["mu"], dtype=float)
        std = np.asarray(out["std"], dtype=float)
        self.assertEqual(mu.shape[0], bd["X_fill"].shape[0])
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(std >= 0.0))


@skip_inference
class TestGp2dimGridRjfSmoke(unittest.TestCase):
    """``run_2DGP_GRID_rjf`` grid builder vs classic for same extrap list."""

    def test_fill_rowcount_matches_legacy_pattern(self):
        import GP2dim_utils_newlog_rjf as grjf

        class D:
            pass

        d = D()
        d.snname = "TEST"
        d.mode = "extend_spectra"
        with tempfile.TemporaryDirectory() as tmp:
            d.save_plot_path = tmp
        d.grids = (np.linspace(3.35, 3.55, 8), [])
        d.grid_norm_info = {"norm1": 4.0, "norm2": 1.8, "offset2": -2.0, "offset": 0.0, "scale_factor": 1.0}
        d.pipeline_wl_min_a = None
        d.pipeline_wl_max_a = None
        d.gp_predict_n_wavelength = 12
        d.gp_predict_wl_step = 0.05
        d.gp_predict_dense_log_phase = False
        d.gp_predict_dense_log_phase_n = 32
        d.gp_2d_anchor_t0 = False
        d.gp_predict_progress = False
        d.verbose = False
        d.gp_print_training_size = False

        extrap = np.linspace(-2.9, -0.9, 5)
        y = np.linspace(-0.1, 0.1, 20)
        ye = np.full_like(y, 0.05)
        x1n = np.linspace(0.85, 0.93, y.size)
        x2n = np.linspace(0.1, 0.5, y.size)

        wls_min = float(np.min(d.grids[0]))
        wls_max = float(np.max(d.grids[0]))
        span_wl = float(wls_max - wls_min)
        _wl_step = float(d.gp_predict_wl_step)
        n_from_step = int(np.ceil(span_wl / _wl_step)) + 1
        n_wl_use = max(2, min(int(d.gp_predict_n_wavelength), n_from_step))
        n_expected = n_wl_use * len(extrap)

        # Patch run_gp_from_bundle to avoid heavy fit
        def _stub(bundle, **_):
            xf = bundle["X_fill"]
            nn = xf.shape[0]
            z = np.zeros(nn, dtype=float)
            return {
                "mu": z + 1.0,
                "mu_raw": z + 2.0,
                "std": np.ones(nn, dtype=float) * 0.01,
                "var": np.ones(nn, dtype=float) * 0.0001,
                "X_fill": xf,
                "log_likelihood": 1.0,
                "total_runtime_seconds": 0.0,
                "config_final": {},
            }

        saved_infer = grjf.run_gp_from_bundle
        saved_export = grjf.maybe_save_gp_minimal_export
        try:
            grjf.run_gp_from_bundle = _stub  # type: ignore

            def _no_export(*_args, **_kw):
                return None

            grjf.maybe_save_gp_minimal_export = _no_export  # type: ignore[attr-defined]

            pts = np.arange(60, dtype=float).reshape(-1, 2)
            vals = np.zeros(pts.shape[0])
            _, _, mf, sf, mrf = grjf.run_2DGP_GRID_rjf(
                d,
                y,
                ye,
                x1n,
                x2n,
                5e-3,
                5e-3,
                extrap,
                prior=True,
                points=pts,
                values=vals,
            )
        finally:
            grjf.run_gp_from_bundle = saved_infer  # type: ignore
            grjf.maybe_save_gp_minimal_export = saved_export  # type: ignore[attr-defined]

        self.assertEqual(mf.size, n_expected)


@skip_inference
class TestGp2dimGridRjfZscoreSmoke(unittest.TestCase):
    """``run_2DGP_GRID_rjf`` with ``GP2dim_utils_newlog_zscore``-style ``grid_norm_info``."""

    def test_no_keyerror_norm1_stub_infer(self):
        import GP2dim_utils_newlog_rjf as grjf

        class D:
            pass

        d = D()
        d.snname = "TEST"
        d.mode = "extend_spectra"
        with tempfile.TemporaryDirectory() as tmp:
            d.save_plot_path = tmp
        d.grids = (np.linspace(3.35, 3.55, 8), [])
        d.grid_norm_info = {
            "offset": 0.0,
            "scale_factor": 1.0,
            "coord_parametrization": "zscore",
            "x1_mean": 3.45,
            "x1_std": 0.1,
            "x2_mean": -1.8,
            "x2_std": 0.45,
            "x2_train_min": -2.5,
        }
        d.pipeline_wl_min_a = None
        d.pipeline_wl_max_a = None
        d.gp_predict_n_wavelength = 12
        d.gp_predict_wl_step = 0.05
        d.gp_predict_dense_log_phase = False
        d.gp_predict_dense_log_phase_n = 32
        d.gp_2d_anchor_t0 = False
        d.gp_predict_progress = False
        d.verbose = False
        d.gp_print_training_size = False

        extrap = np.linspace(-2.9, -0.9, 5)
        y = np.linspace(-0.1, 0.1, 20)
        ye = np.full_like(y, 0.05)
        gn = d.grid_norm_info
        x1n = (np.linspace(3.36, 3.52, y.size) - gn["x1_mean"]) / gn["x1_std"]
        x2n = (np.linspace(-2.0, -1.0, y.size) - gn["x2_mean"]) / gn["x2_std"]

        wls_min = float(np.min(d.grids[0]))
        wls_max = float(np.max(d.grids[0]))
        span_wl = float(wls_max - wls_min)
        n_from_step = int(np.ceil(span_wl / float(d.gp_predict_wl_step))) + 1
        n_wl_use = max(2, min(int(d.gp_predict_n_wavelength), n_from_step))
        n_expected = n_wl_use * len(extrap)

        def _stub(bundle, **_):
            xf = bundle["X_fill"]
            nn = xf.shape[0]
            z = np.zeros(nn, dtype=float)
            return {
                "mu": z + 1.0,
                "mu_raw": z + 2.0,
                "std": np.ones(nn, dtype=float) * 0.01,
                "var": np.ones(nn, dtype=float) * 0.0001,
                "X_fill": xf,
                "log_likelihood": 1.0,
                "total_runtime_seconds": 0.0,
                "config_final": {},
            }

        saved_infer = grjf.run_gp_from_bundle
        saved_export = grjf.maybe_save_gp_minimal_export
        try:
            grjf.run_gp_from_bundle = _stub  # type: ignore

            def _no_export(*_args, **_kw):
                return None

            grjf.maybe_save_gp_minimal_export = _no_export  # type: ignore[attr-defined]

            pts = np.zeros((4, 2), dtype=float)
            vals = np.zeros(4, dtype=float)
            _, _, mf, sf, mrf = grjf.run_2DGP_GRID_rjf(
                d,
                y,
                ye,
                x1n,
                x2n,
                5e-3,
                5e-3,
                extrap,
                prior=False,
                points=pts,
                values=vals,
            )
        finally:
            grjf.run_gp_from_bundle = saved_infer  # type: ignore
            grjf.maybe_save_gp_minimal_export = saved_export  # type: ignore[attr-defined]

        self.assertEqual(mf.size, n_expected)


if __name__ == "__main__":
    unittest.main()
