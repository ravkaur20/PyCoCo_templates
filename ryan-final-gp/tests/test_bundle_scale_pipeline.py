"""Smoke tests for bundle-scale χ² linkage (no collaborator bundle required)."""

from __future__ import annotations

import numpy as np

import bundle_scale_pipeline as bsp


def test_affine_flux_prediction_at_lambda_centered_simple():
    w = np.linspace(9000.0, 9030.0, 31)
    f = np.full_like(w, 2.5)
    ferr = np.full_like(w, 0.1)
    mid = float(w[len(w) // 2])
    mu_hat, sig = bsp.affine_flux_prediction_at_lambda(
        w.copy(),
        f,
        ferr,
        mid + 80.0,
        n_pix_edge=10,
        side="left",
    )
    assert abs(mu_hat - 2.5) < 0.06
    assert sig > 0


def test_solve_epoch_pair_two_overlapping_flats():
    w1 = np.linspace(9800.0, 9950.0, 121)
    w2 = np.linspace(9860.0, 9980.0, 151)
    f1 = np.ones_like(w1) * 3.8
    f2 = np.ones_like(w2) * 1.22
    e1 = np.full_like(w1, 0.05)
    e2 = np.full_like(w2, 0.05)
    s = bsp.solve_pair_scale(
        w1, f1, e1, w2, f2, e2, seam_weight=0.5, overlap_grid=320, seam_band_half_width_aa=50.0
    )
    assert 2.95 < s < 3.25


def test_composite_epoch_linear_does_not_merge_near_phases():
    """Phases within a loose atol must not be merged into one composite (intra-bundle χ²)."""
    gn = dict(_normalized_only=True, x1_mean=0.0, x1_std=1.0, offset=0.0, scale_factor=1.0)
    p0, p1 = -0.831359230, -0.831358570  # >1e-6 apart, same at atol=5e-6
    assert abs(p0 - p1) < 5e-6
    x1 = np.linspace(0.0, 1.0, 40)
    X = np.vstack(
        [
            np.column_stack([x1, np.full(40, p0)]),
            np.column_stack([x1 + 0.01, np.full(40, p1)]),
        ]
    )
    y = np.zeros(80)
    yerr = np.full(80, 0.01)
    w0, _, _ = bsp.composite_epoch_linear(X, y, yerr, gn, p0, phase_atol=5e-6)
    w1, _, _ = bsp.composite_epoch_linear(X, y, yerr, gn, p1, phase_atol=5e-6)
    assert w0.size == 40 and w1.size == 40


def test_intrabundle_scales_accumulate_positive():
    # Two synthetic epochs differing by ~2× and overlapping in λ-center ordering.
    n = 200
    wl = np.linspace(9000.0, 9650.0, n)

    epochs = [-0.8123, -0.812299]  # two distinct epochs (phase column)

    yer = np.full(n, 0.005)

    X_rows = []
    yy = []
    ye = []

    gn = dict(
        _normalized_only=True,
        x1_mean=0.0,
        x1_std=1.0,
        offset=0.0,
        scale_factor=1.0,
    )

    def norm_x_from_w(wave_aa: np.ndarray, wave_ref: np.ndarray = wl) -> np.ndarray:
        u = wave_aa / wave_ref.mean()
        return np.log10(u)

    for ph, fac in zip(epochs, (1.0, 0.5)):
        x1 = norm_x_from_w(wl)
        X_rows.append(np.column_stack([x1, np.full(n, ph)]))
        yy.append(np.full(n, np.log(max(30.0 * fac, 1e-300))))
        ye.append(yer.copy())

    X = np.vstack(X_rows)
    y = np.concatenate(yy)
    yerr = np.concatenate(ye)

    canonical = np.asarray([float(epochs[0]), float(epochs[1])], dtype=float)
    epoch_of_row = np.zeros(X.shape[0], dtype=np.int32)
    epoch_of_row[n:] = 1
    mult_map = bsp.intra_bundle_epoch_scales(
        X,
        y,
        yerr,
        gn,
        canonical,
        np.arange(2),
        epoch_of_row,
        phase_atol=1e-6,
        seam_weight=2.5,
        overlap_grid_points=400,
        seam_band_half_width_aa=50.0,
    )
    assert np.isfinite(mult_map[0])
    assert np.isfinite(mult_map[1])
    assert mult_map[0] == 1.0


def test_solve_gap_seam_scale_disjoint_spectra():
    w1 = np.linspace(4000.0, 5000.0, 80)
    w2 = np.linspace(7000.0, 8000.0, 80)
    f1 = np.ones_like(w1)
    f2 = np.ones_like(w2) * 2.0
    e1 = np.full_like(w1, 0.05)
    e2 = np.full_like(w2, 0.05)
    s = bsp.solve_gap_seam_scale(
        w1, f1, e1, w2, f2, e2, seam_weight=3.0, seam_band_half_width_aa=50.0
    )
    assert np.isfinite(s)
    assert 0.35 < s < 0.65


def test_forward_mst_three_overlapping_epochs():
    gn = dict(
        _normalized_only=True,
        x1_mean=0.0,
        x1_std=1.0,
        offset=0.0,
        scale_factor=1.0,
    )

    def stack(wl0: float, wl1: float, n: int, ph: float, ln_amp: float):
        x1 = np.log10(np.linspace(wl0, wl1, n, dtype=float))
        return np.column_stack([x1, np.full(n, ph)]), np.full(n, float(np.log(max(ln_amp, 1e-300)))), np.full(n, 0.01)

    ph0, ph1, ph2 = -0.910000001, -0.920000002, -0.930000003
    X0, y0, e0 = stack(9000.0, 9500.0, 120, ph0, 40.0)
    X1, y1, e1 = stack(9300.0, 9800.0, 140, ph1, 40.0)
    X2, y2, e2 = stack(9600.0, 10100.0, 130, ph2, 40.0)
    X = np.vstack([X0, X1, X2])
    y = np.concatenate([y0, y1, y2])
    yerr = np.concatenate([e0, e1, e2])
    canonical = np.asarray([ph0, ph1, ph2], dtype=float)
    epoch_of_row = np.zeros(X.shape[0], dtype=np.int32)
    epoch_of_row[120:260] = 1
    epoch_of_row[260:] = 2
    mult = bsp.intra_bundle_epoch_scales(
        X,
        y,
        yerr,
        gn,
        canonical,
        np.arange(3),
        epoch_of_row,
        phase_atol=1e-6,
        seam_weight=2.0,
        overlap_grid_points=320,
        seam_band_half_width_aa=50.0,
    )
    assert len(mult) == 3
    assert all(np.isfinite(v) and v > 0 for v in mult.values())
    data = {}
    for ee in (0, 1, 2):
        data[ee] = bsp.composite_epoch_linear(
            X, y, yerr, gn, bsp.canon_phase(canonical, ee), phase_atol=1e-6
        )
    edges = bsp._forward_mst_edge_indices([0, 1, 2], data)
    assert len(edges) == 2


def test_bluest_representative_epoch_matches_argmin_median_wl():
    gn = dict(
        _normalized_only=True,
        x1_mean=0.0,
        x1_std=1.0,
        offset=0.0,
        scale_factor=1.0,
    )
    ph0, ph1 = -0.500000001, -0.510000002
    n = 40
    X = np.vstack(
        [
            np.column_stack([np.linspace(0.0, 1.0, n), np.full(n, ph0)]),
            np.column_stack([np.linspace(0.25, 1.25, n), np.full(n, ph1)]),
        ]
    )
    y = np.zeros(2 * n)
    yerr = np.full(2 * n, 0.02)
    canon = np.asarray([ph0, ph1], dtype=float)
    epochs_b = np.array([0, 1], dtype=int)
    med_wl = []
    for e in epochs_b:
        w_e, _, _ = bsp.composite_epoch_linear(
            X, y, yerr, gn, bsp.canon_phase(canon, int(e)), phase_atol=1e-6
        )
        med_wl.append(float(np.nanmedian(w_e)) if w_e.size else float("inf"))
    rep_ep = int(epochs_b[int(np.argmin(med_wl))])
    wl_rep, _, _ = bsp.composite_epoch_linear(
        X, y, yerr, gn, bsp.canon_phase(canon, rep_ep), phase_atol=1e-6
    )
    wl0, _, _ = bsp.composite_epoch_linear(
        X, y, yerr, gn, bsp.canon_phase(canon, 0), phase_atol=1e-6
    )
    assert rep_ep == 0
    assert wl_rep.size == wl0.size


def test_composite_epoch_linear_excludes_rows_not_in_epoch_id():
    """Phot (or any) rows sharing rounded x₂ must not enter a spectroscopic epoch composite."""
    gn = dict(_normalized_only=True, x1_mean=0.0, x1_std=1.0, offset=0.0, scale_factor=1.0)
    p0 = -0.831359230
    n = 30
    x1_spec = np.linspace(0.0, 1.0, n)
    x1_phot = np.linspace(0.05, 0.95, n)
    X = np.vstack(
        [
            np.column_stack([x1_spec, np.full(n, p0)]),
            np.column_stack([x1_phot, np.full(n, p0)]),
        ]
    )
    y = np.zeros(2 * n)
    yerr = np.full(2 * n, 0.02)
    eor = np.array([0] * n + [-1] * n, dtype=np.int32)
    w_all, _, _ = bsp.composite_epoch_linear(X, y, yerr, gn, p0, phase_atol=1e-6)
    w_ep0, _, _ = bsp.composite_epoch_linear(
        X, y, yerr, gn, p0, phase_atol=1e-6, epoch_of_row=eor, epoch_id=0
    )
    assert w_all.size == 2 * n
    assert w_ep0.size == n


def test_phot_band_gp_flat_prediction():
    x = np.linspace(-0.9, -0.5, 25, dtype=float)
    y = np.full(25, 3.0)
    e = np.full(25, 0.1)
    pack = bsp._fit_phot_band_gp(x, y, e)
    assert pack is not None
    mu = bsp._mu_phot_gp(-0.7, pack)
    assert abs(mu - 3.0) < 0.08


def test_solve_gap_seam_sloped_abutting_applies_scale():
    """Gap arms with slopes: default path must return s≠1 when levels differ at the seam."""
    w1 = np.linspace(4000.0, 4980.0, 120)
    w2 = np.linspace(5020.0, 6000.0, 120)
    f1 = 1e-15 * (1.0 + 2e-4 * (w1 - 4000.0))
    f2 = 1.35e-15 * (1.0 + 1e-4 * (w2 - 5020.0))
    e1 = np.full_like(w1, 0.05 * 1e-15)
    e2 = np.full_like(w2, 0.05 * 1e-15)
    s = bsp.solve_gap_seam_scale(
        w1, f1, e1, w2, f2, e2, seam_weight=2.0, seam_band_half_width_aa=50.0
    )
    assert np.isfinite(s)
    assert abs(s - 1.0) > 0.02
    assert 0.5 < s < 1.5


def test_gap_veto_min_rel_gain_can_force_unity():
    """Legacy veto: impossible rel-gain threshold must reject the scale (return 1)."""
    w1 = np.linspace(4000.0, 5000.0, 80)
    w2 = np.linspace(7000.0, 8000.0, 80)
    f1 = np.ones_like(w1)
    f2 = np.ones_like(w2) * 2.0
    e1 = np.full_like(w1, 0.05)
    e2 = np.full_like(w2, 0.05)
    s_free = bsp.solve_gap_seam_scale(
        w1, f1, e1, w2, f2, e2, seam_weight=3.0, seam_band_half_width_aa=50.0
    )
    s_veto = bsp.solve_gap_seam_scale(
        w1,
        f1,
        e1,
        w2,
        f2,
        e2,
        seam_weight=3.0,
        seam_band_half_width_aa=50.0,
        gap_veto_min_rel_gain=1.5,
    )
    assert abs(s_free - 1.0) > 0.1
    assert s_veto == 1.0


def test_rough_bundle_log_scales_matches_photometry_ratio():
    """Rough bundle helper: phot vs spec at same phase and overlapping λ → log scale ≈ log(f_phot/f_spec)."""
    gn = dict(_normalized_only=True, x1_mean=0.0, x1_std=1.0, offset=0.0, scale_factor=1.0)
    n = 24
    x1 = np.linspace(1.0, 24.0, n)
    ph = 0.0
    X_p = np.column_stack([x1, np.full(n, ph)])
    y_p = np.full(n, 3.0)
    e_p = np.full(n, 0.05)
    X_s = np.column_stack([x1, np.full(n, ph)])
    y_s = np.full(n, 1.0)
    e_s = np.full(n, 0.05)
    X = np.vstack([X_p, X_s])
    y = np.concatenate([y_p, y_s])
    yerr = np.concatenate([e_p, e_s])
    phot_mask = np.zeros(2 * n, dtype=bool)
    phot_mask[:n] = True
    epoch_of_row = np.full(2 * n, -1, dtype=np.int32)
    epoch_of_row[n:] = 0
    canonical = np.array([float(ph)], dtype=float)
    epochs_b = np.array([0], dtype=int)
    dl, bm = bsp.estimate_bundle_log_scales_rough_phot_wavelength_points(
        X,
        y,
        yerr,
        gn,
        phot_mask,
        canonical,
        epoch_of_row,
        epochs_b,
        phase_epoch_atol=1e-6,
        rough_phot_phase_window_norm=0.1,
    )
    assert bm["anchor_mode"] == "rough_phot_wavelength_points"
    assert bm["any_applied"] is True
    assert abs(float(dl[0]) - np.log(3.0)) < 0.06


def test_estimate_epoch_log_scale_min_points_one():
    """With min_points=1, a single phot row can anchor if spec has one positive point."""
    gn = dict(_normalized_only=True, x1_mean=0.0, x1_std=1.0, offset=0.0, scale_factor=1.0)
    # Under _normalized_only, x1 is interpreted as wavelength in Å (see wavelength_aa_from_x1_norm).
    x1 = np.array([5000.0, 5000.0])
    ph = 0.0
    X = np.column_stack([x1, np.full(2, ph)])
    y = np.array([2.0, 0.0])
    yerr = np.array([0.1, 0.05])
    phot_mask = np.array([True, False])
    wl_spec = np.array([5000.0])
    fl_spec = np.array([1.0])
    dlog, meta = bsp.estimate_epoch_log_scale_from_phot_wavelength_points(
        X=X,
        y=y,
        yerr=yerr,
        gn=gn,
        phot_mask=phot_mask,
        ph_epoch=float(ph),
        wl_spec=wl_spec,
        fl_spec=fl_spec,
        phot_phase_window_norm=0.5,
        min_points=1,
    )
    assert meta.get("applied") is True
    assert abs(float(dlog) - np.log(2.0)) < 1e-9


def test_bundle_pooled_phot_chi2_linear_scale_simple():
    """Pooled χ²: two phot points vs flat spec → s* = weighted LS ratio."""
    gn = dict(_normalized_only=True, x1_mean=0.0, x1_std=1.0, offset=0.0, scale_factor=1.0)
    n = 20
    x1s = np.linspace(0.0, 1.0, n)
    ph_spec = -0.2
    X_s = np.column_stack([x1s, np.full(n, ph_spec)])
    y_s = np.full(n, 1.0)
    yerr_s = np.full(n, 0.02)
    # Two phot points at same phase as spec, λ inside spec range (x1 0..1 maps via gn - use same x1 grid)
    X_p = np.array([[0.3, ph_spec], [0.7, ph_spec]], dtype=float)
    y_p = np.array([4.0, 4.0])
    yerr_p = np.array([0.1, 0.1])
    X = np.vstack([X_p, X_s])
    y = np.concatenate([y_p, y_s])
    yerr = np.concatenate([yerr_p, yerr_s])
    phot_mask = np.array([True, True] + [False] * n, dtype=bool)
    epoch_of_row = np.full(X.shape[0], -1, dtype=np.int32)
    epoch_of_row[2:] = 0
    canonical = np.array([float(ph_spec)], dtype=float)
    epochs_b = np.array([0], dtype=int)
    s_star, meta = bsp.estimate_bundle_pooled_phot_chi2_linear_scale(
        X,
        y,
        yerr,
        gn,
        phot_mask,
        canonical,
        epoch_of_row,
        epochs_b,
        phase_epoch_atol=1e-6,
        rough_phot_phase_window_norm=0.05,
        bundle_phot_pool_max_phase_window_norm=0.2,
        phot_anchor_min_points=1,
    )
    assert meta.get("any_applied") is True
    assert meta.get("phot_anchor_n_points", 0) >= 1
    # spec linear flux 1 at both λ → D=4, σ=0.1 → s* = 4
    assert abs(s_star - 4.0) < 0.08

