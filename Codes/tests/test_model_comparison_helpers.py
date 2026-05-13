"""Smoke tests for Bulla HDF5 model comparison helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_DIR = os.path.dirname(os.path.abspath(__file__))
_CODES = os.path.dirname(_DIR)
_REPO = os.path.dirname(_CODES)
if _CODES not in sys.path:
    sys.path.insert(0, _CODES)

pytest.importorskip("h5py")

HDF5_PATH = os.path.join(
    _REPO,
    "2023_bulla",
    "nph1.0e+07_dyn0.005-0.20-0.20_wind0.050_fiducial.hdf5",
)


@pytest.fixture(scope="module")
def blob():
    from model_comparison_helpers import load_bulla_observables

    return load_bulla_observables(HDF5_PATH)


def test_hdf5_exists():
    assert os.path.isfile(HDF5_PATH)


def test_load_shapes(blob):
    I = blob["I_stokes"]
    assert I.ndim == 3
    n_obs, n_t, n_w = I.shape
    assert n_obs >= 2 and n_t >= 2 and n_w >= 2
    assert blob["time_days"].shape == (n_t,)
    assert blob["wave_rest"].shape == (n_w,)
    assert blob["lbol"].shape[0] == n_obs
    assert blob["lbol"].shape[1] == n_t


def test_cos_theta_grid(blob):
    from model_comparison_helpers import cos_theta_to_theta_deg

    c, th = cos_theta_to_theta_deg(blob["I_stokes"].shape[0])
    assert c[0] == 0.0 and c[-1] == 1.0
    assert th[0] == pytest.approx(90.0)
    assert th[-1] == pytest.approx(0.0)


def test_obs_indices_for_theta_deg_range_nonempty_and_bounded():
    from model_comparison_helpers import (
        cos_theta_to_theta_deg,
        obs_indices_for_theta_deg_range,
    )

    n_obs = 200
    c, th = cos_theta_to_theta_deg(n_obs)
    idx = obs_indices_for_theta_deg_range(n_obs, 13.0, 45.0, inclusive=True)
    assert len(idx) > 10
    for i in idx:
        assert 13.0 - 1e-9 <= th[i] <= 45.0 + 1e-9
    # θ decreases along increasing index on cos-θ-linear grid (90°→0°).
    assert th[idx[0]] >= th[idx[-1]]
    rev = sorted(idx, key=lambda ii: float(th[ii]), reverse=True)
    assert idx == sorted(idx)
    assert th[rev[0]] == pytest.approx(max(th[i] for i in idx))
    assert th[rev[-1]] == pytest.approx(min(th[i] for i in idx))


def test_obs_indices_for_theta_deg_range_swaps_lo_hi_and_inclusive():
    from model_comparison_helpers import (
        cos_theta_to_theta_deg,
        obs_indices_for_theta_deg_range,
    )

    n_obs = 50
    a = obs_indices_for_theta_deg_range(n_obs, 20.0, 30.0)
    b = obs_indices_for_theta_deg_range(n_obs, 30.0, 20.0)
    assert a == b
    _, th = cos_theta_to_theta_deg(n_obs)
    for i in a:
        assert 20.0 <= float(th[i]) <= 30.0
    inclusive_off = obs_indices_for_theta_deg_range(
        n_obs, 13.0, 45.0, inclusive=False
    )
    for i in inclusive_off:
        assert 13.0 < float(th[i]) < 45.0


def test_scaling_finite(blob):
    from model_comparison_helpers import scale_bulla_intensity_to_observer

    x = np.ones(5)
    y = scale_bulla_intensity_to_observer(x, z=0.01, d_lum_mpc=40.0)
    assert np.all(np.isfinite(y))
    assert y[0] > 0


def test_wave_observer():
    from model_comparison_helpers import wave_rest_to_observer

    w = np.array([1000.0, 5000.0])
    wo = wave_rest_to_observer(w, 0.01)
    assert np.allclose(wo, w * 1.01)


def test_interp_and_chi2():
    from model_comparison_helpers import chi2_flux, interp_model_flux_to_wavelengths

    wl = np.linspace(3000.0, 8000.0, 50)
    F = wl**-2
    wl2 = np.linspace(4000.0, 7000.0, 20)
    Fi = interp_model_flux_to_wavelengths(wl, F, wl2)
    assert Fi.shape == wl2.shape
    c2, n = chi2_flux(Fi, Fi, np.full_like(Fi, 0.1))
    assert n == Fi.size
    assert c2 == pytest.approx(0.0)


def test_nearest_obs_index_cos_theta():
    from model_comparison_helpers import nearest_obs_index_for_cos_theta

    grid = np.linspace(0.0, 1.0, 5)
    assert nearest_obs_index_for_cos_theta(grid, 0.23) == 1
    assert nearest_obs_index_for_cos_theta(grid, 1.0) == 4
    assert nearest_obs_index_for_cos_theta(grid, 0.0) == 0
    with pytest.warns(UserWarning):
        assert nearest_obs_index_for_cos_theta(grid, 1.5) == 4
    with pytest.warns(UserWarning):
        assert nearest_obs_index_for_cos_theta(grid, -0.3) == 0


def test_pick_final_by_nearest_phase():
    from model_comparison_helpers import pick_final_by_nearest_phase

    rows = [
        {"fname": "a.txt", "mjd": 1.0, "phase_days": 0.0},
        {"fname": "b.txt", "mjd": 2.0, "phase_days": 2.0},
        {"fname": "c.txt", "mjd": 1.5, "phase_days": 1.0},
    ]
    p = pick_final_by_nearest_phase(rows, 0.8)
    assert p["fname"] == "c.txt"
    assert p["phase_days"] == pytest.approx(1.0)
    assert p["abs_delta_phase_days"] == pytest.approx(0.2)
    assert p["delta_phase_days"] == pytest.approx(0.2)


def test_model_flux_for_epoch(blob):
    from model_comparison_helpers import model_flux_for_epoch_and_angle

    z = 0.01
    d = 40.0
    ph = float(np.median(blob["time_days"]))
    wl, F, _ = model_flux_for_epoch_and_angle(
        blob,
        obs_index=0,
        phase_days_target=ph,
        z=z,
        d_lum_mpc=d,
        time_interp=True,
    )
    m = np.isfinite(F) & (F > 0)
    assert np.count_nonzero(m) > 10


def test_chi2_vs_obs(blob):
    from model_comparison_helpers import chi2_vs_obs_indices, interp_model_flux_to_wavelengths

    z = 0.00984
    d = 43.0
    w_rest = blob["wave_rest"]
    wl_obs = w_rest * (1.0 + z)
    # Fake "data" = middle angle model at median phase
    from model_comparison_helpers import model_flux_for_epoch_and_angle

    ph = float(np.median(blob["time_days"]))
    wl_m, F_ref, _ = model_flux_for_epoch_and_angle(
        blob,
        obs_index=blob["I_stokes"].shape[0] // 2,
        phase_days_target=ph,
        z=z,
        d_lum_mpc=d,
        time_interp=True,
    )
    wl_d = wl_obs[(wl_obs >= np.min(wl_m)) & (wl_obs <= np.max(wl_m))][::5]
    if wl_d.size < 5:
        pytest.skip("wavelength overlap too small")
    F_d = interp_model_flux_to_wavelengths(wl_m, F_ref, wl_d)
    fe = 0.1 * F_d
    mid = blob["I_stokes"].shape[0] // 2
    rows = chi2_vs_obs_indices(
        wl_d,
        F_d,
        fe,
        blob,
        phase_days_target=ph,
        z=z,
        d_lum_mpc=d,
        obs_indices=[0, mid],
        time_interp=True,
    )
    assert len(rows) == 2
    for r in rows:
        assert np.isfinite(r["chi2"]) and r["n_pix"] > 0
    by_idx = {r["obs_index"]: r["chi2"] for r in rows}
    assert by_idx[mid] < by_idx[0]


def test_pooled_chi2_all_epochs_angles_matches_chi2_vs_obs(blob):
    from model_comparison_helpers import (
        chi2_vs_obs_indices,
        interp_model_flux_to_wavelengths,
        model_flux_for_epoch_and_angle,
        pooled_chi2_all_epochs_angles,
    )

    z = 0.00984
    d = 43.0
    ph = float(np.median(blob["time_days"]))
    n_obs = blob["I_stokes"].shape[0]
    wl_m, F_ref, _ = model_flux_for_epoch_and_angle(
        blob,
        obs_index=n_obs // 2,
        phase_days_target=ph,
        z=z,
        d_lum_mpc=d,
        time_interp=True,
    )
    w_rest = blob["wave_rest"]
    wl_obs = w_rest * (1.0 + z)
    wl_d = wl_obs[(wl_obs >= np.min(wl_m)) & (wl_obs <= np.max(wl_m))][::5]
    if wl_d.size < 5:
        pytest.skip("wavelength overlap too small")
    F_d = interp_model_flux_to_wavelengths(wl_m, F_ref, wl_d)
    fe = 0.1 * np.abs(F_d) + 1e-30
    rows = chi2_vs_obs_indices(
        wl_d,
        F_d,
        fe,
        blob,
        phase_days_target=ph,
        z=z,
        d_lum_mpc=d,
        obs_indices=list(range(n_obs)),
        time_interp=True,
    )
    s_chi = sum(float(r["chi2"]) for r in rows)
    s_n = sum(int(r["n_pix"]) for r in rows)
    epochs = [
        {
            "wl_data": wl_d,
            "F_data": F_d,
            "fe_data": fe,
            "phase_model_days": ph,
        }
    ]
    c2, n = pooled_chi2_all_epochs_angles(
        blob, epochs, z=z, d_lum_mpc=d, time_interp=True, uncertainty_scale=1.0
    )
    assert c2 == pytest.approx(s_chi)
    assert n == s_n


def test_pooled_chi2_uncertainty_scale_quarters_chi2(blob):
    from model_comparison_helpers import (
        interp_model_flux_to_wavelengths,
        model_flux_for_epoch_and_angle,
        pooled_chi2_all_epochs_angles,
    )

    z = 0.00984
    d = 43.0
    ph = float(np.median(blob["time_days"]))
    n_obs = blob["I_stokes"].shape[0]
    wl_m, F_ref, _ = model_flux_for_epoch_and_angle(
        blob,
        obs_index=n_obs // 2,
        phase_days_target=ph,
        z=z,
        d_lum_mpc=d,
        time_interp=True,
    )
    w_rest = blob["wave_rest"]
    wl_obs = w_rest * (1.0 + z)
    wl_d = wl_obs[(wl_obs >= np.min(wl_m)) & (wl_obs <= np.max(wl_m))][::5]
    if wl_d.size < 5:
        pytest.skip("wavelength overlap too small")
    F_d = interp_model_flux_to_wavelengths(wl_m, F_ref, wl_d)
    fe = 0.1 * np.abs(F_d) + 1e-30
    epochs = [
        {
            "wl_data": wl_d,
            "F_data": F_d,
            "fe_data": fe,
            "phase_model_days": ph,
        }
    ]
    c2a, _ = pooled_chi2_all_epochs_angles(
        blob, epochs, z=z, d_lum_mpc=d, time_interp=True, uncertainty_scale=1.0
    )
    c2b, _ = pooled_chi2_all_epochs_angles(
        blob, epochs, z=z, d_lum_mpc=d, time_interp=True, uncertainty_scale=2.0
    )
    assert c2b == pytest.approx(c2a / 4.0)


def test_pooled_chi2_obs_indices_subset(blob):
    from model_comparison_helpers import (
        chi2_vs_obs_indices,
        interp_model_flux_to_wavelengths,
        model_flux_for_epoch_and_angle,
        pooled_chi2_all_epochs_angles,
    )

    z = 0.00984
    d = 43.0
    ph = float(np.median(blob["time_days"]))
    n_obs = blob["I_stokes"].shape[0]
    wl_m, F_ref, _ = model_flux_for_epoch_and_angle(
        blob,
        obs_index=n_obs // 2,
        phase_days_target=ph,
        z=z,
        d_lum_mpc=d,
        time_interp=True,
    )
    w_rest = blob["wave_rest"]
    wl_obs = w_rest * (1.0 + z)
    wl_d = wl_obs[(wl_obs >= np.min(wl_m)) & (wl_obs <= np.max(wl_m))][::5]
    if wl_d.size < 5:
        pytest.skip("wavelength overlap too small")
    F_d = interp_model_flux_to_wavelengths(wl_m, F_ref, wl_d)
    fe = 0.1 * np.abs(F_d) + 1e-30
    pick = [1, n_obs // 2]
    rows = chi2_vs_obs_indices(
        wl_d,
        F_d,
        fe,
        blob,
        phase_days_target=ph,
        z=z,
        d_lum_mpc=d,
        obs_indices=pick,
        time_interp=True,
    )
    target = sum(float(r["chi2"]) for r in rows)
    epochs = [
        {
            "wl_data": wl_d,
            "F_data": F_d,
            "fe_data": fe,
            "phase_model_days": ph,
        }
    ]
    c2, n = pooled_chi2_all_epochs_angles(
        blob, epochs, z=z, d_lum_mpc=d, time_interp=True, obs_indices=pick
    )
    assert c2 == pytest.approx(target)


def test_pooled_chi2_rejects_bad_uncertainty_scale(blob):
    from model_comparison_helpers import pooled_chi2_all_epochs_angles

    with pytest.raises(ValueError, match="uncertainty_scale"):
        pooled_chi2_all_epochs_angles(
            blob, [], z=0.01, d_lum_mpc=40.0, uncertainty_scale=-1.0
        )
