"""Helpers for comparing PyCoCo FINAL spectra to Bulla POSSIS models.

HDF5 layout and scaling convention: ``2023_bulla/make_lcs_hdf5.py`` (native spectral
density referenced at **1 Mpc**, ``F_obs ∝ native × (1/D_L)² / (1+z)``).

2019 ASCII (``2019_bulla/README``, ``load_bulla2019_txt``): natives are flux at **10 pc**
(**10⁻⁵ Mpc**); use ``reference_distance_mpc=1e-5`` → ``make_lcs.py``-consistent
``(10⁻⁵/D_L)²/(1+z)``. See ``scale_bulla_intensity_to_observer``.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Sequence

import numpy as np

from comparison_check_log_utils import parse_final_stem, stem_to_spec_mjd

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def require_h5py():
    if h5py is None:
        raise ImportError(
            "h5py is required for Bulla model files. Install with: pip install h5py"
        )


def load_bulla_observables(path: str) -> dict[str, Any]:
    """Load ``observables`` group from a Bulla-style HDF5 file."""
    require_h5py()
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if "observables" not in f:
            raise KeyError("Expected group 'observables' in %s" % path)
        obs = f["observables"]
        stokes = np.asarray(obs["stokes"], dtype=float)
        time_s = np.asarray(obs["time"], dtype=float)
        wave = np.asarray(obs["wave"], dtype=float)
        lbol = np.asarray(obs["lbol"], dtype=float)
    I = stokes[..., 0]
    t_days = time_s / (60.0 * 60.0 * 24.0)
    return {
        "I_stokes": I,
        "time_days": t_days,
        "wave_rest": wave,
        "lbol": lbol,
        "stokes_shape": stokes.shape,
        # Native spectral flux corresponds to luminosity-distance reference used in
        # ``make_lcs_hdf5.py`` (scaling ``(1/D_Mpc)**2``, i.e. D_ref ≈ 1 Mpc).
        "reference_distance_mpc": 1.0,
    }


def load_bulla2019_txt(path: str) -> dict[str, Any]:
    """Load a Dhawan et al. (2019) Bulla-format ``.txt`` spectral cube.

    Header (first three lines): ``Nobs``, ``Nwave``, then
    ``Ntime t_i t_f`` (days). Data start on line 4: for each viewing angle
    a block of ``Nwave`` rows; column 0 is rest wavelength (Å), columns
    ``1 … Ntime`` are flux F_λ (erg s⁻¹ cm⁻² Å⁻¹) at **10 pc** (``10^{-5}`` Mpc);

    On load, ``reference_distance_mpc`` is **1e-5** so ``scale_bulla_intensity_to_observer``
    matches ``2019_bulla/make_lcs.py``.

    Bin centers in time are ``t_i + Dt * (it + 0.5)`` with
    ``Dt = (t_f - t_i) / Ntime`` (README). Viewing-angle rows follow

    cos θ linearly from 0 to 1 (`cos_theta_to_theta_deg`).
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw0, raw1, raw2 = f.readline(), f.readline(), f.readline()
    try:
        n_obs = int(float(raw0.split()[0]))
        n_wave = int(float(raw1.split()[0]))
        p2 = raw2.split()
        n_time = int(float(p2[0]))
        t_i = float(p2[1])
        t_f = float(p2[2])
    except (IndexError, ValueError) as ex:
        raise ValueError("Invalid header in %s" % path) from ex

    a = np.genfromtxt(path, skip_header=3)
    rows_needed = int(n_obs) * int(n_wave)
    if a.ndim != 2:
        raise ValueError("Expected 2D numeric table in %s after header" % path)
    if int(a.shape[0]) < rows_needed:
        raise ValueError(
            "Need at least %d data rows in %s, got %d"
            % (rows_needed, path, int(a.shape[0]))
        )
    n_col = int(a.shape[1])
    if n_col < int(n_time) + 1:
        raise ValueError(
            "%s: need ≥ 1 + %d columns for flux, got %d"
            % (path, int(n_time), n_col)
        )

    wave_rest = np.asarray(a[0:n_wave, 0], dtype=float)
    I_stokes = np.empty((n_obs, n_time, n_wave), dtype=float)
    for obs in range(n_obs):
        sl = slice(obs * n_wave, (obs + 1) * n_wave)
        block = np.asarray(a[sl, :], dtype=float)
        wave_b = np.asarray(block[:, 0], dtype=float)
        if obs > 0 and not np.allclose(wave_b, wave_rest, rtol=0.0, atol=1.0):
            warnings.warn(
                "Wavelength column differs in observer block %d of %s "
                "(continuing with first block grid)."
                % (obs, os.path.basename(path)),
                UserWarning,
                stacklevel=2,
            )
        flux_bt = block[:, 1 : 1 + n_time].astype(float)
        I_stokes[obs, :, :] = flux_bt.T

    dt = (t_f - t_i) / float(n_time)
    time_days = t_i + dt * (np.arange(n_time, dtype=float) + 0.5)
    return {
        "I_stokes": I_stokes,
        "time_days": time_days,
        "wave_rest": wave_rest,
        "source": "bulla2019_txt",
        "path": path,
        "reference_distance_mpc": 1e-5,
    }


def observer_wavelength_bounds(
    blob: dict[str, Any],
    z: float,
    *,
    wave_is_rest: bool = True,
) -> tuple[float, float]:
    """Min/max observer-frame wavelength (Å) over the model spectral grid."""
    w_rest = blob["wave_rest"]
    wl_obs = (
        wave_rest_to_observer(w_rest, z)
        if wave_is_rest
        else np.asarray(w_rest, dtype=float)
    )
    return float(np.min(wl_obs)), float(np.max(wl_obs))


def prepare_epochs_overlap(
    blob: dict[str, Any],
    epochs: list[dict[str, Any]],
    *,
    z: float,
    wave_is_rest: bool = True,
    min_pix_per_epoch: int = 3,
) -> list[dict[str, Any]]:
    """
    Keep only epochs whose ``phase_model_days`` lies inside the model time grid
    (inclusive endpoints, matching :func:`interpolate_model_along_time_regular`).

    For each kept epoch, subset ``(wl_data, F_data, fe_data)`` to wavelengths
    overlapping the observer-frame model grid. Epochs with fewer than
    ``min_pix_per_epoch`` finite overlap pixels are dropped. Other keys on each
    epoch dict are preserved.
    """
    tmod = np.asarray(blob["time_days"], dtype=float)
    t_lo = float(np.min(tmod))
    t_hi = float(np.max(tmod))
    wl_lo_mod, wl_hi_mod = observer_wavelength_bounds(blob, z, wave_is_rest=wave_is_rest)
    need = int(min_pix_per_epoch)
    if need < 1:
        raise ValueError("min_pix_per_epoch must be >= 1")

    out: list[dict[str, Any]] = []
    for ep in epochs:
        ph = float(ep["phase_model_days"])
        if ph < t_lo or ph > t_hi:
            continue
        wl_d = np.asarray(ep["wl_data"], dtype=float)
        F_d = np.asarray(ep["F_data"], dtype=float)
        fe_d = np.asarray(ep["fe_data"], dtype=float)
        mask = (
            np.isfinite(wl_d)
            & np.isfinite(F_d)
            & np.isfinite(fe_d)
            & (wl_d >= wl_lo_mod)
            & (wl_d <= wl_hi_mod)
        )
        if int(np.count_nonzero(mask)) < need:
            continue
        trimmed = dict(ep)
        trimmed["wl_data"] = wl_d[mask]
        trimmed["F_data"] = F_d[mask]
        trimmed["fe_data"] = fe_d[mask]
        out.append(trimmed)
    return out


def cos_theta_to_theta_deg(n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Match ``make_lcs_hdf5.py``: cos θ linearly from 0 to 1, θ in degrees."""
    cos_t = np.linspace(0.0, 1.0, int(n_obs))
    theta_deg = np.arccos(cos_t) * 180.0 / np.pi
    return cos_t, theta_deg


def obs_indices_for_theta_deg_range(
    n_obs: int,
    theta_lo_deg: float = 13.0,
    theta_hi_deg: float = 45.0,
    *,
    inclusive: bool = True,
) -> list[int]:
    """
    Observer indices whose polar angle θ (degrees on Bulla's ``cos θ`` linear grid)
    falls in ``[theta_lo_deg, theta_hi_deg]``.

    Passing the returned list as ``obs_indices`` to ``pooled_chi2_all_epochs_angles``
    restricts the sum over angles to that band (hard support / “prior” on θ).

    **Weighting caveat:** Equal contribution per index is roughly uniform on discrete
    ``cos θ``, not strictly uniform π(θ) on θ unless you introduce Jacobian or bin-width
    weights on top of χ² pooling.
    """
    n = int(n_obs)
    if n <= 0:
        return []
    lo = float(theta_lo_deg)
    hi = float(theta_hi_deg)
    if lo > hi:
        lo, hi = hi, lo
    _cos_t, theta = cos_theta_to_theta_deg(n)
    theta = np.asarray(theta, dtype=float)
    if inclusive:
        mask = (theta >= lo) & (theta <= hi)
    else:
        mask = (theta > lo) & (theta < hi)
    return sorted(int(i) for i in np.flatnonzero(mask))


def scale_bulla_intensity_to_observer(
    I_native: np.ndarray,
    z: float,
    d_lum_mpc: float,
    *,
    reference_distance_mpc: float = 1.0,
) -> np.ndarray:
    """
    Map native spectral flux density F_lambda to the cosmological observer.

    ``reference_distance_mpc`` is **D_ref** (Mpc) such that natives are defined at
    distance **D_ref**; observer flux scales as native **times** ``(D_ref / D_L)² /
    (1+z)``.

    HDF5 loaders set **D_ref = 1** (``make_lcs_hdf5.py``).
    2019 ASCII loaders set **D_ref = 1e-5** (**10 pc**; ``make_lcs.py``).
    """
    pref = float(reference_distance_mpc)
    dl = float(d_lum_mpc)
    if pref <= 0 or dl <= 0:
        raise ValueError(
            "reference_distance_mpc and d_lum_mpc must be positive (got pref=%s, dl=%s)"
            % (pref, dl)
        )
    arr = np.asarray(I_native, dtype=float)
    return arr * (pref / dl) ** 2 / (1.0 + float(z))


def wave_rest_to_observer(wave_rest: np.ndarray, z: float) -> np.ndarray:
    return np.asarray(wave_rest, dtype=float) * (1.0 + float(z))


def nearest_time_index(t_model_days: np.ndarray, t_target_days: float) -> int:
    t_model_days = np.asarray(t_model_days, dtype=float)
    return int(np.nanargmin(np.abs(t_model_days - float(t_target_days))))


def interpolate_model_along_time_regular(
    F_cube: np.ndarray,
    t_model: np.ndarray,
    t_target: float,
) -> np.ndarray:
    """
    Linear interpolation in time; returns NaN if t_target is outside [min(t), max(t)].
    """
    t_model = np.asarray(t_model, dtype=float)
    F_cube = np.asarray(F_cube, dtype=float)
    lo, hi = float(np.min(t_model)), float(np.max(t_model))
    if t_target < lo or t_target > hi:
        return np.full(F_cube.shape[1], np.nan)
    return np.array(
        [
            float(np.interp(t_target, t_model, F_cube[:, j]))
            for j in range(F_cube.shape[1])
        ]
    )


def interp_model_flux_to_wavelengths(
    wl_model: np.ndarray,
    F_on_model_grid: np.ndarray,
    wl_target: np.ndarray,
) -> np.ndarray:
    """1D interpolation of model flux onto ``wl_target`` (Å)."""
    wl_model = np.asarray(wl_model, dtype=float)
    F_on_model_grid = np.asarray(F_on_model_grid, dtype=float)
    wl_target = np.asarray(wl_target, dtype=float)
    order = np.argsort(wl_model)
    wl_s = wl_model[order]
    F_s = F_on_model_grid[order]
    return np.interp(wl_target, wl_s, F_s, left=np.nan, right=np.nan)


def chi2_flux(
    F_data: np.ndarray,
    F_model: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, int]:
    """χ² = Σ ((D-M)/σ)² over pixels with finite values and σ > 0."""
    F_data = np.asarray(F_data, dtype=float).ravel()
    F_model = np.asarray(F_model, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    m = (
        np.isfinite(F_data)
        & np.isfinite(F_model)
        & np.isfinite(sigma)
        & (sigma > 0)
    )
    if not np.any(m):
        return float("nan"), 0
    r = (F_data[m] - F_model[m]) / sigma[m]
    return float(np.sum(r**2)), int(np.count_nonzero(m))


def model_flux_for_epoch_and_angle(
    blob: dict[str, Any],
    *,
    obs_index: int,
    phase_days_target: float,
    z: float,
    d_lum_mpc: float,
    wave_is_rest: bool = True,
    time_interp: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return (wl_obs_angstrom, F_lambda_observer, itime_nearest_or_used).

    ``F_lambda_observer`` scales natives with ``scale_bulla_intensity_to_observer`` using
    ``blob['reference_distance_mpc']`` (**1 Mpc** for HDF5 — ``make_lcs_hdf5``;
    **10⁻⁵ Mpc** for 2019 ASCII — ``make_lcs``).
    """
    I = blob["I_stokes"]
    if obs_index < 0 or obs_index >= I.shape[0]:
        raise IndexError("obs_index %d out of range [0, %d)" % (obs_index, I.shape[0]))
    t = blob["time_days"]
    w_rest = blob["wave_rest"]
    wl_obs = wave_rest_to_observer(w_rest, z) if wave_is_rest else np.asarray(w_rest, float)
    cube = I[obs_index]
    if time_interp:
        F_native = interpolate_model_along_time_regular(cube, t, phase_days_target)
    else:
        it = nearest_time_index(t, phase_days_target)
        F_native = cube[it].astype(float)
    ref_mpc = float(blob.get("reference_distance_mpc", 1.0))
    F_obs = scale_bulla_intensity_to_observer(
        F_native, z, d_lum_mpc, reference_distance_mpc=ref_mpc
    )
    it_used = (
        nearest_time_index(t, phase_days_target) if not time_interp else -1
    )
    return wl_obs, F_obs, int(it_used)


def chi2_vs_obs_indices(
    wl_data: np.ndarray,
    F_data: np.ndarray,
    fe_data: np.ndarray,
    blob: dict[str, Any],
    *,
    phase_days_target: float,
    z: float,
    d_lum_mpc: float,
    obs_indices: list[int] | range,
    wave_is_rest: bool = True,
    time_interp: bool = True,
) -> list[dict[str, Any]]:
    """Compute χ² for each observer index at one data epoch."""
    rows = []
    w_rest = blob["wave_rest"]
    wl_obs_grid = (
        wave_rest_to_observer(w_rest, z) if wave_is_rest else np.asarray(w_rest, float)
    )
    for oi in obs_indices:
        wl_m, F_m, _ = model_flux_for_epoch_and_angle(
            blob,
            obs_index=oi,
            phase_days_target=phase_days_target,
            z=z,
            d_lum_mpc=d_lum_mpc,
            wave_is_rest=wave_is_rest,
            time_interp=time_interp,
        )
        F_on_data = interp_model_flux_to_wavelengths(wl_m, F_m, wl_data)
        c2, n = chi2_flux(F_data, F_on_data, fe_data)
        cos_t, th_deg = cos_theta_to_theta_deg(blob["I_stokes"].shape[0])
        rows.append(
            {
                "obs_index": oi,
                "cos_theta": float(cos_t[oi]),
                "theta_deg": float(th_deg[oi]),
                "chi2": c2,
                "n_pix": n,
            }
        )
    return rows


def pooled_chi2_all_epochs_angles(
    blob: dict[str, Any],
    epochs: list[dict[str, Any]],
    *,
    z: float,
    d_lum_mpc: float,
    wave_is_rest: bool = True,
    time_interp: bool = True,
    uncertainty_scale: float = 1.0,
    obs_indices: Sequence[int] | None = None,
    restrict_to_overlap: bool = False,
    min_pix_per_epoch: int = 3,
) -> tuple[float, int]:
    """
    Sum χ² and effective pixel counts over every epoch in ``epochs``.

    If ``obs_indices`` is ``None``, sums over all observer indices ``0 … N_obs-1``.
    Otherwise sums only over the given indices (entries outside ``[0, N_obs)`` are
    skipped). Each epoch dict needs ``wl_data``, ``F_data``, ``fe_data``,
    ``phase_model_days``. Data σ are multiplied by ``uncertainty_scale`` before
    ``chi2_flux``.

    With ``restrict_to_overlap=True``, only epochs/phases inside the model time
    extent are used and each epoch is wavelength-trimmed to the intersection of
    data pixels with the observer-frame model spectral range (via
    :func:`prepare_epochs_overlap`).
    """
    if uncertainty_scale <= 0:
        raise ValueError("uncertainty_scale must be > 0")
    sc = float(uncertainty_scale)
    n_obs = blob["I_stokes"].shape[0]
    if obs_indices is None:
        oi_loop: list[int] = list(range(n_obs))
    else:
        oi_loop = []
        for raw in obs_indices:
            ii = int(raw)
            if 0 <= ii < n_obs:
                oi_loop.append(ii)
        if not oi_loop:
            return 0.0, 0
    eps = epochs
    if restrict_to_overlap:
        eps = prepare_epochs_overlap(
            blob,
            epochs,
            z=z,
            wave_is_rest=wave_is_rest,
            min_pix_per_epoch=min_pix_per_epoch,
        )
    tot_c2, tot_n = 0.0, 0
    for ep in eps:
        wl_d = np.asarray(ep["wl_data"], dtype=float)
        F_d = np.asarray(ep["F_data"], dtype=float)
        fe_d = np.asarray(ep["fe_data"], dtype=float) * sc
        ph = float(ep["phase_model_days"])
        for oi in oi_loop:
            wl_m, F_m, _ = model_flux_for_epoch_and_angle(
                blob,
                obs_index=oi,
                phase_days_target=ph,
                z=z,
                d_lum_mpc=d_lum_mpc,
                wave_is_rest=wave_is_rest,
                time_interp=time_interp,
            )
            F_on = interp_model_flux_to_wavelengths(wl_m, F_m, wl_d)
            c2, n = chi2_flux(F_d, F_on, fe_d)
            if np.isfinite(c2):
                tot_c2 += float(c2)
                tot_n += int(n)
    return tot_c2, tot_n


def nearest_obs_index_for_cos_theta(
    cos_theta_grid: np.ndarray, cos_target: float
) -> int:
    """
    Index of the nearest point on ``cos_theta_grid`` to ``cos_target``.

    Values outside ``[0, 1]`` are **clamped** to that interval (with ``UserWarning``)
    before choosing the nearest grid point.
    """
    cos_theta_grid = np.asarray(cos_theta_grid, dtype=float).ravel()
    ct = float(cos_target)
    if ct < 0.0 or ct > 1.0:
        warnings.warn(
            "cos_target=%s is outside [0, 1]; clamping for nearest-grid lookup."
            % ct,
            UserWarning,
            stacklevel=2,
        )
        ct = float(np.clip(ct, 0.0, 1.0))
    return int(np.argmin(np.abs(cos_theta_grid - ct)))


def enumerate_final_spectrum_phases(
    data_dir: str,
    coco_path: str,
    snname: str,
    datalc_path: str | None,
    t0_mjd: float,
    final_suffixes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """
    One row per ``*.txt`` in ``data_dir``, with spectrum MJD and phase ``MJD - t0``.

    If ``final_suffixes`` is set, only files whose names end with one of those strings
    are included.
    """
    data_dir = os.path.abspath(data_dir)
    rows: list[dict[str, Any]] = []
    for fname in sorted(f for f in os.listdir(data_dir) if f.endswith(".txt")):
        if final_suffixes is not None and not any(
            fname.endswith(s) for s in final_suffixes
        ):
            continue
        stem = parse_final_stem(fname)
        mjd = float(
            stem_to_spec_mjd(
                stem, coco_path, snname, datalc_path=datalc_path
            )
        )
        phase_days = float(mjd - float(t0_mjd))
        rows.append({"fname": fname, "mjd": mjd, "phase_days": phase_days})
    rows.sort(key=lambda r: r["phase_days"])
    return rows


def pick_final_by_nearest_phase(
    rows: list[dict[str, Any]], target_phase_days: float
) -> dict[str, Any]:
    """Pick the row with smallest ``|phase_days - target_phase_days|``."""
    if not rows:
        raise ValueError("pick_final_by_nearest_phase: no rows")
    target = float(target_phase_days)

    def _abs_delta(r: dict[str, Any]) -> float:
        return abs(float(r["phase_days"]) - target)

    best = min(rows, key=_abs_delta)
    delta = float(best["phase_days"]) - target
    return {
        "fname": best["fname"],
        "mjd": float(best["mjd"]),
        "phase_days": float(best["phase_days"]),
        "target_phase_days": target,
        "delta_phase_days": delta,
        "abs_delta_phase_days": abs(delta),
    }
