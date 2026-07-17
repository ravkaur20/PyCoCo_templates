"""Log-space spectrum mangling helpers extracted from ``5_Mangle_spectra_KN_log.ipynb``.

Phase 1 provides I/O, mask apply/demangle, and band-flux / mangling-mask fitting for reuse
in notebook 5 and the future iterative GP+mangle driver.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

SPEC_DTYPE = np.dtype([("wls", "<f8"), ("flux", "<f8"), ("fluxerr", "<f8")])
LOG_SPEC_DTYPE = SPEC_DTYPE  # mangled on disk: log10(wls), log10(flux)


def err_to_log10(flux: np.ndarray, err_flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    err_flux = np.asarray(err_flux, dtype=float)
    return err_flux / (flux * np.log(10.0))


def err_from_log10(logflux: np.ndarray, logerr_flux: np.ndarray) -> np.ndarray:
    return np.log(10.0) * np.power(10.0, logflux) * logerr_flux


def calc_lam_avg(wls: np.ndarray, transmission: np.ndarray) -> float:
    from scipy import integrate

    return float(
        integrate.trapezoid(transmission * wls, wls)
        / integrate.trapezoid(transmission, wls)
    )


def calc_lam_eff(wls: np.ndarray, transmission: np.ndarray, flux: np.ndarray) -> float:
    from scipy import integrate

    return float(
        integrate.trapezoid(transmission * flux * wls, wls)
        / integrate.trapezoid(transmission * flux, wls)
    )


def load_linear_spectrum(path: str) -> np.ndarray:
    """Load ``wls, flux, fluxerr`` (linear Å and linear Fλ); drop NaN/nonpositive flux."""
    raw = np.genfromtxt(
        path, dtype=None, encoding="utf-8", names=["wls", "flux", "fluxerr"]
    )
    mask = (
        np.isfinite(raw["wls"])
        & np.isfinite(raw["flux"])
        & np.isfinite(raw["fluxerr"])
        & (raw["flux"] > 0.0)
    )
    return raw[mask]


def load_mangled_spectrum(path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load mangled file (3 or 4 columns). Returns ``(log_spec, mangling_mask_or_None)``."""
    try:
        raw = np.genfromtxt(
            path,
            dtype=None,
            encoding="utf-8",
            names=["wls", "flux", "fluxerr", "mangling_mask"],
        )
        mask = raw["mangling_mask"]
    except (ValueError, OSError):
        raw = np.genfromtxt(
            path, dtype=None, encoding="utf-8", names=["wls", "flux", "fluxerr"]
        )
        mask = None
    log_spec = np.array(
        list(zip(raw["wls"], raw["flux"], raw["fluxerr"])), dtype=LOG_SPEC_DTYPE
    )
    if mask is not None and np.all(np.isfinite(mask)):
        return log_spec, np.asarray(mask, dtype=float)
    return log_spec, None


def save_mangled_spectrum(
    path: str,
    wls_linear: np.ndarray,
    log_flux: np.ndarray,
    log_flux_err: np.ndarray,
    mangling_mask: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("#wls\tflux\tfluxerr\tmangling_mask\n")
        for w, f, fe, mm in zip(wls_linear, log_flux, log_flux_err, mangling_mask):
            fout.write("%E\t%E\t%E\t%E\n" % (w, f, fe, mm))


def apply_mangling_mask_linear(
    wls: np.ndarray,
    flux: np.ndarray,
    fluxerr: np.ndarray,
    mangling_mask_log: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply additive log-space mask; return linear wls, log10 flux, log10 err."""
    log_flux = np.log10(np.clip(flux, 1e-30, None)) + np.asarray(mangling_mask_log, dtype=float)
    log_err = err_to_log10(flux, fluxerr)  # approximate unchanged shape in log
    return np.asarray(wls, dtype=float), log_flux, log_err


def demangle_log_spectrum(
    log_flux: np.ndarray, mangling_mask_log: np.ndarray
) -> np.ndarray:
    return np.asarray(log_flux, dtype=float) - np.asarray(mangling_mask_log, dtype=float)


def _resolve_filter_path(filter_path: str, filter_name: str, snname: str, csp_sne: Sequence[str]) -> str:
    if "swift" in filter_name:
        return os.path.join(filter_path, "Swift", "%s.dat" % filter_name)
    if snname in csp_sne:
        return os.path.join(filter_path, "Site3_CSP", "%s.txt" % filter_name)
    return os.path.join(filter_path, "GeneralFilters", "%s.dat" % filter_name)


def band_flux_trapz(
    spec_wls: np.ndarray,
    spec_flux: np.ndarray,
    spec_fluxerr: np.ndarray,
    filter_name: str,
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
) -> tuple[float, float, float, float, float, float]:
    """Trapezoid synthetic photometry (linear wls / linear flux)."""
    from scipy import integrate, interpolate

    filt_path = _resolve_filter_path(filter_path, filter_name, snname, csp_sne)
    filt = np.genfromtxt(
        filt_path, dtype=None, encoding="utf-8", names=["wls", "flux"]
    )
    min_wls = float(np.min(filt["wls"]))
    max_wls = float(np.max(filt["wls"]))
    lam_avg = calc_lam_avg(filt["wls"], filt["flux"])

    cut = (spec_wls > min_wls) & (spec_wls < max_wls)
    wls_c = np.asarray(spec_wls, dtype=float)[cut]
    flux_c = np.asarray(spec_flux, dtype=float)[cut]
    ferr_c = np.asarray(spec_fluxerr, dtype=float)[cut]

    filt_interp = interpolate.interp1d(filt["wls"], filt["flux"], kind="linear")(wls_c)
    filt_xlam = filt_interp * wls_c
    lam_eff = calc_lam_eff(wls_c, filt_interp, flux_c)
    denom = integrate.trapezoid(filt_xlam, wls_c)
    raw_phot = integrate.trapezoid(filt_xlam * flux_c, wls_c) / denom
    raw_phot_err = (
        integrate.trapezoid((filt_xlam * ferr_c) ** 2, wls_c) ** 0.5 / denom
    )
    return lam_avg, lam_eff, float(raw_phot), float(raw_phot_err), min_wls, max_wls


def compute_mangling_mask(
    raw_spec: np.ndarray,
    phot4mangling_row: Mapping[str, Any],
    avail_filters: Sequence[str],
    filter_mjd_dict: Mapping[str, Mapping[str, float]],
    *,
    filter_path: str,
    snname: str = "",
    csp_sne: Sequence[str] = (),
    photometry_target: str = "gp_fit",
    kernel_divide: int = 800,
    min_gp_metric: float = 0.09,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    """Fit wavelength-dependent log-space mangling mask (NB5 logic).

    Returns ``(mangling_mask, log_mangled_flux, log_mangled_err, meta)`` or ``None``.
    """
    from scipy import optimize

    try:
        import george
        from george.kernels import Matern32Kernel
    except ImportError as exc:
        raise ImportError("george is required for compute_mangling_mask") from exc

    if george is None or Matern32Kernel is None:
        raise ImportError("george is required for compute_mangling_mask")

    spec_mjd = float(phot4mangling_row["spec_mjd"])
    log_diffs: list[float] = []
    log_diffs_err: list[float] = []
    log_wls_eff: list[float] = []
    used_filters: list[str] = []

    wls = np.asarray(raw_spec["wls"], dtype=float)
    flux = np.asarray(raw_spec["flux"], dtype=float)
    fluxerr = np.asarray(raw_spec["fluxerr"], dtype=float)

    for filt in avail_filters:
        if filt not in filter_mjd_dict:
            continue
        band = filter_mjd_dict[filt]
        if band["min_mjd"] == band["max_mjd"]:
            continue
        if not (band["min_mjd"] <= spec_mjd <= band["max_mjd"]):
            continue

        if photometry_target == "gp_fit":
            fitted_logphot = float(phot4mangling_row["%s_fit_log_flux" % filt])
            fitted_logphot_err = float(phot4mangling_row["%s_fit_log_fluxerr" % filt])
            in_mjd = bool(phot4mangling_row["%s_inrange" % filt])
        else:
            raise NotImplementedError(
                "photometry_target=%r not implemented in Phase 1" % photometry_target
            )

        lam_avg, lam_eff, raw_phot, raw_phot_err, min_wls, max_wls = band_flux_trapz(
            wls, flux, fluxerr, filt, filter_path=filter_path, snname=snname, csp_sne=csp_sne
        )
        if raw_phot <= 0.0:
            continue
        condition = (max_wls > np.min(wls)) & (min_wls < np.max(wls)) & (raw_phot > 0.0)
        if not in_mjd or not condition:
            continue

        raw_logphot = np.log10(raw_phot)
        raw_logphot_err = raw_phot_err / (raw_phot * np.log(10.0))
        log_diffs.append(fitted_logphot - raw_logphot)
        log_diffs_err.append(float(np.sqrt(fitted_logphot_err ** 2 + raw_logphot_err ** 2)))
        log_wls_eff.append(float(np.log10(lam_eff)))
        used_filters.append(filt)

    if len(log_diffs) < 1:
        return None

    log_diffs_a = np.asarray(log_diffs, dtype=float)
    log_wls_eff_a = np.asarray(log_wls_eff, dtype=float)
    log_diffs_err_a = np.asarray(log_diffs_err, dtype=float)

    if len(wls) > 10 ** 4:
        full_wls = wls[:: max(1, int(len(wls) / 5000.0))]
    else:
        full_wls = wls
    full_log_wls = np.log10(full_wls)

    norm_wls = float(np.median(full_log_wls))
    full_log_wls_normed = full_log_wls - norm_wls
    log_wls_eff_normed = log_wls_eff_a - norm_wls
    norm_diff = float(np.median(log_diffs_a))
    log_diffs_normed = log_diffs_a - norm_diff

    k = np.var(log_diffs_normed) * Matern32Kernel(5.0)
    gp = george.GP(k)
    gp.compute(np.atleast_2d(log_wls_eff_normed).T, log_diffs_err_a)
    p0 = gp.get_parameter_vector()

    def ll(p: np.ndarray) -> float:
        gp.set_parameter_vector(p)
        scale = float(np.exp(gp.get_parameter_dict()["kernel:k2:metric:log_M_0_0"]))
        if scale < min_gp_metric:
            return np.inf
        return -gp.lnlikelihood(log_diffs_normed, quiet=True)

    def grad_ll(p: np.ndarray) -> np.ndarray:
        gp.set_parameter_vector(p)
        return -gp.grad_lnlikelihood(log_diffs_normed, quiet=True)

    results = optimize.minimize(ll, p0, jac=grad_ll)
    gp.set_parameter_vector(results.x)
    mu_log, cov = gp.predict(log_diffs_normed, full_log_wls - norm_wls)
    std_log = np.sqrt(np.diag(cov))

    if len(wls) > 10 ** 4:
        mu_full_log = np.interp(np.log10(wls), full_log_wls, mu_log)
        std_full_log = np.interp(np.log10(wls), full_log_wls, std_log)
    else:
        mu_full_log = mu_log
        std_full_log = std_log

    mangling_mask = mu_full_log + norm_diff
    raw_log_flux = np.log10(np.clip(flux, 1e-30, None))
    mangled_log = raw_log_flux + mangling_mask
    mangled_log_err = np.sqrt(
        std_full_log ** 2 + (fluxerr / (flux * np.log(10.0))) ** 2
    )
    meta = {
        "used_filters": used_filters,
        "norm_diff": norm_diff,
        "n_filters": len(used_filters),
    }
    return mangling_mask, mangled_log, mangled_log_err, meta
