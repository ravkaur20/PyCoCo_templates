"""Notebook 7 ``*_ryanv2`` helper: propagate extended linear spectrum without GP re-mangle.

Set ``REMANGLE_IDENTITY_ONLY = True`` (globals in the rimangle notebook); re-mangling then uses
identity masks (no ``GP_interpolation_mangle``), which differs from ``REMANGLE_MAX_ITERATIONS=0``
where one GP-based mask is still evaluated.
"""

from __future__ import annotations

import numpy as np

_DTYPE_SPEC = np.dtype(
    [
        ("wls", "<f8"),
        ("flux", "<f8"),
        ("fluxerr", "<f8"),
    ]
)


def identity_no_overlap(sn_obj, flux_floor: float, rem_tol: float, rem_max: int) -> bool:
    """No in-MJD overlapping photometry: copy extended spectrum to iteration 0 and finish.

    ``sn_obj`` is the rimangle notebook instance (methods / attributes mirror ``*_rjf``).
    """
    _lin = sn_obj.ext_spec_linear
    w = np.asarray(_lin["wls"], dtype=float)
    fl = np.maximum(np.asarray(_lin["flux"], dtype=float), float(flux_floor))
    fe = np.asarray(_lin["fluxerr"], dtype=float)
    sn_obj.mangled_spec = {
        0: np.asarray(list(zip(w, fl, fe)), dtype=_DTYPE_SPEC)
    }
    ones = np.ones_like(fl, dtype=float)
    zeros = np.zeros_like(fl, dtype=float)
    sn_obj.mangling_mask = {0: (ones, zeros)}
    sn_obj.magled_photometry_dict = {
        0: {
            "eff_wls": np.zeros(0, dtype=float),
            "fitted_phot": [],
            "used_filters": [],
        }
    }
    sn_obj.final_mangled_spec = sn_obj.mangled_spec[0]
    denom = np.asarray(_lin["flux"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sn_obj.mangling_mask_FINAL = np.nan_to_num(
            np.asarray(sn_obj.final_mangled_spec["flux"], dtype=float) / denom,
            nan=1.0,
            posinf=1.0,
        )
    try:
        _sm = float(sn_obj.phot4mangling["spec_mjd"].values[0])
    except Exception:
        _sm = float("nan")
    sn_obj.remangle_diagnostics = {
        "status": "identity_only_no_overlap",
        "spec_file": sn_obj.spec_file,
        "spec_mjd": _sm,
        "output_stem": None,
        "used_filters": [],
        "eff_wls_A": [],
        "ratios_final": [],
        "ratios_err_final": [],
        "max_abs_ratio_minus_1": float("nan"),
        "passed_1pct": True,
        "ratio_tol": float(rem_tol),
        "last_iter": 0,
        "n_refinement_loops": 0,
        "max_refinement_loops": int(rem_max),
        "hit_refinement_cap": False,
        "still_above_tol_after_remangle": False,
        "REMANGLE_IDENTITY_ONLY": True,
    }
    print(
        "=" * 72
        + "\nREMANGLE_IDENTITY_ONLY: no in-MJD overlapping photometry; "
        + "extended spectrum copied unchanged to mangled/FINAL intermediates.\n"
        + "=" * 72
    )
    return True


def setup_identity_iteration_zero(
    sn_obj,
    *,
    flux_floor: float,
    wls_eff,
    used_filters,
) -> None:
    """Build ``mangled_spec[0]`` and matching phot dict from extended spectrum + identity mask.

    ``self.mangling_mask[0]`` is the authoritative iteration‑0 `(mask, mask_err)` pair on the extended
    wavelength grid—``mangle_iteration_function`` assigns locals from this for gate‑0 diagnostics.
    """
    _lin = sn_obj.ext_spec_linear
    w = np.asarray(_lin["wls"], dtype=float)
    fl = np.maximum(np.asarray(_lin["flux"], dtype=float), float(flux_floor))
    fe = np.asarray(_lin["fluxerr"], dtype=float)
    sn_obj.mangled_spec = {
        0: np.asarray(list(zip(w, fl, fe)), dtype=_DTYPE_SPEC)
    }
    ones = np.ones_like(fl, dtype=float)
    zeros = np.zeros_like(fl, dtype=float)
    sn_obj.mangling_mask = {0: (ones, zeros)}
    ulist = [u for u in used_filters] if used_filters is not None else []
    pl: list[float] = []
    for filt in ulist:
        pl.append(sn_obj.band_flux(filt, use_what=0)[1])
    w_eff = np.asarray(wls_eff, dtype=float).ravel()
    sn_obj.magled_photometry_dict = {
        0: {"eff_wls": w_eff, "fitted_phot": pl, "used_filters": ulist}
    }


def wire_final_mangled_spec_from_extended_linear(sn_obj, flux_floor: float = 0.0) -> None:
    """Set ``final_mangled_spec`` from ``ext_spec_linear`` (no REMANGLE / GP iteration).

    Call after ``load_extended_spec()`` on a rimangle class instance when writing FINAL products
    from notebook-6 extended spectra only (``save_FINAL_spectrum()``).
    """
    _lin = sn_obj.ext_spec_linear
    w = np.asarray(_lin["wls"], dtype=float)
    fl = np.maximum(np.asarray(_lin["flux"], dtype=float), float(flux_floor))
    fe = np.asarray(_lin["fluxerr"], dtype=float)
    sn_obj.final_mangled_spec = np.asarray(list(zip(w, fl, fe)), dtype=_DTYPE_SPEC)
