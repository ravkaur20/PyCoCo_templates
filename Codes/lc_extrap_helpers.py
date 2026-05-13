"""
Shared helpers for LC early-time Bazin extrapolation error handling.
Imported by 2_LC_modelRising_KN_fullfit_log.ipynb (sys.path must include Codes/).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EXTRA_ERR_ABS_CAP_DEFAULT",
    "EXTRA_ERR_REL_DATAERR_MAX_DEFAULT",
    "EXTRA_COV_CONDITION_MAX_DEFAULT",
    "clip_extrap_uncertainties",
    "covariance_is_bad",
]


EXTRA_ERR_ABS_CAP_DEFAULT = 0.5  # max σ in normalized-flux units on synthetic early points
EXTRA_ERR_REL_DATAERR_MAX_DEFAULT = None  # e.g. 8.0 → cap at 8 * median(fluxerr_)
EXTRA_COV_CONDITION_MAX_DEFAULT = 5e13


def clip_extrap_uncertainties(
    newpts_err: np.ndarray,
    flux_: np.ndarray,
    fluxerr_: np.ndarray,
    abs_cap: float | None = EXTRA_ERR_ABS_CAP_DEFAULT,
    rel_med_max: float | None = EXTRA_ERR_REL_DATAERR_MAX_DEFAULT,
) -> np.ndarray:
    """Clip positive extrapolation sigmas to avoid huge plot noise from unstable MC."""
    ee = np.asarray(newpts_err, dtype=float).copy()
    ee = np.where(np.isfinite(ee), ee, np.nan)
    # treat non-finite as large then clip
    ee = np.nan_to_num(ee, nan=abs_cap if abs_cap is not None else 1.0, posinf=abs_cap or 1.0, neginf=0.0)
    ee = np.clip(ee, 0.0, np.inf)
    if abs_cap is not None:
        ee = np.minimum(ee, float(abs_cap))
    if rel_med_max is not None and fluxerr_ is not None and len(fluxerr_) > 0:
        med = float(np.nanmedian(np.asarray(fluxerr_, dtype=float)))
        if np.isfinite(med) and med > 0:
            ee = np.minimum(ee, float(rel_med_max) * med)
    return ee


def covariance_is_bad(
    cov: np.ndarray,
    cond_max: float = EXTRA_COV_CONDITION_MAX_DEFAULT,
) -> bool:
    """True if pcov is unusable for multivariate_normal sampling."""
    c = np.asarray(cov, dtype=float)
    if c.size == 0 or np.any(np.isinf(c)) or np.any(np.isnan(c)):
        return True
    d = np.diag(c)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return True
    try:
        cond = float(np.linalg.cond(c))
    except np.linalg.LinAlgError:
        return True
    return (not np.isfinite(cond)) or (cond > cond_max)
