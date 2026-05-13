"""Shared KN log-pipeline paths and naming (extend vs extrapolate; spliced vs full_gp).

``COCO_PATH`` defaults to the ``PyCoCo_templates`` repo root Detected relative to this file,
or override with environment variable ``COCO_PATH``.
"""

from __future__ import annotations

import math
import os

# ---------------------------------------------------------------------------
# Repo / SN defaults
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_COCO = os.path.abspath(os.path.join(_DIR, ".."))

COCO_PATH: str = os.environ.get("COCO_PATH", _DEFAULT_COCO).rstrip("/") + os.sep

# Default supernova name (override per notebook).
SNNAME_DEFAULT: str = "AT2017gfo"

# Explosion / merger time (MJD) for phase axes (e.g. 7.5_alternate ``prepare_trapz_gp_comparison``
# when ``mjd0`` is None). Add entries for other events; notebooks may also pass ``mjd0=`` explicitly.
SN_EXPLOSION_MJD: dict[str, float] = {
    "AT2017gfo": 57982.52851852,
}

# Optional global wavelength limits for GP prediction / lookup (Angstrom). None = auto from data.
PIPELINE_WL_MIN_A: float | None = None
PIPELINE_WL_MAX_A: float | None = None

# Optional George diagonal jitter (linear variance); 0 = off. Passed to ``GP(white_noise=log(...))`` in
# ``GP2dim_utils_newlog.run_2DGP_GRID`` (George >=0.4 has no ``kernels.WhiteKernel``).
# input the variance here (sigma squared not sigma)
GP_WHITE_NOISE: float = 0.1**2
# When ``USE_TWO_D_GP_LINEAR_FLUX`` is True (notebook 6 imports ``GP2dim_utils_newlog_linear_flux``), use this
# jitter on **scaled linear-flux** targets if set; ``None`` falls back to ``GP_WHITE_NOISE``.
GP_WHITE_NOISE_LINEAR: float = 1e-20

# Notebook 6: import ``GP2dim_utils_newlog_linear_flux`` instead of ``GP2dim_utils_newlog`` (same log10 axes;
# GP targets are affine-scaled linear flux instead of ln-flux). Default False: ln-flux GP stays positive via ``exp``.
USE_TWO_D_GP_LINEAR_FLUX: bool = False

# Notebook 6 (ln-flux path only): import ``GP2dim_utils_newlog_zscore`` so GP *coordinates* are training z-scores
# on log10(λ) and log10(phase days). Ignored when ``USE_TWO_D_GP_LINEAR_FLUX`` is True (linear module wins).
USE_TWO_D_GP_ZSCORE_COORDS: bool = True

# After ``run_2DGP_GRID`` / ``run_2DGP_GRID_linear``: write ``gp_minimal_bundle.npz`` + meta JSON under
# ``save_plot_path/GP_EXPORT_SUBDIR/`` (training ``X,y,yerr``, ``y_compute``, prediction ``X_fill``, kernel
# scales, white noise, prior arrays, ``grid_norm_info``). Or set ``spec_class.gp_export_minimal = True`` to
# force export even when this is False.
GP_EXPORT_MINIMAL: bool = True
GP_EXPORT_SUBDIR: str = "gp_minimal_export"

# ``transform2LOG_reshape``: if True (default), ``sigma_ln(F) = sqrt(ln(1 + (sigma_F/F)^2))`` with
# ``sigma_F = |F| ln(10) sigma_log10``. If False, legacy ``sigma_ln = sigma_log10 * ln(10)`` (log-base chain rule).
GP_LN_FLUX_ERR_FROM_RELATIVE: bool = True

# Minimum training sigma from spread of scaled ``y`` (legacy used 1e-4). ``None`` or ``0`` = off—use
# propagated ``yerr`` only (then every point must have strictly positive error or set ``GP_YERR_ABS_FLOOR``).
GP_YERR_FLOOR_FRAC: float | None = None
# Absolute floor on scaled training ``yerr`` after optional spread floor (0 = off).
GP_YERR_ABS_FLOOR: float = 0.0

# Notebook 6 / run_2DGP_GRID: optionally merge extra prediction phases, evenly spaced in log10(phase
# days) between the current min/max column (``np.logspace`` on linear days). Default off = legacy behavior.
GP_PREDICT_DENSE_LOG_PHASE: bool = False
GP_PREDICT_DENSE_LOG_PHASE_N: int = 64

# Notebook 6 / ``run_2DGP_GRID``: optional pseudo training points at fixed log10(phase days) with
# capped log10 flux (faint), one per unique training wavelength node—mirrors LC anchor idea in 2D.
# Default off. Does not write files; mangling still uses raw photometry only.
GP_2D_ANCHOR_T0: bool = False
GP_2D_T0_ANCHOR_LOG_PHASE: float = -8.0
GP_2D_T0_ANCHOR_LOG10_FLUX_CAP: float = -50.0
GP_2D_T0_ANCHOR_LOG10_FLUX_ERR: float = 2.0

# Notebook 4: add synthetic (pseudo) log-phase / log-flux training point before ``gp.compute`` so each
# band's 1D GP can pull toward faint flux at explosion. Marked SUDO in ``clipped_extended_data``.
ANCHOR_T0_IN_LC_GP: bool = True

# After LC fit: append a row to ``fitted_phot_logspace_*.dat`` (does not refit the 1D GP; 2-D-oriented).
APPEND_T0_ROW_TO_LOGSPACE_AFTER_FIT: bool = False

# Subfolder names under ``Outputs/<SN>/twodim/<mode>/``
SUBDIR_SPLICED: str = "spliced"
SUBDIR_FULL_GP: str = "full_gp"
SUBDIR_DIAGNOSTICS: str = "diagnostics"

# ---------------------------------------------------------------------------
# Collaborator ``gp_rjf`` parallel outputs (notebook 6/7 *_rjf.ipynb)
# Layout: ``Outputs/<SN>/twodim_rjf/<extend|extrapolate>/spliced|full_gp``
# plus ``RE_mangled_spectra_2dim`` under that product dir for NB7 RJF re-mangled writes.
# Keeps classic ``twodim/`` products untouched on side-by-side runs.
# ---------------------------------------------------------------------------
TWODIM_RJF_SUBDIR_ROOT: str = "twodim_rjf"
# Notebook 7 RJF: re-mangled spectra/plots live under the product folder (non-legacy) or ``*_rjf`` at SN root (legacy).
RJF_REMANGLED_SUBDIR: str = "RE_mangled_spectra_2dim"
# Prior LinearNDInterpolator cache for RJF GP (recommended under diagnostics per run).
GP_RJF_PRIOR_CACHE_SUBDIR: str = "diagnostics/rjf_prior_cache"
# Extra contour PDFs comparing raw GP μ vs mono+blue post-processed μ (same grid as NB6 plots).
GP_RJF_PLOT_RAW_AND_PROCESSED: bool = False

# Keyword overrides merged into collaborator ``gp_collab_rjf.run_inference.DEFAULT_KWARGS`` at fit time.
# Default: match ``gp_rjf/WRITEUP.md`` production run ``matern52_addw_addt_linear_opt_v5`` (§2 table:
# additive Matern 5/2 on λ and phase, linear prior mean, jitter floors, L-BFGS subsample 2500, mono+blue).
# Warm-start metrics / weights reproduce their *post-fit* hypers on their bundle; re-optimization still runs
# per object so your hypers will adapt to local data.
_gp_rjf_v5_kw: dict[str, object] = dict(
    kernel_time="matern52",
    kernel_wls="matern52",
    additive_time=True,
    additive_wls=True,
    mean="linear",
    phot_spec_threshold=50,
    # Warm-start SHORT/LONG squared George metrics (= table ``metric_*`` in WRITEUP; ℓ ≈ √metric).
    lw_short=0.0258,
    lw2=5.90,
    w_short_w=0.003,
    lt_short=0.126,
    lt2=7.23,
    w_short_t=0.027,
    lw=None,
    lt=None,
    log_amp=float(math.log(0.0135)),
    sigma_phot=0.012,
    sigma_spec=0.005,
    enforce_mono_early=True,
    enforce_blue_early=True,
    early_time_cutoff=-4.0,
    mono_floor_fraction=0.5,
    mono_min_slope=0.005,
    mono_smoothing_scale=0.3,
    optimize=True,
    max_iter=60,
    optimize_subsample=2500,
    seed=0,
    predict_chunk=10000,
    predict_train=True,
)
GP_RJF_KWARGS: dict = dict(_gp_rjf_v5_kw)

# Short mode labels for directory names (not the GP2DIM_Class.mode strings).
MODE_EXTEND_SHORT: str = "extend"
MODE_EXTRAPOLATE_SHORT: str = "extrapolate"

# Legacy flat folder (pre–dual-product layout). Set ``USE_LEGACY_TWODIM_LAYOUT = True`` in your
# notebook to write/read extended spectra only from ``TwoDextended_spectra``.
USE_LEGACY_TWODIM_LAYOUT: bool = False
LEGACY_TWODIM_EXTENDED_DIRNAME: str = "TwoDextended_spectra"
# Legacy flat folder for RJF runs (parallel notebook 6 avoids deleting classic ``TwoDextended_spectra``).
LEGACY_TWODIM_RJF_DIRNAME: str = "TwoDextended_spectra_rjf"


def outputs_root(coco_path: str | None = None) -> str:
    base = coco_path or COCO_PATH
    return os.path.join(os.path.normpath(base), "Outputs")


def twodim_mode_to_short(gp_mode: str) -> str:
    if gp_mode == "extend_spectra":
        return MODE_EXTEND_SHORT
    if gp_mode == "extrapolate_spectra":
        return MODE_EXTRAPOLATE_SHORT
    raise ValueError("Unknown GP mode %r (expect extend_spectra or extrapolate_spectra)" % gp_mode)


def twodim_extended_base(output_dir: str, snname: str, gp_mode: str) -> str:
    """Directory that holds ``spliced/``, ``full_gp/``, ``diagnostics/`` (``save_plot_path``)."""
    if USE_LEGACY_TWODIM_LAYOUT:
        return os.path.join(output_dir.rstrip(os.sep), snname, LEGACY_TWODIM_EXTENDED_DIRNAME)
    short = twodim_mode_to_short(gp_mode)
    return os.path.join(output_dir.rstrip(os.sep), snname, "twodim", short)


def twodim_product_dir(output_dir: str, snname: str, gp_mode: str, product: str) -> str:
    """Extended spectra for one product: ``spliced`` or ``full_gp``."""
    if product not in (SUBDIR_SPLICED, SUBDIR_FULL_GP):
        raise ValueError("product must be %r or %r" % (SUBDIR_SPLICED, SUBDIR_FULL_GP))
    if USE_LEGACY_TWODIM_LAYOUT:
        return twodim_extended_base(output_dir, snname, gp_mode)
    return os.path.join(twodim_extended_base(output_dir, snname, gp_mode), product)


def twodim_diagnostics_dir(output_dir: str, snname: str, gp_mode: str) -> str:
    if USE_LEGACY_TWODIM_LAYOUT:
        return twodim_extended_base(output_dir, snname, gp_mode)
    return os.path.join(twodim_extended_base(output_dir, snname, gp_mode), SUBDIR_DIAGNOSTICS)


def twodim_rjf_extended_base(output_dir: str, snname: str, gp_mode: str) -> str:
    """Same role as ``twodim_extended_base`` but rooted under ``twodim_rjf/``."""
    if USE_LEGACY_TWODIM_LAYOUT:
        return os.path.join(
            output_dir.rstrip(os.sep), snname, LEGACY_TWODIM_RJF_DIRNAME
        )
    short = twodim_mode_to_short(gp_mode)
    return os.path.join(output_dir.rstrip(os.sep), snname, TWODIM_RJF_SUBDIR_ROOT, short)


def twodim_rjf_product_dir(output_dir: str, snname: str, gp_mode: str, product: str) -> str:
    if product not in (SUBDIR_SPLICED, SUBDIR_FULL_GP):
        raise ValueError("product must be %r or %r" % (SUBDIR_SPLICED, SUBDIR_FULL_GP))
    if USE_LEGACY_TWODIM_LAYOUT:
        return twodim_rjf_extended_base(output_dir, snname, gp_mode)
    return os.path.join(twodim_rjf_extended_base(output_dir, snname, gp_mode), product)


def twodim_rjf_final_branch(gp_mode_short: str, product: str) -> str:
    """Path segment ``twodim_rjf/<short>/<product>`` for FINAL_spectra_2dim (NB7 Rimangle)."""
    return os.path.join(
        TWODIM_RJF_SUBDIR_ROOT,
        gp_mode_short.replace("\\", "/").strip("/"),
        product.replace("\\", "/").strip("/"),
    )


def final_spectra_twodim_branch(
    gp_mode_short: str, product: str, *, use_rjf: bool
) -> str:
    """Branch under ``FINAL_spectra_2dim`` for 7.5 comparison / spectra / alternate notebooks.

    If ``use_rjf`` is False: ``<mode_short>/<product>`` (classic ``twodim`` pipeline, same as
    ``comparison_check_log_utils.twodim_final_branch``). If True: ``twodim_rjf/<mode>/<product>``
    (NB6/7 ``*_rjf`` outputs). Returns forward slashes for portable ``resolve_final_directory`` splits.

    ``USE_LEGACY_TWODIM_LAYOUT`` is not handled here; use ``twodim_branch=None`` or a manual branch.
    """
    ms = gp_mode_short.replace("\\", "/").strip("/")
    pr = product.replace("\\", "/").strip("/")
    if not use_rjf:
        return "%s/%s" % (ms, pr)
    return twodim_rjf_final_branch(ms, pr).replace("\\", "/")


def twodim_rjf_remangled_dir(output_dir: str, snname: str, gp_mode: str, product: str) -> str:
    """Directory for notebook 7 ``*_rjf`` re-mangled spectra and diagnostic PNGs.

    Non-legacy: ``twodim_rjf/<mode>/<product>/RE_mangled_spectra_2dim`` under ``Outputs/<SN>/``.

    Legacy layout: ``Outputs/<SN>/RE_mangled_spectra_2dim_rjf`` (parallel to classic ``RE_mangled_spectra_2dim``).
    """
    if USE_LEGACY_TWODIM_LAYOUT:
        return os.path.join(
            output_dir.rstrip(os.sep), snname, "RE_mangled_spectra_2dim_rjf"
        )
    return os.path.join(
        twodim_rjf_product_dir(output_dir, snname, gp_mode, product), RJF_REMANGLED_SUBDIR
    )


def raw_photometry_path(coco_path: str | None, snname: str) -> str:
    base = coco_path or COCO_PATH
    return os.path.join(
        os.path.normpath(base),
        "Inputs",
        "Photometry",
        "1_LCs_flux_raw",
        "%s.dat" % snname,
    )


def band_mjd_ranges_json_path(output_dir: str, snname: str) -> str:
    return os.path.join(
        output_dir.rstrip(os.sep), snname, "%s_band_mjd_ranges.json" % snname
    )


def final_spectra_branch_dir(
    coco_path: str | None,
    snname: str,
    *,
    extension_type: str = "2dim",
    twodim_branch: str | None = None,
) -> str:
    """Top FINAL directory, optionally including ``extend/spliced``-style branch.

    ``twodim_branch``: e.g. ``"extend/spliced"`` or ``None`` for legacy flat
    ``Outputs/<SN>/FINAL_spectra_2dim``.
    """
    base = coco_path or COCO_PATH
    out = os.path.join(
        os.path.normpath(base), "Outputs", snname, "FINAL_spectra_%s" % extension_type
    )
    if twodim_branch:
        out = os.path.join(out, *twodim_branch.replace("\\", "/").split("/"))
    return out
