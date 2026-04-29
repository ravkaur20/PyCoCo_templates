"""Shared KN log-pipeline paths and naming (extend vs extrapolate; spliced vs full_gp).

``COCO_PATH`` defaults to the ``PyCoCo_templates`` repo root Detected relative to this file,
or override with environment variable ``COCO_PATH``.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Repo / SN defaults
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_COCO = os.path.abspath(os.path.join(_DIR, ".."))

COCO_PATH: str = os.environ.get("COCO_PATH", _DEFAULT_COCO).rstrip("/") + os.sep

# Default supernova name (override per notebook).
SNNAME_DEFAULT: str = "AT2017gfo"

# Optional global wavelength limits for GP prediction / lookup (Angstrom). None = auto from data.
PIPELINE_WL_MIN_A: float | None = None
PIPELINE_WL_MAX_A: float | None = None

# Optional George diagonal jitter (linear variance); 0 = off. Passed to ``GP(white_noise=log(...))`` in
# ``GP2dim_utils_newlog.run_2DGP_GRID`` (George >=0.4 has no ``kernels.WhiteKernel``).
GP_WHITE_NOISE: float = 0.0

# Notebook 6 / run_2DGP_GRID: optionally merge extra prediction phases, evenly spaced in log10(phase
# days) between the current min/max column (``np.logspace`` on linear days). Default off = legacy behavior.
GP_PREDICT_DENSE_LOG_PHASE: bool = False
GP_PREDICT_DENSE_LOG_PHASE_N: int = 64

# Notebook 4: add synthetic (pseudo) log-phase / log-flux training point before ``gp.compute`` so each
# band's 1D GP can pull toward faint flux at explosion. Marked SUDO in ``clipped_extended_data``.
ANCHOR_T0_IN_LC_GP: bool = True

# After LC fit: append a row to ``fitted_phot_logspace_*.dat`` (does not refit the 1D GP; 2-D-oriented).
APPEND_T0_ROW_TO_LOGSPACE_AFTER_FIT: bool = False

# Subfolder names under ``Outputs/<SN>/twodim/<mode>/``
SUBDIR_SPLICED: str = "spliced"
SUBDIR_FULL_GP: str = "full_gp"
SUBDIR_DIAGNOSTICS: str = "diagnostics"

# Short mode labels for directory names (not the GP2DIM_Class.mode strings).
MODE_EXTEND_SHORT: str = "extend"
MODE_EXTRAPOLATE_SHORT: str = "extrapolate"

# Legacy flat folder (pre–dual-product layout). Set ``USE_LEGACY_TWODIM_LAYOUT = True`` in your
# notebook to write/read extended spectra only from ``TwoDextended_spectra``.
USE_LEGACY_TWODIM_LAYOUT: bool = False
LEGACY_TWODIM_EXTENDED_DIRNAME: str = "TwoDextended_spectra"


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
