"""Decode Ryan GP bundle training spectroscopy to observer Å vs linear flux for FINAL overlays."""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np

# Matches ``ryan-updates/py_files/bundle_preprocess.YERR_DISABLED`` — omit from overlays.
_YERR_DISABLED = np.float64(1e30)


def _ryan_py_files_dir(repo_root: str | None = None) -> str:
    codes = os.path.dirname(os.path.abspath(__file__))
    root = repo_root or os.path.dirname(codes)
    return os.path.join(root, "ryan-updates", "py_files")


def prepend_ryan_py_files(repo_root: str | None = None) -> str:
    p = os.path.abspath(_ryan_py_files_dir(repo_root=repo_root))
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def load_grid_norm(
    bundle_npz_path: str,
    *,
    meta_json_path: str | None = None,
    config_json_path: str | None = None,
) -> dict[str, Any]:
    """``grid_norm_info`` for latent → linear flux and coordinate denorms.

    Prefer ``meta_json_path``. If sibling meta missing and NPZ lacks meta, optionally
    read ``grid_norm_info`` from a ``run_gp`` ``config.json`` (iteration snapshot).
    """
    prepend_ryan_py_files()
    import bundle_meta as bmeta

    gn = bmeta.grid_norm_from_bundle_or_meta(bundle_npz_path, meta_path=meta_json_path)
    if gn.get("_normalized_only") and config_json_path and os.path.isfile(config_json_path):
        with open(config_json_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg.get("grid_norm_info"), dict):
            out = dict(cfg["grid_norm_info"])
            out["_normalized_only"] = False
            gn = out
            return gn
    return gn


def _final_branch_directories(
    which_sed: str,
    dir_spliced: Optional[str],
    dir_full_gp: Optional[str],
) -> list[tuple[str, str]]:
    which_sed = str(which_sed).lower()
    if which_sed not in ("spliced", "full_gp", "both"):
        raise ValueError("which_sed must be spliced, full_gp, or both")
    out: list[tuple[str, str]] = []
    if which_sed in ("spliced", "both") and dir_spliced:
        out.append(("spliced", dir_spliced))
    if which_sed in ("full_gp", "both") and dir_full_gp:
        out.append(("full_gp", dir_full_gp))
    return out


def nearest_final_observer_bundle(
    q_mjd: float,
    *,
    which_sed: str,
    dir_spliced: Optional[str],
    dir_full_gp: Optional[str],
    coco_path: str,
    snname: str,
    t0_mjd: float,
    datalc_path: str,
    final_flux_on_disk: str | float = "auto",
    final_suffixes: tuple | None = None,
) -> tuple[float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """First successful ``nearest_final_spectrum_native`` match (same branch order as ``plot_native_epoch``).

    Returns ``(phase_days_after_t0, spec_mjd, wl_aa_observer, flux_linear, filepath_str)``.
    On failure wl/flux/path are ``None`` and phases fall back to the query MJD.
    """
    from comparison_check_log_utils import nearest_final_spectrum_native

    for _tag, fdir in _final_branch_directories(which_sed, dir_spliced, dir_full_gp):
        if not fdir:
            continue
        try:
            smjd, sed_wl, sed_fl, fn = nearest_final_spectrum_native(
                fdir,
                float(q_mjd),
                coco_path,
                snname,
                flux_on_disk=final_flux_on_disk,
                datalc_path=datalc_path,
                final_suffixes=final_suffixes,
            )
            pmd = float(smjd) - float(t0_mjd)
            return (
                pmd,
                float(smjd),
                np.asarray(sed_wl, dtype=float),
                np.asarray(sed_fl, dtype=float),
                str(fn),
            )
        except Exception as e:
            warnings.warn("nearest FINAL failed for overlay (%s): %s" % (_tag, e))

    pmd = float(q_mjd) - float(t0_mjd)
    warnings.warn(
        "nearest_final_observer_bundle: no FINAL spectrum; phases from query Δt %.6f d"
        % pmd
    )
    return pmd, float(q_mjd), None, None, None


def matched_observer_phase_days_nearest_final(
    q_mjd: float,
    *,
    which_sed: str,
    dir_spliced: Optional[str],
    dir_full_gp: Optional[str],
    coco_path: str,
    snname: str,
    t0_mjd: float,
    datalc_path: str,
    final_flux_on_disk: str | float = "auto",
    final_suffixes: tuple | None = None,
) -> tuple[float, float]:
    """Observer-frame phase (days since ``t0_mjd``) for the FINAL spectrum nearest ``q_mjd``.

    Mirrors the first successfully matched branch from ``plot_native_epoch`` ordering.
    Returns ``(phase_days, spec_mjd)``.
    """
    pmd, smjd, _w, _f, _fp = nearest_final_observer_bundle(
        q_mjd,
        which_sed=which_sed,
        dir_spliced=dir_spliced,
        dir_full_gp=dir_full_gp,
        coco_path=coco_path,
        snname=snname,
        t0_mjd=t0_mjd,
        datalc_path=datalc_path,
        final_flux_on_disk=final_flux_on_disk,
        final_suffixes=final_suffixes,
    )
    return pmd, smjd


def median_overlap_rescale_like_plot_native_epoch(
    comp_wl: np.ndarray,
    comp_flux: np.ndarray,
    ref_wl: np.ndarray,
    ref_flux: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Median rescale companions to plotted SED (same semantics as ``plot_native_epoch``, ``normalize_median``).

    ``comp_flux := (comp_flux / nanmedian(comp)) * nanmedian(SED_overlap)``
    using pixels of ``ref`` whose wavelengths lie between min/max finite ``comp_wl``.
    Invalid conditions return flux unchanged with ``applied=False`` metadata.
    """
    cw = np.asarray(comp_wl, dtype=float).ravel()
    cf = np.asarray(comp_flux, dtype=float).ravel()
    rw = np.asarray(ref_wl, dtype=float).ravel()
    rf = np.asarray(ref_flux, dtype=float).ravel()

    md: dict[str, Any] = {"applied": False}
    if cw.size != cf.size:
        warnings.warn(
            "median_overlap_rescale: wavelength/flux shape mismatch (%d vs %d)"
            % (cw.size, cf.size)
        )
        return np.asarray(cf, dtype=float), md

    m_comp = np.isfinite(cf)
    if not np.any(m_comp):
        warnings.warn("median_overlap_rescale: no finite comparison flux")
        return np.asarray(cf, dtype=float), md

    comp_med = float(np.nanmedian(cf[m_comp]))

    if not np.isfinite(comp_med) or comp_med <= 0.0:
        warnings.warn(
            "median_overlap_rescale: comparison median invalid (%.12g)"
            % float(comp_med)
        )
        return np.asarray(cf, dtype=float), md

    min_wl = float(np.nanmin(cw[m_comp]))
    max_wl = float(np.nanmax(cw[m_comp]))

    m_sed_overlap = (
        (rw >= min_wl) & (rw <= max_wl) & np.isfinite(rf)
    )

    if not np.any(m_sed_overlap):
        warnings.warn("median_overlap_rescale: no overlapping SED region")
        return np.asarray(cf, dtype=float), md

    sed_med = float(np.nanmedian(rf[m_sed_overlap]))
    if not np.isfinite(sed_med):
        warnings.warn("median_overlap_rescale: SED overlap median not finite")
        return np.asarray(cf, dtype=float), md

    scale = sed_med / comp_med
    out = np.asarray(cf * scale, dtype=float)
    md["applied"] = True
    md["scale"] = float(scale)
    md["comp_med"] = float(comp_med)
    md["sed_overlap_med"] = float(sed_med)
    md["overlap_wl_band"] = (min_wl, max_wl)
    return out, md


def apply_median_overlap_to_segments(
    segments: list[dict[str, Any]],
    ref_wl: np.ndarray,
    ref_flux: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deep-copy segments and median-rescale ``flux_lin`` / ``eflux_lin`` vs ``ref``.

    Metadata list parallels segments (skipped entries get ``applied=False``).
    """
    out: list[dict[str, Any]] = []
    mds: list[dict[str, Any]] = []
    for sg in segments:
        g = dict(sg)
        cw = np.asarray(g["wl_aa"], dtype=float)
        cf = np.asarray(g["flux_lin"], dtype=float)
        ef_raw = np.asarray(g["eflux_lin"], dtype=float)
        cfs, mdi = median_overlap_rescale_like_plot_native_epoch(
            cw,
            cf,
            ref_wl,
            ref_flux,
        )
        scale = mdi.get("scale")
        if scale is None or not mdi.get("applied"):
            ef = ef_raw.copy()
        else:
            ef = ef_raw * float(scale)
        g["flux_lin"] = cfs
        g["eflux_lin"] = ef
        out.append(g)
        mds.append(mdi)
    return out, mds


def scaled_training_segments_observer_aa(
    bundle_npz_path: str,
    *,
    observer_phase_days: float,
    meta_json_path: str | None = None,
    config_json_path: str | None = None,
    phot_spec_threshold: int = 50,
    phase_pick: str = "observer_window",
    phase_window_days: float | None = 0.2,
    spectrum_tolerance_norm: float = 0.05,
    mask_telluric_rows: bool = True,
    normalize_median_to_sed: bool = False,
    sed_ref_wl: np.ndarray | None = None,
    sed_ref_flux: np.ndarray | None = None,
    overlap_arm_scale: bool = True,
    gap_factor: float = 35.0,
    min_abs_gap_norm: float = 3e-3,
    spec_phase_decimals: int = 9,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return spectroscopy λ-segments decoded to observer Å (+ linear flux) near ``observer_phase_days``.

    ``phase_pick``:

    - ``observer_window`` (default, mangled-parity): all discrete spectroscopic ``X[:,1]`` keys whose
      observer phase-days differ from ``observer_phase_days`` by at most ``phase_window_days``
      (use the same numeric value as ``plot_native_epoch(..., time_window=...)`` for comparable
      multiplicity).
    - ``nearest_cluster`` (Ryan / ``plot_results``): nearest training phase in normalized coordinates,
      then ± ``spectrum_tolerance_norm`` in normalized ``x₂``.

    Optional ``normalize_median_to_sed`` applies the same per-segment overlap-median flux rescale as
    ``plot_native_epoch(normalize_median=True)`` when ``sed_ref_*`` arrays are supplied.
    Segment dict keys: ``wl_aa``, ``flux_lin``, ``eflux_lin``, ``sp_phase_norm``,
    ``observer_phase_days``, ``scale`` (Ryan overlap-arm multiplier).
    """
    prepend_ryan_py_files()
    import gp_utils as gu
    from plot_results import (
        _build_scaled_spec_overlay_rows,
        denorm_ln_wavelength,
        norm_x2_from_phase_days,
        phase_days_from_norm_x2,
    )

    bd = np.load(bundle_npz_path, allow_pickle=False)
    try:
        X = np.asarray(bd["X"], dtype=float)
        y_train = np.asarray(bd["y"], dtype=float)
        yerr_train = np.asarray(bd["yerr"], dtype=float)
        obs_cls = bd["train_obs_class"] if "train_obs_class" in bd.files else None
        if obs_cls is not None:
            obs_cls = np.asarray(obs_cls)
        plot_row_good = np.isfinite(yerr_train.ravel())
        yer = np.asarray(bd["yerr"], dtype=float).ravel()
        plot_row_good &= yer < float(_YERR_DISABLED)
        if mask_telluric_rows and "telluric_bad_mask" in bd.files:
            tm = np.asarray(bd["telluric_bad_mask"], dtype=bool).ravel()
            if tm.size == X.shape[0]:
                plot_row_good &= ~tm
            elif tm.size:
                warnings.warn(
                    "telluric_bad_mask length mismatch (%d vs N=%d); ignoring"
                    % (tm.size, X.shape[0])
                )
    finally:
        bd.close()

    gn = load_grid_norm(
        bundle_npz_path,
        meta_json_path=meta_json_path,
        config_json_path=config_json_path,
    )

    pt = gu.effective_point_class(X, threshold=phot_spec_threshold, train_obs_class=obs_cls)
    spec_mask = pt == gu.SPEC

    if gn.get("_normalized_only"):
        raise ValueError(
            "bundle lacks grid_norm_info; pass meta_json_path or config_json_path with grid_norm_info"
        )

    target_observer_phase_days = float(observer_phase_days)
    target_x2 = float(
        norm_x2_from_phase_days(np.asarray([target_observer_phase_days], dtype=float), gn)[0]
    )
    spec_phases_train = (
        np.unique(X[spec_mask, 1]) if np.any(spec_mask) else np.array([], dtype=float)
    )
    if spec_phases_train.size == 0:
        return [], {"reason": "no_spec_rows", "target_x2": target_x2}

    pick = str(phase_pick).strip().lower()
    if pick not in ("observer_window", "nearest_cluster"):
        raise ValueError(
            'phase_pick must be "observer_window" or "nearest_cluster", got %r'
            % (phase_pick,)
        )

    if pick == "observer_window":
        win = phase_window_days
        if win is None:
            warnings.warn(
                'phase_pick="observer_window" but phase_window_days is None '
                "; falling back to nearest_cluster semantics"
            )
            pick = "nearest_cluster"
        else:
            ph_days_each = phase_days_from_norm_x2(spec_phases_train.astype(float), gn)
            nw = np.sort(
                spec_phases_train[
                    np.abs(ph_days_each - target_observer_phase_days) <= float(win)
                ]
            )
            if nw.size > 0:
                near_spec_phases = nw
                spec_phase = float(
                    near_spec_phases[
                        np.argmin(
                            np.abs(
                                phase_days_from_norm_x2(near_spec_phases.astype(float), gn)
                                - target_observer_phase_days
                            )
                        )
                    ]
                )
            else:
                warnings.warn(
                    "observer_window (±%.6g d) matched zero spectroscopic epochs; snapping "
                    "to nearest discrete training phase" % float(win)
                )
                spec_idx = int(np.nanargmin(np.abs(spec_phases_train - target_x2)))
                spec_phase = float(spec_phases_train[spec_idx])
                near_spec_phases = np.asarray([spec_phase], dtype=float)

    elif pick == "nearest_cluster":
        spec_idx = int(np.nanargmin(np.abs(spec_phases_train - target_x2)))
        spec_phase = float(spec_phases_train[spec_idx])
        near_spec_phases = np.sort(
            spec_phases_train[
                np.abs(spec_phases_train - spec_phase) <= float(spectrum_tolerance_norm)
            ]
        )

    scaled_rows, n_ov = _build_scaled_spec_overlay_rows(
        X,
        y_train,
        yerr_train,
        gn,
        near_spec_phases,
        spec_mask,
        overlap_scale=bool(overlap_arm_scale),
        gap_factor=float(gap_factor),
        min_abs_gap_norm=float(min_abs_gap_norm),
        plot_row_mask=plot_row_good,
        spec_phase_decimals=int(spec_phase_decimals),
    )

    out_segments: list[dict[str, Any]] = []
    for rw in scaled_rows:
        wl_n = np.asarray(rw["wl_norm"], dtype=float)
        wl_aa = np.power(10.0, np.asarray(denorm_ln_wavelength(wl_n, gn), dtype=float))
        sc = float(rw["scale"])
        fl = np.asarray(rw["flux_lin"], dtype=float) * sc
        ef = np.asarray(rw["eflux_lin"], dtype=float) * sc
        sp_phase = float(rw["sp_phase"])
        obs_ph_seg = float(
            phase_days_from_norm_x2(np.asarray([sp_phase], dtype=float), gn)[0]
        )
        out_segments.append(
            {
                "wl_aa": wl_aa,
                "flux_lin": fl,
                "eflux_lin": ef,
                "sp_phase_norm": sp_phase,
                "observer_phase_days": obs_ph_seg,
                "scale": sc,
            }
        )

    phase_days_near = phase_days_from_norm_x2(np.asarray(near_spec_phases, dtype=float), gn)
    snapped_phase_days = phase_days_from_norm_x2(np.array([spec_phase], dtype=float), gn)[0]
    observer_arg = float(target_observer_phase_days)

    if bool(normalize_median_to_sed):
        if sed_ref_wl is None or sed_ref_flux is None:
            warnings.warn(
                "normalize_median_to_sed=True but sed_ref_wl/sed_ref_flux missing "
                "; skipping overlap median rescale"
            )
            norm_meta: dict[str, Any] | list = {}
        else:
            out_segments, norm_meta = apply_median_overlap_to_segments(
                out_segments,
                sed_ref_wl,
                sed_ref_flux,
            )
    else:
        norm_meta = {}

    meta = {
        "target_x2_requested": target_x2,
        "nearest_spec_x2_anchor": float(spec_phase),
        "observer_phase_days_target_arg": observer_arg,
        "observer_phase_days_snap_anchor_observer": float(snapped_phase_days),
        "phase_pick": pick,
        "phase_window_days": float(phase_window_days)
        if (pick == "observer_window" and phase_window_days is not None)
        else None,
        "spectrum_tolerance_norm": float(spectrum_tolerance_norm),
        "near_spec_phase_norm_keys": np.asarray(near_spec_phases, dtype=float),
        "near_observer_phase_days": np.asarray(phase_days_near, dtype=float),
        "n_overlap_links": int(n_ov),
        "n_segments": len(out_segments),
        "median_normalize_meta": norm_meta,
    }

    return out_segments, meta


def _palette_from_color_args(
    color: str,
    colors: Union[str, Sequence[str], None],
) -> list[str]:
    """Return a non-empty color list for per-exposure cycling.

    ``colors is None`` → single ``color``. ``colors`` as str → single entry.
    """
    if colors is None:
        return [str(color)]
    if isinstance(colors, str):
        return [colors]
    seq = [str(x).strip() for x in colors if str(x).strip()]
    if not seq:
        warnings.warn(
            "plot_scaled_training_segments: empty colors sequence; using color=%r" % color
        )
        return [str(color)]
    return seq


def plot_scaled_training_segments(
    ax: Any,
    segments: list[dict[str, Any]],
    *,
    color: str = "#d62728",
    colors: Union[str, Sequence[str], None] = None,
    linestyle: str = "-",
    lw: float = 1.5,
    alpha: float = 0.92,
    zorder: float = 4.6,
    legend_phase_dp: int = 2,
    label_prefix: str = "GP training",
    draw_caps: bool = False,
) -> None:
    """Overlay Ryan-style decoded spectroscopy segments as **lines** on an Å vs F_λ axis.

    Each contiguous λ-arm is plotted sorted by wavelength. Legend entries mirror input
    spectra style: ``t=<phase> days (<label_prefix>)`` once per exposure (distinct
    ``observer_phase_days``), so multi-arm spectra do not clutter the legend.

    Pass ``colors`` as a **sequence** to cycle **one color per distinct exposure**
    (same rounding as the legend). All arms at the same phase share one color.
    If ``colors`` is ``None``, use ``color`` for every segment.
    """
    palette = _palette_from_color_args(color, colors)
    phase_to_line_color: dict[Any, str] = {}
    next_palette_k = 0

    def _color_for_phase_key(ph_k: Any) -> str:
        nonlocal next_palette_k
        if ph_k not in phase_to_line_color:
            phase_to_line_color[ph_k] = palette[next_palette_k % len(palette)]
            next_palette_k += 1
        return phase_to_line_color[ph_k]

    seen_phase_keys: set[Any] = set()
    for sg in segments:
        wl = np.asarray(sg["wl_aa"], dtype=float)
        fl = np.asarray(sg["flux_lin"], dtype=float)
        ef = np.asarray(sg["eflux_lin"], dtype=float)
        if wl.size == 0:
            continue
        order = np.argsort(wl)
        wl_o = wl[order]
        fl_o = fl[order]
        ef_o = ef[order]

        obs_ph = sg.get("observer_phase_days")
        if obs_ph is not None and np.isfinite(float(obs_ph)):
            ph_key = round(float(obs_ph), max(0, int(legend_phase_dp)))
        else:
            ph_key = None

        line_c = _color_for_phase_key(ph_key)

        lbl = None
        if obs_ph is not None and np.isfinite(float(obs_ph)):
            if ph_key not in seen_phase_keys:
                seen_phase_keys.add(ph_key)
                dp = max(0, int(legend_phase_dp))
                lbl = "t=%.*f days" % (dp, float(obs_ph))
        elif not seen_phase_keys and label_prefix:
            seen_phase_keys.add(None)
            lbl = str(label_prefix)

        if draw_caps and np.any(np.isfinite(ef_o)) and np.nanmax(ef_o) > 0:
            lo = np.clip(fl_o - ef_o, 0.0, None)
            hi = fl_o + ef_o
            ax.fill_between(
                wl_o,
                lo,
                hi,
                color=line_c,
                alpha=0.12,
                linewidth=0,
                zorder=zorder - 0.1,
            )

        ax.plot(
            wl_o,
            fl_o,
            color=line_c,
            ls=linestyle,
            lw=float(lw),
            alpha=float(alpha),
            label=lbl,
            zorder=zorder,
        )
