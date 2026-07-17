#!/usr/bin/env python3
"""Two-panel intra-bundle diagnostic matching ``spec_bundle_*_gp`` style (log10 λ vs linear flux).

``bundle_scale_pipeline.composite_epoch_linear`` now restricts rows by ``epoch_of_row ==
epoch_id`` whenever the scaler (or this script) passes those arguments, so **photometry
rows cannot enter a spectroscopic epoch composite** even if they share the same rounded
``x₂`` as an exposure.

The overview PNG’s dashed **GP mean** is still only a model on a grid (not photometry as
spectra).  ``--overview-plot-mask`` is deprecated for MST composites (scaler uses full epochs).

Replays **intra-bundle MST propagation** exactly: same arm-corrected ``y`` as
``bundle_scale_pipeline``, ``intra_bundle_epoch_scale_trace`` (not consecutive raw pairwise
edges — the second χ² compares **cumulatively scaled** flux on one arm to raw redder flux).

Each panel shows:
  * **Raw** bluer and redder (faded / dashed)
  * **Scaled** bluer (= raw, anchor unchanged) and redder × **s** (solid, overview-like linewidth)
  * Shaded **overlap** (narrow ``overlap_shade_aa`` ≈ seam half-width each side of λ seam, clipped)
    and **seam** Å bands from ``pair_scale_report`` (full χ² overlap extent is ``overlap_aa`` in JSON)

This does **not** modify your NPZ.

Example::

    python3 diagnose_spec_bundle_pair_scaling.py \\
        --bundle gp_bundle_collab_fixes.npz \\
        --meta gp_minimal_bundle_meta.json \\
        --bundle-id 3 \\
        --out runs/my_run/figs/overview/bundle_3_pair_scaling.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import bundle_meta as bmeta
import bundle_preprocess as bpre
import bundle_scale_pipeline as bsp
import gp_utils as gu
import spectrum_bundles as sb
from plot_results import wl_linear_aa_to_plot_x1


def _load_enrich(path: str | None) -> dict[str, np.ndarray] | None:
    if not path or not str(path).strip():
        return None
    p = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.isfile(p):
        print(f"[diagnose] WARNING: enrich not found {p!r}, using phase-only times", file=sys.stderr)
        return None
    z = np.load(p, allow_pickle=True)
    try:
        return {k: np.asarray(z[k]) for k in z.files}
    finally:
        z.close()


def _overview_style_row_mask(
    X: np.ndarray,
    yerr: np.ndarray,
    *,
    telluric_bad_mask: np.ndarray | None,
) -> np.ndarray:
    """Match ``plot_bands_gp_overview`` spectral row gate (telluric + finite yerr + not disabled)."""
    good = np.ones(X.shape[0], dtype=bool)
    if telluric_bad_mask is not None:
        tm = np.asarray(telluric_bad_mask, dtype=bool).ravel()
        if tm.shape[0] == X.shape[0]:
            good &= ~tm
    good &= np.isfinite(yerr.ravel())
    good &= yerr.ravel() < float(bpre.YERR_DISABLED)
    return good


def _shade_geometry(ax: plt.Axes, geom: dict[str, object], gn: dict, *, with_labels: bool) -> None:
    lbl_ov = "overlap χ² (±seam half-width, clipped)" if with_labels else "_nolegend_"
    lbl_rb = "ref seam band (affine)" if with_labels else "_nolegend_"
    lbl_mb = "mov seam band (affine)" if with_labels else "_nolegend_"
    ov = geom.get("overlap_shade_aa")
    if ov is not None and isinstance(ov, (tuple, list)) and float(ov[1]) > float(ov[0]) + 1e-9:
        xs = wl_linear_aa_to_plot_x1(np.asarray([float(ov[0]), float(ov[1])], dtype=float), gn)
        ax.axvspan(float(np.min(xs)), float(np.max(xs)), color="green", alpha=0.18, label=lbl_ov)
    rb = geom["ref_seam_band_aa"]
    mb = geom["mov_seam_band_aa"]
    xrb = wl_linear_aa_to_plot_x1(np.asarray([float(rb[0]), float(rb[1])], dtype=float), gn)
    ax.axvspan(float(np.min(xrb)), float(np.max(xrb)), color="orange", alpha=0.16, label=lbl_rb)
    xmb = wl_linear_aa_to_plot_x1(np.asarray([float(mb[0]), float(mb[1])], dtype=float), gn)
    ax.axvspan(float(np.min(xmb)), float(np.max(xmb)), color="cyan", alpha=0.13, label=lbl_mb)


def _flux_ylim_from_series(ax: plt.Axes, series: list[np.ndarray]) -> None:
    parts = [np.asarray(s, dtype=float).ravel() for s in series if np.asarray(s, dtype=float).size]
    if not parts:
        return
    yy = np.concatenate(parts)
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return
    lo, hi = float(np.nanmin(yy)), float(np.nanmax(yy))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    pad = 0.04 * (hi - lo + 1e-30)
    ax.set_ylim(lo - pad, hi + pad)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", "-b", required=True, help="input training npz")
    p.add_argument("--meta", default=None, help="grid norm JSON")
    p.add_argument("--enrich", default=None, help="optional enrich.npz (MJD times)")
    p.add_argument("--bundle-id", type=int, required=True, help="time-bundle label, e.g. 3")
    p.add_argument("--out", "-o", required=True, help="output PNG path")
    p.add_argument("--max-bundle-minutes", type=float, default=5.0)
    p.add_argument("--phase-epoch-atol", type=float, default=5e-6)
    p.add_argument("--seam-weight", type=float, default=1.0)
    p.add_argument("--overlap-grid-points", type=int, default=256)
    p.add_argument("--seam-fit-half-width-aa", type=float, default=50.0)
    p.add_argument("--arm-gap-factor", type=float, default=35.0)
    p.add_argument("--arm-min-gap-norm", type=float, default=3e-3)
    p.add_argument("--phot-spec-threshold", type=int, default=50)
    p.add_argument(
        "--overview-plot-mask",
        action="store_true",
        help="build per-epoch composites using the same telluric+yerr row mask as plot_bands_gp_overview "
        "(disjoint λ in the PNG vs wide overlap in bundle_scale χ²)",
    )
    ns = p.parse_args(argv)

    gn = bmeta.grid_norm_from_bundle_or_meta(ns.bundle, meta_path=ns.meta)
    enrich = _load_enrich(ns.enrich)

    bd = np.load(ns.bundle, allow_pickle=False)
    try:
        X = np.asarray(bd["X"], dtype=float)
        y = np.asarray(bd["y"], dtype=float)
        yerr = np.asarray(bd["yerr"], dtype=float)
        train_obs = bd["train_obs_class"] if "train_obs_class" in bd.files else None
        telluric = bd["telluric_bad_mask"] if "telluric_bad_mask" in bd.files else None
    finally:
        bd.close()

    row_mask_plot: np.ndarray | None = None
    if ns.overview_plot_mask:
        row_mask_plot = _overview_style_row_mask(X, yerr, telluric_bad_mask=telluric)
        print(
            "[diagnose] WARNING: MST replay uses full spectroscopic epochs (matches "
            "bundle_scale_pipeline); --overview-plot-mask does not gate these composites.",
            file=sys.stderr,
        )

    point_class = gu.effective_point_class(
        X,
        train_obs_class=np.asarray(train_obs) if train_obs is not None else None,
        threshold=int(ns.phot_spec_threshold),
    )
    spec_mask = point_class == gu.SPEC
    canonical_phases, epoch_of_row = bsp.unique_spec_epochs(X, spec_mask)
    n_eps = int(canonical_phases.size)
    if n_eps == 0:
        print("[diagnose] no spectroscopic epochs", file=sys.stderr)
        return 2

    t_epoch = bsp.time_per_epoch(X, canonical_phases, gn, enrich)
    labels_eps = sb.cluster_by_time(t_epoch, max_delta_minutes=float(ns.max_bundle_minutes))
    bid = int(ns.bundle_id)
    ep_ids = np.flatnonzero(labels_eps == bid).astype(int)
    if ep_ids.size == 0:
        print(f"[diagnose] no epochs with labels_eps=={bid}", file=sys.stderr)
        return 2

    y_arm, yerr_arm = bsp.apply_intra_epoch_arm_scaling(
        X,
        y,
        yerr,
        gn,
        epoch_of_row,
        canonical_phases,
        phase_atol=float(ns.phase_epoch_atol),
        seam_weight=float(ns.seam_weight),
        overlap_grid=int(ns.overlap_grid_points),
        seam_band_half_width_aa=float(ns.seam_fit_half_width_aa),
        arm_gap_factor=float(ns.arm_gap_factor),
        arm_min_gap_norm=float(ns.arm_min_gap_norm),
    )

    _, trace, elist_o, _ = bsp.intra_bundle_epoch_scale_trace(
        X,
        y_arm,
        yerr_arm,
        gn,
        canonical_phases,
        np.asarray(ep_ids, dtype=int),
        epoch_of_row,
        phase_atol=float(ns.phase_epoch_atol),
        seam_weight=float(ns.seam_weight),
        overlap_grid_points=int(ns.overlap_grid_points),
        seam_band_half_width_aa=float(ns.seam_fit_half_width_aa),
    )
    if not trace:
        print("[diagnose] empty MST trace (need ≥2 epochs with valid composites?)", file=sys.stderr)
        return 2

    data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ee in elist_o:
        w, f, e = bsp.composite_epoch_linear(
            X,
            y_arm,
            yerr_arm,
            gn,
            bsp.canon_phase(canonical_phases, int(ee)),
            phase_atol=float(ns.phase_epoch_atol),
            epoch_of_row=epoch_of_row,
            epoch_id=int(ee),
        )
        data[int(ee)] = (w, f, e)

    n_ep = len(elist_o)
    n_trace = len(trace)
    n_rows = n_trace
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, max(n_ep, 2)))
    ep_to_ci = {int(elist_o[i]): i for i in range(n_ep)}
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 4.0 * n_rows), sharex=True, squeeze=False)
    axes = np.atleast_1d(axes).ravel()

    print(f"[diagnose] bundle_id={bid} median-λ order elist={elist_o} (index0=bluest)")

    for k, row in enumerate(trace):
        ax = axes[k]
        e_lo = int(row["epoch_bluer"])
        e_hi = int(row["epoch_redder"])
        wl_b = np.asarray(row["wl_bluer_aa"], dtype=float)
        f_bs = np.asarray(row["f_bluer_in"], dtype=float).ravel()
        wl_r = np.asarray(row["wl_redder_aa"], dtype=float)
        fr_in = np.asarray(row["f_redder_in"], dtype=float).ravel()
        s_ij = float(row["s_ij"])
        case = str(row.get("case", ""))
        geom = row["geometry"]
        mode = str(geom["mode"])

        wl_raw_lo, f_raw_lo, _ = data[e_lo]
        wl_raw_hi, f_raw_hi, _ = data[e_hi]
        c_lo = colors[ep_to_ci[e_lo]]
        c_hi = colors[ep_to_ci[e_hi]]

        x_rlo = wl_linear_aa_to_plot_x1(wl_raw_lo, gn)
        x_rhi = wl_linear_aa_to_plot_x1(wl_raw_hi, gn)
        x_sb = wl_linear_aa_to_plot_x1(wl_b, gn)
        x_sr = wl_linear_aa_to_plot_x1(wl_r, gn)

        ax.plot(x_rlo, f_raw_lo, "--", color=c_lo, lw=1.15, alpha=0.5, label=f"epoch {e_lo} composite (raw)")
        ax.plot(x_rhi, f_raw_hi, "--", color=c_hi, lw=1.15, alpha=0.5, label=f"epoch {e_hi} composite (raw)")
        ax.plot(x_sb, f_bs, "-", color=c_lo, lw=2.2, alpha=0.95, label=f"epoch {e_lo} ref (solver)")
        ax.plot(x_sr, fr_in * s_ij, "-", color=c_hi, lw=2.2, alpha=0.95, label=f"epoch {e_hi} mov×s (solver)")

        _shade_geometry(ax, geom, gn, with_labels=(k == 0))

        ax.set_ylabel("flux (linear)")
        ax.set_title(
            f"Spectral bundle {bid} — MST edge {k + 1}/{n_trace}: {e_lo}→{e_hi} ({case}); "
            f"s={s_ij:.6g}; mode={mode}; Δt≤{float(ns.max_bundle_minutes):g} min"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best", ncol=1)
        _flux_ylim_from_series(ax, [f_raw_lo, f_raw_hi, f_bs, fr_in * s_ij])

        print(
            f"[diagnose] panel {k + 1}: MST {e_lo}→{e_hi}  s={s_ij:.12g}  mode={mode}  case={case}  "
            f"overlap_extent_aa={geom.get('overlap_aa')}  overlap_shade_aa={geom.get('overlap_shade_aa')}"
        )

    axes[-1].set_xlabel("log10(wavelength)")
    fig.tight_layout()
    out = os.path.abspath(os.path.expanduser(ns.out))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[diagnose] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
