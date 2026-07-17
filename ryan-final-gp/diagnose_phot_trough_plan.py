#!/usr/bin/env python3
"""Execute bundled diagnostics for phot trough / bundle 0–1 follow-up.

Sections (stdout):
  * Test 0 — photometry at a fixed normalized log-λ ``x₁`` (needs ``--meta`` for physical units).
  * Test 3 — intra-bundle MST pair scales for spectral ``spec_bundle_id`` in {0..3}
    (same arm scaling + solver defaults as ``diagnose_spec_bundle_pair_scaling.py``).
  * Test 8 — row sanity for training rows with ``spec_bundle_id`` in {0,1}:
    duplicate (x₁,x₂) buckets, telluric mask hits, disabled ``yerr``, finite checks.

Example::

    python diagnose_phot_trough_plan.py \\
        --bundle gp_work_scaled.npz --meta gp_scaled_bundle_meta.json \\
        --x1-target -0.876717272314
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import bundle_meta as bmeta
import bundle_preprocess as bpre
import bundle_scale_pipeline as bsp
import gp_utils as gu
import spectrum_bundles as sb
from plot_results import linear_flux_yerr, phase_days_from_norm_x2, scaled_ln_to_linear


def _test0_phot_at_x1(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    pc: np.ndarray,
    gn: dict,
    *,
    x1_target: float,
    x1_atol: float,
) -> None:
    phot = np.flatnonzero(pc == gu.PHOT)
    sel = phot[np.abs(X[phot, 0] - float(x1_target)) < float(x1_atol)]
    print("\n=== Test 0: photometry at x1_target (normalized log10 λ) ===")
    print(f"x1_target={x1_target:.15g}  atol={x1_atol:g}  n_phot={sel.size}")
    if sel.size == 0:
        return
    ph = phase_days_from_norm_x2(X[sel, 1], gn)
    order = np.argsort(ph)
    print("idx   phase_d      x2_norm          flux_linear    yerr_linear")
    for i in sel[order]:
        p = float(phase_days_from_norm_x2(np.array([X[i, 1]]), gn)[0])
        fl = float(scaled_ln_to_linear(np.array([y[i]]), gn)[0])
        el = float(linear_flux_yerr(np.array([y[i]]), np.array([yerr[i]]), gn)[0])
        print(f"{int(i):5d} {p:10.6f} {X[i,1]:16.8f} {fl:12.4e} {el:12.4e}")


def _test3_intra_bundle_scales(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    pc: np.ndarray,
    gn: dict,
    *,
    bundle_ids: list[int],
    max_bundle_minutes: float,
    phase_epoch_atol: float,
    seam_weight: float,
    overlap_grid_points: int,
    seam_band_half_width_aa: float,
    arm_gap_factor: float,
    arm_min_gap_norm: float,
    enrich: dict[str, np.ndarray] | None,
) -> None:
    print("\n=== Test 3: intra-bundle MST edges (spec bundles) ===")
    spec_mask = pc == gu.SPEC
    canonical_phases, epoch_of_row = bsp.unique_spec_epochs(X, spec_mask)
    if canonical_phases.size == 0:
        print("no spectroscopic epochs")
        return
    t_epoch = bsp.time_per_epoch(X, canonical_phases, gn, enrich)
    labels_eps = sb.cluster_by_time(t_epoch, max_delta_minutes=float(max_bundle_minutes))

    y_arm, yerr_arm = bsp.apply_intra_epoch_arm_scaling(
        X,
        y,
        yerr,
        gn,
        epoch_of_row,
        canonical_phases,
        phase_atol=float(phase_epoch_atol),
        seam_weight=float(seam_weight),
        overlap_grid=int(overlap_grid_points),
        seam_band_half_width_aa=float(seam_band_half_width_aa),
        arm_gap_factor=float(arm_gap_factor),
        arm_min_gap_norm=float(arm_min_gap_norm),
    )

    for bid in bundle_ids:
        ep_ids = np.flatnonzero(labels_eps == int(bid)).astype(int)
        print(f"\n--- spectral bundle (spec_bundle_id) {bid}: n_epochs={ep_ids.size} ---")
        if ep_ids.size < 2:
            print("  (<2 epochs in this cluster — no MST edges)")
            continue
        mult, trace, elist_o, _ = bsp.intra_bundle_epoch_scale_trace(
            X,
            y_arm,
            yerr_arm,
            gn,
            canonical_phases,
            np.asarray(ep_ids, dtype=int),
            epoch_of_row,
            phase_atol=float(phase_epoch_atol),
            seam_weight=float(seam_weight),
            overlap_grid_points=int(overlap_grid_points),
            seam_band_half_width_aa=float(seam_band_half_width_aa),
        )
        print(f"  median-λ epoch order: {elist_o}")
        print(f"  cumulative mult (DFS from bluest): { {k: float(v) for k, v in sorted(mult.items())} }")
        for k, row in enumerate(trace):
            geom = row["geometry"]
            print(
                f"  edge {k + 1}/{len(trace)}: {int(row['epoch_bluer'])}→{int(row['epoch_redder'])}  "
                f"s_ij={float(row['s_ij']):.8g}  mode={geom.get('mode')!r}  "
                f"case={row.get('case')!r}"
            )


def _test8_spec_bundle_row_sanity(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    spec_bid: np.ndarray,
    telluric: np.ndarray | None,
    *,
    bundle_ids: list[int],
    dup_decimals: int,
) -> None:
    print("\n=== Test 8: row sanity (spec_bundle_id 0 and 1) ===")
    ydl = float(bpre.YERR_DISABLED)
    for bid in bundle_ids:
        m = spec_bid == int(bid)
        idx = np.flatnonzero(m)
        print(f"\n--- spec_bundle_id {bid}: n_rows={idx.size} ---")
        if idx.size == 0:
            continue
        x1r = np.round(X[idx, 0], int(dup_decimals))
        x2r = np.round(X[idx, 1], int(dup_decimals))
        keys = np.stack([x1r, x2r], axis=1)
        # bucket counts via unique rows
        uniq, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        dups = int(np.sum(cnt > 1))
        print(f"  unique (x1,x2) buckets @ {dup_decimals} decimals: {uniq.shape[0]}  buckets_with_count>1: {dups}")
        if dups:
            bad = cnt > 1
            for j in np.flatnonzero(bad)[:12]:
                nij = int(cnt[j])
                print(f"    key x1={uniq[j,0]:.9g} x2={uniq[j,1]:.9g}  n={nij}")
        ye = yerr[idx].ravel()
        print(
            f"  yerr: n_nan={int(np.sum(~np.isfinite(ye)))}  "
            f"n_disabled(>={ydl:g})={int(np.sum(ye >= ydl))}  "
            f"min={float(np.nanmin(ye)):.4g}  max={float(np.nanmax(ye)):.4g}"
        )
        if telluric is not None and telluric.size == X.shape[0]:
            tm = telluric[idx].astype(bool)
            print(f"  telluric_bad_mask true: {int(np.sum(tm))}/{idx.size}")
        else:
            print("  telluric_bad_mask: absent or length mismatch — skipped")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--bundle", "-b", default=os.path.join(here, "gp_work_scaled.npz"))
    p.add_argument("--meta", default=None, help="grid_norm JSON (e.g. gp_scaled_bundle_meta.json)")
    p.add_argument("--enrich", default=None, help="optional enrich.npz for Test 3 time clustering")
    p.add_argument("--x1-target", type=float, default=-0.876717272314)
    p.add_argument("--x1-atol", type=float, default=1e-9)
    p.add_argument("--phot-spec-threshold", type=int, default=50)
    p.add_argument("--test0", action="store_true", help="run Test 0 (default: all)")
    p.add_argument("--test3", action="store_true", help="run Test 3 only")
    p.add_argument("--test8", action="store_true", help="run Test 8 only")
    p.add_argument("--dup-decimals", type=int, default=9)
    p.add_argument("--max-bundle-minutes", type=float, default=5.0)
    p.add_argument("--phase-epoch-atol", type=float, default=5e-6)
    p.add_argument("--seam-weight", type=float, default=1.0)
    p.add_argument("--overlap-grid-points", type=int, default=256)
    p.add_argument("--seam-fit-half-width-aa", type=float, default=50.0)
    p.add_argument("--arm-gap-factor", type=float, default=35.0)
    p.add_argument("--arm-min-gap-norm", type=float, default=3e-3)
    ns = p.parse_args(argv)

    run_all = not (ns.test0 or ns.test3 or ns.test8)
    do0 = run_all or ns.test0
    do3 = run_all or ns.test3
    do8 = run_all or ns.test8

    bundle_path = ns.bundle if os.path.isabs(ns.bundle) else os.path.join(here, ns.bundle)
    if not os.path.isfile(bundle_path):
        print(f"ERROR: bundle not found {bundle_path!r}", file=sys.stderr)
        return 2

    meta_path = ns.meta
    if meta_path and not os.path.isabs(meta_path):
        meta_path = os.path.join(here, meta_path)
    gn = bmeta.grid_norm_from_bundle_or_meta(bundle_path, meta_path=meta_path)
    if gn.get("_normalized_only") and (do0 or do3):
        print(
            "ERROR: grid norm is identity fallback; pass --meta gp_scaled_bundle_meta.json "
            "(or place correct *_meta.json beside the bundle).",
            file=sys.stderr,
        )
        return 2

    enrich: dict[str, np.ndarray] | None = None
    if ns.enrich:
        ep = ns.enrich if os.path.isabs(ns.enrich) else os.path.join(here, ns.enrich)
        if os.path.isfile(ep):
            z = np.load(ep, allow_pickle=True)
            try:
                enrich = {k: np.asarray(z[k]) for k in z.files}
            finally:
                z.close()

    bd = np.load(bundle_path, allow_pickle=False)
    try:
        X = np.asarray(bd["X"], dtype=float)
        y = np.asarray(bd["y"], dtype=float)
        yerr = np.asarray(bd["yerr"], dtype=float)
        train_obs = bd["train_obs_class"] if "train_obs_class" in bd.files else None
        telluric = np.asarray(bd["telluric_bad_mask"], dtype=bool) if "telluric_bad_mask" in bd.files else None
        spec_bid = np.asarray(bd["spec_bundle_id"], dtype=np.int32).ravel() if "spec_bundle_id" in bd.files else None
    finally:
        bd.close()

    pc = gu.effective_point_class(
        X,
        threshold=int(ns.phot_spec_threshold),
        train_obs_class=np.asarray(train_obs) if train_obs is not None else None,
    )

    if do0:
        _test0_phot_at_x1(X, y, yerr, pc, gn, x1_target=ns.x1_target, x1_atol=ns.x1_atol)

    if do3:
        _test3_intra_bundle_scales(
            X,
            y,
            yerr,
            pc,
            gn,
            bundle_ids=[0, 1, 2, 3],
            max_bundle_minutes=ns.max_bundle_minutes,
            phase_epoch_atol=ns.phase_epoch_atol,
            seam_weight=ns.seam_weight,
            overlap_grid_points=ns.overlap_grid_points,
            seam_band_half_width_aa=ns.seam_fit_half_width_aa,
            arm_gap_factor=ns.arm_gap_factor,
            arm_min_gap_norm=ns.arm_min_gap_norm,
            enrich=enrich,
        )

    if do8:
        if spec_bid is None or spec_bid.shape[0] != X.shape[0]:
            print("\n=== Test 8: skipped (no spec_bundle_id) ===", file=sys.stderr)
        else:
            _test8_spec_bundle_row_sanity(
                X,
                y,
                yerr,
                spec_bid,
                telluric,
                bundle_ids=[0, 1],
                dup_decimals=int(ns.dup_decimals),
            )

    print("\n[diagnose_phot_trough_plan] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
