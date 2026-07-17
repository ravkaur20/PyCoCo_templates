#!/usr/bin/env python3
"""Print training density and residual stats in a physical log10(wavelength) window.

Supports plan follow-up for bundle-3 bluest-λ mismatch (§2b) and near-duplicate spec phases (gp-followup).

Example::

    python diagnose_gp_training_window.py \\
        --bundle gp_work_scaled.npz --meta gp_scaled_bundle_meta.json \\
        --log-wl-min 3.5 --log-wl-max 3.72 \\
        --run-dir runs/my_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import bundle_scale_pipeline as bsp
import gp_utils as gu
from plot_results import denorm_ln_wavelength, linear_flux_yerr, scaled_ln_to_linear, scatter_train_vector_to_bundle


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", "-b", required=True, help="training npz (scaled or raw)")
    p.add_argument("--meta", default=None, help="grid_norm JSON (default: discover like bundle_scale_pipeline)")
    p.add_argument("--run-dir", default=None, help="runs/<tag> dir with predictions.npz for residual slice")
    p.add_argument("--log-wl-min", type=float, default=3.5)
    p.add_argument("--log-wl-max", type=float, default=3.72)
    p.add_argument("--phot-spec-threshold", type=int, default=50)
    p.add_argument("--dup-phase-round", type=int, default=9, help="decimals on x2 for duplicate-phase bucketing")
    p.add_argument("--dup-phase-min-count", type=int, default=2)
    args = p.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    bundle_path = args.bundle if os.path.isabs(args.bundle) else os.path.join(here, args.bundle)
    if not os.path.isfile(bundle_path):
        print(f"ERROR: bundle not found: {bundle_path!r}", file=sys.stderr)
        return 2

    gn = bsp.load_grid_norm(bundle_path, meta_override=args.meta)
    bd = np.load(bundle_path, allow_pickle=False)
    X = np.asarray(bd["X"], dtype=float)
    y = np.asarray(bd["y"], dtype=float)
    yerr = np.asarray(bd["yerr"], dtype=float)
    train_obs = bd["train_obs_class"] if "train_obs_class" in bd.files else None
    pc = gu.effective_point_class(
        X,
        threshold=int(args.phot_spec_threshold),
        train_obs_class=np.asarray(train_obs) if train_obs is not None else None,
    )
    bd.close()

    log_wl = denorm_ln_wavelength(X[:, 0], gn)
    m = (log_wl >= float(args.log_wl_min)) & (log_wl <= float(args.log_wl_max)) & np.isfinite(X[:, 0])
    m_spec = m & (pc == gu.SPEC)
    m_phot = m & (pc == gu.PHOT)
    n_s = int(np.sum(m_spec))
    n_p = int(np.sum(m_phot))
    print(f"[window] log10(lambda) in [{args.log_wl_min:g}, {args.log_wl_max:g}]")
    print(f"[window] spec rows={n_s} phot rows={n_p}")

    if n_s > 0:
        x2s = np.round(X[m_spec, 1], int(args.dup_phase_round))
        uniq, cnt = np.unique(x2s, return_counts=True)
        dups = uniq[cnt >= int(args.dup_phase_min_count)]
        print(f"[dup-phase] spec epochs (x2 rounded to {args.dup_phase_round}d) with count>={args.dup_phase_min_count}: {dups.size}")
        if dups.size and dups.size <= 24:
            for u in dups:
                c = int(np.sum(x2s == u))
                print(f"    x2≈{u:g}  n={c}")

    if args.run_dir:
        rd = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(here, args.run_dir)
        pred_path = os.path.join(rd, "predictions.npz")
        if not os.path.isfile(pred_path):
            print(f"WARN: no predictions at {pred_path!r}", file=sys.stderr)
            return 0
        pr = np.load(pred_path, allow_pickle=False)
        try:
            mu_tr = np.asarray(pr["mu_train"], dtype=float).ravel()
            if mu_tr.shape[0] != X.shape[0]:
                mu_tr2 = scatter_train_vector_to_bundle(pr, mu_tr, int(X.shape[0]))
                if mu_tr2 is None:
                    print("WARN: mu_train length mismatch bundle X (no train_row_index_orig)", file=sys.stderr)
                    return 0
                mu_tr = mu_tr2
        finally:
            pr.close()
        if n_s > 0:
            m_fit = m_spec & np.isfinite(mu_tr)
            if np.any(m_fit):
                y_lin = scaled_ln_to_linear(y[m_fit], gn)
                mu_lin = scaled_ln_to_linear(mu_tr[m_fit], gn)
                sig_lin = linear_flux_yerr(y[m_fit], yerr[m_fit], gn)
                rat = mu_lin / np.maximum(np.abs(y_lin), 1e-40) - 1.0
                print(
                    f"[resid] spec |mu/y-1|: mean={float(np.mean(np.abs(rat))):.4g} "
                    f"median={float(np.median(np.abs(rat))):.4g}"
                )
                print(
                    f"[resid] spec |mu-y|/sig: mean={float(np.mean(np.abs(mu_lin - y_lin) / np.maximum(sig_lin, 1e-40))):.4g}"
                )

        cfg = os.path.join(rd, "config.json")
        if os.path.isfile(cfg):
            with open(cfg, encoding="utf-8") as f:
                cj = json.load(f)
            k = cj.get("config", {})
            print(
                "[run] metric_t={:.4g} metric_w={:.4g} weight_t_short={:.4g} weight_w_short={:.4g}".format(
                    float(k.get("metric_t", float("nan"))),
                    float(k.get("metric_w", float("nan"))),
                    float(k.get("weight_t_short", float("nan"))),
                    float(k.get("weight_w_short", float("nan"))),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
