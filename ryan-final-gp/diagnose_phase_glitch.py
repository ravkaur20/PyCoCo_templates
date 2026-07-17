#!/usr/bin/env python3
"""Diagnostics for localized artifacts on the photometric phase axis (e.g. ~0.5 d).

Implements the checklist from the phase-glitch diagnostic plan:

- Pin physical phase vs normalized ``x₂`` (``grid_norm_info`` / ``*_meta.json``).
- Hull / ``LinearNDInterpolator`` NaN scan (dense phot curve without NN fallback).
- Optional ``gp.predict`` vs stored latent interpolation when ``config.json`` matches the bundle.
- ``mu`` vs ``mu_raw`` on ``X_fill`` at the nearest grid node to a band+phase query.
- Photometry coverage and suggested ``plot_bands_gp_overview.py`` ablation flags.

Examples::

    python diagnose_phase_glitch.py -b gp_work_scaled.npz -p runs/my_run/predictions.npz \\
        --run-config runs/my_run/config.json --phase-days 0.5

    python diagnose_phase_glitch.py -b bundle.npz -p predictions.npz --run-config config.json \\
        --enrich enrich.npz --phase-days 0.5 --x1-round 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

import numpy as np

import bundle_meta as bmeta
import gp_utils as gu
from plot_bands_gp_overview import (
    interp_latent_gp_linear_only,
    photometry_pseudo_wavelength_groups,
    _interp_latent_gp_on_rows,
    _norm_x2_for_dense_time_axis,
    _time_axis,
    _x1_on_dense_time_grid,
)
from plot_results import (
    norm_x2_from_phase_days,
    phase_days_from_norm_x2,
    scaled_ln_to_linear,
    scatter_train_vector_to_bundle,
)
from run_gp import _exclude_rows_by_spec_bundle_id, _parse_exclude_spec_bundle_ids

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_run_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _kernel_config_from_saved(d: dict) -> gu.KernelConfig:
    cfg = gu.KernelConfig(
        name_t=str(d["name_t"]),
        name_w=str(d["name_w"]),
        additive_t=bool(d["additive_t"]),
        additive_w=bool(d["additive_w"]),
    )
    for n in cfg.free_param_names():
        setattr(cfg, n, float(d[n]))
    return cfg


def load_training_like_run_gp(bundle_path: str, cfg: dict[str, Any]) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
]:
    """Apply the same row selection as ``run_gp`` (train-include + exclude spec bundles)."""
    bundle = np.load(bundle_path, allow_pickle=False)
    X = np.asarray(bundle["X"], dtype=float)
    y = np.asarray(bundle["y"], dtype=float).ravel()
    yerr = np.asarray(bundle["yerr"], dtype=float).ravel()
    train_row_index_orig = np.arange(int(X.shape[0]), dtype=np.int64)

    train_obs_override: Optional[np.ndarray] = None
    if "train_obs_class" in bundle.files:
        train_obs_override = np.asarray(bundle["train_obs_class"])

    spec_bid_train: Optional[np.ndarray] = None
    if "spec_bundle_id" in bundle.files:
        spec_bid_train = np.asarray(bundle["spec_bundle_id"], dtype=np.int32).ravel()

    ti = cfg.get("train_include_file")
    if ti:
        mpath = os.path.abspath(os.path.expanduser(str(ti)))
        inc = np.load(mpath, allow_pickle=False)
        if "include" in inc.files:
            mask = np.asarray(inc["include"], dtype=bool).ravel()
        elif "mask" in inc.files:
            mask = np.asarray(inc["mask"], dtype=bool).ravel()
        else:
            raise ValueError(f"{mpath!r}: expected 'include' or 'mask'")
        if mask.shape[0] != X.shape[0]:
            raise ValueError(f"train-include length {mask.shape[0]} != N_train {X.shape[0]}")
        ke = np.nonzero(mask)[0]
        X, y, yerr = X[ke], y[ke], yerr[ke]
        train_row_index_orig = train_row_index_orig[ke]
        if train_obs_override is not None:
            train_obs_override = train_obs_override[ke]
        if spec_bid_train is not None:
            spec_bid_train = spec_bid_train[ke]

    ex_arg = cfg.get("exclude_spec_bundle_ids")
    excl_arg: Optional[str]
    if ex_arg is None:
        excl_arg = None
    elif isinstance(ex_arg, (list, tuple)):
        excl_arg = ",".join(str(x) for x in ex_arg)
    else:
        excl_arg = str(ex_arg)
    exclude_ids = _parse_exclude_spec_bundle_ids(excl_arg)
    if exclude_ids:
        if spec_bid_train is None:
            raise ValueError("exclude_spec_bundle_ids in config but bundle lacks spec_bundle_id")
        keep_b = ~np.isin(
            np.asarray(spec_bid_train, dtype=np.int32),
            np.asarray(exclude_ids, dtype=np.int32),
        )
        train_row_index_orig = train_row_index_orig[keep_b]
        X, y, yerr, train_obs_override, _n_drop = _exclude_rows_by_spec_bundle_id(
            X,
            y,
            yerr,
            spec_bid_train,
            exclude_ids,
            train_obs_override,
        )

    pth = int(cfg.get("phot_spec_threshold", 50))
    point_class = gu.effective_point_class(X, threshold=pth, train_obs_class=train_obs_override)
    return X, y, yerr, point_class, train_row_index_orig, bundle


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", "-b", required=True)
    p.add_argument("--predictions", "-p", required=True)
    p.add_argument("--run-config", default=None)
    p.add_argument("--meta", default=None)
    p.add_argument("--enrich", default=None)
    p.add_argument("--phase-days", type=float, default=0.5)
    p.add_argument("--phase-window", type=float, default=0.2)
    p.add_argument("--phot-lc-time-step-days", type=float, default=0.05)
    p.add_argument("--phot-lc-x1-mode", choices=("track", "median"), default="track")
    p.add_argument("--posterior-kind", choices=("train", "grid_pp", "grid_raw"), default="train")
    p.add_argument("--pseudo-band-digits", type=int, default=4)
    p.add_argument("--phot-pseudo-grouping", choices=("rounded", "unique_x1"), default="rounded")
    p.add_argument("--phot-unique-x1-decimals", type=int, default=12)
    p.add_argument("--all-phot-bands", action="store_true")
    args = p.parse_args(argv)

    gn = bmeta.grid_norm_from_bundle_or_meta(args.bundle, meta_path=args.meta)
    u2 = float(norm_x2_from_phase_days(np.array([float(args.phase_days)], dtype=float), gn)[0])
    print(f"[diagnose] phase_days={args.phase_days:g} -> norm_x2={u2:.12g}")
    print(f"[diagnose] x2_mean={gn.get('x2_mean')} x2_std={gn.get('x2_std')}")

    print(
        "\nAblations:\n"
        f"  python plot_bands_gp_overview.py -b {args.bundle} -p {args.predictions} "
        "--phot-lc-x1-mode median\n"
        f"  python plot_bands_gp_overview.py -b {args.bundle} -p {args.predictions} "
        "--phot-lc-time-step-days 0\n"
        f"  python plot_bands_gp_overview.py -b {args.bundle} -p {args.predictions} "
        "--posterior-kind grid_pp\n"
    )

    bundle_path = os.path.abspath(os.path.expanduser(args.bundle))
    X = np.asarray(np.load(bundle_path, allow_pickle=False)["X"], dtype=float)
    preds = np.load(os.path.abspath(os.path.expanduser(args.predictions)), allow_pickle=False)
    enrich = np.load(args.enrich, allow_pickle=False) if args.enrich else None

    run_cfg: Optional[dict[str, Any]] = None
    if args.run_config:
        rc = os.path.abspath(os.path.expanduser(args.run_config))
        if os.path.isfile(rc):
            run_cfg = _load_run_config(rc)
            X, _y, _ye, point_class, _i, bundle = load_training_like_run_gp(bundle_path, run_cfg)
        else:
            print(f"[diagnose] WARN: missing {rc!r}", file=sys.stderr)
            bundle = np.load(bundle_path, allow_pickle=False)
            point_class = gu.effective_point_class(
                X,
                threshold=50,
                train_obs_class=np.asarray(bundle["train_obs_class"]) if "train_obs_class" in bundle.files else None,
            )
    else:
        bundle = np.load(bundle_path, allow_pickle=False)
        tobs = np.asarray(bundle["train_obs_class"]) if "train_obs_class" in bundle.files else None
        point_class = gu.effective_point_class(X, threshold=50, train_obs_class=tobs)

    groups: dict[str, np.ndarray] = {}
    phot_m = point_class == gu.PHOT
    if enrich is not None and ("band_name" in enrich.files or "band_id" in enrich.files):
        labels = np.array([""] * X.shape[0], dtype=object)
        if "band_name" in enrich.files:
            bn = enrich["band_name"]
            for i in range(min(X.shape[0], bn.shape[0])):
                labels[i] = str(bn[i])
        elif "band_id" in enrich.files:
            bid = enrich["band_id"]
            for i in range(min(X.shape[0], bid.shape[0])):
                labels[i] = f"id_{int(bid[i])}"
        use_real = np.any(labels[phot_m] != "")
    else:
        use_real = False
    if use_real:
        d: dict[str, list[int]] = {}
        for i in np.nonzero(phot_m)[0]:
            lab = str(labels[i]) if labels[i] else "unknown"
            d.setdefault(lab, []).append(int(i))
        groups = {k: np.asarray(v, dtype=int) for k, v in d.items()}
    else:
        groups = photometry_pseudo_wavelength_groups(
            X,
            np.nonzero(phot_m)[0],
            gn,
            grouping=str(args.phot_pseudo_grouping),
            pseudo_band_digits=int(args.pseudo_band_digits),
            unique_x1_decimals=int(args.phot_unique_x1_decimals),
            max_unique_panels=10_000,
        )

    pk = str(args.posterior_kind)
    band_items = list(groups.items())
    if not args.all_phot_bands:
        band_items = band_items[: min(3, len(band_items))]

    for lab, idx in band_items:
        tt = _time_axis(X, gn, enrich if enrich is not None else None)[idx]
        t_dense, Xq = build_dense_X_query(
            X,
            idx,
            tt,
            gn,
            enrich,
            time_step=float(args.phot_lc_time_step_days),
            x1_mode=str(args.phot_lc_x1_mode),
        )
        fs = preds.files
        if pk == "train":
            mu_ref = scatter_train_vector_to_bundle(preds, preds["mu_train"], int(X.shape[0]))
            if mu_ref is None:
                print(f"[diagnose] skip band {lab!r}: mu_train align failed")
                continue
            ok_rows = np.isfinite(mu_ref)
            sites, latent = X[ok_rows], np.asarray(mu_ref[ok_rows], dtype=float).ravel()
        else:
            if "X_fill" not in fs:
                print("[diagnose] grid posterior kinds need X_fill in predictions.npz", file=sys.stderr)
                return 1
            sites = np.asarray(preds["X_fill"], dtype=float)
            mk = "mu_raw" if pk == "grid_raw" and "mu_raw" in fs else "mu"
            latent = np.asarray(preds[mk], dtype=float).ravel()
        lin_only = interp_latent_gp_linear_only(sites, latent, Xq)
        phys = phase_days_from_norm_x2(Xq[:, 1], gn)
        wm = np.isfinite(phys) & (phys >= args.phase_days - args.phase_window) & (
            phys <= args.phase_days + args.phase_window
        )
        n_nan_w = int(np.sum(wm & ~np.isfinite(lin_only)))
        nn_vals = _interp_latent_gp_on_rows(sites, latent, Xq)
        d_jump = np.abs(scaled_ln_to_linear(nn_vals, gn) - scaled_ln_to_linear(lin_only, gn))
        mx = float(np.nanmax(np.where(wm, d_jump, np.nan))) if np.any(wm) else float("nan")
        print(
            f"  band {lab[:56]!r}  "
            f"hull_NaN_in_window={n_nan_w}  "
            f"max_lin_vs_nn_jump_in_window={mx:.3g}  "
            f"n_dense={int(t_dense.size)}"
        )

    return 0


def build_dense_X_query(
    X: np.ndarray,
    idx: np.ndarray,
    tt: np.ndarray,
    gn: dict,
    enrich: Any,
    *,
    time_step: float,
    x1_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    step = float(time_step)
    tlo, thi = float(np.nanmin(tt)), float(np.nanmax(tt))
    t_dense = np.arange(tlo, thi + 0.5 * step, step, dtype=float)
    if t_dense.size < 2:
        t_dense = np.asarray([tlo, thi], dtype=float)
    x2_q = _norm_x2_for_dense_time_axis(t_dense, tt, X[idx, 1], gn, enrich)
    ok = np.isfinite(x2_q)
    t_dense, x2_q = t_dense[ok], x2_q[ok]
    x1_all = np.asarray(X[idx, 0], dtype=float)
    if str(x1_mode).lower() == "median":
        x1_col = np.full(t_dense.shape[0], float(np.median(x1_all)), dtype=float)
    else:
        x1_col = _x1_on_dense_time_grid(t_dense, tt, x1_all)
    return t_dense, np.column_stack([x1_col, x2_q])


if __name__ == "__main__":
    sys.exit(main())