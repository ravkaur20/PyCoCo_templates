#!/usr/bin/env python3
"""Plot training data only: one plot per spectrum and one plot per light curve.

This script is intentionally GP-free so data handling can be validated first.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import bundle_meta as bmeta
import bundle_preprocess as bpre
import gp_utils as gu
from plot_results import denorm_ln_wavelength, linear_flux_yerr, phase_days_from_norm_x2, scaled_ln_to_linear


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "gp_minimal_bundle.npz")


def _load_enrich(path: Optional[str]) -> Optional[dict[str, np.ndarray]]:
    if not path:
        return None
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        print(f"[separated] WARNING: enrich file missing: {p!r}", file=sys.stderr)
        return None
    d = np.load(p, allow_pickle=True)
    out = {k: np.asarray(d[k]) for k in d.files}
    d.close()
    return out


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def plot_light_curves(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    point_class: np.ndarray,
    gn: dict,
    enrich: Optional[dict[str, np.ndarray]],
    out_dir: str,
    *,
    pseudo_band_digits: int,
) -> int:
    phot_m = point_class == gu.PHOT
    if not np.any(phot_m):
        print("[separated] no photometry rows")
        return 0

    labels = np.full(X.shape[0], "", dtype=object)
    if enrich is not None:
        if "band_name" in enrich:
            bn = np.asarray(enrich["band_name"]).ravel()
            for i in range(min(labels.shape[0], bn.shape[0])):
                labels[i] = str(bn[i])
        elif "band_id" in enrich:
            bid = np.asarray(enrich["band_id"]).ravel()
            for i in range(min(labels.shape[0], bid.shape[0])):
                labels[i] = f"id_{int(bid[i])}"

    groups: dict[str, np.ndarray]
    if np.any(labels[phot_m] != ""):
        g = defaultdict(list)
        for i in np.nonzero(phot_m)[0]:
            g[str(labels[i])].append(int(i))
        groups = {k: np.asarray(v, dtype=int) for k, v in g.items()}
    else:
        k = np.round(np.asarray(X[:, 0], dtype=float), int(pseudo_band_digits))
        g = defaultdict(list)
        for i in np.nonzero(phot_m)[0]:
            g[f"log10λ_norm≈{k[i]:.4f}"].append(int(i))
        groups = {kk: np.asarray(v, dtype=int) for kk, v in g.items()}

    if enrich is not None and "mjd" in enrich and len(enrich["mjd"]) >= X.shape[0]:
        t = np.asarray(enrich["mjd"], dtype=float)[: X.shape[0]]
        xlab = "MJD"
    else:
        t = phase_days_from_norm_x2(X[:, 1], gn)
        xlab = "phase (days)"

    n_written = 0
    for lab, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if idx.size == 0:
            continue
        tt = t[idx]
        ff = scaled_ln_to_linear(y[idx], gn)
        ee = linear_flux_yerr(y[idx], yerr[idx], gn)
        o = np.argsort(tt)

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.errorbar(tt[o], ff[o], yerr=ee[o], fmt="o", ms=4, lw=0.5, elinewidth=0.6, alpha=0.85)
        ax.set_title(f"Photometry: {lab} (N={idx.size})")
        ax.set_xlabel(xlab)
        ax.set_ylabel("flux (linear)")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        fn = f"phot_{_safe_name(str(lab))}.png"
        fig.savefig(os.path.join(out_dir, fn), dpi=150)
        plt.close(fig)
        n_written += 1
    return n_written


def plot_spectra(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    point_class: np.ndarray,
    gn: dict,
    out_dir: str,
    *,
    phase_match_atol: float,
    include_disabled_telluric: bool,
) -> int:
    spec_m = point_class == gu.SPEC
    if not np.any(spec_m):
        print("[separated] no spectroscopy rows")
        return 0

    good = np.isfinite(yerr)
    if not include_disabled_telluric:
        good &= np.asarray(yerr, dtype=float) < float(bpre.YERR_DISABLED)

    phases = bpre.canonical_sorted_phases(X[spec_m, 1], atol=phase_match_atol)
    n_written = 0
    for i, ph in enumerate(phases):
        m = spec_m & np.isclose(X[:, 1], float(ph), rtol=0.0, atol=phase_match_atol) & good
        idx = np.nonzero(m)[0]
        if idx.size < 2:
            continue
        wl = denorm_ln_wavelength(X[idx, 0], gn)
        ff = scaled_ln_to_linear(y[idx], gn)
        ee = linear_flux_yerr(y[idx], yerr[idx], gn)
        o = np.argsort(wl)

        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        ax.errorbar(wl[o], ff[o], yerr=ee[o], fmt=".", ms=2, lw=0.4, elinewidth=0.5, alpha=0.85)
        ax.set_title(f"Spectrum phase_norm={ph:.8g} (N={idx.size})")
        ax.set_xlabel("log10(wavelength)")
        ax.set_ylabel("flux (linear)")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        fn = f"spec_{i:03d}_phase_{ph:+.6f}.png".replace("+", "p").replace("-", "m")
        fig.savefig(os.path.join(out_dir, fn), dpi=150)
        plt.close(fig)
        n_written += 1
    return n_written


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", "-b", default=DEFAULT_BUNDLE)
    p.add_argument("--meta", default=None)
    p.add_argument("--enrich", default=None)
    p.add_argument("--output-dir", "-o", default=os.path.join(HERE, "runs", "separated_data_plots"))
    p.add_argument("--phot-spec-threshold", type=int, default=50)
    p.add_argument("--pseudo-band-digits", type=int, default=4)
    p.add_argument(
        "--phase-match-atol",
        type=float,
        default=0.0,
        help="merge spec rows whose X[:,1] differs by less than this (0 = one plot per distinct phase float)",
    )
    p.add_argument("--include-disabled-telluric", action="store_true")
    p.add_argument(
        "--train-include",
        default=None,
        help="optional npz with bool 'include' or 'mask' (length N_train); same semantics as run_gp — "
        "plot only rows kept in that GP fit",
    )
    args = p.parse_args(argv)

    gn = bmeta.grid_norm_from_bundle_or_meta(args.bundle, meta_path=args.meta)
    enrich = _load_enrich(args.enrich)

    b = np.load(args.bundle, allow_pickle=False)
    try:
        X = np.asarray(b["X"], dtype=float)
        y = np.asarray(b["y"], dtype=float)
        yerr = np.asarray(b["yerr"], dtype=float)
        obs = np.asarray(b["train_obs_class"]) if "train_obs_class" in b.files else None
    finally:
        b.close()

    if args.train_include:
        mp = os.path.abspath(os.path.expanduser(args.train_include))
        inc = np.load(mp, allow_pickle=False)
        try:
            if "include" in inc.files:
                row_ok = np.asarray(inc["include"], dtype=bool).ravel()
            elif "mask" in inc.files:
                row_ok = np.asarray(inc["mask"], dtype=bool).ravel()
            else:
                raise ValueError(f"{mp!r}: expected 'include' or 'mask'")
        finally:
            inc.close()
        if row_ok.shape[0] != X.shape[0]:
            raise ValueError(f"train-include length {row_ok.shape[0]} != bundle N {X.shape[0]}")
        ke = np.flatnonzero(row_ok)
        X, y, yerr = X[ke], y[ke], yerr[ke]
        if obs is not None:
            obs = obs[ke]
        if enrich is not None:
            en2: dict[str, np.ndarray] = {}
            for k, v in enrich.items():
                vv = np.asarray(v)
                if vv.shape[0] >= ke.max() + 1:
                    en2[k] = vv[ke]
                else:
                    en2[k] = vv
            enrich = en2
        print(f"[separated] applied train-include {mp!r}: N={ke.size}/{row_ok.size}")

    point_class = gu.effective_point_class(X, train_obs_class=obs, threshold=args.phot_spec_threshold)

    out_phot = os.path.join(args.output_dir, "light_curves")
    out_spec = os.path.join(args.output_dir, "spectra")
    os.makedirs(out_phot, exist_ok=True)
    os.makedirs(out_spec, exist_ok=True)

    n_ph = plot_light_curves(
        X,
        y,
        yerr,
        point_class,
        gn,
        enrich,
        out_phot,
        pseudo_band_digits=args.pseudo_band_digits,
    )
    n_sp = plot_spectra(
        X,
        y,
        yerr,
        point_class,
        gn,
        out_spec,
        phase_match_atol=args.phase_match_atol,
        include_disabled_telluric=bool(args.include_disabled_telluric),
    )
    print(f"[separated] wrote {n_ph} light-curve plots -> {out_phot}")
    print(f"[separated] wrote {n_sp} spectrum plots -> {out_spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
