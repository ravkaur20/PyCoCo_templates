#!/usr/bin/env python3
"""Photometry SED-style views in a physical phase window (diagnostic for outlier λ).

For each photometry row in ``phase_days in [phase_min, phase_max]``, plots
``log10(wavelength)`` vs linear flux.  Two panels:

  * **Binned SEDs**: phases grouped into equal-width bins (default 0.05 d); within each bin,
    one curve (sorted by λ) connecting phot points — smooth evolution shows nested curves;
    a bad wavelength at one epoch stands away from neighbors.
  * **Scatter**: all points colored by phase (continuous) — outliers break the color trend at a λ.

Example::

    python plot_phot_sed_phase_window.py \\
        --bundle gp_work_scaled.npz --meta gp_scaled_bundle_meta.json \\
        --phase-min 0.3 --phase-max 0.8 \\
        --out runs/my_run/figs/phot_sed_0p3_0p8d.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import bundle_meta as bmeta
import gp_utils as gu
from plot_results import denorm_ln_wavelength, linear_flux_yerr, phase_days_from_norm_x2, scaled_ln_to_linear


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--bundle", "-b", default=os.path.join(here, "gp_work_scaled.npz"))
    p.add_argument("--meta", default=None)
    p.add_argument("--phot-spec-threshold", type=int, default=50)
    p.add_argument("--phase-min", type=float, default=0.3)
    p.add_argument("--phase-max", type=float, default=0.8)
    p.add_argument("--bin-days", type=float, default=0.05, help="phase bin width in days for panel A")
    p.add_argument("--out", "-o", required=True)
    ns = p.parse_args(argv)

    bundle_path = ns.bundle if os.path.isabs(ns.bundle) else os.path.join(here, ns.bundle)
    meta_path = ns.meta
    if meta_path and not os.path.isabs(meta_path):
        meta_path = os.path.join(here, meta_path)

    gn = bmeta.grid_norm_from_bundle_or_meta(bundle_path, meta_path=meta_path)
    if gn.get("_normalized_only"):
        print("ERROR: pass --meta with grid_norm_info", file=sys.stderr)
        return 2

    bd = np.load(bundle_path, allow_pickle=False)
    try:
        X = np.asarray(bd["X"], dtype=float)
        y = np.asarray(bd["y"], dtype=float)
        yerr = np.asarray(bd["yerr"], dtype=float)
        obs = bd["train_obs_class"] if "train_obs_class" in bd.files else None
    finally:
        bd.close()

    pc = gu.effective_point_class(
        X,
        threshold=int(ns.phot_spec_threshold),
        train_obs_class=np.asarray(obs) if obs is not None else None,
    )
    phot = pc == gu.PHOT
    ph = phase_days_from_norm_x2(X[:, 1], gn)
    m = phot & np.isfinite(ph) & (ph >= float(ns.phase_min)) & (ph <= float(ns.phase_max))
    idx = np.flatnonzero(m)
    if idx.size < 4:
        print(f"[phot_sed] only {idx.size} phot rows in window — skipping", file=sys.stderr)
        return 1

    logwl = denorm_ln_wavelength(X[idx, 0], gn)
    flux = scaled_ln_to_linear(y[idx], gn)
    sig = linear_flux_yerr(y[idx], yerr[idx], gn)
    phs = ph[idx]

    out = os.path.abspath(os.path.expanduser(ns.out))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10.5, 8.6), sharex=True)

    # Panel A: binned curves (phase bins along the window)
    lo, hi = float(ns.phase_min), float(ns.phase_max)
    bw = float(ns.bin_days)
    nbin = max(int(np.ceil((hi - lo) / bw)), 1)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.92, max(nbin, 2)))
    bi = 0
    t0 = lo
    while t0 < hi - 1e-12:
        t1 = min(t0 + bw, hi)
        last = t1 >= hi - 1e-9
        mb = (phs >= t0) & (phs <= t1) if last else (phs >= t0) & (phs < t1)
        if int(np.sum(mb)) >= 3:
            j = np.flatnonzero(mb)
            o = np.argsort(logwl[j])
            jj = j[o]
            color = cmap[bi % len(cmap)]
            cen = 0.5 * (t0 + t1)
            ax0.plot(
                logwl[jj],
                flux[jj],
                "-o",
                ms=3,
                lw=0.9,
                color=color,
                alpha=0.85,
                label=f"{cen:.3f} d (n={jj.size})",
            )
            bi += 1
        t0 = t1

    ax0.set_ylabel("flux (linear)")
    ax0.set_title(
        f"Photometry SEDs (binned phases, Δ={bw:g} d) — [{lo:g}, {hi:g}] d — N={idx.size}"
    )
    ax0.grid(alpha=0.25)
    if bi <= 16:
        ax0.legend(fontsize=6, loc="best", ncol=2)
    else:
        ax0.legend(fontsize=5, loc="upper left", ncol=3, framealpha=0.92)

    sc = ax1.scatter(logwl, flux, c=phs, s=18, cmap="viridis", alpha=0.82, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.02)
    cb.set_label("phase (days)")
    ax1.set_xlabel("log10(wavelength)")
    ax1.set_ylabel("flux (linear)")
    ax1.set_title("Photometry (color = phase); outlier λ breaks vertical trend")
    ax1.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[phot_sed] wrote {out!r}  (N={idx.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
