"""Plot diagnostics for a single GP run.

Usage:
    python plot_results.py --tag matern52_linear_opt
    python plot_results.py --tag matern32_nearest_baseline_jitter

Reads:
    runs/<tag>/predictions.npz  (written by run_gp.py)
    runs/<tag>/config.json
    gp_minimal_bundle.npz       (training X/y/yerr)

Writes:
    runs/<tag>/figs/*.{pdf,png}

Note: the un-normalization metadata (gp_minimal_bundle_meta.json) is missing,
so axes are left in *normalized* coordinates and ``scaled_ln_to_linear`` /
``phase_days_from_norm_x2`` are placeholder identities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import cycle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import gp_utils as gu


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "gp_minimal_bundle.npz")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "runs")


GRID_NORM_INFO: dict = {
    "x1_mean": 0.0,
    "x1_std": 1.0,
    "x2_mean": 0.0,
    "x2_std": 1.0,
    "offset": 0.0,
    "scale_factor": 1.0,
}


def scaled_ln_to_linear(mu: np.ndarray, offset: float, scale_factor: float) -> np.ndarray:
    return mu


def phase_days_from_norm_x2(x2: np.ndarray, gn: dict) -> np.ndarray:
    return x2


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_wavelength_slice_figure(
    x1_fill: np.ndarray,
    x2_fill: np.ndarray,
    mu_fill: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    yerr_train: np.ndarray,
    point_class: np.ndarray,
    gn: dict,
    *,
    use_log10_phase_axis: bool,
    log_y: bool,
    save_path: str,
    suptitle: Optional[str] = None,
    overlay_training: bool = True,
) -> None:
    offset = gn["offset"]
    scale_factor = gn["scale_factor"]
    x1m, x1s = float(gn["x1_mean"]), float(gn["x1_std"])
    x2m, x2s = float(gn["x2_mean"]), float(gn["x2_std"])

    fit_wls = np.unique(x1_fill)[::10]
    len_wls = len(fit_wls)
    if len_wls < 4:
        print(f"[plot_results]   only {len_wls} slices; skipping {save_path}")
        return

    color_iter = cycle(plt.cm.gnuplot(np.linspace(0.05, 0.95, len_wls)))

    fig = plt.figure(figsize=(11, 7))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=9, y=1.02)

    quarter = len_wls // 4
    panel_slices = [
        (1, slice(0, quarter)),
        (2, slice(quarter, 2 * quarter)),
        (3, slice(2 * quarter, 3 * quarter)),
        (4, slice(3 * quarter, len_wls)),
    ]

    for panel_idx, sl in panel_slices:
        ax = plt.subplot(2, 2, panel_idx)
        wls_in_panel = fit_wls[sl]
        if wls_in_panel.size == 0:
            continue
        ax.set_title(
            "log10(wl): %.3f-%.3f"
            % (
                min(x1m + x1s * wls_in_panel),
                max(x1m + x1s * wls_in_panel),
            ),
            fontsize=10,
        )
        for i in wls_in_panel:
            mask = x1_fill == i
            if not mask.any():
                continue
            xv = (x2m + x2s * x2_fill[mask]) if use_log10_phase_axis else phase_days_from_norm_x2(x2_fill[mask], gn)
            yv = scaled_ln_to_linear(mu_fill[mask], offset, scale_factor)
            order = np.argsort(xv)
            ax.plot(xv[order], yv[order], color=next(color_iter), lw=0.8, alpha=0.8)
        if overlay_training:
            wls_min = wls_in_panel.min()
            wls_max = wls_in_panel.max()
            wls_pad = 0.5 * (np.unique(x1_fill).max() - np.unique(x1_fill).min()) / max(len(np.unique(x1_fill)) - 1, 1)
            wls_mask = (X_train[:, 0] >= wls_min - wls_pad) & (X_train[:, 0] <= wls_max + wls_pad)
            for cls_name, marker, alpha, label in (
                (gu.PHOT, "o", 0.9, "phot"),
                (gu.SPEC, ".", 0.5, "spec"),
            ):
                cm = wls_mask & (point_class == cls_name)
                if not cm.any():
                    continue
                xv = (x2m + x2s * X_train[cm, 1]) if use_log10_phase_axis else phase_days_from_norm_x2(X_train[cm, 1], gn)
                yv = scaled_ln_to_linear(y_train[cm], offset, scale_factor)
                ax.errorbar(
                    xv,
                    yv,
                    yerr=yerr_train[cm],
                    fmt=marker,
                    ms=3 if cls_name == gu.PHOT else 1.5,
                    lw=0.5,
                    elinewidth=0.4,
                    color="k",
                    alpha=alpha,
                    label=label,
                )
            ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("log10(phase days)" if use_log10_phase_axis else "Phase (days)")
        ax.set_ylabel("flux (linear)")
        if log_y:
            ax.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.92] if suptitle else None)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_heatmap(
    X_fill: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    save_path: str,
    cmap: str = "viridis",
    overlay_training_phases: Optional[np.ndarray] = None,
) -> None:
    wls = np.unique(X_fill[:, 0])
    phases = np.unique(X_fill[:, 1])
    grid = np.full((wls.size, phases.size), np.nan, dtype=float)
    wls_idx = {v: i for i, v in enumerate(wls)}
    phase_idx = {v: i for i, v in enumerate(phases)}
    for k in range(X_fill.shape[0]):
        i = wls_idx[X_fill[k, 0]]
        j = phase_idx[X_fill[k, 1]]
        grid[i, j] = values[k]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(phases, wls, grid, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label=cbar_label)
    if overlay_training_phases is not None and overlay_training_phases.size:
        for p in np.unique(overlay_training_phases):
            ax.axvline(p, color="white", lw=0.2, alpha=0.5)
    ax.set_xlabel("normalized log10(phase days)")
    ax.set_ylabel("normalized log10(wavelength)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_training_coverage(
    X: np.ndarray,
    y: np.ndarray,
    point_class: np.ndarray,
    X_fill: np.ndarray,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    spec_mask = point_class == gu.SPEC
    phot_mask = point_class == gu.PHOT
    sc = ax.scatter(X[spec_mask, 1], X[spec_mask, 0], c=y[spec_mask], s=3, cmap="viridis", label=f"spec (n={spec_mask.sum()})")
    ax.scatter(X[phot_mask, 1], X[phot_mask, 0], c=y[phot_mask], s=14, cmap="viridis",
               edgecolors="red", linewidths=0.6, label=f"phot (n={phot_mask.sum()})")
    fig.colorbar(sc, ax=ax, label="y (training)")
    ax.set_xlim(X_fill[:, 1].min(), X_fill[:, 1].max())
    ax.set_ylim(X_fill[:, 0].min(), X_fill[:, 0].max())
    ax.set_xlabel("normalized log10(phase days)")
    ax.set_ylabel("normalized log10(wavelength)")
    ax.set_title("Training coverage (red rim = phot, dot = spec)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_residual_histograms(
    y: np.ndarray,
    mu_train: np.ndarray,
    sigma_eff: np.ndarray,
    point_class: np.ndarray,
    save_path: str,
) -> None:
    resid = mu_train - y
    norm = resid / np.maximum(sigma_eff, 1e-30)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(resid, bins=80, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("mu_train - y")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"raw residuals (mean={resid.mean():.3g}, std={resid.std():.3g})")

    bins = np.linspace(-6, 6, 80)
    for cls, color, label in (
        (gu.PHOT, "indianred", "phot"),
        (gu.SPEC, "steelblue", "spec"),
    ):
        m = point_class == cls
        if not m.any():
            continue
        axes[1].hist(
            np.clip(norm[m], -6, 6),
            bins=bins,
            histtype="step",
            lw=1.5,
            color=color,
            label=f"{label} (n={m.sum()}, std={norm[m].std():.2f})",
        )
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel("(mu_train - y) / sigma_eff")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"normalized residuals (target std=1)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_spectrum_figure(
    X_fill: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    yerr_train: np.ndarray,
    point_class: np.ndarray,
    requested_phases: np.ndarray,
    save_path: str,
    near_sim_tol: float = 0.05,
) -> None:
    """Spectrum (mu vs wavelength) at the closest available training spectra,
    overlaying *all* near-simultaneous spectra.

    For each requested normalized-log10(phase) value:

    1. Snap to the *nearest spec training phase* (an actual spectrum that
       exists in the data, even if it isn't exactly at the requested time).
    2. Find every other spec training phase within ``near_sim_tol`` of that
       chosen spec phase. These count as "near-simultaneous" spectra and
       are all overlaid (with different colors), since some spectra are
       very close in time but cover non-overlapping wavelength ranges.
    3. Snap the chosen spec phase to the nearest X_fill grid phase and plot
       the GP prediction (mu +/- 1 sigma) at that grid phase.
    4. Optionally overlay phot training points within ``near_sim_tol`` of
       the chosen spec phase for cross-context.
    """
    phases_pred = np.unique(X_fill[:, 1])
    if phases_pred.size < 2:
        return

    spec_mask = point_class == gu.SPEC
    spec_phases_train = np.unique(X_train[spec_mask, 1]) if spec_mask.any() else np.array([])
    if spec_phases_train.size == 0:
        print("[plot_results]   no spec training data; skipping spectrum figure")
        return

    n = len(requested_phases)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.1 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, requested in zip(axes, requested_phases):
        spec_idx = int(np.argmin(np.abs(spec_phases_train - requested)))
        spec_phase = float(spec_phases_train[spec_idx])

        # Every spec phase within near_sim_tol of the chosen spec phase.
        near_spec_phases = np.sort(
            spec_phases_train[np.abs(spec_phases_train - spec_phase) <= near_sim_tol]
        )

        # Snap chosen spec phase to grid for the GP prediction.
        grid_idx = int(np.argmin(np.abs(phases_pred - spec_phase)))
        grid_phase = float(phases_pred[grid_idx])

        mask = X_fill[:, 1] == grid_phase
        wls = X_fill[mask, 0]
        m = mu[mask]
        s = std[mask]
        order = np.argsort(wls)
        wls = wls[order]; m = m[order]; s = s[order]
        ax.fill_between(wls, m - s, m + s, color="steelblue", alpha=0.20)
        ax.plot(
            wls, m,
            color="steelblue", lw=1.4,
            label=f"GP @ phase {grid_phase:.3g}",
        )

        spec_colors = plt.cm.viridis(
            np.linspace(0.05, 0.85, max(near_spec_phases.size, 1))
        )
        n_spec_total = 0
        for sp_phase, sp_color in zip(near_spec_phases, spec_colors):
            sp_mask = (X_train[:, 1] == sp_phase) & spec_mask
            n_at = int(sp_mask.sum())
            n_spec_total += n_at
            if not n_at:
                continue
            delta = sp_phase - spec_phase
            ax.errorbar(
                X_train[sp_mask, 0],
                y_train[sp_mask],
                yerr=yerr_train[sp_mask],
                fmt=".",
                ms=2,
                lw=0.4,
                elinewidth=0.4,
                color=sp_color,
                alpha=0.85,
                label=f"spec phase {sp_phase:.4g} (?={delta:+.3g}, n={n_at})",
            )

        phot_mask = point_class == gu.PHOT
        phot_near = phot_mask & (np.abs(X_train[:, 1] - spec_phase) <= near_sim_tol)
        n_phot_at = int(phot_near.sum())
        if n_phot_at:
            ax.errorbar(
                X_train[phot_near, 0],
                y_train[phot_near],
                yerr=yerr_train[phot_near],
                fmt="o",
                ms=3,
                lw=0.4,
                elinewidth=0.4,
                mfc="none",
                mec="red",
                ecolor="red",
                alpha=0.7,
                label=f"phot within ?={near_sim_tol:g} (n={n_phot_at})",
            )
        ax.set_title(
            f"requested log10(phase)={requested:.3g} -> spec phase {spec_phase:.4g} "
            f"(?req={spec_phase - requested:+.3g}); "
            f"{near_spec_phases.size} near-simultaneous spec phase(s) within ?={near_sim_tol:g}, "
            f"{n_spec_total} spec points total",
            fontsize=8,
        )
        ax.set_ylabel("mu")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc="best", ncol=1)
    axes[-1].set_xlabel("normalized log10(wavelength)")
    fig.suptitle(
        "Spectra at the nearest available training-spectrum phase (GP +/- 1 sigma)\n"
        "all near-simultaneous spec phases overlaid in viridis; phot in red\n"
        "NB: axes are *normalized* until grid_norm_info JSON arrives",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def _make_phase_profile_figure(
    X_fill: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    yerr_train: np.ndarray,
    point_class: np.ndarray,
    save_path: str,
    n_wls: int = 6,
) -> None:
    """Phase profile (mu vs phase, +/- std band) at a few fixed wavelengths.

    Picks ``n_wls`` evenly-spaced wavelengths from X_fill and overlays the
    training points whose normalized wavelength is the closest of all
    prediction wavelengths.
    """
    wls_pred = np.unique(X_fill[:, 0])
    pick = wls_pred[np.linspace(0, wls_pred.size - 1, n_wls).astype(int)]

    fig, axes = plt.subplots(n_wls, 1, figsize=(10, 1.6 * n_wls), sharex=True)
    if n_wls == 1:
        axes = [axes]

    for ax, wls in zip(axes, pick):
        mask = X_fill[:, 0] == wls
        ph = X_fill[mask, 1]
        m = mu[mask]
        s = std[mask]
        order = np.argsort(ph)
        ph = ph[order]; m = m[order]; s = s[order]
        ax.fill_between(ph, m - s, m + s, color="steelblue", alpha=0.25, label="+/- 1 sigma")
        ax.plot(ph, m, color="steelblue", lw=1.0, label="mu")

        # Pick training points whose nearest wls (in the full X_fill grid) is this one.
        if X_train.size:
            nearest = wls_pred[np.argmin(np.abs(wls_pred[None, :] - X_train[:, [0]]), axis=1)]
            sel = nearest == wls
            if sel.any():
                for cls, marker, color in (
                    (gu.PHOT, "o", "red"),
                    (gu.SPEC, ".", "k"),
                ):
                    cm = sel & (point_class == cls)
                    if not cm.any():
                        continue
                    ax.errorbar(
                        X_train[cm, 1],
                        y_train[cm],
                        yerr=yerr_train[cm],
                        fmt=marker,
                        ms=3 if cls == gu.PHOT else 2,
                        lw=0.4,
                        elinewidth=0.4,
                        color=color,
                        alpha=0.7,
                    )
        ax.set_ylabel(f"wls={wls:.3f}", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("normalized log10(phase days)")
    fig.suptitle("Phase profiles at fixed wavelengths (mu +/- 1 sigma, training overlaid)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plot_results]   wrote {save_path}")
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="matern52_linear_opt")
    p.add_argument("--bundle", default=DEFAULT_BUNDLE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--spectrum-phases",
        type=str,
        default="-2 -1 0 0.5 1",
        help="space-separated list of normalized log10(phase) values for spectrum plots",
    )
    p.add_argument(
        "--spectrum-tolerance",
        type=float,
        default=0.05,
        help="tolerance (normalized log10(phase) units) for treating other "
             "spec training phases as 'near-simultaneous' to the chosen phase. "
             "All such spectra are overlaid in the spectrum panel.",
    )
    args = p.parse_args(argv)

    run_dir = os.path.join(args.output_dir, args.tag)
    pred_path = os.path.join(run_dir, "predictions.npz")
    config_path = os.path.join(run_dir, "config.json")
    figs_dir = os.path.join(run_dir, "figs")

    if not os.path.exists(pred_path):
        print(f"[plot_results] ERROR: {pred_path} not found. Run run_gp.py first.", file=sys.stderr)
        return 1
    _ensure_dir(figs_dir)

    print(f"[plot_results] loading {pred_path}")
    preds = np.load(pred_path, allow_pickle=False)
    X_fill = preds["X_fill"]
    mu = preds["mu"]
    std = preds["std"]
    point_class = preds["point_class_train"]
    sigma_eff = preds["sigma_eff_train"] if "sigma_eff_train" in preds.files else None
    mu_train = preds["mu_train"] if "mu_train" in preds.files else None

    print(f"[plot_results] loading {args.bundle}")
    bundle = np.load(args.bundle, allow_pickle=False)
    X = bundle["X"]
    y = bundle["y"]
    yerr = bundle["yerr"]

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        suptitle = (
            f"tag={args.tag} | log_lik={cfg.get('log_likelihood_at_compute', float('nan')):.2f} | "
            f"sigma_phot={cfg['config'].get('sigma_phot', float('nan')):.4g} | "
            f"sigma_spec={cfg['config'].get('sigma_spec', float('nan')):.4g}"
        )
    else:
        suptitle = f"tag={args.tag}"
    print(f"[plot_results] {suptitle}")

    print(f"[plot_results] mu range [{mu.min():.4g}, {mu.max():.4g}], "
          f"std range [{std.min():.4g}, {std.max():.4g}]")

    _make_wavelength_slice_figure(
        X_fill[:, 0], X_fill[:, 1], mu,
        X, y, yerr, point_class, GRID_NORM_INFO,
        use_log10_phase_axis=True,
        log_y=True,
        save_path=os.path.join(figs_dir, "gp_results_wavelength_slices.pdf"),
        suptitle=suptitle,
    )
    _make_wavelength_slice_figure(
        X_fill[:, 0], X_fill[:, 1], mu,
        X, y, yerr, point_class, GRID_NORM_INFO,
        use_log10_phase_axis=False,
        log_y=False,
        save_path=os.path.join(figs_dir, "gp_results_wavelength_slices_linear_phase_linear_flux.pdf"),
        suptitle=suptitle + " (linear phase placeholder)",
    )

    train_phases = np.unique(X[:, 1])
    _make_heatmap(
        X_fill, mu,
        title=f"GP posterior mean (mu) - {args.tag}",
        cbar_label="mu",
        save_path=os.path.join(figs_dir, "gp_mu_heatmap.png"),
        overlay_training_phases=train_phases,
    )
    _make_heatmap(
        X_fill, std,
        title=f"GP posterior std - {args.tag}",
        cbar_label="std",
        save_path=os.path.join(figs_dir, "gp_std_heatmap.png"),
        cmap="magma",
        overlay_training_phases=train_phases,
    )
    _make_training_coverage(X, y, point_class, X_fill, os.path.join(figs_dir, "training_coverage.png"))

    _make_phase_profile_figure(
        X_fill, mu, std, X, y, yerr, point_class,
        save_path=os.path.join(figs_dir, "gp_mu_phase_profiles.png"),
    )

    requested_phases = np.array(
        [float(s) for s in args.spectrum_phases.split() if s.strip()],
        dtype=float,
    )
    if requested_phases.size:
        _make_spectrum_figure(
            X_fill, mu, std, X, y, yerr, point_class,
            requested_phases=requested_phases,
            save_path=os.path.join(figs_dir, "gp_spectra.png"),
            near_sim_tol=args.spectrum_tolerance,
        )

    if mu_train is not None and sigma_eff is not None:
        _make_residual_histograms(
            y, mu_train, sigma_eff, point_class,
            save_path=os.path.join(figs_dir, "training_residuals.png"),
        )

    print("[plot_results] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
