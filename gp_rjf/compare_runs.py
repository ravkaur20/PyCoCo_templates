"""Compare two or more GP runs side-by-side.

Usage:
    python compare_runs.py matern32_nearest_baseline_jitter matern52_linear_opt
    python compare_runs.py --tags A B C --output runs/compare_ABC

Reads each ``runs/<tag>/predictions.npz`` + ``config.json``; writes
side-by-side heatmaps, overlaid phase profiles, and a summary table to
``runs/compare_<...>/`` (or the path supplied via ``--output``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "runs")


def _grid_from_X_fill(X_fill: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wls = np.unique(X_fill[:, 0])
    phases = np.unique(X_fill[:, 1])
    grid = np.full((wls.size, phases.size), np.nan)
    wls_idx = {v: i for i, v in enumerate(wls)}
    phase_idx = {v: i for i, v in enumerate(phases)}
    for k in range(X_fill.shape[0]):
        grid[wls_idx[X_fill[k, 0]], phase_idx[X_fill[k, 1]]] = values[k]
    return wls, phases, grid


def _strip_heatmap(
    runs: list[dict],
    field: str,
    cmap: str,
    title: str,
    save_path: str,
) -> None:
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    vmin = min(np.min(r[field]) for r in runs)
    vmax = max(np.max(r[field]) for r in runs)
    for ax, r in zip(axes, runs):
        wls, phases, grid = _grid_from_X_fill(r["X_fill"], r[field])
        im = ax.pcolormesh(phases, wls, grid, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(r["tag"], fontsize=10)
        ax.set_xlabel("normalized log10(phase days)")
    axes[0].set_ylabel("normalized log10(wavelength)")
    fig.suptitle(title)
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cax, label=field)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[compare_runs] wrote {save_path}")
    plt.close(fig)


def _overlaid_phase_profiles(
    runs: list[dict],
    save_path: str,
    n_wls: int = 6,
) -> None:
    ref = runs[0]
    wls_pred = np.unique(ref["X_fill"][:, 0])
    pick = wls_pred[np.linspace(0, wls_pred.size - 1, n_wls).astype(int)]
    fig, axes = plt.subplots(n_wls, 1, figsize=(11, 1.7 * n_wls), sharex=True)
    if n_wls == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for ax, wls in zip(axes, pick):
        for r, color in zip(runs, colors):
            mask = r["X_fill"][:, 0] == wls
            ph = r["X_fill"][mask, 1]
            mu = r["mu"][mask]
            std = r["std"][mask]
            order = np.argsort(ph)
            ph = ph[order]; mu = mu[order]; std = std[order]
            ax.fill_between(ph, mu - std, mu + std, color=color, alpha=0.12)
            ax.plot(ph, mu, color=color, lw=1.0, label=r["tag"])
        ax.set_ylabel(f"wls={wls:.3f}", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="upper right", ncol=min(len(runs), 3))
    axes[-1].set_xlabel("normalized log10(phase days)")
    fig.suptitle("Overlaid phase profiles (mu +/- 1 sigma) at fixed wavelengths")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[compare_runs] wrote {save_path}")
    plt.close(fig)


def _summary_table(runs: list[dict]) -> str:
    headers = [
        "tag",
        "log_lik",
        "sigma_phot",
        "sigma_spec",
        "metric_w",
        "metric_t",
        "extra_w",
        "extra_t",
        "chi2_phot",
        "chi2_spec",
    ]
    lines = ["\t".join(headers)]
    for r in runs:
        cfg = r.get("config", {})
        cfg_inner = cfg.get("config", {})
        extra_w = ""
        if cfg_inner.get("additive_w"):
            extra_w = (
                f"metric_w2={cfg_inner.get('metric_w2', float('nan')):.4g}, "
                f"w_short={cfg_inner.get('weight_w_short', float('nan')):.3f}"
            )
        extra_t = ""
        if cfg_inner.get("additive_t"):
            extra_t = (
                f"metric_t2={cfg_inner.get('metric_t2', float('nan')):.4g}, "
                f"w_short={cfg_inner.get('weight_t_short', float('nan')):.3f}"
            )
        row = [
            r["tag"],
            f"{cfg.get('log_likelihood_at_compute', float('nan')):.2f}",
            f"{cfg_inner.get('sigma_phot', float('nan')):.4g}",
            f"{cfg_inner.get('sigma_spec', float('nan')):.4g}",
            f"{cfg_inner.get('metric_w', float('nan')):.4g}",
            f"{cfg_inner.get('metric_t', float('nan')):.4g}",
            extra_w,
            extra_t,
            f"{cfg.get('chi2_per_n_phot', float('nan')):.3f}" if cfg.get('chi2_per_n_phot') is not None else "n/a",
            f"{cfg.get('chi2_per_n_spec', float('nan')):.3f}" if cfg.get('chi2_per_n_spec') is not None else "n/a",
        ]
        lines.append("\t".join(row))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tags", nargs="*", help="run tags (positional)")
    p.add_argument("--tags", dest="tags_flag", nargs="+", help="alternative way to pass tags")
    p.add_argument("--output", default=None)
    p.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    args = p.parse_args(argv)

    tags = args.tags or args.tags_flag or []
    if len(tags) < 2:
        print("[compare_runs] ERROR: need at least 2 tags", file=sys.stderr)
        return 1

    output = args.output or os.path.join(args.runs_dir, "compare_" + "_vs_".join(tags))
    os.makedirs(output, exist_ok=True)
    print(f"[compare_runs] output dir: {output}")

    runs = []
    for tag in tags:
        run_dir = os.path.join(args.runs_dir, tag)
        pred = os.path.join(run_dir, "predictions.npz")
        cfg = os.path.join(run_dir, "config.json")
        if not os.path.exists(pred):
            print(f"[compare_runs] missing {pred}", file=sys.stderr)
            return 1
        d = np.load(pred, allow_pickle=False)
        cfg_data = {}
        if os.path.exists(cfg):
            with open(cfg) as f:
                cfg_data = json.load(f)
        runs.append({
            "tag": tag,
            "X_fill": d["X_fill"],
            "mu": d["mu"],
            "std": d["std"],
            "config": cfg_data,
        })

    _strip_heatmap(
        runs, "mu", cmap="viridis",
        title="Posterior mean (mu)",
        save_path=os.path.join(output, "compare_mu_heatmaps.png"),
    )
    _strip_heatmap(
        runs, "std", cmap="magma",
        title="Posterior std",
        save_path=os.path.join(output, "compare_std_heatmaps.png"),
    )
    _overlaid_phase_profiles(
        runs,
        save_path=os.path.join(output, "compare_phase_profiles.png"),
    )

    table = _summary_table(runs)
    print(table)
    with open(os.path.join(output, "summary.tsv"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"[compare_runs] wrote {os.path.join(output, 'summary.tsv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
