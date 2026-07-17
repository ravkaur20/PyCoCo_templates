#!/usr/bin/env python3
"""Alternate ``run_gp`` with spectroscopic bundle rescaling against the GP fill-grid surface.

Each iteration:

1. Fits the GP (optionally warm-started from the previous iteration's ``config.json``).
2. Interpolates ``mu_raw`` or ``mu`` on ``X_fill`` to every spectroscopic training ``(x₁,x₂)``.
3. For each ``spec_bundle_id``, applies one linear flux multiplier (WLS vs the surface) using
   :func:`bundle_scale_pipeline.apply_epoch_linear_multiplier`, then writes the next bundle NPZ.

Logs JSON lines to ``<workspace>/iteration_log.jsonl`` and writes summary plots under
``<workspace>/metrics/``. Optional targeted overview panels via ``--diag-bundles``.

Example (set paths once, then run; only ``run_gp`` flags go after ``--``)::

    GP_ROOT="$(pwd)"
    BUNDLE="${GP_ROOT}/gp_work_scaled_nophot_m8767_m8217.npz"
    META="${GP_ROOT}/gp_scaled_bundle_meta.json"
    WORKSPACE="${GP_ROOT}/runs/my_surface_iter"
    RUNS_DIR="${GP_ROOT}/runs"

    python iterate_gp_surface_bundle_scale.py \\
        --input-bundle "${BUNDLE}" \\
        --meta "${META}" \\
        --workspace "${WORKSPACE}" \\
        --runs-dir "${RUNS_DIR}" \\
        --gp-tag-prefix mysurf \\
        --max-iters 20 \\
        --diag-bundles 3,5 \\
        --diag-full-overview-interval 5 \\
        --plot-results-each-iter \\
        -- --additive-time --additive-wls \\
        --kernel-time matern52 --kernel-wls matern52 --mean linear \\
        --meta-json "${META}"

See also ``docs/RUNNING_MY_SURFACE_ITER.md`` for a full runbook.

Only arguments accepted by ``run_gp.py`` may appear **after** a lone ``--``.
Driver flags (``--diag-bundles``, ``--diag-full-overview-interval``, ``--plot-results-each-iter``, etc.)
must appear **before** ``--``. Summary plots under ``<workspace>/metrics/`` are written when the
outer loop exits; ``iteration_log.jsonl`` grows each iteration. ``run_gp`` outputs live under
``<runs-dir>/<gp-tag-prefix>_kNN>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

import bundle_meta as bmeta
import bundle_scale_pipeline as bsp
import gp_grid_interp as ggi
import gp_utils as gu
from plot_results import linear_flux_yerr, scaled_ln_to_linear

HERE = os.path.dirname(os.path.abspath(__file__))


def _copy_npz(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)


def _load_bundle_arrays(path: str) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    try:
        return {k: np.asarray(z[k]) for k in z.files}
    finally:
        z.close()


def _save_npz(path: str, data: dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **data)


def _parse_int_set(s: Optional[str]) -> Optional[set[int]]:
    if not s or not str(s).strip():
        return None
    return {int(x.strip()) for x in str(s).split(",") if x.strip()}


def _strip_misplaced_iterate_flags_from_run_gp_extra(extra: list[str]) -> list[str]:
    """Remove iterate-driver-only flags if the user put them after ``--`` (run_gp would reject them)."""
    drop_next_value = {"--diag-full-overview-interval"}
    drop_alone = {"--plot-results-each-iter"}
    out: list[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok in drop_alone:
            print(
                f"[surface-iter] WARN: removed {tok!r} from run_gp arguments "
                "(it belongs on iterate_gp_surface_bundle_scale.py *before* a lone `--`).",
                file=sys.stderr,
            )
            i += 1
            continue
        if tok in drop_next_value:
            nxt = extra[i + 1] if i + 1 < len(extra) else None
            print(
                f"[surface-iter] WARN: removed {tok!r} and value {nxt!r} from run_gp arguments "
                f"(pass it on this driver before a lone `--`).",
                file=sys.stderr,
            )
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def _surface_bundle_scales(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    point_class: np.ndarray,
    spec_bundle_id: np.ndarray,
    X_fill: np.ndarray,
    mu_fill: np.ndarray,
    gn: dict,
    *,
    clip_max: float,
) -> tuple[dict[int, float], int]:
    """Return (bundle_id -> linear multiplier), n_nn_interp_fallback."""
    spec_m = point_class == gu.SPEC
    sb = np.asarray(spec_bundle_id, dtype=np.int32).ravel()
    mu_lat, n_nn = ggi.interp_latent_gp_at_fill_rows(X_fill, mu_fill, X)
    bids = sorted({int(b) for b in np.unique(sb[spec_m]).tolist() if int(b) >= 0})
    scales: dict[int, float] = {}
    for b in bids:
        R = np.flatnonzero(spec_m & (sb == int(b)))
        if R.size < 2:
            scales[b] = 1.0
            continue
        f_lin = scaled_ln_to_linear(y[R], gn)
        mu_lin = scaled_ln_to_linear(mu_lat[R], gn)
        sig = linear_flux_yerr(y[R], yerr[R], gn)
        w = 1.0 / np.maximum(sig * sig, 1e-60)
        num = float(np.sum(w * f_lin * mu_lin))
        den = float(np.sum(w * f_lin * f_lin))
        if den <= 1e-60 or not np.isfinite(num):
            m = 1.0
        else:
            m = num / den
        m = float(np.clip(m, 1.0 / float(clip_max), float(clip_max)))
        scales[int(b)] = m
    return scales, int(n_nn)


def _apply_bundle_scales_inplace(
    data: dict[str, np.ndarray],
    gn: dict,
    point_class: np.ndarray,
    spec_bundle_id: np.ndarray,
    scales: dict[int, float],
) -> None:
    y = np.asarray(data["y"], dtype=float).copy()
    ye = np.asarray(data["yerr"], dtype=float).copy()
    sb = np.asarray(spec_bundle_id, dtype=np.int32).ravel()
    spec_m = point_class == gu.SPEC
    for b, mult in scales.items():
        if abs(float(np.log(mult))) < 1e-15:
            continue
        R = np.flatnonzero(spec_m & (sb == int(b)))
        if R.size == 0:
            continue
        y, ye = bsp.apply_epoch_linear_multiplier(y, ye, gn, R, mult=float(mult))
    data["y"] = y
    data["yerr"] = ye


def _read_run_metrics(cfg_path: str) -> dict[str, Any]:
    with open(cfg_path, encoding="utf-8") as f:
        cj = json.load(f)
    inner = cj.get("config", {})
    n_phot = cj.get("n_phot")
    n_spec = cj.get("n_spec")
    n_train = None
    if isinstance(n_phot, int) and isinstance(n_spec, int):
        n_train = int(n_phot) + int(n_spec)
    return {
        "chi2_per_n_total": cj.get("chi2_per_n_total"),
        "chi2_per_n_phot": cj.get("chi2_per_n_phot"),
        "chi2_per_n_spec": cj.get("chi2_per_n_spec"),
        "log_likelihood_at_compute": cj.get("log_likelihood_at_compute"),
        "n_phot": n_phot,
        "n_spec": n_spec,
        "n_train": n_train,
        "config": dict(inner) if isinstance(inner, dict) else {},
    }


def _write_metrics_plots(
    records: list[dict[str, Any]],
    out_dir: str,
    *,
    highlight_bundles: set[int],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    it = np.arange(len(records), dtype=int)
    chi_t = np.array([float(r["chi2_per_n_total"]) for r in records])
    chi_s = np.array([float(r["chi2_per_n_spec"]) for r in records])
    ll = np.array([float(r["log_likelihood_at_compute"]) for r in records])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(it, chi_t, "o-", label=r"$\chi^2/N$ total")
    ax.plot(it, chi_s, "s-", label=r"$\chi^2/N$ spec")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\chi^2/N$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chi2_vs_iter.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(it, ll, "o-", color="darkgreen")
    ax.set_xlabel("iteration")
    ax.set_ylabel("log likelihood (full train compute)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loglik_vs_iter.png"), dpi=150)
    plt.close(fig)

    keys = [
        ("metric_t", "metric_t"),
        ("metric_w", "metric_w"),
        ("metric_t2", "metric_t2"),
        ("metric_w2", "metric_w2"),
        ("weight_t_short", "weight_t_short"),
        ("weight_w_short", "weight_w_short"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = np.asarray(axes).ravel()
    for ax, (k, _) in zip(axes, keys):
        ys = []
        for r in records:
            v = r.get("config", {}).get(k)
            ys.append(float(v) if v is not None and np.isfinite(float(v)) else float("nan"))
        ax.plot(it, np.asarray(ys, dtype=float), "o-", ms=3)
        ax.set_title(k)
        ax.set_xlabel("iter")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lengthscales_vs_iter.png"), dpi=150)
    plt.close(fig)

    all_bids: set[int] = set()
    for r in records:
        all_bids.update(int(x) for x in r.get("bundle_scales", {}).keys())
    if all_bids:
        fig, ax = plt.subplots(figsize=(10, 5))
        for b in sorted(all_bids):
            ys = [float(r["bundle_scales"].get(str(b), r["bundle_scales"].get(b, 1.0))) for r in records]
            lw = 2.2 if b in highlight_bundles else 0.9
            al = 0.95 if b in highlight_bundles else 0.35
            ax.plot(it, ys, "-o", ms=3, lw=lw, alpha=al, label=f"bundle {b}")
        ax.axhline(1.0, color="k", ls="--", lw=0.6)
        ax.set_xlabel("iteration")
        ax.set_ylabel("linear scale factor (this iter)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bundle_linear_scale_vs_iter.png"), dpi=150)
        plt.close(fig)

    if highlight_bundles:
        fig, axes = plt.subplots(1, len(highlight_bundles), figsize=(5 * len(highlight_bundles), 4), squeeze=False)
        axes = axes.ravel()
        cum: dict[int, float] = {b: 1.0 for b in highlight_bundles}
        for ax, b in zip(axes, sorted(highlight_bundles)):
            cvs = []
            for r in records:
                m = float(r["bundle_scales"].get(str(b), r["bundle_scales"].get(b, 1.0)))
                cum[b] *= m
                cvs.append(cum[b])
            ax.semilogy(it, np.asarray(cvs, dtype=float), "o-", color="navy")
            ax.set_title(f"bundle {b} cumulative linear scale")
            ax.set_xlabel("iteration")
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "highlight_bundles_cumulative_scale.png"), dpi=150)
        plt.close(fig)


def _write_per_epoch_bundle_scaling(
    workspace: str,
    bundle_path_iter0: str,
    highlight_bundles: set[int],
    records: list[dict[str, Any]],
) -> None:
    """CSV + evolution plots for each (bundle, spec-epoch) in highlight bundles (plan § diagnostics)."""
    if not highlight_bundles or not records:
        return
    bd = _load_bundle_arrays(bundle_path_iter0)
    X = np.asarray(bd["X"], dtype=float)
    obs = bd.get("train_obs_class")
    pc = gu.effective_point_class(
        X,
        threshold=50,
        train_obs_class=np.asarray(obs) if obs is not None else None,
    )
    if "spec_bundle_id" not in bd:
        return
    sbid = np.asarray(bd["spec_bundle_id"], dtype=np.int32).ravel()
    spec_m = pc == gu.SPEC
    canon_phases, eor = bsp.unique_spec_epochs(X, spec_m)
    scale_dir = os.path.join(workspace, "scaling")
    os.makedirs(scale_dir, exist_ok=True)

    for b in sorted(highlight_bundles):
        R = np.flatnonzero(spec_m & (sbid == int(b)))
        if R.size == 0:
            continue
        epoch_ids = sorted({int(eor[i]) for i in R})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        it = np.arange(len(records), dtype=int)
        for epi in epoch_ids:
            rows_csv: list[str] = [
                "iteration,linear_scale_this_iter,cumulative_linear_scale,canonical_phase,spec_bundle_id,spec_epoch_id\n"
            ]
            phase = float(canon_phases[epi]) if epi < canon_phases.size else float("nan")
            cum = 1.0
            ys: list[float] = []
            for r in records:
                k = int(r["iteration"])
                m = float(r["bundle_scales"].get(str(b), r["bundle_scales"].get(b, 1.0)))
                cum *= m
                ys.append(cum)
                rows_csv.append(f"{k},{m:.16e},{cum:.16e},{phase:.9g},{b},{epi}\n")
            name = f"cumulative_scale_bundle_{b}_epoch_{epi:04d}.csv"
            with open(os.path.join(scale_dir, name), "w", encoding="utf-8") as cf:
                cf.writelines(rows_csv)
            ax.semilogy(
                it,
                np.asarray(ys, dtype=float),
                "-o",
                ms=3,
                alpha=0.85,
                label=f"ep {epi} (x₂≈{phase:.3f})",
            )
        ax.axhline(1.0, color="k", ls="--", lw=0.6)
        ax.set_xlabel("iteration")
        ax.set_ylabel("cumulative linear scale (product over iters)")
        ax.set_title(f"spec_bundle {b}: per-epoch cumulative scale")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(scale_dir, f"spec_bundle_{b}_scale_evolution.png"), dpi=150)
        plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Arguments after a lone `--` are passed only to run_gp.py. "
            "Use --diag-bundles, --diag-full-overview-interval, --plot-results-each-iter, "
            "and other driver options *before* `--`."
        ),
    )
    p.add_argument("--input-bundle", "-i", required=True, help="scaled training NPZ (starting point)")
    p.add_argument("--meta", required=True, help="grid_norm JSON, e.g. gp_scaled_bundle_meta.json")
    p.add_argument(
        "--workspace",
        "-w",
        required=True,
        help="directory for iter_XX/bundle.npz, logs, and metrics (created)",
    )
    p.add_argument(
        "--runs-dir",
        default=os.path.join(HERE, "runs"),
        help="where run_gp writes runs/<gp_tag>/ (default: ./runs)",
    )
    p.add_argument(
        "--gp-tag-prefix",
        default="surf_iter",
        help="run_gp tag will be <prefix>_k{iteration:02d}",
    )
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--bundle-scale-clip", type=float, default=10.0, help="clip linear multiplier to [1/C, C]")
    p.add_argument(
        "--converge-max-log-scale",
        type=float,
        default=5e-4,
        help="stop if max_b |log m_b| < this after an iteration's scale solve",
    )
    p.add_argument(
        "--converge-delta-chi2-spec",
        type=float,
        default=5e-4,
        help="also require |Δ chi2_spec/N| < this vs previous iteration",
    )
    p.add_argument(
        "--surface-mu-key",
        choices=("mu_raw", "mu"),
        default="mu_raw",
        help="fill-grid latent vector for surface interpolation (mu_raw avoids mono/blue warp)",
    )
    p.add_argument("--diag-bundles", default=None, metavar="IDS", help="comma spec_bundle ids for overview subset")
    p.add_argument(
        "--diag-full-overview-interval",
        type=int,
        default=0,
        help="if >0, run full plot_bands_gp_overview every N iterations (0=never; phot+all spec panels)",
    )
    p.add_argument(
        "--plot-results-each-iter",
        action="store_true",
        help="also run plot_results.py for each run_gp tag",
    )
    p.add_argument(
        "--run-gp-max-iter",
        type=int,
        default=60,
        help="forwarded as run_gp --max-iter unless overridden after --",
    )
    p.add_argument(
        "run_gp_argv",
        nargs=argparse.REMAINDER,
        default=[],
        help="optional: pass -- then run_gp.py flags (e.g. -- --additive-time --additive-wls ...)",
    )
    args = p.parse_args(argv)
    extra = list(args.run_gp_argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    extra = _strip_misplaced_iterate_flags_from_run_gp_extra(extra)

    inp = os.path.abspath(os.path.expanduser(args.input_bundle))
    meta = os.path.abspath(os.path.expanduser(args.meta))
    workspace = os.path.abspath(os.path.expanduser(args.workspace))
    runs_dir = os.path.abspath(os.path.expanduser(args.runs_dir))
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(os.path.join(workspace, "metrics"), exist_ok=True)

    if not os.path.isfile(inp):
        print(f"ERROR: input bundle not found: {inp!r}", file=sys.stderr)
        return 2
    if not os.path.isfile(meta):
        print(f"ERROR: meta not found: {meta!r}", file=sys.stderr)
        return 2

    gn = bmeta.grid_norm_from_bundle_or_meta(inp, meta_path=meta)
    if gn.get("_normalized_only"):
        print("ERROR: meta must provide grid_norm_info", file=sys.stderr)
        return 2

    diag_bundles = _parse_int_set(args.diag_bundles)
    highlight = diag_bundles if diag_bundles else {3, 5}

    iter0 = os.path.join(workspace, "iter_00", "bundle.npz")
    _copy_npz(inp, iter0)

    print(
        f"[surface-iter] workspace={workspace!r} (iter_XX bundles, iteration_log.jsonl, "
        f"metrics/*.png after loop ends, scaling/ after loop ends)",
        flush=True,
    )
    print(
        f"[surface-iter] run_gp output dirs: {runs_dir!r} / {args.gp_tag_prefix}_kNN",
        flush=True,
    )
    if not diag_bundles:
        print(
            "[surface-iter] note: no --diag-bundles → no per-iter plot_bands_gp_overview under "
            "iter_XX/figs/ (JSONL still updates each iter).",
            flush=True,
        )

    records: list[dict[str, Any]] = []
    log_path = os.path.join(workspace, "iteration_log.jsonl")
    with open(log_path, "w", encoding="utf-8"):
        pass
    prev_chi_spec: Optional[float] = None
    py = sys.executable
    run_gp_py = os.path.join(HERE, "run_gp.py")

    for k in range(int(args.max_iters)):
        bundle_path = os.path.join(workspace, f"iter_{k:02d}", "bundle.npz")
        gp_tag = f"{args.gp_tag_prefix}_k{k:02d}"
        run_dir = os.path.join(runs_dir, gp_tag)
        cfg_path = os.path.join(run_dir, "config.json")
        pred_path = os.path.join(run_dir, "predictions.npz")

        cmd = [
            py,
            run_gp_py,
            "-i",
            bundle_path,
            "--meta-json",
            meta,
            "-o",
            runs_dir,
            "-t",
            gp_tag,
        ]
        if "--max-iter" not in extra:
            cmd += ["--max-iter", str(int(args.run_gp_max_iter))]
        if k > 0:
            wp = os.path.join(runs_dir, f"{args.gp_tag_prefix}_k{k-1:02d}", "config.json")
            if os.path.isfile(wp):
                cmd += ["--warm-start-config-json", wp]
        if not any(x == "--additive-time" for x in extra):
            cmd.append("--additive-time")
        if not any(x == "--additive-wls" for x in extra):
            cmd.append("--additive-wls")
        if not any(x == "--kernel-time" for x in extra):
            cmd += ["--kernel-time", "matern52"]
        if not any(x == "--kernel-wls" for x in extra):
            cmd += ["--kernel-wls", "matern52"]
        if not any(x == "--mean" for x in extra):
            cmd += ["--mean", "linear"]
        cmd += extra

        print("[surface-iter]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        shutil.copy2(pred_path, os.path.join(workspace, f"iter_{k:02d}", "predictions.npz"))
        if os.path.isfile(cfg_path):
            shutil.copy2(cfg_path, os.path.join(workspace, f"iter_{k:02d}", "config.json"))

        metrics = _read_run_metrics(cfg_path)
        pr = np.load(pred_path, allow_pickle=False)
        try:
            X_fill = np.asarray(pr["X_fill"], dtype=float)
            mu_key = str(args.surface_mu_key)
            if mu_key not in pr.files:
                print(f"ERROR: predictions missing {mu_key!r}", file=sys.stderr)
                return 2
            mu_fill = np.asarray(pr[mu_key], dtype=float).ravel()
        finally:
            pr.close()

        bd = _load_bundle_arrays(bundle_path)
        X = np.asarray(bd["X"], dtype=float)
        y = np.asarray(bd["y"], dtype=float)
        yerr = np.asarray(bd["yerr"], dtype=float)
        obs = bd.get("train_obs_class")
        pc = gu.effective_point_class(
            X,
            threshold=50,
            train_obs_class=np.asarray(obs) if obs is not None else None,
        )
        if "spec_bundle_id" not in bd:
            print("ERROR: bundle lacks spec_bundle_id", file=sys.stderr)
            return 2
        sbid = np.asarray(bd["spec_bundle_id"], dtype=np.int32).ravel()

        scales, n_nn = _surface_bundle_scales(
            X,
            y,
            yerr,
            pc,
            sbid,
            X_fill,
            mu_fill,
            gn,
            clip_max=float(args.bundle_scale_clip),
        )
        max_log = max(abs(float(np.log(v))) for v in scales.values()) if scales else 0.0
        chi_spec = float(metrics.get("chi2_per_n_spec") or float("nan"))
        dchi = abs(chi_spec - prev_chi_spec) if prev_chi_spec is not None and np.isfinite(chi_spec) else float("inf")

        rec: dict[str, Any] = {
            "iteration": k,
            "gp_tag": gp_tag,
            "bundle_scales": {str(b): float(v) for b, v in sorted(scales.items())},
            "max_abs_log_scale": float(max_log),
            "n_nn_interp_fallback": int(n_nn),
            **metrics,
        }
        records.append(rec)
        with open(log_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(rec, default=str) + "\n")
        print(
            f"[surface-iter] iteration {k} logged → {log_path!r} ; run_gp dir → {run_dir!r}",
            flush=True,
        )

        if diag_bundles:
            ov_dir = os.path.join(workspace, f"iter_{k:02d}", "figs", "overview")
            ob = [
                py,
                os.path.join(HERE, "plot_bands_gp_overview.py"),
                "--bundle",
                bundle_path,
                "--meta",
                meta,
                "--predictions",
                pred_path,
                "--output-dir",
                ov_dir,
                "--expect-pipeline-bundle",
                "--only-spec-bundle-ids",
                ",".join(str(b) for b in sorted(diag_bundles)),
            ]
            print("[surface-iter]", " ".join(ob))
            subprocess.run(ob, check=True)

        full_iv = int(args.diag_full_overview_interval)
        if full_iv > 0 and (k % full_iv == 0 or k == int(args.max_iters) - 1):
            ov_full = os.path.join(workspace, f"iter_{k:02d}", "figs", "overview_full")
            obf = [
                py,
                os.path.join(HERE, "plot_bands_gp_overview.py"),
                "--bundle",
                bundle_path,
                "--meta",
                meta,
                "--predictions",
                pred_path,
                "--output-dir",
                ov_full,
                "--expect-pipeline-bundle",
            ]
            print("[surface-iter]", " ".join(obf))
            subprocess.run(obf, check=True)

        if bool(args.plot_results_each_iter):
            pr_cmd = [
                py,
                os.path.join(HERE, "plot_results.py"),
                "--tag",
                gp_tag,
                "--bundle",
                bundle_path,
                "--meta",
                meta,
                "--heatmap-raw",
            ]
            print("[surface-iter]", " ".join(pr_cmd))
            subprocess.run(pr_cmd, check=True)

        converged = (
            max_log < float(args.converge_max_log_scale)
            and prev_chi_spec is not None
            and dchi < float(args.converge_delta_chi2_spec)
        )
        if np.isfinite(chi_spec):
            prev_chi_spec = chi_spec

        if converged:
            print(f"[surface-iter] convergence at k={k} (max|log m|={max_log:g}, |Δχ²_spec/N|={dchi:g})")
            break

        if k == int(args.max_iters) - 1:
            print("[surface-iter] reached max-iters without strict convergence")
            break

        data = _load_bundle_arrays(bundle_path)
        _apply_bundle_scales_inplace(data, gn, pc, sbid, scales)
        next_dir = os.path.join(workspace, f"iter_{k+1:02d}", "bundle.npz")
        _save_npz(next_dir, data)

    _write_per_epoch_bundle_scaling(workspace, iter0, highlight, records)
    _write_metrics_plots(records, os.path.join(workspace, "metrics"), highlight_bundles=highlight)
    print(f"[surface-iter] wrote metrics under {os.path.join(workspace, 'metrics')!r}")
    print(f"[surface-iter] log {log_path!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
