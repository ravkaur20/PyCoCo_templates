#!/usr/bin/env python3
"""End-to-end pipeline: optional preprocess → bundle scale → GP → diagnostics.

Stages
------
1. **Preprocess** (optional): ``--corrections-json`` runs ``bundle_preprocess``; or
   ``--skip-preprocess`` uses ``-i`` unchanged (e.g. already-fixed ``gp_bundle_collab_fixes.npz``).
2. **bundle_scale_pipeline**: time-bundles spectra (``--max-bundle-minutes``), **intra-epoch λ-arm**
   overlap+seam scaling (fixes jumps when a single exposure has multiple orders/chips), then
   inter-epoch intra-bundle overlap+seam χ² scaling (**relative** scaling), then by default **global**
   photometric anchoring (``--global-scale-iters`` **1**): **band + inner GP** when both **enrich**
   npz and **filter YAML** exist; otherwise **rough / pooled χ²** anchoring from photometry rows in
   ``X`` (**no enrich required**). Use ``--skip-global-phot-anchor`` or ``--global-scale-iters 0`` for
   relative-only tests.
3. **run_gp.py**: final GP on photometry + **relative- and absolute-scaled** spectra (writes ``runs/<tag>/``).
4. **Plots** (same suite as before): ``plot_results.py``, ``plot_bands_gp_overview.py``,
   ``outlier_pipeline.py``, ``plot_outliers.py``.  Spectral bundle panels default to
   ``--spec-phase-decimals 9`` so nearly-coincident spectroscopic epochs are **not** merged
   into one λ-sorted polyline (which looked like broken relative scaling).

Example (quick test: **relative scaling only**; no photometric anchor)::

    python run_full_pipeline.py \\
        -i gp_bundle_collab_fixes.npz --skip-preprocess \\
        --skip-global-phot-anchor \\
        -t my_run --output-prefix gp_work

Example (default: **phot anchor on** without enrich — rough / pooled χ² uses phot rows in ``X``)::

    python run_full_pipeline.py \\
        -i gp_bundle_collab_fixes.npz --skip-preprocess \\
        -t my_run --output-prefix gp_work

Example (optional **band** anchor: enrich + filter YAML; two global anchor iterations)::

    python run_full_pipeline.py \\
        -i gp_bundle_collab_fixes.npz --skip-preprocess \\
        --enrich enrich.npz \\
        --filter-config configs/filter_pipeline.example.yaml \\
        --global-scale-iters 2 \\
        -t my_run --output-prefix gp_work

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence

import bundle_scale_pipeline as bsp


HERE = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: Sequence[str]) -> None:
    print("[full-pipeline]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _abs_existing(path: str | None) -> str | None:
    if not path or not str(path).strip():
        return None
    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    return p if os.path.isfile(p) else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", default=os.path.join(HERE, "gp_minimal_bundle.npz"))
    p.add_argument("--tag", "-t", default="my_final_run")
    p.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--output-prefix", default="gp_pipeline")
    p.add_argument("--meta", default=None, help="passed to bundle_scale_pipeline and overview plots")
    p.add_argument("--bundle-minutes", type=float, default=5.0)
    p.add_argument("--z-threshold", type=float, default=5.0)
    p.add_argument("--min-spec-rows-per-phase", type=int, default=32)
    p.add_argument("--phot-spec-threshold", type=int, default=50)

    p.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="use -i directly as bundle_scale input (e.g. gp_bundle_collab_fixes.npz)",
    )
    p.add_argument(
        "--corrections-json",
        default=None,
        help="if set, run bundle_preprocess with this JSON (overrides --skip-preprocess for the first step)",
    )

    p.add_argument("--skip-optimize", action="store_true")
    p.add_argument("--kernel-time", default="matern52")
    p.add_argument("--kernel-wls", default="matern52")

    p.add_argument(
        "--enrich",
        default=None,
        help="training-row-aligned enrich npz; if omitted, same auto-discovery as bundle_scale_pipeline "
        "(<input_stem>_enrich.npz or enrich.npz beside the bundle).",
    )
    p.add_argument(
        "--filter-config",
        default=None,
        help="YAML with TRDS roots + band aliases; if omitted, uses configs/filter_pipeline*.yaml "
        "in the repo when present.",
    )
    p.add_argument(
        "--global-scale-iters",
        type=int,
        default=1,
        help="synthetic-phot global iterations forwarded to bundle_scale_pipeline (default 1). "
        "Use 0 for relative-only scaling. Values >0 run phot anchor: band+synth only if enrich and "
        "filter YAML are both present; otherwise rough / pooled χ² (no enrich required).",
    )
    p.add_argument(
        "--skip-global-phot-anchor",
        action="store_true",
        help="relative-only: forwards --skip-global-phot-anchor to bundle_scale_pipeline (disables phot anchor)",
    )
    p.add_argument(
        "--phase-tolerance-norm-global",
        type=float,
        default=0.06,
        help="bundle_scale_pipeline: legacy knob (phot anchor now uses per-band GP at each spec epoch's phase)",
    )
    p.add_argument(
        "--gp-tag-prefix",
        default="bscale",
        help="tag prefix for internal GP runs during global scaling (bundle_scale_pipeline)",
    )
    p.add_argument(
        "--seam-fit-half-width-aa",
        type=float,
        default=50.0,
        help="Å half-width for seam linear fits in bundle_scale_pipeline (default 50 ≈ 100 Å window)",
    )
    p.add_argument(
        "--lw-short",
        type=float,
        default=0.02,
        help="warm-start short wavelength metric when using --additive-wls (run_gp and global-scale inner GP)",
    )
    p.add_argument(
        "--phot-lc-time-step-days",
        type=float,
        default=0.05,
        help="plot_bands_gp_overview: Δt for dense phot GP curves (MJD or phase days; 0 = legacy train-row segments)",
    )
    p.add_argument(
        "--phot-pseudo-grouping",
        choices=("rounded", "unique_x1"),
        default="rounded",
        help="plot_bands_gp_overview: pseudo-band panels when enrich has no band labels (display only)",
    )
    p.add_argument(
        "--phot-unique-x1-decimals",
        type=int,
        default=12,
        help="plot_bands_gp_overview: key rounding for unique_x1 grouping (-1 = exact floats)",
    )
    p.add_argument(
        "--phot-max-unique-panels-warn",
        type=int,
        default=500,
        help="plot_bands_gp_overview: stderr warn threshold for many unique_x1 panels (0 = off)",
    )
    p.add_argument("--arm-gap-factor", type=float, default=35.0)
    p.add_argument("--arm-min-gap-norm", type=float, default=3e-3)

    ns = p.parse_args(argv)

    py = sys.executable
    inp = os.path.abspath(os.path.expanduser(ns.input))
    runs_dir = os.path.abspath(os.path.expanduser(ns.runs_dir))
    stem = ns.output_prefix
    pre = os.path.abspath(f"{stem}_preprocessed.npz")
    scaled = os.path.abspath(f"{stem}_scaled.npz")
    meta_arg = _abs_existing(ns.meta)

    os.makedirs(runs_dir, exist_ok=True)

    work_bundle = inp
    if ns.corrections_json:
        cj = os.path.abspath(os.path.expanduser(str(ns.corrections_json).strip()))
        if not os.path.isfile(cj):
            print(f"[full-pipeline] ERROR: corrections json not found: {cj!r}", file=sys.stderr)
            return 2
        cmd_pre = [
            py,
            os.path.join(HERE, "bundle_preprocess.py"),
            "preprocess",
            "--input",
            inp,
            "--output",
            pre,
            "--corrections-json",
            cj,
        ]
        _run(cmd_pre)
        work_bundle = pre
    elif not ns.skip_preprocess:
        print(
            "[full-pipeline] NOTE: no preprocess (--corrections-json) and not --skip-preprocess; "
            "using input bundle as-is. Use --skip-preprocess when input is already corrected.",
            file=sys.stderr,
        )

    enrich_abs = _abs_existing(ns.enrich) or bsp.discover_enrich_npz(work_bundle, None)
    filt_abs = _abs_existing(ns.filter_config) or bsp.discover_filter_config_yaml(None)
    if enrich_abs and not _abs_existing(ns.enrich):
        print(
            f"[full-pipeline] NOTE: auto-discovered enrich npz {enrich_abs!r}",
            file=sys.stderr,
        )
    if filt_abs and not _abs_existing(ns.filter_config):
        print(
            f"[full-pipeline] NOTE: auto-discovered filter config {filt_abs!r}",
            file=sys.stderr,
        )

    gscale = int(ns.global_scale_iters)
    skip_ga = bool(ns.skip_global_phot_anchor)
    if skip_ga:
        if gscale > 0:
            print(
                "[full-pipeline] NOTE: --skip-global-phot-anchor → relative-only "
                "(ignoring --global-scale-iters for photometric anchoring).",
                file=sys.stderr,
            )
        gscale = 0
    elif not skip_ga and gscale < 1:
        gscale = 1
        if enrich_abs and filt_abs:
            print(
                "[full-pipeline] NOTE: enrich + filter config present; using --global-scale-iters=1 "
                "(band+synth photometric anchor). Pass --skip-global-phot-anchor for relative-only spec.",
                file=sys.stderr,
            )
        else:
            print(
                "[full-pipeline] NOTE: using --global-scale-iters=1 (photometric anchor). "
                "Without enrich + filter YAML, bundle_scale_pipeline uses rough / pooled χ² "
                "photometry anchoring only (no enrich required).",
                file=sys.stderr,
            )

    if gscale > 0 and (enrich_abs is None or filt_abs is None):
        print(
            "[full-pipeline] NOTE: enrich npz and/or filter YAML missing — global phot anchor "
            "uses training photometry vs spectra (rough / pooled χ²); inner run_gp for "
            "band synthesis is skipped.",
            file=sys.stderr,
        )
    if gscale < 1 and not skip_ga:
        print(
            "[full-pipeline] WARNING: final GP trains on **relative-only** scaled spectroscopy "
            "(--global-scale-iters 0).",
            file=sys.stderr,
        )

    bscale_cmd: list[str] = [
        py,
        os.path.join(HERE, "bundle_scale_pipeline.py"),
        "--input",
        work_bundle,
        "--output",
        scaled,
        "--max-bundle-minutes",
        str(float(ns.bundle_minutes)),
        "--overlap-grid-points",
        "256",
        "--seam-weight",
        "1.0",
        "--phot-spec-threshold",
        str(int(ns.phot_spec_threshold)),
        "--seam-fit-half-width-aa",
        str(float(ns.seam_fit_half_width_aa)),
        "--arm-gap-factor",
        str(float(ns.arm_gap_factor)),
        "--arm-min-gap-norm",
        str(float(ns.arm_min_gap_norm)),
    ]
    if meta_arg:
        bscale_cmd += ["--meta", meta_arg]

    bscale_cmd += ["--global-scale-iters", str(int(gscale))]
    if skip_ga:
        bscale_cmd.append("--skip-global-phot-anchor")

    if gscale > 0 and enrich_abs and filt_abs:
        bscale_cmd += [
            "--enrich",
            enrich_abs,
            "--filter-config",
            filt_abs,
            "--phase-tolerance-norm-global",
            str(float(ns.phase_tolerance_norm_global)),
            "--run-gp",
            "--runs-dir",
            runs_dir,
            "--gp-tag-prefix",
            str(ns.gp_tag_prefix),
            "--",
            "--kernel-time",
            str(ns.kernel_time),
            "--kernel-wls",
            str(ns.kernel_wls),
            "--mean",
            "linear",
            "--additive-time",
            "--additive-wls",
            "--max-iter",
            "60",
            "--lw-short",
            str(float(ns.lw_short)),
        ]
        if ns.skip_optimize:
            bscale_cmd.append("--no-optimize")
        else:
            bscale_cmd.append("--optimize")

    _run(bscale_cmd)

    gp_cmd = [
        py,
        os.path.join(HERE, "run_gp.py"),
        "--input",
        scaled,
        "--output-dir",
        runs_dir,
        "--tag",
        str(ns.tag),
        "--kernel-time",
        str(ns.kernel_time),
        "--kernel-wls",
        str(ns.kernel_wls),
        "--mean",
        "linear",
        "--additive-time",
        "--additive-wls",
        "--phot-spec-threshold",
        str(int(ns.phot_spec_threshold)),
        "--lw-short",
        str(float(ns.lw_short)),
        "--max-iter",
        "60",
    ]
    if ns.skip_optimize:
        gp_cmd.append("--no-optimize")
    else:
        gp_cmd.append("--optimize")
    _run(gp_cmd)

    plot_res = [
        py,
        os.path.join(HERE, "plot_results.py"),
        "--tag",
        str(ns.tag),
        "--bundle",
        scaled,
        "--output-dir",
        runs_dir,
    ]
    if meta_arg:
        plot_res += ["--meta", meta_arg]
    _run(plot_res)

    overview = [
        py,
        os.path.join(HERE, "plot_bands_gp_overview.py"),
        "--bundle",
        scaled,
        "--tag",
        str(ns.tag),
        "--output-dir",
        os.path.join(runs_dir, str(ns.tag), "figs", "overview"),
        "--expect-pipeline-bundle",
        "--plot-residuals-vs-gp",
        "--spec-ratio-vs-gp",
        "--min-spec-rows-per-phase",
        str(int(ns.min_spec_rows_per_phase)),
        "--bundle-minutes",
        str(float(ns.bundle_minutes)),
        "--phot-spec-threshold",
        str(int(ns.phot_spec_threshold)),
    ]
    if meta_arg:
        overview += ["--meta", meta_arg]
    if enrich_abs:
        overview += ["--enrich", enrich_abs]
    if filt_abs:
        overview += ["--filter-config", filt_abs]
    overview += ["--phot-lc-time-step-days", str(float(ns.phot_lc_time_step_days))]
    overview += [
        "--phot-pseudo-grouping",
        str(ns.phot_pseudo_grouping),
        "--phot-unique-x1-decimals",
        str(int(ns.phot_unique_x1_decimals)),
        "--phot-max-unique-panels-warn",
        str(int(ns.phot_max_unique_panels_warn)),
    ]
    _run(overview)

    _run(
        [
            py,
            os.path.join(HERE, "outlier_pipeline.py"),
            "--run-dir",
            os.path.join(runs_dir, str(ns.tag)),
            "--bundle",
            scaled,
            "--z-threshold",
            str(float(ns.z_threshold)),
            "--iteration",
            "0",
        ]
    )

    po = [
        py,
        os.path.join(HERE, "plot_outliers.py"),
        "--tag",
        str(ns.tag),
        "--output-dir",
        runs_dir,
        "--bundle",
        scaled,
        "--iteration",
        "0",
        "--z-threshold",
        str(float(ns.z_threshold)),
    ]
    if meta_arg:
        po += ["--meta", meta_arg]
    if enrich_abs:
        po += ["--enrich", enrich_abs]
    _run(po)

    print("\n[full-pipeline] done")
    print(f"[full-pipeline] scaled bundle: {scaled}")
    print(f"[full-pipeline] run dir: {os.path.join(runs_dir, str(ns.tag))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
