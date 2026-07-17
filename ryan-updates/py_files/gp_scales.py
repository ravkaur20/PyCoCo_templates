#!/usr/bin/env python3
"""Summarize GP kernel metrics from ``runs/<tag>/config.json``: bounds context + heuristic scales.

Wavelength (``metric_w``, ``metric_w2``): maps normalized-``u`` step to
``Δ(log₁₀ λ)`` then to **velocity** via ``v/c ≈ ln(10) · Δ(log₁₀ λ)`` (first-order
Doppler: ``Δλ/λ ≈ v/c``). Time pieces still report ``Δt`` at 1 day for the same
``|Δu| ~ 1/√(5m)`` Matern heuristic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from typing import Any

import gp_utils as gu

C_KMS = 299_792.458  # km/s


def _du_char(m: float) -> float:
    """Order-one Matern-5/2 scale in normalized ``u`` (``√(5m)|Δu| ~ 1``)."""
    m = float(m)
    if m <= 0.0 or not math.isfinite(m):
        return float("nan")
    return 1.0 / math.sqrt(5.0 * m)


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        print(f"[gp_scales] ERROR: config not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def _bounds_table(additive_t: bool, additive_w: bool) -> list[tuple[str, tuple[float, float]]]:
    cfg = gu.KernelConfig(additive_t=additive_t, additive_w=additive_w)
    names = cfg.free_param_names()
    bounds = cfg.default_bounds()
    return list(zip(names, bounds))


def main(argv: list[str] | None = None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    default_runs = here / "runs"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "tag",
        nargs="?",
        default=os.environ.get("GP_TAG", "").strip() or None,
        help="run tag (directory under --runs-dir). Else set env GP_TAG or pass --config.",
    )
    p.add_argument(
        "--runs-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("GP_RUNS_DIR", str(default_runs))).expanduser(),
        help=f"parent of <tag>/ (default: env GP_RUNS_DIR or {default_runs})",
    )
    p.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="explicit config.json (overrides tag + --runs-dir)",
    )
    args = p.parse_args(argv)

    runs_dir = args.runs_dir.expanduser().resolve()
    if args.config is not None:
        cfgp = args.config.expanduser().resolve()
    elif args.tag:
        cfgp = runs_dir / str(args.tag) / "config.json"
    else:
        p.print_help()
        print(
            "\n[gp_scales] ERROR: pass TAG, or --config PATH, or set GP_TAG.",
            file=sys.stderr,
        )
        return 2

    c = _load_config(cfgp)
    inner = c.get("config", {})
    if not isinstance(inner, dict):
        print("[gp_scales] ERROR: config.json missing dict 'config'", file=sys.stderr)
        return 2

    gn = c.get("grid_norm_info") or {}
    x1s = float(gn.get("x1_std", float("nan")))
    x2s = float(gn.get("x2_std", float("nan")))

    additive_t = bool(inner.get("additive_t", False))
    additive_w = bool(inner.get("additive_w", False))

    print(cfgp)
    print(f"additive_t={additive_t}  additive_w={additive_w}")
    if not gn or gn.get("_normalized_only"):
        print(
            "[gp_scales] WARN: no usable grid_norm_info in config; "
            "velocity/time heuristics need x1_std / x2_std.",
            file=sys.stderr,
        )

    overrides = c.get("kernel_bound_overrides") or {}
    if overrides:
        print("kernel_bound_overrides:", json.dumps(overrides, indent=2))
    else:
        print("kernel_bound_overrides: {}")

    print("\nOptimizer parameter boxes (``gp_utils.KernelConfig.default_bounds()``; "
          "``run_gp`` tightens some edges via ``kernel_bound_overrides`` / CLI):")
    for name, bound in _bounds_table(additive_t, additive_w):
        v = inner.get(name)
        lo, hi = bound
        vs = f"{v!r}" if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)) else "n/a"
        print(f"  {name:22s}  [{lo:.6g}, {hi:.6g}]   value={vs}")

    print("\nHeuristic scales (per Matern term; additive mixes with weight_*_short):")

    def report_wavelength(label: str, m_raw: Any) -> None:
        if m_raw is None:
            return
        m = float(m_raw)
        du = _du_char(m)
        dlog10lam = x1s * du if math.isfinite(x1s) and math.isfinite(du) else float("nan")
        # Δλ/λ ≈ ln(10) Δ(log10 λ); interpret as v/c for line-of-sight Doppler.
        v_over_c = math.log(10.0) * dlog10lam if math.isfinite(dlog10lam) else float("nan")
        v_kms = v_over_c * C_KMS if math.isfinite(v_over_c) else float("nan")
        print(f"{label}: m={m:.6g}")
        print(
            f"    heuristic  Δ(log10 λ) ≈ {dlog10lam:.5g} dex  |  "
            f"v/c ≈ {v_over_c:.5g}  |  v ≈ {v_kms:.5g} km/s"
        )

    def report_time(label: str, m_raw: Any) -> None:
        if m_raw is None:
            return
        m = float(m_raw)
        du = _du_char(m)
        dt1 = 1.0 * math.log(10.0) * x2s * du if math.isfinite(x2s) and math.isfinite(du) else float("nan")
        print(f"{label}: m={m:.6g}")
        print(f"    heuristic  Δt @ 1 day ≈ {dt1:.5g} d")

    for label, key in [
        ("λ short (metric_w)", "metric_w"),
        ("λ long  (metric_w2)", "metric_w2"),
        ("t short (metric_t)", "metric_t"),
        ("t long  (metric_t2)", "metric_t2"),
    ]:
        v = inner.get(key)
        if v is None:
            continue
        if key.startswith("metric_w"):
            report_wavelength(label, v)
        else:
            report_time(label, v)

    print(
        "mixture weights: weight_w_short =",
        inner.get("weight_w_short"),
        "  weight_t_short =",
        inner.get("weight_t_short"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
