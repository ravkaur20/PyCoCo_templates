#!/usr/bin/env python3
"""Map photometry training rows (``PHOT`` class) to pseudo x₁ buckets and physical λ (Å).

Reads a GP training ``*.npz`` (e.g. notebook 6 ``gp_minimal_export/gp_minimal_bundle.npz``) and optional
``*_meta.json`` with ``grid_norm_info``. For each rounded ``X[:, 0]`` bucket used by spectroscopy-strip
helpers (defaults ``round_digits=4``), reports row counts and representative physical wavelength::

    wl_log_physical = grid_norm_info.x1_mean + grid_norm_info.x1_std * u

consistent with ``ryan_gp.bundle_scale_pipeline`` / ``ryan_gp.strip_photometry_bands.py``.

Examples::

    python trace_phot_x1_buckets.py --bundle /path/to/gp_minimal_bundle.npz
    python trace_phot_x1_buckets.py \\
        --bundle /path/to/gp_minimal_bundle.npz \\
        --meta /path/to/gp_minimal_bundle_meta.json \\
        --bands -0.8767,-0.8217 \\
        --round-digits 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PHOT = "phot"
SPEC = "spec"


def _classify_points(
    X: np.ndarray, threshold: int = 50, round_decimals: int = 9
) -> np.ndarray:
    """Mirrors ``ryan_gp.gp_utils.classify_points`` (no george dependency)."""
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be (N, 2); got %s" % (X.shape,))
    phase_round = np.round(X[:, 1], round_decimals)
    uphases, inv = np.unique(phase_round, return_inverse=True)
    n_phases = uphases.size
    counts = np.zeros(n_phases, dtype=int)
    for k in range(n_phases):
        rows = inv == k
        counts[k] = np.unique(np.round(X[rows, 0], round_decimals)).size
    is_phot_phase = counts < threshold
    return np.where(is_phot_phase[inv], PHOT, SPEC)


def _effective_point_class(
    X: np.ndarray,
    *,
    threshold: int,
    train_obs_class: np.ndarray | None,
) -> np.ndarray:
    """Mirrors ``ryan_gp.gp_utils.effective_point_class`` (no george dependency)."""
    if train_obs_class is None:
        return _classify_points(X, threshold=threshold, round_decimals=9)
    n = X.shape[0]
    raw = np.asarray(train_obs_class).ravel()
    if raw.shape[0] != n:
        raise ValueError(
            "train_obs_class length %d != N=%d" % (raw.shape[0], n)
        )
    out = np.empty(n, dtype="<U8")
    for i in range(n):
        v = raw[i]
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("ascii", errors="ignore")
        s = str(v).strip().lower()
        if s in ("phot", "p", "1", "true", "yes"):
            out[i] = PHOT
        elif s in ("spec", "s", "0", "false", "no"):
            out[i] = SPEC
        elif isinstance(raw[i], (np.integer, int)):
            out[i] = PHOT if int(raw[i]) != 0 else SPEC
        else:
            raise ValueError(
                "train_obs_class[%d]=%r not recognized (phot/spec or 0/1)"
                % (i, raw[i])
            )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, help="gp_minimal_bundle.npz or collaborator bundle")
    p.add_argument("--meta", default=None, help="*_meta.json (default <bundle>_meta.json if present)")
    p.add_argument("--round-digits", type=int, default=4)
    p.add_argument("--phot-spec-threshold", type=int, default=50)
    p.add_argument(
        "--bands",
        default=None,
        help="Optional comma-separated target keys (printed with match counts); defaults from strip helper",
    )
    ns = p.parse_args(argv)

    bundle_path = ns.bundle if os.path.isabs(ns.bundle) else os.path.join(os.getcwd(), ns.bundle)
    if not os.path.isfile(bundle_path):
        print("ERROR bundle not found: %r" % bundle_path, file=sys.stderr)
        return 2

    meta_path = ns.meta
    if meta_path is None:
        cand = os.path.splitext(bundle_path)[0] + "_meta.json"
        meta_path = cand if os.path.isfile(cand) else None
    else:
        meta_path = meta_path if os.path.isabs(meta_path) else os.path.join(os.getcwd(), meta_path)

    bd = np.load(bundle_path, allow_pickle=False)
    try:
        X = np.asarray(bd["X"], dtype=float)
        obs = bd["train_obs_class"] if "train_obs_class" in bd.files else None
        tobs = np.asarray(obs) if obs is not None else None
        phot_mask = _effective_point_class(
            X.astype(float),
            threshold=ns.phot_spec_threshold,
            train_obs_class=tobs,
        ) == PHOT
        rk = np.round(X[:, 0], int(ns.round_digits))
        rd = int(ns.round_digits)

        gn: dict | None = None
        if meta_path and os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            gn = meta.get("grid_norm_info")

        bands = [-0.8767, -0.8217]
        if ns.bands:
            bands = []
            for part in ns.bands.split(","):
                part = part.strip()
                if part:
                    bands.append(float(part))

        print("[trace_phot_x1_buckets] bundle=%s" % bundle_path)
        print("  X.shape=%s  phot_points=%d  round_digits=%d" % (X.shape, int(phot_mask.sum()), rd))

        uniq = sorted({float(u) for u in rk[phot_mask]})
        print("  unique rounded phot x1 buckets (rounded to %d dp): %d" % (rd, len(uniq)))

        x1_mean = float(gn["x1_mean"]) if gn else float("nan")
        x1_std = float(gn["x1_std"]) if gn else float("nan")
        coord = gn.get("coord_parametrization") if gn else None
        meta_note = coord or ("no-meta" if gn is None else "present")

        for key in uniq[:200]:
            m = phot_mask & (rk == np.round(key, rd))
            n = int(m.sum())
            u_mean = float(np.mean(X[m, 0])) if n else float("nan")
            if gn and np.isfinite(x1_mean) and np.isfinite(x1_std) and abs(x1_std) > 0:
                wl_log = x1_mean + x1_std * u_mean
                lam = float(np.power(10.0, wl_log))
            else:
                lam = float("nan")
                wl_log = float("nan")
            print(
                "    rounded_x1=%r  n_rows=%d  mean_u=%.8g  log10_lambda_phys=%.6g  lambda_A≈%.4g [%s]"
                % (np.round(key, rd), n, u_mean, wl_log, lam, meta_note)
            )
        if len(uniq) > 200:
            print("    ... truncated (%d buckets total)" % len(uniq))

        if bands:
            targets = np.asarray(bands, dtype=float)
            phot_idx = np.where(phot_mask)[0]
            sub_rk = rk[phot_idx]
            hit = np.isin(np.round(sub_rk, rd), np.round(targets, rd))
            print(
                "[strip analogue] band targets=%s touching_phot_rows=%s"
                % (bands, int(hit.sum()))
            )

        if not uniq:
            print("  WARN: zero photometric rows (threshold / train_obs_class?)", file=sys.stderr)
        return 0
    finally:
        bd.close()


if __name__ == "__main__":
    raise SystemExit(main())
