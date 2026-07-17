"""Residual-based outlier flags from GP training predictions; persists JSON artifacts.

Full LC / spectrum / bundle plotting needs enriched tables (band, spectrum_id, MJD).
This module implements standardized residual cuts and saves machine-readable outputs.

When ``predictions.npz`` includes ``train_row_index_orig`` (subset fits from ``run_gp``),
``flagged_train_indices`` in the JSON are **original bundle row indices** (length matches
the training bundle ``X``). The optional ``--write-mask`` builds an ``include`` array of
that same bundle length. Re-run this pipeline after upgrading if old JSON used fit-local indices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class OutlierSummary:
    iteration: int
    n_total: int
    n_flagged: int
    z_threshold: float
    flagged_train_indices: list[int]
    norm_resid_at_flagged: list[float]

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def load_predictions(path: str):
    d = np.load(path, allow_pickle=False)
    mu_t = np.asarray(d["mu_train"], dtype=float)
    sig = np.asarray(d["sigma_eff_train"], dtype=float)
    pc = np.asarray(d["point_class_train"])
    return d, mu_t, sig, pc


def flag_outliers(
    pred_path: str,
    *,
    bundle_npz: Optional[str] = None,
    z_threshold: float = 5.0,
    iteration: int = 0,
) -> OutlierSummary:
    d, mu_t, sig, _pc = load_predictions(pred_path)
    if "y_train" in d.files:
        y = np.asarray(d["y_train"], dtype=float)
    elif bundle_npz and os.path.isfile(bundle_npz):
        b = np.load(bundle_npz, allow_pickle=False)
        y = np.asarray(b["y"], dtype=float)
        b.close()
    else:
        raise ValueError("predictions.npz must contain y_train, or pass bundle_npz to load y")
    if y.shape[0] != mu_t.shape[0]:
        raise ValueError(f"y N={y.shape[0]} vs mu_train N={mu_t.shape[0]} — use matching bundle")

    resid = mu_t - y
    norm = resid / np.maximum(sig, 1e-30)
    bad = np.abs(norm) > z_threshold
    idx_fit = np.nonzero(bad)[0].astype(int).tolist()

    bundle_n: Optional[int] = None
    if bundle_npz and os.path.isfile(bundle_npz):
        b2 = np.load(bundle_npz, allow_pickle=False)
        bundle_n = int(np.asarray(b2["X"]).shape[0])
        b2.close()

    if "train_row_index_orig" in d.files:
        oi = np.asarray(d["train_row_index_orig"], dtype=np.int64).ravel()
        if oi.size != mu_t.shape[0]:
            raise ValueError(
                f"train_row_index_orig length {oi.size} vs mu_train {mu_t.shape[0]}"
            )
        if bundle_n is None:
            raise ValueError("bundle_npz is required when predictions contain train_row_index_orig")
        if int(oi.max()) >= bundle_n or int(oi.min()) < 0:
            raise ValueError("train_row_index_orig out of range for bundle X")
        idx_bundle = [int(oi[i]) for i in idx_fit]
        n_mask_rows = bundle_n
    else:
        idx_bundle = list(idx_fit)
        n_mask_rows = int(y.size)

    summary = OutlierSummary(
        iteration=iteration,
        n_total=n_mask_rows,
        n_flagged=int(len(idx_bundle)),
        z_threshold=float(z_threshold),
        flagged_train_indices=idx_bundle,
        norm_resid_at_flagged=[float(norm[i]) for i in idx_fit],
    )
    return summary


def save_outlier_report(run_dir: str, summary: OutlierSummary) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"outliers_iter{summary.iteration}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_jsonable(), f, indent=2)
    return path


def build_include_mask(n_train: int, summary: OutlierSummary) -> np.ndarray:
    m = np.ones(n_train, dtype=bool)
    for i in summary.flagged_train_indices:
        if 0 <= i < n_train:
            m[i] = False
    return m


def save_train_mask(path: str, include: np.ndarray) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez(path, include=include.astype(bool))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="runs/<tag>/ containing predictions.npz")
    p.add_argument(
        "--bundle",
        default=None,
        help="training bundle with y matching predictions (default: gp_minimal_bundle.npz in cwd)",
    )
    p.add_argument("--z-threshold", type=float, default=5.0)
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--write-mask", default=None, help="optional path to write train_include.npz")
    args = p.parse_args(argv)

    pred = os.path.join(args.run_dir, "predictions.npz")
    if not os.path.isfile(pred):
        print(f"[outlier_pipeline] ERROR: {pred} not found", file=sys.stderr)
        return 1

    here = os.path.dirname(os.path.abspath(__file__))
    bundle = args.bundle or os.path.join(here, "gp_minimal_bundle.npz")
    if not os.path.isfile(bundle):
        print(f"[outlier_pipeline] ERROR: bundle {bundle!r} not found", file=sys.stderr)
        return 1

    summ = flag_outliers(pred, bundle_npz=bundle, z_threshold=args.z_threshold, iteration=args.iteration)
    out_json = save_outlier_report(args.run_dir, summ)
    print(f"[outlier_pipeline] wrote {out_json}")
    print(f"[outlier_pipeline] flagged {summ.n_flagged}/{summ.n_total} at |z|>{args.z_threshold}")

    if args.write_mask:
        mask = build_include_mask(summ.n_total, summ)
        save_train_mask(args.write_mask, mask)
        print(f"[outlier_pipeline] wrote mask {args.write_mask} ({mask.sum()} kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
