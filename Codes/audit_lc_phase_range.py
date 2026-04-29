#!/usr/bin/env python3
"""Compare phase coverage in fitted_phot (linear flux) vs fitted_phot_logspace (Log_Phase).

Run from repo root or anywhere: ``python Codes/audit_lc_phase_range.py --sn AT2017gfo``

Uses only the standard library (csv). Does not modify data.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys


def _read_tsv(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        rows = list(r)
        return (r.fieldnames or [], rows)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser(description="Audit LC phase range: linear vs logspace files.")
    ap.add_argument("--sn", default="AT2017gfo", help="Supernova name (Outputs/<SN>/).")
    ap.add_argument(
        "--t0",
        type=float,
        default=57982.528,
        help="Explosion MJD for linear file (phase = MJD - t0).",
    )
    ap.add_argument(
        "--repo",
        default=repo,
        help="Repository root (contains Outputs/).",
    )
    args = ap.parse_args()
    out_dir = os.path.join(args.repo, "Outputs", args.sn)
    f_lin = os.path.join(out_dir, f"fitted_phot_{args.sn}.dat")
    f_log = os.path.join(out_dir, f"fitted_phot_logspace_{args.sn}.dat")
    for p in (f_lin, f_log):
        if not os.path.isfile(p):
            print(f"Missing file: {p}", file=sys.stderr)
            return 1

    cols_lin, rows_lin = _read_tsv(f_lin)
    cols_log, rows_log = _read_tsv(f_log)
    if not rows_lin or not rows_log:
        print("Empty LC file.", file=sys.stderr)
        return 1

    mjds = []
    for row in rows_lin:
        m = row.get("MJD")
        if m is None or m == "":
            continue
        try:
            mjds.append(float(m))
        except ValueError:
            continue
    if mjds:
        phases = [m - args.t0 for m in mjds]
        pmin, pmax = min(phases), max(phases)
        print("=== fitted_phot (linear flux) ===")
        print(f"  Rows: {len(rows_lin)}  MJD rows parsed: {len(mjds)}")
        print(f"  Phase (days) min / max: {pmin:.6g} / {pmax:.6g}")
    else:
        print("=== fitted_phot: no MJD column parsed ===")

    lp_vals = []
    for row in rows_log:
        v = row.get("Log_Phase")
        if v is None or v == "":
            continue
        try:
            lp_vals.append(float(v))
        except ValueError:
            continue
    if lp_vals:
        lmin, lmax = min(lp_vals), max(lp_vals)
        print("=== fitted_phot_logspace ===")
        print(f"  Rows: {len(rows_log)}  Log_Phase rows: {len(lp_vals)}")
        print(f"  Log_Phase min / max: {lmin:.8g} / {lmax:.8g}")
        print(
            f"  Equivalent phase (days) min / max: {10 ** lmin:.6g} / {10 ** lmax:.6g}"
        )
    else:
        print("=== fitted_phot_logspace: no Log_Phase parsed ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
