"""Plot flagged training outliers (phot LC + spectrum overlays).

Reads ``runs/<tag>/predictions.npz``, bundle ``y``/``X``, ``runs/<tag>/config.json``
for ``grid_norm_info``, and ``runs/<tag>/outliers_iter{k}.json`` (or recomputes flags).

Without enriched metadata (bands, MJD), photometry uses **pseudo-bands** = bins of
normalized ``X[:,0]`` (log wavelength proxy). The smooth curve is **GP posterior**
``mu_train`` sorted by phase (not synthetically photometered spectra).

Optional enriched sidecar ``--enrich NPZ`` may contain aligned length-N arrays::

    band_id (int32) or band_name (object/U bytes) — photometry band per row; use -1 / '' for spec
    mjd (float64) — time for LC x-axis when present (else phase days from bundle meta)

Also optional ``synth_phot_{band}`` float arrays or table — see ``--enrich-help`` in code.

Writes figures under ``runs/<tag>/figs/outliers/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import bundle_meta as bmeta
import bundle_preprocess as bpre
import gp_utils as gu
from outlier_pipeline import OutlierSummary, flag_outliers, save_outlier_report
from plot_results import (
    _build_scaled_spec_overlay_rows,
    denorm_ln_wavelength,
    linear_flux_yerr,
    phase_days_from_norm_x2,
    scaled_ln_to_linear,
)


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "gp_minimal_bundle.npz")
DEFAULT_RUNS = os.path.join(HERE, "runs")


def _load_gn(run_dir: str, bundle: str, meta_path: Optional[str]) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg.get("grid_norm_info"), dict):
            gn = dict(cfg["grid_norm_info"])
            gn["_normalized_only"] = False
            return gn
    return bmeta.grid_norm_from_bundle_or_meta(bundle, meta_path=meta_path)


def _load_enrich(path: Optional[str]) -> Optional[dict[str, np.ndarray]]:
    if not path:
        return None
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        print(f"[plot_outliers] WARNING: enrich file missing {p!r}", file=sys.stderr)
        return None
    d = np.load(p, allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


def _phot_band_labels(enrich: Optional[dict], n: int, phot_mask: np.ndarray) -> np.ndarray:
    """Return object array of band label per row (empty string if unknown)."""
    labels = np.full(n, "", dtype=object)
    if enrich is None:
        return labels
    if "band_name" in enrich:
        bn = enrich["band_name"]
        for i in range(min(n, bn.shape[0])):
            labels[i] = str(bn[i])
    elif "band_id" in enrich:
        ids = enrich["band_id"]
        for i in range(min(n, ids.shape[0])):
            labels[i] = f"band_{int(ids[i])}"
    return labels


def _time_axis_days(X: np.ndarray, gn: dict, enrich: Optional[dict]) -> np.ndarray:
    if enrich is not None and "mjd" in enrich:
        mjd = np.asarray(enrich["mjd"], dtype=float).ravel()
        if mjd.size >= X.shape[0]:
            return mjd[: X.shape[0]]
    return phase_days_from_norm_x2(X[:, 1], gn)


def _pseudo_band_key(x1_norm: np.ndarray, ndigits: int = 4) -> np.ndarray:
    return np.round(np.asarray(x1_norm, dtype=float), ndigits)


def _load_outlier_summary(
    run_dir: str,
    iteration: int,
    pred_path: str,
    bundle: str,
    z_threshold: float,
    recompute: bool,
) -> OutlierSummary:
    json_path = os.path.join(run_dir, f"outliers_iter{iteration}.json")
    if not recompute and os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        fields = {k: d[k] for k in OutlierSummary.__dataclass_fields__ if k in d}
        return OutlierSummary(**fields)
    summ = flag_outliers(pred_path, bundle_npz=bundle, z_threshold=z_threshold, iteration=iteration)
    save_outlier_report(run_dir, summ)
    return summ


def _smooth_gp_curve(phases: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by phase and interpolate for a continuous-looking model line."""
    phases = np.asarray(phases, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    ok = np.isfinite(phases) & np.isfinite(mu)
    phases, mu = phases[ok], mu[ok]
    if phases.size < 2:
        return phases, mu
    order = np.argsort(phases)
    phases, mu = phases[order], mu[order]
    # collapse duplicate phases (median mu)
    uniq = []
    mus = []
    i0 = 0
    for i in range(1, phases.size + 1):
        if i == phases.size or phases[i] != phases[i - 1]:
            sl = slice(i0, i)
            uniq.append(phases[i - 1])
            mus.append(float(np.median(mu[sl])))
            i0 = i
    px = np.asarray(uniq, dtype=float)
    my = np.asarray(mus, dtype=float)
    if px.size < 2:
        return px, my
    dense = np.linspace(px.min(), px.max(), max(80, px.size * 4))
    yi = np.interp(dense, px, my)
    return dense, yi


def _parse_int_csv(s: str) -> set[int]:
    out: set[int] = set()
    for part in s.replace(",", " ").split():
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def _parse_synth_sidecar(enrich: Optional[dict[str, np.ndarray]]) -> Optional[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Build ``{band_key: (time_array, flux_array)}`` from optional enrich npz.

    Convention 1: global ``synth_times`` + ``synth_flux`` (one curve) -> key ``"default"``.

    Convention 2: per-band keys ``synth_times_<label>`` and ``synth_flux_<label>`` (same <label>).
    """
    if not enrich:
        return None
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "synth_times" in enrich and "synth_flux" in enrich:
        out["default"] = (enrich["synth_times"].ravel(), enrich["synth_flux"].ravel())
    for k in list(enrich.keys()):
        if k.startswith("synth_times_"):
            lab = k[len("synth_times_") :]
            fk = f"synth_flux_{lab}"
            if fk in enrich:
                out[lab] = (enrich[k].ravel(), enrich[fk].ravel())
    return out or None


def plot_phot_outliers(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    mu_train: np.ndarray,
    point_class: np.ndarray,
    flagged: set[int],
    gn: dict,
    enrich: Optional[dict],
    out_dir: str,
    *,
    pseudo_band_digits: int = 4,
    synth_curves: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
) -> None:
    """One figure per band (or pseudo-band): time vs linear flux + smooth GP line; outliers marked."""
    phot_m = point_class == gu.PHOT
    if not phot_m.any():
        print("[plot_outliers] no photometry points in training")
        return

    n_phot_figs = 0
    t_days = _time_axis_days(X, gn, enrich)
    flux = scaled_ln_to_linear(y, gn)
    ferr = linear_flux_yerr(y, yerr, gn)
    mu_lin = scaled_ln_to_linear(mu_train, gn)

    labels = _phot_band_labels(enrich, X.shape[0], phot_m)
    use_band = enrich is not None and (
        ("band_name" in enrich or "band_id" in enrich)
        and np.any(labels[phot_m] != "")
    )

    groups: dict[str, np.ndarray]
    if use_band:
        groups = defaultdict(list)
        for i in np.nonzero(phot_m)[0]:
            lab = str(labels[i]) if labels[i] else "unknown"
            groups[lab].append(i)
        groups = {k: np.asarray(v, dtype=int) for k, v in groups.items()}
    else:
        keys = _pseudo_band_key(X[:, 0], pseudo_band_digits)
        groups = defaultdict(list)
        for i in np.nonzero(phot_m)[0]:
            groups[float(keys[i])].append(i)
        groups = {f"log10λ_norm≈{k:.4f}": np.asarray(v, dtype=int) for k, v in groups.items()}

    for lab, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if idx.size == 0:
            continue
        out_idx = [j for j in idx if j in flagged]
        if not out_idx:
            continue

        fig, ax = plt.subplots(figsize=(9, 4))
        ti = t_days[idx]
        fi = flux[idx]
        ei = ferr[idx]
        mi = mu_lin[idx]

        ax.errorbar(ti, fi, yerr=ei, fmt="o", ms=4, color="gray", alpha=0.45, label="phot")

        synth_line = False
        if synth_curves:
            curve = synth_curves.get(lab) or synth_curves.get("default")
            if curve is not None:
                sx, sy = curve
                ax.plot(sx, sy, "-", color="darkgreen", lw=2.0, alpha=0.85, label="synth phot (sidecar)")
                synth_line = True

        px, py = _smooth_gp_curve(ti, mi)
        lbl_curve = "GP @ train (smooth)" if synth_line else "model (GP posterior, smooth)"
        ax.plot(px, py, "-", color="steelblue", lw=1.8, alpha=0.9, label=lbl_curve)

        if out_idx:
            ax.scatter(
                t_days[out_idx],
                flux[out_idx],
                s=120,
                facecolors="none",
                edgecolors="crimson",
                linewidths=2.2,
                zorder=5,
                label=f"outlier (n={len(out_idx)})",
            )

        ax.set_xlabel("MJD" if enrich is not None and "mjd" in enrich else "phase (days)")
        ax.set_ylabel("flux (linear)")
        ttl = lab[:80] + ("…" if len(lab) > 80 else "")
        ax.set_title(f"Photometry outliers — {ttl}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in lab)[:60]
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phot_outliers_{safe}.png"), dpi=150)
        plt.close(fig)
        n_phot_figs += 1
        print(f"[plot_outliers] wrote phot_outliers_{safe}.png")

    if n_phot_figs == 0:
        print("[plot_outliers] no photometric outliers (no band/pseudo-band contains a flagged phot point)")


def plot_light_curves_flagged_phot(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    mu_train: np.ndarray,
    point_class: np.ndarray,
    flagged: set[int],
    gn: dict,
    enrich: Optional[dict],
    out_dir: str,
    *,
    pseudo_band_digits: int = 4,
) -> None:
    """One LC panel per flagged row classified as photometry (highlight that epoch)."""
    t_days = _time_axis_days(X, gn, enrich)
    flux = scaled_ln_to_linear(y, gn)
    ferr = linear_flux_yerr(y, yerr, gn)
    mu_lin = scaled_ln_to_linear(mu_train, gn)
    keys = _pseudo_band_key(X[:, 0], pseudo_band_digits)

    phot_flagged = sorted([i for i in flagged if point_class[i] == gu.PHOT])
    if not phot_flagged:
        return

    for i in phot_flagged:
        kb = float(keys[i])
        band_mask = point_class == gu.PHOT
        same_band = band_mask & np.isclose(keys, kb, rtol=0.0, atol=10 ** (-pseudo_band_digits))
        idx = np.nonzero(same_band)[0]
        if idx.size == 0:
            idx = np.array([i])

        fig, ax = plt.subplots(figsize=(9, 4))
        ti = t_days[idx]
        fi = flux[idx]
        ei = ferr[idx]
        mi = mu_lin[idx]
        ax.errorbar(ti, fi, yerr=ei, fmt="o", ms=4, color="gray", alpha=0.5, label="phot same pseudo-band")
        px, py = _smooth_gp_curve(ti, mi)
        ax.plot(px, py, "-", color="steelblue", lw=1.6, alpha=0.9, label="GP @ train (smooth)")
        ax.scatter(
            [t_days[i]],
            [flux[i]],
            s=200,
            facecolors="none",
            edgecolors="crimson",
            linewidths=3.0,
            zorder=8,
            label=f"flagged train idx={i}",
        )
        ax.set_xlabel("MJD" if enrich is not None and "mjd" in enrich else "phase (days)")
        ax.set_ylabel("flux (linear)")
        ax.set_title(
            f"Photometry LC — flagged outlier idx={i}\n"
            f"pseudo-band log10λ_norm≈{kb:.4f} (n={idx.size} phot pts in bin)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phot_lc_flagged_idx_{i}.png"), dpi=150)
        plt.close(fig)
        print(f"[plot_outliers] wrote phot_lc_flagged_idx_{i}.png")


def plot_spec_outliers(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    point_class: np.ndarray,
    flagged: set[int],
    gn: dict,
    out_dir: str,
    *,
    near_tol: float = 0.05,
    overlap_scale: bool = True,
    gap_factor: float = 35.0,
    min_gap_norm: float = 3e-3,
) -> None:
    """For each flagged spec row, plot λ vs flux with bundle context (overlap-scaled)."""
    spec_m = point_class == gu.SPEC
    plot_row_good = np.isfinite(yerr) & (np.asarray(yerr, dtype=float) < float(bpre.YERR_DISABLED))
    flagged_spec = sorted([i for i in flagged if spec_m[i]])
    if not flagged_spec:
        print("[plot_outliers] no spectroscopic outliers")
        return

    phases_unique = np.unique(X[spec_m, 1])
    # Group flagged indices by exposure phase (exact float match)
    by_phase: dict[float, list[int]] = defaultdict(list)
    for i in flagged_spec:
        by_phase[float(X[i, 1])].append(i)

    for ph, idx_list in sorted(by_phase.items()):
        near = phases_unique[np.abs(phases_unique - ph) <= near_tol]
        if near.size == 0:
            near = np.array([ph])

        scaled_rows, _n_ov = _build_scaled_spec_overlay_rows(
            X,
            y,
            yerr,
            gn,
            np.sort(near),
            spec_m,
            overlap_scale=overlap_scale,
            gap_factor=gap_factor,
            min_abs_gap_norm=min_gap_norm,
            plot_row_mask=plot_row_good,
            spec_phase_decimals=9,
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0.05, 0.92, max(len(near), 1)))

        for sg in scaled_rows:
            sp_ph = float(sg["sp_phase"])
            color = colors[int(np.argmin(np.abs(near - sp_ph))) % len(colors)]
            wl = denorm_ln_wavelength(sg["wl_norm"], gn)
            yp = sg["flux_lin"] * float(sg["scale"])
            lw = 2.8 if abs(sp_ph - ph) < 1e-9 else 1.0
            ax.plot(wl, yp, "-", color=color, lw=lw, alpha=0.85)

        for i in idx_list:
            ax.scatter(
                denorm_ln_wavelength(np.array([X[i, 0]]), gn),
                scaled_ln_to_linear(np.array([y[i]]), gn) * 1.0,
                s=140,
                facecolors="none",
                edgecolors="crimson",
                linewidths=2.5,
                zorder=6,
                label="flagged pixel" if i == idx_list[0] else None,
            )

        ax.set_xlabel("log10(wavelength)")
        ax.set_ylabel("flux (linear)")
        ax.set_title(
            f"Spectrum outliers — phase_norm={ph:.6g}\n"
            f"thick line = flagged exposure; bundle context ± tol={near_tol:g}"
        )
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.grid(alpha=0.25)
        fn = f"spec_outliers_phase_{ph:.6f}".replace(".", "p")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{fn}.png"), dpi=150)
        plt.close(fig)
        print(f"[plot_outliers] wrote {fn}.png")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument(
        "--output-dir",
        default=DEFAULT_RUNS,
        help=f"parent of <tag>/ (same as run_gp --output-dir). Default: {DEFAULT_RUNS!r}",
    )
    p.add_argument("--bundle", default=DEFAULT_BUNDLE)
    p.add_argument("--meta", default=None)
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--z-threshold", type=float, default=5.0)
    p.add_argument("--recompute-outliers", action="store_true")
    p.add_argument("--enrich", default=None, help="optional npz with band_name/band_id, mjd, ...")
    p.add_argument("--near-phase-tol", type=float, default=0.05, help="spec context phases within this norm log-phase")
    p.add_argument("--no-spec-overlap-scale", action="store_true")
    p.add_argument(
        "--plot-as-phot-indices",
        default="",
        help="comma-separated train indices to treat as photometry in these plots only "
             "(light curves + exclude from spec panels); use after `bundle_preprocess.py find`",
    )
    args = p.parse_args(argv)

    tag = str(args.tag).strip()
    if not tag:
        print(
            "[plot_outliers] ERROR: --tag is empty. Pass the same name as ``run_gp.py -t``.",
            file=sys.stderr,
        )
        return 1

    args.output_dir = os.path.abspath(os.path.expanduser(str(args.output_dir)))
    run_dir = os.path.join(args.output_dir, tag)
    pred_path = os.path.join(run_dir, "predictions.npz")
    if not os.path.isfile(pred_path):
        print(f"[plot_outliers] ERROR: {pred_path} missing", file=sys.stderr)
        return 1

    preds = np.load(pred_path, allow_pickle=False)
    if "mu_train" not in preds.files:
        print("[plot_outliers] ERROR: predictions need mu_train (--predict-train)", file=sys.stderr)
        return 1
    mu_train = np.asarray(preds["mu_train"], dtype=float)

    bundle = np.load(args.bundle, allow_pickle=False)
    X = np.asarray(bundle["X"], dtype=float)
    y = np.asarray(bundle["y"], dtype=float)
    yerr = np.asarray(bundle["yerr"], dtype=float)

    subset_oi: Optional[np.ndarray] = None
    if "train_row_index_orig" in preds.files:
        subset_oi = np.asarray(preds["train_row_index_orig"], dtype=np.int64).ravel()
        if mu_train.shape[0] != subset_oi.size:
            print(
                "[plot_outliers] ERROR: train_row_index_orig length does not match mu_train",
                file=sys.stderr,
            )
            return 1
        if int(subset_oi.max()) >= X.shape[0] or int(subset_oi.min()) < 0:
            print("[plot_outliers] ERROR: train_row_index_orig out of range for bundle X", file=sys.stderr)
            return 1
        X, y, yerr = X[subset_oi], y[subset_oi], yerr[subset_oi]
        print(
            f"[plot_outliers] sliced bundle to GP training subset N={X.shape[0]} "
            f"(full N={int(bundle['X'].shape[0])}) via train_row_index_orig"
        )

    gn = _load_gn(run_dir, args.bundle, args.meta)
    enrich = _load_enrich(args.enrich)

    summ = _load_outlier_summary(
        run_dir,
        args.iteration,
        pred_path,
        args.bundle,
        args.z_threshold,
        args.recompute_outliers,
    )
    flagged_bundle = set(summ.flagged_train_indices)
    if subset_oi is not None:
        oi_to_j = {int(subset_oi[j]): j for j in range(subset_oi.size)}
        flagged = {oi_to_j[i] for i in flagged_bundle if i in oi_to_j}
    else:
        flagged = {i for i in flagged_bundle if 0 <= i < X.shape[0]}
    print(f"[plot_outliers] {summ.n_flagged} flagged / {summ.n_total}")

    plot_as_phot = _parse_int_csv(args.plot_as_phot_indices)
    if "train_obs_class" in bundle.files:
        train_obs_plot = np.asarray(bundle["train_obs_class"]).astype("<U8").copy()
        if subset_oi is not None:
            train_obs_plot = train_obs_plot[subset_oi]
    else:
        train_obs_plot = np.asarray(gu.classify_points(X)).astype("<U8")
    if subset_oi is not None:
        oi_to_j_phot = {int(subset_oi[j]): j for j in range(subset_oi.size)}
        for bi in plot_as_phot:
            lj = oi_to_j_phot.get(int(bi))
            if lj is not None and 0 <= lj < X.shape[0]:
                train_obs_plot[lj] = gu.PHOT
    else:
        for i in plot_as_phot:
            if 0 <= i < X.shape[0]:
                train_obs_plot[i] = gu.PHOT
    if plot_as_phot:
        print(f"[plot_outliers] overriding rows as phot for figures: {sorted(plot_as_phot)}")

    point_class = gu.effective_point_class(X, train_obs_class=train_obs_plot)

    out_dir = os.path.join(run_dir, "figs", "outliers")
    os.makedirs(out_dir, exist_ok=True)

    synth_curves = _parse_synth_sidecar(enrich)

    plot_phot_outliers(
        X,
        y,
        yerr,
        mu_train,
        point_class,
        flagged,
        gn,
        enrich,
        out_dir,
        synth_curves=synth_curves,
    )

    plot_light_curves_flagged_phot(
        X,
        y,
        yerr,
        mu_train,
        point_class,
        flagged,
        gn,
        enrich,
        out_dir,
    )

    plot_spec_outliers(
        X,
        y,
        yerr,
        point_class,
        flagged,
        gn,
        out_dir,
        near_tol=args.near_phase_tol,
        overlap_scale=not args.no_spec_overlap_scale,
    )

    print(f"[plot_outliers] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

