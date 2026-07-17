#!/usr/bin/env python3
"""Grid scan: compare PyCoCo FINAL spectrum (or many epochs) to Bulla (2023) HDF5 models.

For each model file, evaluate χ² and reduced χ² = χ² / n_pix on the Cartesian grid of
native model times × all observer angles. Reduced χ² here uses ν ≈ n_pix (no free parameters
per grid cell); interpret with care when comparing to fits with explicit degrees of freedom.

If ``--spectrum`` is omitted, spectra are picked automatically from your FINAL directory for
each target phase in ``--target-phases`` (default 0.5, 1.5, … 10.5 days), using the same
nearest-epoch logic as ``model_comparison.ipynb``.

Examples:
  python model_chi2.py --out-csv scan.csv --out-plot scan.png
  python model_chi2.py --spectrum /path/to/spec.txt --out-csv one.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _codes_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_codes_on_path() -> None:
    d = _codes_dir()
    if d not in sys.path:
        sys.path.insert(0, d)


def default_repo_root() -> str:
    _ensure_codes_on_path()
    import pipeline_config as pconf

    return os.path.normpath(pconf.COCO_PATH.rstrip(os.sep))


def default_bulla_dir() -> str:
    return os.path.join(default_repo_root(), "2023_bulla")


def iter_hdf5_paths(bulla_dir: str) -> list[str]:
    pat = os.path.join(os.path.abspath(bulla_dir), "*.hdf5")
    return sorted(glob.glob(pat))


def read_sn_redshift(coco_path: str, snname: str) -> float:
    root = coco_path.rstrip(os.sep)
    path = os.path.join(root, "Inputs", "SNe_Info", "info.dat")
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    row = df.loc[df["Name"] == snname]
    return float(row.iloc[0]["z"])


def default_target_phases_sequence() -> list[float]:
    """Phases in days: 0.5, 1.5, …, 10.5 (step 1 day)."""
    return [0.5 + float(i) for i in range(11)]


def parse_target_phases(s: str | None) -> list[float]:
    if s is None or not str(s).strip():
        return default_target_phases_sequence()
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        return default_target_phases_sequence()
    return out


def load_observed_final(path: str, *, flux_on_disk: str = "auto") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from comparison_check_log_utils import deduplicate_wavelength_flux, read_final_spectrum_linear

    wl, F, fe = read_final_spectrum_linear(path, flux_on_disk=flux_on_disk)
    m = np.isfinite(wl) & np.isfinite(F) & np.isfinite(fe)
    wl, F, fe = wl[m], F[m], fe[m]
    order = np.argsort(wl)
    wl, F, fe = wl[order], F[order], fe[order]
    return deduplicate_wavelength_flux(wl, F, fe)


def chi2_red_grid(
    blob: dict[str, Any],
    wl_data: np.ndarray,
    F_data: np.ndarray,
    fe_data: np.ndarray,
    *,
    z: float,
    d_lum_mpc: float,
    wave_is_rest: bool = True,
    time_interp: bool = True,
    phase_offset_days: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    χ²_red at each (obs_index, native time).

    Returns
    -------
    chi2_red : (N_obs, N_t), may contain nan
    cos_theta : (N_obs,)
    t_days : (N_t,) native model time axis (evaluation uses ``t + phase_offset_days``)
    """
    chi2_red, _, n_pix_mat, cos_theta, t_days = chi2_and_red_grids(
        blob,
        wl_data,
        F_data,
        fe_data,
        z=z,
        d_lum_mpc=d_lum_mpc,
        wave_is_rest=wave_is_rest,
        time_interp=time_interp,
        phase_offset_days=phase_offset_days,
    )
    if np.nanmax(n_pix_mat) > 0 and np.percentile(n_pix_mat[n_pix_mat > 0], 5) < max(
        3, int(0.1 * wl_data.size)
    ):
        warnings.warn(
            "Many grid cells have few valid pixels (check wavelength coverage vs model).",
            UserWarning,
            stacklevel=2,
        )
    return chi2_red, cos_theta, t_days


def chi2_and_red_grids(
    blob: dict[str, Any],
    wl_data: np.ndarray,
    F_data: np.ndarray,
    fe_data: np.ndarray,
    *,
    z: float,
    d_lum_mpc: float,
    wave_is_rest: bool = True,
    time_interp: bool = True,
    phase_offset_days: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return chi2_red, chi2_raw, n_pix_matrix, cos_theta, t_days."""
    from model_comparison_helpers import (
        chi2_flux,
        cos_theta_to_theta_deg,
        interp_model_flux_to_wavelengths,
        model_flux_for_epoch_and_angle,
    )

    I = blob["I_stokes"]
    n_obs, n_t = I.shape[0], I.shape[1]
    t_days = np.asarray(blob["time_days"], dtype=float)
    cos_theta, _ = cos_theta_to_theta_deg(n_obs)
    chi2_red = np.full((n_obs, n_t), np.nan, dtype=float)
    chi2_raw = np.full((n_obs, n_t), np.nan, dtype=float)
    n_pix_g = np.zeros((n_obs, n_t), dtype=int)
    for i in range(n_obs):
        for j in range(n_t):
            t_target = float(t_days[j]) + float(phase_offset_days)
            wl_m, F_m, _ = model_flux_for_epoch_and_angle(
                blob,
                obs_index=i,
                phase_days_target=t_target,
                z=z,
                d_lum_mpc=d_lum_mpc,
                wave_is_rest=wave_is_rest,
                time_interp=time_interp,
            )
            F_on = interp_model_flux_to_wavelengths(wl_m, F_m, wl_data)
            c2, n_pix = chi2_flux(F_data, F_on, fe_data)
            n_pix_g[i, j] = n_pix
            if n_pix > 0 and np.isfinite(c2):
                chi2_raw[i, j] = c2
                chi2_red[i, j] = c2 / float(n_pix)
    return chi2_red, chi2_raw, n_pix_g, cos_theta, t_days


def rows_from_grid(
    model_path: str,
    chi2_red: np.ndarray,
    cos_theta: np.ndarray,
    t_days: np.ndarray,
    *,
    chi2_raw: np.ndarray | None = None,
    n_pix_grid: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand (N_obs, N_t) grids into long-form records."""
    base = os.path.basename(model_path)
    rows: list[dict[str, Any]] = []
    n_obs, n_t = chi2_red.shape
    for i in range(n_obs):
        for j in range(n_t):
            row: dict[str, Any] = {
                "model_path": model_path,
                "model_basename": base,
                "obs_index": i,
                "cos_theta": float(cos_theta[i]),
                "theta_deg": float(np.arccos(np.clip(cos_theta[i], -1.0, 1.0)) * 180.0 / np.pi),
                "t_model_days": float(t_days[j]),
                "chi2_red": float(chi2_red[i, j])
                if np.isfinite(chi2_red[i, j])
                else float("nan"),
            }
            if chi2_raw is not None:
                row["chi2"] = (
                    float(chi2_raw[i, j]) if np.isfinite(chi2_raw[i, j]) else float("nan")
                )
            if n_pix_grid is not None:
                row["n_pix"] = int(n_pix_grid[i, j])
            if extra:
                row.update(extra)
            rows.append(row)
    return rows


def plot_summary_bar(
    df: pd.DataFrame,
    out_path: str,
    *,
    metric: str = "chi2_red",
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting")
    if df.empty:
        raise ValueError("no rows to plot")
    sub = df.groupby("model_basename", as_index=False)[metric].min()
    sub = sub.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(sub))))
    y = np.arange(len(sub))
    colors = ["crimson" if yi == 0 else "steelblue" for yi in range(len(sub))]
    ax.barh(y, sub[metric].values, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["model_basename"].values, fontsize=8)
    ax.set_xlabel(
        r"min [%s] over ($\theta$, model time)  (ν ≈ $n_{\rm pix}$ per cell)"
        % metric.replace("_", " ")
    )
    ax.set_title("Bulla HDF5 models: best grid cell per file (red = best)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap_best(
    chi2_red: np.ndarray,
    cos_theta: np.ndarray,
    t_days: np.ndarray,
    out_path: str,
    *,
    title: str,
    mark_ij: tuple[int, int] | None = None,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting")
    fig, ax = plt.subplots(figsize=(10, 5))
    n_obs, _nt = chi2_red.shape
    extent = (float(t_days[0]), float(t_days[-1]), -0.5, float(n_obs) - 0.5)
    im = ax.imshow(
        np.ma.masked_invalid(chi2_red),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis_r",
    )
    ax.set_xlabel("Model time (days)")
    ax.set_ylabel(r"Observer index ($\cos\theta$ increases with index)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=r"$\chi^2_{\rm red}$")
    if mark_ij is not None:
        i, j = mark_ij
        ax.plot(float(t_days[j]), float(i), "r+", ms=16, mew=2, label="best cell")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def resolve_final_data_dir(
    coco_path: str,
    sn_name: str,
    *,
    data_dir: str | None,
    final_variant: str,
    use_rjf: bool,
    use_ryanv2: bool = False,
) -> str:
    """Same FINAL layout as ``model_comparison.ipynb`` when ``data_dir`` is None."""
    if data_dir:
        return os.path.abspath(data_dir)
    import pipeline_config as pconf
    from comparison_check_log_utils import resolve_final_directory

    tw = pconf.final_spectra_twodim_branch(
        pconf.MODE_EXTRAPOLATE_SHORT,
        pconf.SUBDIR_FULL_GP,
        use_rjf=use_rjf and not use_ryanv2,
        use_ryanv2=use_ryanv2,
    )
    return resolve_final_directory(
        coco_path, sn_name, final_variant, twodim_branch=tw
    )


def scan_all_models(
    paths: list[str],
    wl_data: np.ndarray,
    F_data: np.ndarray,
    fe_data: np.ndarray,
    *,
    z: float,
    d_lum_mpc: float,
    wave_is_rest: bool,
    time_interp: bool,
    phase_offset_days: float,
    row_extra: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Run χ² grid for each HDF5 path; return long rows and globally best row (with _matrix keys)."""
    from model_comparison_helpers import load_bulla_observables

    all_rows: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None

    for pi, path in enumerate(paths):
        print("[%d/%d]" % (pi + 1, len(paths)), os.path.basename(path))
        blob = load_bulla_observables(path)
        chi2_red, chi2_raw, n_pix_g, cos_theta, t_days = chi2_and_red_grids(
            blob,
            wl_data,
            F_data,
            fe_data,
            z=z,
            d_lum_mpc=d_lum_mpc,
            wave_is_rest=wave_is_rest,
            time_interp=time_interp,
            phase_offset_days=phase_offset_days,
        )
        rows = rows_from_grid(
            path,
            chi2_red,
            cos_theta,
            t_days,
            chi2_raw=chi2_raw,
            n_pix_grid=n_pix_g,
            extra=row_extra,
        )
        all_rows.extend(rows)
        for r in rows:
            if not np.isfinite(r["chi2_red"]):
                continue
            if global_best is None or r["chi2_red"] < global_best["chi2_red"]:
                global_best = {**r}
                global_best["_chi2_red_matrix"] = chi2_red
                global_best["_cos_theta"] = cos_theta
                global_best["_t_days"] = t_days

    return all_rows, global_best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bulla-dir",
        default=None,
        help="Directory containing *.hdf5 (default: <repo>/2023_bulla)",
    )
    p.add_argument(
        "--spectrum",
        default=None,
        help="Path to one FINAL spectrum .txt. If omitted, spectra are chosen from the FINAL directory for each --target-phases value (default phases: 0.5…10.5 d).",
    )
    p.add_argument(
        "--target-phases",
        default=None,
        help="Comma-separated target phase days (MJD−t0); used only when --spectrum is omitted. Default: 0.5,1.5,…,10.5",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="FINAL spectra directory (default: resolve via pipeline_config like model_comparison.ipynb)",
    )
    p.add_argument(
        "--final-variant",
        default="as_observed",
        help="Subdirectory under FINAL_spectra_2dim branch (default: as_observed)",
    )
    p.add_argument(
        "--use-rjf-final",
        action="store_true",
        default=True,
        help="Use RJF twodim branch (default: True)",
    )
    p.add_argument(
        "--no-rjf-final",
        action="store_false",
        dest="use_rjf_final",
        help="Disable RJF branch for resolve_final_directory",
    )
    p.add_argument(
        "--ryanv2-final",
        action="store_true",
        default=False,
        help="Use twodim_ryanv2 FINAL branch (overrides --use-rjf-final for path resolution)",
    )
    p.add_argument(
        "--t0-mjd",
        type=float,
        default=None,
        help="Explosion/reference MJD for phase (default: SN_EXPLOSION_MJD[sn-name])",
    )
    p.add_argument(
        "--datalc-path",
        default=None,
        help="Directory with late LCs for stem_to_spec_mjd (default: …/4_LCs_late_extrapolated)",
    )
    p.add_argument(
        "--phase-match-max-warn",
        type=float,
        default=0.15,
        help="Warn when |Δphase| exceeds this after nearest-spectrum pick (days)",
    )
    p.add_argument(
        "--final-suffixes",
        default=None,
        help="Optional comma-separated filename suffixes to restrict FINAL *.txt (e.g. _FINAL_spec_FL.txt)",
    )
    p.add_argument(
        "--flux-on-disk",
        default="auto",
        help="Passed to read_final_spectrum_linear (default: auto)",
    )
    p.add_argument(
        "--coco-path",
        default=None,
        help="PyCoCo repo root (default: pipeline_config.COCO_PATH)",
    )
    p.add_argument(
        "--sn-name",
        default="AT2017gfo",
        help="SNe name for reading z from info.dat when --z omitted",
    )
    p.add_argument("--z", type=float, default=None, help="Override redshift")
    p.add_argument(
        "--d-lum-mpc",
        type=float,
        default=None,
        help="Luminosity distance (Mpc); default from Planck18(z)",
    )
    p.add_argument("--wave-is-rest", action="store_true", default=True)
    p.add_argument("--no-wave-is-rest", action="store_false", dest="wave_is_rest")
    p.add_argument("--time-interp", action="store_true", default=True)
    p.add_argument("--no-time-interp", action="store_false", dest="time_interp")
    p.add_argument(
        "--phase-offset-days",
        type=float,
        default=0.0,
        help="Added to each native model time when evaluating (default: 0)",
    )
    p.add_argument("--out-csv", default=None, help="Write long-form results CSV")
    p.add_argument("--out-plot", default=None, help="Summary bar chart (min χ²_red per file)")
    p.add_argument(
        "--heatmap-best",
        default=None,
        help="Write heatmap PNG for the globally best model file",
    )
    p.add_argument("--top-k", type=int, default=10, help="Print K best grid points")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_codes_on_path()
    import pipeline_config as pconf
    from astropy.cosmology import Planck18

    from model_comparison_helpers import enumerate_final_spectrum_phases, pick_final_by_nearest_phase

    coco = (
        os.path.normpath(args.coco_path.rstrip(os.sep)) + os.sep
        if args.coco_path
        else pconf.COCO_PATH
    )
    repo = os.path.normpath(coco.rstrip(os.sep))
    bulla_dir = args.bulla_dir or os.path.join(repo, "2023_bulla")
    paths = iter_hdf5_paths(bulla_dir)
    if not paths:
        print("No *.hdf5 in", bulla_dir, file=sys.stderr)
        return 1

    z = float(args.z) if args.z is not None else read_sn_redshift(coco, args.sn_name)
    d_lum_mpc = (
        float(args.d_lum_mpc)
        if args.d_lum_mpc is not None
        else float(Planck18.luminosity_distance(z).to("Mpc").value)
    )

    datalc_path = args.datalc_path or os.path.join(
        coco, "Inputs", "Photometry", "4_LCs_late_extrapolated"
    )
    t0_mjd = (
        float(args.t0_mjd)
        if args.t0_mjd is not None
        else float(pconf.SN_EXPLOSION_MJD[args.sn_name])
    )

    suffixes: tuple[str, ...] | None = None
    if args.final_suffixes:
        suffixes = tuple(
            s.strip() for s in args.final_suffixes.split(",") if s.strip()
        )

    print("z =", z, "D_L (Mpc) =", d_lum_mpc)
    print("Models:", len(paths), "in", bulla_dir)

    all_rows: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None

    if args.spectrum:
        wl_data, F_data, fe_data = load_observed_final(
            args.spectrum, flux_on_disk=args.flux_on_disk
        )
        print("Loaded spectrum:", args.spectrum, "npix=", wl_data.size)
        extra = {
            "target_phase_days": float("nan"),
            "spectrum_fname": os.path.basename(args.spectrum),
            "spectrum_phase_days": float("nan"),
            "delta_phase_days": float("nan"),
        }
        rows, gb = scan_all_models(
            paths,
            wl_data,
            F_data,
            fe_data,
            z=z,
            d_lum_mpc=d_lum_mpc,
            wave_is_rest=args.wave_is_rest,
            time_interp=args.time_interp,
            phase_offset_days=args.phase_offset_days,
            row_extra=extra,
        )
        all_rows.extend(rows)
        global_best = gb
    else:
        data_dir = resolve_final_data_dir(
            coco,
            args.sn_name,
            data_dir=args.data_dir,
            final_variant=args.final_variant,
            use_rjf=args.use_rjf_final,
            use_ryanv2=args.ryanv2_final,
        )
        phase_list = parse_target_phases(args.target_phases)
        print("FINAL dir:", data_dir)
        print("Target phases (days):", phase_list)
        phase_rows = enumerate_final_spectrum_phases(
            data_dir,
            coco,
            args.sn_name,
            datalc_path,
            t0_mjd=t0_mjd,
            final_suffixes=suffixes,
        )
        if not phase_rows:
            print("No *.txt FINAL spectra in", data_dir, file=sys.stderr)
            return 1

        for tgt in phase_list:
            picked = pick_final_by_nearest_phase(phase_rows, float(tgt))
            if picked["abs_delta_phase_days"] > float(args.phase_match_max_warn):
                warnings.warn(
                    "Target phase %.2f d: nearest spectrum has |Δphase|=%.4f d (file %s)"
                    % (tgt, picked["abs_delta_phase_days"], picked["fname"]),
                    UserWarning,
                )
            spec_path = os.path.join(data_dir, picked["fname"])
            wl_data, F_data, fe_data = load_observed_final(
                spec_path, flux_on_disk=args.flux_on_disk
            )
            print(
                "--- Phase target %.2f d -> %s (phase=%.4f d, Δ=%.4f) npix=%d ---"
                % (
                    tgt,
                    picked["fname"],
                    picked["phase_days"],
                    picked["delta_phase_days"],
                    wl_data.size,
                )
            )
            extra = {
                "target_phase_days": float(tgt),
                "spectrum_fname": picked["fname"],
                "spectrum_phase_days": float(picked["phase_days"]),
                "delta_phase_days": float(picked["delta_phase_days"]),
            }
            rows, gb = scan_all_models(
                paths,
                wl_data,
                F_data,
                fe_data,
                z=z,
                d_lum_mpc=d_lum_mpc,
                wave_is_rest=args.wave_is_rest,
                time_interp=args.time_interp,
                phase_offset_days=args.phase_offset_days,
                row_extra=extra,
            )
            all_rows.extend(rows)
            if gb is not None:
                if global_best is None or gb["chi2_red"] < global_best["chi2_red"]:
                    global_best = gb

    df = pd.DataFrame(all_rows)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print("Wrote", args.out_csv)

    if global_best:
        msg = (
            "\nGlobal best (min χ²_red): %s target_phase=%s spectrum=%s obs_index=%d cosθ=%.4f t=%.4f χ²_red=%.4f chi2=%s n_pix=%s"
            % (
                global_best["model_basename"],
                global_best.get("target_phase_days", "—"),
                global_best.get("spectrum_fname", "—"),
                global_best["obs_index"],
                global_best["cos_theta"],
                global_best["t_model_days"],
                global_best["chi2_red"],
                global_best.get("chi2", "—"),
                global_best.get("n_pix", "—"),
            )
        )
        print(msg)

    if args.top_k > 0 and not df.empty:
        valid = df[np.isfinite(df["chi2_red"])].sort_values("chi2_red").head(args.top_k)
        cols = [
            c
            for c in (
                "target_phase_days",
                "spectrum_fname",
                "model_basename",
                "obs_index",
                "cos_theta",
                "t_model_days",
                "chi2_red",
                "chi2",
                "n_pix",
            )
            if c in valid.columns
        ]
        print("\nTop", args.top_k, "by χ²_red:")
        print(valid[cols].to_string(index=False))

    if args.out_plot:
        if plt is None:
            print("matplotlib not installed; skipping --out-plot", file=sys.stderr)
        elif not df.empty:
            base, ext = os.path.splitext(args.out_plot)
            if not ext:
                ext = ".png"
            if args.spectrum or "target_phase_days" not in df.columns:
                plot_path = base + ext
                plot_summary_bar(df, plot_path, metric="chi2_red")
                print("Wrote", plot_path)
            else:
                phases = sorted(
                    float(x)
                    for x in df["target_phase_days"].dropna().unique()
                    if np.isfinite(x)
                )
                if len(phases) <= 1:
                    plot_summary_bar(df, base + ext, metric="chi2_red")
                    print("Wrote", base + ext)
                else:
                    for ph in phases:
                        sub = df[df["target_phase_days"] == ph]
                        plot_path = "%s_phase%g%s" % (base, ph, ext)
                        plot_summary_bar(sub, plot_path, metric="chi2_red")
                        print("Wrote", plot_path)

    if args.heatmap_best:
        if plt is None:
            print("matplotlib not installed; skipping --heatmap-best", file=sys.stderr)
        elif global_best is not None:
            M = global_best["_chi2_red_matrix"]
            ct = global_best["_cos_theta"]
            td = global_best["_t_days"]
            i_best = int(global_best["obs_index"])
            j_best = int(np.argmin(np.abs(td - float(global_best["t_model_days"]))))
            t_ph = global_best.get("target_phase_days")
            sp = global_best.get("spectrum_fname", "")
            title_bits = ["Best model:", str(global_best["model_basename"])]
            if sp:
                title_bits.append(str(sp))
            if t_ph is not None and np.isfinite(t_ph):
                title_bits.append("data phase target=%g d" % float(t_ph))
            plot_heatmap_best(
                M,
                ct,
                td,
                args.heatmap_best,
                title=" ".join(title_bits),
                mark_ij=(i_best, j_best),
            )
            print("Wrote", args.heatmap_best)

    return 0


if __name__ == "__main__":
    if main() != 0:
        raise SystemExit(1)