#!/usr/bin/env python3
"""Configurable runner for the 2-D Gaussian-process SED fit.

Usage examples:

    # Recommended: Matern 5/2 on both axes, linear-griddata mean,
    # per-class jitter, hyperparameters fit by L-BFGS-B.
    python run_gp.py --tag matern52_linear_opt

    # Original collaborator setup with per-class jitter (no optimization):
    python run_gp.py \
        --kernel-time matern32 --kernel-wls matern32 \
        --mean nearest --no-optimize \
        --tag matern32_nearest_baseline_jitter

    # Two length scales per axis (color + features on wavelength,
    # short + long on time):
    python run_gp.py \
        --additive-wls --additive-time \
        --tag matern52_addw_addt_linear_opt

Outputs (per tag):
- runs/<tag>/predictions.npz : X_fill, mu, std, var, mu_train, std_train,
                               train_row_index_orig (maps each fitted row to the original bundle
                               row index when ``--train-include`` or ``--exclude-spec-bundle-ids``
                               subsets rows; same length as mu_train), point_class_train,
                               sigma_eff_train, y_train, yerr_train, ...

- runs/<tag>/config.json     : full config + optimized hyperparameters,
                               log-likelihood, runtimes.

If overview heatmaps show **vertical striping** in ``μ`` *and* ``μ_raw``, the GP is
interpolating aggressively in time (often from near-simultaneous spectra that disagree
after scaling). With ``--additive-time``, the optimizer may put almost all weight on the
high-``metric_t2`` branch. Use ``--logit-weight-t-min`` / ``--log-metric-t2-max`` (see
``--help``) to bias toward smoother temporal structure without changing the bundle.

To **omit spectroscopic time bundles** from the fit (e.g. one bundle driving a bad trade-off),
use ``--exclude-spec-bundle-ids 1`` on NPZ files that include ``spec_bundle_id`` from
``bundle_scale_pipeline``. Photometry rows use ``spec_bundle_id=-1``; listing ``-1`` is an error.

If ``--no-enforce-mono-early`` / ``--no-enforce-blue-early``: ``mu_raw`` in ``predictions.npz`` is always the
pre-post-process GP mean; ``mu`` includes these steps when enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Optional, Sequence

import george
import numpy as np
from scipy.optimize import minimize

import bundle_meta as bmeta
import gp_utils as gu

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "gp_minimal_bundle.npz")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "runs")
PREDICT_CHUNK = 10_000


def _fmt_arr(a: np.ndarray) -> str:
    if a.size == 0:
        return "<empty>"
    finite = np.isfinite(a)
    if not finite.any():
        return f"all non-finite (n_nan={int((~finite).sum())})"
    f = a[finite]
    return (
        f"shape={a.shape} dtype={a.dtype} "
        f"min={f.min():.6g} mean={f.mean():.6g} max={f.max():.6g} "
        f"n_nan={int((~finite).sum())}"
    )


def _auto_tag(args: argparse.Namespace) -> str:
    parts = [args.kernel_time]
    if args.additive_time:
        parts.append("addt")
    parts.append(args.kernel_wls)
    if args.additive_wls:
        parts.append("addw")
    parts.append(args.mean)
    parts.append("opt" if args.optimize else "fixed")
    return "_".join(parts)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _build_initial_config(args: argparse.Namespace, bundle) -> gu.KernelConfig:
    if args.additive_wls:
        lw = float(args.lw) if args.lw is not None else float(args.lw_short)
        lw2 = float(args.lw2)
    else:
        lw = float(args.lw) if args.lw is not None else float(bundle["kernel_wls_scale"])
        lw2 = lw * 16.0
    if args.additive_time:
        lt = float(args.lt) if args.lt is not None else float(args.lt_short)
        lt2 = float(args.lt2)
    else:
        lt = float(args.lt) if args.lt is not None else float(bundle["kernel_time_scale"])
        lt2 = lt * 16.0
    log_amp = float(args.log_amp) if args.log_amp is not None else float(np.log(float(bundle["y_var_scale"])))
    return gu.KernelConfig(
        name_t=args.kernel_time,
        name_w=args.kernel_wls,
        additive_t=args.additive_time,
        additive_w=args.additive_wls,
        log_amp=log_amp,
        log_metric_t=float(np.log(lt)),
        log_metric_w=float(np.log(lw)),
        log_metric_t2=float(np.log(lt2)),
        log_metric_w2=float(np.log(lw2)),
        logit_weight_t=_logit(args.w_short_t),
        logit_weight_w=_logit(args.w_short_w),
        log_sigma_phot=float(np.log(max(float(args.sigma_phot), 1e-6))),
        log_sigma_spec=float(np.log(max(float(args.sigma_spec), 1e-6))),
    )



def _enforce_monotone_early(
    X_fill: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    cutoff: float,
    floor_fraction: float = 0.5,
    slope_window: int = 5,
    min_slope: float = 0.005,
    smoothing_scale: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-wavelength, enforce a smooth monotone-increasing-in-phase
    extrapolation that asymptotes to the GP for ``phase >> cutoff`` and to a
    linear extrapolation (matched in value and slope to the GP at the cutoff)
    for ``phase << cutoff``.

    Algorithm per wavelength slice:

    1. ``t_cutoff, mu_cutoff`` = first grid phase >= cutoff.
    2. Slope at cutoff = linear-fit slope over the next ``slope_window``
       grid points; floored at ``min_slope`` so extrapolation is increasing.
    3. ``mu_extrap(t) = mu_cutoff + slope * (t - t_cutoff)``, floored at
       ``floor_fraction * mu_cutoff``. Computed across the *whole* grid.
    4. Blend with the GP via ``w(t) = 0.5 * (1 + tanh((t - cutoff)/scale))``:
       ``mu_out(t) = (1 - w) * mu_extrap(t) + w * mu_GP(t)``.

    Because mu_extrap matches the GP value and slope at the cutoff, and w is
    a tanh, the blend is C-infinity smooth (no kink, no curvature jump) and
    the constraint is fully active for ``t << cutoff - 2*scale`` while the GP
    is intact for ``t >> cutoff + 2*scale``.

    Returns the new ``mu``, the unchanged ``std``, and the number of grid
    points whose mu changed by more than 1e-9.
    """
    mu_out = mu.copy()
    wls_unique = np.unique(X_fill[:, 0])
    n_modified = 0
    scale = max(float(smoothing_scale), 1e-6)
    for wls in wls_unique:
        mask = X_fill[:, 0] == wls
        idx = np.where(mask)[0]
        ph = X_fill[idx, 1]
        order = np.argsort(ph)
        idx_sorted = idx[order]
        ph_sorted = ph[order]
        early_mask = ph_sorted < cutoff
        if not early_mask.any() or early_mask.all():
            continue
        first_in = int(np.where(~early_mask)[0][0])
        t_cutoff = float(ph_sorted[first_in])
        mu_cutoff = float(mu_out[idx_sorted[first_in]])
        n_inside = int((~early_mask).sum())
        win = min(slope_window, n_inside - 1) if n_inside >= 2 else 0
        if win >= 1:
            t_w = ph_sorted[first_in:first_in + win + 1]
            mu_w = mu_out[idx_sorted[first_in:first_in + win + 1]]
            slope = float(np.polyfit(t_w - t_cutoff, mu_w - mu_cutoff, 1)[0])
        else:
            slope = 0.0
        slope = max(slope, float(min_slope))
        # Linear extrapolation across the *whole* phase range for this wls.
        mu_extrap_full = mu_cutoff + slope * (ph_sorted - t_cutoff)
        mu_floor = float(floor_fraction) * mu_cutoff
        mu_extrap_full = np.maximum(mu_extrap_full, mu_floor)
        # Tanh blend: 0 at far left -> all extrap, 1 at far right -> all GP.
        w = 0.5 * (1.0 + np.tanh((ph_sorted - t_cutoff) / scale))
        mu_blend = (1.0 - w) * mu_extrap_full + w * mu_out[idx_sorted]
        diff = np.abs(mu_blend - mu_out[idx_sorted])
        n_modified += int(np.sum(diff > 1e-9))
        mu_out[idx_sorted] = mu_blend
    return mu_out, std, n_modified



def _enforce_blue_early(
    X_fill: np.ndarray,
    mu: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, int]:
    """For each phase column with phase < cutoff, enforce mu non-increasing
    as wavelength increases (i.e. the spectrum is monotonic-decreasing in
    wls, which means flux is monotonic-increasing toward shorter wls).

    Implemented by sorting wls ascending and applying ``np.minimum.accumulate``.
    Plateaus may appear where the GP was non-monotonic; this is fine because
    the GP at log_t < cutoff is being smoothed/extrapolated anyway.
    """
    mu_out = mu.copy()
    phases_unique = np.unique(X_fill[:, 1])
    n_modified = 0
    for ph in phases_unique:
        if ph >= cutoff:
            continue
        mask = X_fill[:, 1] == ph
        idx = np.where(mask)[0]
        wls = X_fill[idx, 0]
        order = np.argsort(wls)
        idx_sorted = idx[order]
        mu_in = mu_out[idx_sorted]
        mu_corr = np.minimum.accumulate(mu_in)
        n_modified += int(np.sum(mu_corr != mu_in))
        mu_out[idx_sorted] = mu_corr
    return mu_out, n_modified


def _tightened_optimizer_bounds(
    cfg: gu.KernelConfig,
    *,
    log_metric_t_min: Optional[float] = None,
    log_metric_w_min: Optional[float] = None,
    log_metric_t2_max: Optional[float] = None,
    log_metric_w2_min: Optional[float] = None,
    log_metric_w2_max: Optional[float] = None,
    logit_weight_t_min: Optional[float] = None,
    logit_weight_t_max: Optional[float] = None,
    logit_weight_w_min: Optional[float] = None,
    logit_weight_w_max: Optional[float] = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    bounds = [list(b) for b in cfg.default_bounds()]
    names = cfg.free_param_names()
    idx = {n: i for i, n in enumerate(names)}
    msgs: list[str] = []

    def tighten(n: str, lo: Optional[float] = None, hi: Optional[float] = None) -> None:
        if n not in idx:
            return
        i = idx[n]
        a, b = bounds[i]
        if lo is not None:
            na = max(a, float(lo))
            if na > b:
                raise ValueError(f"bound override for {n}: new lower {na} exceeds upper {b}")
            a = na
        if hi is not None:
            nb = min(b, float(hi))
            if a > nb:
                raise ValueError(f"bound override for {n}: lower {a} exceeds new upper {nb}")
            b = nb
        if (a, b) != tuple(bounds[i]):
            msgs.append(f"{n}: ({bounds[i][0]:.6g},{bounds[i][1]:.6g}) -> ({a:.6g},{b:.6g})")
        bounds[i] = [a, b]

    if log_metric_t_min is not None:
        tighten("log_metric_t", lo=log_metric_t_min)
    if log_metric_w_min is not None:
        tighten("log_metric_w", lo=log_metric_w_min)
    if log_metric_t2_max is not None:
        tighten("log_metric_t2", hi=log_metric_t2_max)
    if log_metric_w2_min is not None:
        i = idx["log_metric_w2"]
        a, b = bounds[i]
        na = float(log_metric_w2_min)
        if na > b:
            raise ValueError(f"log_metric_w2_min={na} exceeds current upper bound {b}")
        if (na, b) != (a, b):
            msgs.append(f"log_metric_w2: ({a:.6g},{b:.6g}) -> ({na:.6g},{b:.6g})")
        bounds[i] = [na, b]
    if log_metric_w2_max is not None:
        tighten("log_metric_w2", hi=log_metric_w2_max)
    if logit_weight_t_min is not None:
        tighten("logit_weight_t", lo=logit_weight_t_min)
    if logit_weight_t_max is not None:
        tighten("logit_weight_t", hi=logit_weight_t_max)
    if logit_weight_w_min is not None:
        tighten("logit_weight_w", lo=logit_weight_w_min)
    if logit_weight_w_max is not None:
        tighten("logit_weight_w", hi=logit_weight_w_max)
    return [(float(a), float(b)) for a, b in bounds], msgs


def _clip_theta_to_bounds(theta: Sequence[float], bounds: list[tuple[float, float]]) -> np.ndarray:
    out = np.empty(len(bounds), dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        out[i] = float(np.clip(theta[i], lo, hi))
    return out


def _parse_exclude_spec_bundle_ids(arg: Optional[str]) -> list[int]:
    if arg is None or not str(arg).strip():
        return []
    out: list[int] = []
    for part in str(arg).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p, 10))
    return out


def _exclude_rows_by_spec_bundle_id(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    spec_bundle_id: np.ndarray,
    exclude_ids: Sequence[int],
    train_obs_override: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Drop training rows whose ``spec_bundle_id`` is in ``exclude_ids``."""
    if not exclude_ids:
        return X, y, yerr, train_obs_override, 0
    if int(spec_bundle_id.shape[0]) != int(X.shape[0]):
        raise ValueError(
            f"spec_bundle_id length {spec_bundle_id.shape[0]} != N_train {X.shape[0]}"
        )
    sb = np.asarray(spec_bundle_id, dtype=np.int32).ravel()
    ex = np.asarray(list(exclude_ids), dtype=np.int32)
    drop = np.isin(sb, ex)
    n_drop = int(np.sum(drop))
    if n_drop == 0:
        return X, y, yerr, train_obs_override, 0
    keep = ~drop
    Xo, yo, yeo = X[keep], y[keep], yerr[keep]
    obs = train_obs_override[keep] if train_obs_override is not None else None
    return Xo, yo, yeo, obs, n_drop


def _make_neg_ll(
    base_cfg: gu.KernelConfig,
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    point_class: np.ndarray,
    mean_model,
):
    eval_count: dict = {"n": 0, "best": np.inf, "best_theta": None}

    def neg_ll(theta: np.ndarray) -> float:
        cfg = gu.KernelConfig(**asdict(base_cfg))
        try:
            cfg.update_from_vector(theta)
        except Exception:
            return 1e15
        try:
            kernel = gu.build_kernel(cfg)
        except (ValueError, FloatingPointError):
            return 1e15
        gp = george.GP(kernel, mean=mean_model) if mean_model is not None else george.GP(kernel)
        sigma_phot = float(np.exp(cfg.log_sigma_phot))
        sigma_spec = float(np.exp(cfg.log_sigma_spec))
        diag = gu.compute_diagonal(yerr, point_class, sigma_phot, sigma_spec)
        try:
            gp.compute(X, diag)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            return 1e15
        ll = gp.log_likelihood(y)
        eval_count["n"] += 1
        if not np.isfinite(ll):
            return 1e15
        val = -ll
        if val < eval_count["best"]:
            eval_count["best"] = val
            eval_count["best_theta"] = np.asarray(theta, dtype=float).copy()
        return float(val)

    return neg_ll, eval_count


def _predict_chunked(gp: george.GP, y: np.ndarray, X_query: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    n = X_query.shape[0]
    mu = np.empty(n, dtype=float)
    var = np.empty(n, dtype=float)
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        m, v = gp.predict(y, X_query[s0:s1], return_var=True)
        mu[s0:s1] = m
        var[s0:s1] = v
    return mu, var


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", default=DEFAULT_BUNDLE)
    p.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--tag", "-t", default=None)
    p.add_argument(
        "--meta-json",
        default=None,
        help="explicit path to *_meta.json (grid_norm_info); default is <input_stem>_meta.json beside --input",
    )
    p.add_argument(
        "--warm-start-config-json",
        default=None,
        help="optional prior run_gp config.json; inner 'config' block seeds optimizer hyperparameters "
        "(after --lw/--lt overrides and before L-BFGS-B; values are clipped to bounds)",
    )
    p.add_argument(
        "--train-include",
        default=None,
        help="optional npz with bool array 'include' or 'mask' (length N_train); False rows dropped",
    )
    p.add_argument(
        "--exclude-spec-bundle-ids",
        default=None,
        metavar="IDS",
        help=(
            "comma-separated spectroscopic ``spec_bundle_id`` integers to omit from training "
            "(after --train-include). Requires ``spec_bundle_id`` in the bundle NPZ. "
            "Photometry uses id -1 — do not list -1."
        ),
    )

    p.add_argument("--kernel-time", choices=gu.KERNEL_NAMES, default="matern52")
    p.add_argument("--kernel-wls", choices=gu.KERNEL_NAMES, default="matern52")
    p.add_argument("--additive-time", dest="additive_time", action="store_true", default=False)
    p.add_argument("--no-additive-time", dest="additive_time", action="store_false")
    p.add_argument("--additive-wls", dest="additive_wls", action="store_true", default=False)
    p.add_argument("--no-additive-wls", dest="additive_wls", action="store_false")

    p.add_argument("--mean", choices=gu.MEAN_NAMES, default="linear")
    p.add_argument("--phot-spec-threshold", type=int, default=50)

    p.add_argument("--lw", type=float, default=None, help="warm-start metric (squared length scale) on wavelength axis (single-scale, or 'short' if additive)")
    p.add_argument("--lt", type=float, default=None, help="warm-start metric on time axis (single-scale, or 'short' if additive)")
    p.add_argument("--lw2", type=float, default=16.0, help="warm-start LONG metric on wavelength axis (only used if --additive-wls); default 16 (length~4) matches 'color spans full spectrum'")
    p.add_argument("--lt2", type=float, default=16.0, help="warm-start LONG metric on time axis (only used if --additive-time); default 16 (length~4) bridges gaps")
    p.add_argument(
        "--lw-short",
        type=float,
        default=0.02,
        help="warm-start SHORT metric on wavelength axis when --additive-wls (larger => finer spectral structure; "
        "optimizer can move within bounds in gp_utils)",
    )
    p.add_argument("--lt-short", type=float, default=0.04, help="warm-start SHORT metric on time axis when --additive-time; default 0.04 (length~0.2) for intra-cluster smoothing")
    p.add_argument("--w-short-w", type=float, default=0.4, help="initial weight on short-wavelength scale (0..1); 0.4 = mild bias toward long (color)")
    p.add_argument("--w-short-t", type=float, default=0.4, help="initial weight on short-time scale (0..1); 0.4 = mild bias toward long")
    p.add_argument("--log-amp", type=float, default=None)
    p.add_argument("--sigma-phot", type=float, default=0.02)
    p.add_argument("--sigma-spec", type=float, default=0.01)

    p.add_argument("--optimize", dest="optimize", action="store_true", default=True)
    p.add_argument("--no-optimize", dest="optimize", action="store_false")
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument(
        "--optimize-subsample",
        type=int,
        default=2500,
        help=(
            "O(N^3); subsampling 2.5k saves >40x). 0 disables subsampling. "
            "All photometry rows are always kept; only spectroscopy may be subsampled."
        ),
    )
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--log-metric-t-min", type=float, default=None)
    p.add_argument("--log-metric-w-min", type=float, default=None)
    p.add_argument("--log-metric-t2-max", type=float, default=None)
    p.add_argument(
        "--log-metric-w2-min",
        type=float,
        default=None,
        help="optional lower bound on log_metric_w2 (log of George λ metric); "
        "use to allow a genuinely smooth long-λ component (default lower is 1.0)",
    )
    p.add_argument("--log-metric-w2-max", type=float, default=None)
    p.add_argument("--logit-weight-t-min", type=float, default=None)
    p.add_argument(
        "--logit-weight-t-max",
        type=float,
        default=None,
        help="optional upper bound on logit_weight_t (caps weight on short time kernel)",
    )
    p.add_argument("--logit-weight-w-min", type=float, default=None)
    p.add_argument(
        "--logit-weight-w-max",
        type=float,
        default=None,
        help="optional upper bound on logit_weight_w (caps weight on short-wavelength kernel)",
    )

    p.add_argument(
        "--enforce-mono-early",
        dest="enforce_mono_early",
        action="store_true",
        default=True,
        help="post-process X_fill μ for phase < --early-time-cutoff (default ON)",
    )
    p.add_argument("--no-enforce-mono-early", dest="enforce_mono_early", action="store_false")
    p.add_argument(
        "--enforce-blue-early",
        dest="enforce_blue_early",
        action="store_true",
        default=True,
        help="post-process X_fill μ for early phases: cumulative min in wls (default ON)",
    )
    p.add_argument("--no-enforce-blue-early", dest="enforce_blue_early", action="store_false")
    p.add_argument("--early-time-cutoff", type=float, default=-4.0, help="apply early-time constraints below this normalized log10(phase)")
    p.add_argument("--mono-floor-fraction", type=float, default=0.5, help="floor mu in extrapolation region at floor_fraction * mu_cutoff to prevent steep extrapolations from going absurdly low")
    p.add_argument("--mono-min-slope", type=float, default=0.005, help="minimum allowed slope for the early-time linear extrapolation (forces increasing even if GP slope is ~0 or negative)")
    p.add_argument("--mono-smoothing-scale", type=float, default=0.3, help="tanh blend scale around the cutoff (in normalized log10(phase) units) - the join is fully GP for t > cutoff + 2*scale and fully extrap for t < cutoff - 2*scale")

    p.add_argument("--predict-train", dest="predict_train", action="store_true", default=True)
    p.add_argument("--no-predict-train", dest="predict_train", action="store_false")

    p.add_argument("--chunk", type=int, default=PREDICT_CHUNK)
    args = p.parse_args(argv)

    args.output_dir = os.path.abspath(os.path.expanduser(str(args.output_dir)))

    tag = args.tag or _auto_tag(args)
    run_dir = os.path.join(args.output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run_gp] tag = {tag}")
    print(f"[run_gp] output dir = {run_dir}")

    print(f"[run_gp] loading {args.input}")
    bundle = np.load(args.input, allow_pickle=False)
    X = bundle["X"]
    y = bundle["y"]
    yerr = bundle["yerr"]
    X_fill = bundle["X_fill"]
    train_row_index_orig = np.arange(int(X.shape[0]), dtype=np.int64)

    train_obs_override = None
    if "train_obs_class" in bundle.files:
        train_obs_override = np.asarray(bundle["train_obs_class"])

    spec_bid_train: Optional[np.ndarray] = None
    if "spec_bundle_id" in bundle.files:
        spec_bid_train = np.asarray(bundle["spec_bundle_id"], dtype=np.int32).ravel()
        if spec_bid_train.shape[0] != X.shape[0]:
            raise ValueError(
                f"spec_bundle_id length {spec_bid_train.shape[0]} != N_train {X.shape[0]}"
            )

    if args.train_include:
        mpath = os.path.abspath(os.path.expanduser(args.train_include))
        inc = np.load(mpath, allow_pickle=False)
        if "include" in inc.files:
            mask = np.asarray(inc["include"], dtype=bool).ravel()
        elif "mask" in inc.files:
            mask = np.asarray(inc["mask"], dtype=bool).ravel()
        else:
            raise ValueError(f"{mpath!r}: expected 'include' or 'mask' array")
        if mask.shape[0] != X.shape[0]:
            raise ValueError(f"train-include length {mask.shape[0]} != N_train {X.shape[0]}")
        ke = np.nonzero(mask)[0]
        X, y, yerr = X[ke], y[ke], yerr[ke]
        train_row_index_orig = train_row_index_orig[ke]
        if train_obs_override is not None:
            train_obs_override = train_obs_override[ke]
        if spec_bid_train is not None:
            spec_bid_train = spec_bid_train[ke]
        print(
            f"[run_gp] applied train-include {mpath!r}: kept {ke.size}/{mask.size} rows"
        )

    exclude_ids = _parse_exclude_spec_bundle_ids(args.exclude_spec_bundle_ids)
    if exclude_ids:
        if -1 in exclude_ids:
            raise ValueError(
                "[run_gp] -1 in --exclude-spec-bundle-ids would drop photometry "
                "(spec_bundle_id=-1 for phot rows); remove -1 from the list."
            )
        if spec_bid_train is None:
            raise ValueError(
                "[run_gp] --exclude-spec-bundle-ids requires spec_bundle_id in the bundle NPZ"
            )
        n_before = int(X.shape[0])
        keep_b = ~np.isin(
            np.asarray(spec_bid_train, dtype=np.int32),
            np.asarray(exclude_ids, dtype=np.int32),
        )
        train_row_index_orig = train_row_index_orig[keep_b]
        X, y, yerr, train_obs_override, n_drop = _exclude_rows_by_spec_bundle_id(
            X, y, yerr, spec_bid_train, exclude_ids, train_obs_override
        )
        if n_drop == 0:
            print(
                f"[run_gp] WARN: --exclude-spec-bundle-ids {exclude_ids!r}: no rows matched; fit unchanged"
            )
        else:
            print(
                f"[run_gp] excluded spec_bundle_id in {sorted(set(exclude_ids))}: "
                f"dropped {n_drop} / {n_before} training rows"
            )

    point_class = gu.effective_point_class(
        X,
        threshold=args.phot_spec_threshold,
        train_obs_class=train_obs_override,
    )
    n_phot = int((point_class == gu.PHOT).sum())
    n_spec = int((point_class == gu.SPEC).sum())
    print(f"[run_gp] classification (threshold={args.phot_spec_threshold}): "
          f"phot={n_phot}, spec={n_spec}")

    mean_model = gu.build_mean(
        args.mean,
        prior_pts=bundle["prior_points"] if "prior_points" in bundle.files else None,
        prior_val=bundle["prior_values"] if "prior_values" in bundle.files else None,
        cache_workdir=HERE,
    )
    print(f"[run_gp] mean = {args.mean} -> {type(mean_model).__name__ if mean_model else 'None'}")

    cfg0 = _build_initial_config(args, bundle)
    if args.warm_start_config_json:
        wpath = os.path.abspath(os.path.expanduser(str(args.warm_start_config_json).strip()))
        if not os.path.isfile(wpath):
            print(f"[run_gp] ERROR: --warm-start-config-json not found: {wpath!r}", file=sys.stderr)
            return 2
        with open(wpath, encoding="utf-8") as wf:
            wj = json.load(wf)
        inner = wj.get("config")
        if not isinstance(inner, dict):
            print(
                f"[run_gp] ERROR: {wpath!r} missing dict 'config' block",
                file=sys.stderr,
            )
            return 2
        for key in ("additive_t", "additive_w"):
            if key in inner and bool(inner[key]) != bool(getattr(cfg0, key, None)):
                print(
                    f"[run_gp] WARN: warm-start JSON {key}={inner[key]!r} != current {getattr(cfg0, key)!r}; "
                    "keeping current kernel layout (only matching free params are applied)",
                    file=sys.stderr,
                )
        applied = cfg0.apply_saved_inner_config(inner)
        print(f"[run_gp] warm-start from {wpath!r}: applied {len(applied)} hyperparameter(s)")
    print("[run_gp] initial config:", json.dumps(cfg0.as_dict(), indent=2))

    t_total = time.time()
    ll0: Optional[float] = None
    ll_final: Optional[float] = None

    if args.optimize:
        if args.optimize_subsample and args.optimize_subsample > 0 and args.optimize_subsample < X.shape[0]:
            rng = np.random.default_rng(args.seed)
            phot_idx = np.where(point_class == gu.PHOT)[0]
            spec_idx = np.where(point_class == gu.SPEC)[0]
            n_spec_keep = max(args.optimize_subsample - phot_idx.size, 100)
            spec_keep = rng.choice(spec_idx, size=min(n_spec_keep, spec_idx.size), replace=False)
            sub_idx = np.sort(np.concatenate([phot_idx, spec_keep]))
            X_opt = X[sub_idx]
            y_opt = y[sub_idx]
            yerr_opt = yerr[sub_idx]
            cls_opt = point_class[sub_idx]
            print(
                f"[run_gp] optimization on subsample N={sub_idx.size} "
                f"(phot={phot_idx.size}, spec={spec_keep.size}); "
                f"final fit will use full N={X.shape[0]}"
            )
        else:
            X_opt, y_opt, yerr_opt, cls_opt = X, y, yerr, point_class
            print(f"[run_gp] optimization on full N={X.shape[0]}")

        print("[run_gp] optimizing hyperparameters with L-BFGS-B...")
        neg_ll, counter = _make_neg_ll(cfg0, X_opt, y_opt, yerr_opt, cls_opt, mean_model)
        theta0 = cfg0.to_vector()
        bounds, bound_msgs = _tightened_optimizer_bounds(
            cfg0,
            log_metric_t_min=args.log_metric_t_min,
            log_metric_w_min=args.log_metric_w_min,
            log_metric_t2_max=args.log_metric_t2_max,
            log_metric_w2_min=args.log_metric_w2_min,
            log_metric_w2_max=args.log_metric_w2_max,
            logit_weight_t_min=args.logit_weight_t_min,
            logit_weight_t_max=args.logit_weight_t_max,
            logit_weight_w_min=args.logit_weight_w_min,
            logit_weight_w_max=args.logit_weight_w_max,
        )
        if bound_msgs:
            print("[run_gp] kernel optimizer bound overrides:")
            for line in bound_msgs:
                print(f"  {line}")
        theta0 = _clip_theta_to_bounds(theta0, bounds)
        ll0 = float(-neg_ll(theta0))
        print(f"  initial log-likelihood (subsample) = {ll0:.4f}")
        t0 = time.time()
        res = minimize(
            neg_ll,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": args.max_iter, "disp": False, "ftol": 1e-7},
        )
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.2f} s, neg_ll evals = {counter['n']}")
        print(f"  optimizer success = {res.success}, message = {res.message!r}")
        if counter["best_theta"] is not None and -counter["best"] >= -res.fun:
            cfg0.update_from_vector(counter["best_theta"])
            ll_final = float(-counter["best"])
        else:
            cfg0.update_from_vector(res.x)
            ll_final = float(-res.fun)
        print(f"  final log-likelihood = {ll_final:.4f}")
    else:
        elapsed = 0.0
        counter = {"n": 0, "best": np.inf, "best_theta": None}
        print("[run_gp] optimization skipped (--no-optimize)")

    cfg_final = cfg0
    print("[run_gp] final config:", json.dumps(cfg_final.as_dict(), indent=2))

    kernel = gu.build_kernel(cfg_final)
    gp = george.GP(kernel, mean=mean_model) if mean_model is not None else george.GP(kernel)
    sigma_phot = float(np.exp(cfg_final.log_sigma_phot))
    sigma_spec = float(np.exp(cfg_final.log_sigma_spec))
    diag = gu.compute_diagonal(yerr, point_class, sigma_phot, sigma_spec)

    print("[run_gp] gp.compute(X, diag) ...")
    t0 = time.time()
    gp.compute(X, diag)
    print(f"[run_gp]   done in {time.time() - t0:.2f} s")

    log_lik = float(gp.log_likelihood(y))
    print(f"[run_gp] log-likelihood = {log_lik:.4f}")

    print(
        f"[run_gp] predicting on X_fill (N={X_fill.shape[0]}) in chunks of {args.chunk} ..."
    )
    t0 = time.time()
    mu, var = _predict_chunked(gp, y, X_fill, args.chunk)
    n_neg = int((var < 0).sum())
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    mu_raw = mu.copy()
    print(f"[run_gp]   prediction took {time.time() - t0:.2f} s, neg variances clipped: {n_neg}")
    print(f"[run_gp]   mu  (raw): {_fmt_arr(mu)}")
    print(f"[run_gp]   std (raw): {_fmt_arr(std)}")

    n_modified_mono = 0
    n_modified_blue = 0
    if args.enforce_mono_early:
        mu, std, n_modified_mono = _enforce_monotone_early(
            X_fill,
            mu,
            std,
            args.early_time_cutoff,
            floor_fraction=args.mono_floor_fraction,
            min_slope=args.mono_min_slope,
            smoothing_scale=args.mono_smoothing_scale,
        )
        print(
            f"[run_gp] enforced monotonic early-time (tanh-blend, cutoff={args.early_time_cutoff}, scale={args.mono_smoothing_scale}): "
            f"{n_modified_mono} grid points adjusted"
        )
    if args.enforce_blue_early:
        mu, n_modified_blue = _enforce_blue_early(X_fill, mu, args.early_time_cutoff)
        print(
            f"[run_gp] enforced blue early-time (cutoff={args.early_time_cutoff}): "
            f"{n_modified_blue} grid points adjusted (cumulative-min in wls)"
        )
    if args.enforce_mono_early or args.enforce_blue_early:
        print(f"[run_gp]   mu  (post): {_fmt_arr(mu)}")

    mu_train = std_train = None
    chi2 = chi2_phot = chi2_spec = None
    if args.predict_train:
        print(f"[run_gp] predicting on X (N={X.shape[0]}) for residuals ...")
        t0 = time.time()
        mu_train, var_train = _predict_chunked(gp, y, X, args.chunk)
        var_train = np.maximum(var_train, 0.0)
        std_train = np.sqrt(var_train)
        resid = mu_train - y
        sigma_eff = diag.copy()
        norm_resid = resid / sigma_eff
        within = int((np.abs(resid) <= sigma_eff).sum())
        chi2 = float(np.sum(norm_resid**2))
        chi2_phot = float(np.sum(norm_resid[point_class == gu.PHOT] ** 2))
        chi2_spec = float(np.sum(norm_resid[point_class == gu.SPEC] ** 2))
        print(f"[run_gp]   prediction took {time.time() - t0:.2f} s")
        print(f"[run_gp] residuals (mu_train - y):")
        print(f"  mean = {resid.mean():.6g}, std = {resid.std():.6g}, max|.| = {np.abs(resid).max():.6g}")
        print(f"  within +/- sigma_eff: {within}/{y.size} ({100*within/y.size:.1f}%)")
        print(f"  chi^2/N total = {chi2/y.size:.4f}")
        print(f"  chi^2/N phot  = {chi2_phot/max(n_phot,1):.4f} (n={n_phot})")
        print(f"  chi^2/N spec  = {chi2_spec/max(n_spec,1):.4f} (n={n_spec})")

    pred_path = os.path.join(run_dir, "predictions.npz")
    save_payload = dict(
        X_fill=X_fill,
        mu=mu,
        std=std,
        var=var,
        mu_raw=mu_raw,
        point_class_train=point_class,
        sigma_eff_train=diag,
        y_train=y,
        yerr_train=yerr,
    )
    if mu_train is not None:
        save_payload["mu_train"] = mu_train
        save_payload["std_train"] = std_train
        if int(train_row_index_orig.shape[0]) != int(np.asarray(mu_train).shape[0]):
            raise RuntimeError("internal: train_row_index_orig length must match mu_train")
        save_payload["train_row_index_orig"] = train_row_index_orig
    np.savez(pred_path, **save_payload)
    print(f"[run_gp] wrote {pred_path}")

    inp_abs = os.path.abspath(args.input)
    meta_path_opt = (
        os.path.abspath(os.path.expanduser(args.meta_json)) if args.meta_json else None
    )
    meta_bundle = bmeta.load_bundle_meta(inp_abs, meta_path=meta_path_opt)
    sibling_meta = os.path.abspath(bmeta.bundle_meta_json_path(inp_abs))

    config_payload = {
        "tag": tag,
        "input": os.path.abspath(args.input),
        "train_obs_class_in_bundle": bool(train_obs_override is not None),
        "train_include_file": os.path.abspath(args.train_include) if args.train_include else None,
        "exclude_spec_bundle_ids": sorted(set(exclude_ids)) if exclude_ids else None,
        "kernel_time": args.kernel_time,
        "kernel_wls": args.kernel_wls,
        "additive_time": args.additive_time,
        "additive_wls": args.additive_wls,
        "mean": args.mean,
        "phot_spec_threshold": args.phot_spec_threshold,
        "n_phot": n_phot,
        "n_spec": n_spec,
        "optimize": args.optimize,
        "max_iter": args.max_iter,
        "neg_ll_evals": counter.get("n", 0),
        "optimize_seconds": elapsed,
        "log_likelihood_initial": ll0 if args.optimize else None,
        "log_likelihood_final": ll_final if args.optimize else log_lik,
        "log_likelihood_at_compute": log_lik,
        "n_neg_var_clipped": n_neg,
        "enforce_mono_early": args.enforce_mono_early,
        "enforce_blue_early": args.enforce_blue_early,
        "early_time_cutoff": args.early_time_cutoff,
        "mono_floor_fraction": args.mono_floor_fraction,
        "mono_min_slope": args.mono_min_slope,
        "mono_smoothing_scale": args.mono_smoothing_scale,
        "n_modified_mono": n_modified_mono,
        "n_modified_blue": n_modified_blue,
        "kernel_bound_overrides": {
            k: v
            for k, v in (
                ("log_metric_t_min", args.log_metric_t_min),
                ("log_metric_w_min", args.log_metric_w_min),
                ("log_metric_t2_max", args.log_metric_t2_max),
                ("log_metric_w2_min", args.log_metric_w2_min),
                ("log_metric_w2_max", args.log_metric_w2_max),
                ("logit_weight_t_min", args.logit_weight_t_min),
                ("logit_weight_t_max", args.logit_weight_t_max),
                ("logit_weight_w_min", args.logit_weight_w_min),
                ("logit_weight_w_max", args.logit_weight_w_max),
            )
            if v is not None
        },
        "chi2_per_n_total": chi2 / y.size if chi2 is not None else None,
        "chi2_per_n_phot": chi2_phot / max(n_phot, 1) if chi2_phot is not None else None,
        "chi2_per_n_spec": chi2_spec / max(n_spec, 1) if chi2_spec is not None else None,
        "config": cfg_final.as_dict(),
        "total_runtime_seconds": time.time() - t_total,
        "warm_start_config_json": (
            os.path.abspath(os.path.expanduser(str(args.warm_start_config_json)))
            if args.warm_start_config_json
            else None
        ),
    }
    tried_meta_paths = sibling_meta if meta_path_opt is None else meta_path_opt
    if meta_bundle and isinstance(meta_bundle.get("grid_norm_info"), dict):
        config_payload["grid_norm_info"] = dict(meta_bundle["grid_norm_info"])
        config_payload["bundle_meta_json"] = meta_path_opt or sibling_meta
    else:
        print(
            f"[run_gp] no bundle meta (tried {tried_meta_paths!r}); "
            "pass --meta-json /path/to/*_meta.json or place meta beside input; "
            "plot_results.py/plot_bands_gp_overview need grid_norm_info for physical axes.",
            file=sys.stderr,
        )

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    print(f"[run_gp] wrote {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
