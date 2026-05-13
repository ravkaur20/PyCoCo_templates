# Time-evolving SED Gaussian Process — README and integration guide

This directory contains a small self-contained pipeline that fits a 2-D Gaussian
Process to the SED data in `gp_minimal_bundle.npz`, produces diagnostic plots,
and supports side-by-side comparison of multiple configurations. This document
is both the README (how to use it as-is) and the integration guide (how to drop
the pieces into a larger codebase). For the *why*, see `WRITEUP.pdf`.

## Contents

| File | Role |
|---|---|
| `gp_utils.py` | Library: kernel/mean factories, `KernelConfig` dataclass, point classification, jitter diagonal |
| `run_gp.py` | CLI: load bundle, build kernel + mean, optimize, predict, post-process, save |
| `plot_results.py` | CLI: load a run, produce all diagnostic figures |
| `compare_runs.py` | CLI: overlay two or more runs (heatmaps, phase profiles, summary table) |
| `gp_minimal_bundle.npz` | Input data (collaborator-supplied) |
| `gp.info` | Collaborator's notes on the bundle |
| `plot.py` | Collaborator's original plot script (kept for reference) |
| `prior_linear_interp.pkl` | Cached `LinearNDInterpolator` + `NearestNDInterpolator` for the prior mean (rebuilt automatically if deleted) |
| `runs/<tag>/predictions.npz` | Per-run predictions (see schema below) |
| `runs/<tag>/config.json` | Per-run config + metrics |
| `runs/<tag>/figs/` | Per-run figures |
| `WRITEUP.pdf` | Change log + rationale |

## Installation

Python 3.10+ recommended. Dependencies:

```bash
pip install numpy scipy matplotlib george
```

`george` is the only non-standard dep (Gaussian-process library, MIT-licensed).
There is no `requirements.txt` yet; add one if you want to pin versions.

## Quick start

End-to-end, three commands:

```bash
# 1. Train + predict + post-process. ~140 s on a laptop.
python run_gp.py --tag my_run --kernel-time matern52 --kernel-wls matern52 \
    --additive-time --additive-wls --mean linear --optimize \
    --enforce-mono-early --enforce-blue-early

# 2. Make all diagnostic figures.
python plot_results.py --tag my_run

# 3. Compare against the production run.
python compare_runs.py --tags my_run matern52_addw_addt_linear_opt_v5
```

Outputs land under `runs/my_run/`. The "production" reference is
`runs/matern52_addw_addt_linear_opt_v5/`.

## CLI reference

### `run_gp.py`

```text
--tag NAME                    output dir is runs/<tag>/ (auto-named from flags if omitted)
--input PATH                  input bundle (default: gp_minimal_bundle.npz)
--output-dir DIR              parent of runs/

--kernel-time {matern32,matern52,exp_squared,rational_quadratic}
--kernel-wls  {matern32,matern52,exp_squared,rational_quadratic}
--additive-time / --no-additive-time      sum of two same-family kernels on time axis
--additive-wls  / --no-additive-wls       same on wls axis
--mean {none,constant,nearest,linear}     mean function (linear is default)
--phot-spec-threshold INT     min n_wls/phase to call a phase "spec" (default 50)

# Warm starts (all metrics are squared length scales in george's convention)
--lw / --lt / --lw2 / --lt2                wls/time short/long metrics
--lw-short / --lt-short                    short-scale warm start when additive
--w-short-w / --w-short-t                  initial weight on short scale (0..1)
--log-amp / --sigma-phot / --sigma-spec    amplitude and per-class jitters

# Optimization
--optimize / --no-optimize                  L-BFGS-B on the marginal likelihood
--max-iter INT                              L-BFGS-B max iterations
--optimize-subsample N                      use N random points (default 2500;
                                            all phot are kept; 0 disables subsampling)
--seed INT                                  rng seed for the subsample

# Early-time monotonicity / blue-spectrum constraints (post-processing)
--enforce-mono-early / --no-enforce-mono-early    default ON
--enforce-blue-early / --no-enforce-blue-early    default ON
--early-time-cutoff FLOAT                  normalized log10(phase) cutoff (default -4)
--mono-floor-fraction FLOAT                 floor mu at floor_fraction * mu_cutoff
--mono-min-slope FLOAT                      min slope of the linear extrapolation
--mono-smoothing-scale FLOAT                tanh blend scale around the cutoff (default 0.3)

--predict-train / --no-predict-train        predict on X to get residuals (default ON)
--chunk INT                                 chunk size for chunked predict (default 10000)
```

The post-processing constraints are described in detail in `WRITEUP.pdf` §4.5–4.6.

### `plot_results.py`

```text
--tag NAME                       run to plot (under runs/)
--bundle PATH                    bundle to read training X/y/yerr from
--spectrum-phases "STRING"       space-separated normalized log-phase values
                                 to take spectra at (default "-2 -1 0 0.5 1")
--spectrum-tolerance FLOAT       within how much (normalized log-phase) to count
                                 a spec phase as near-simultaneous to the
                                 chosen one (default 0.05)
```

Always reads `runs/<tag>/predictions.npz` and `runs/<tag>/config.json`,
writes to `runs/<tag>/figs/`. Six figures per run:

- `gp_results_wavelength_slices.pdf` — collaborator-style multi-panel slice plot
- `gp_results_wavelength_slices_linear_phase_linear_flux.pdf` — same on linear axes
- `gp_mu_heatmap.png` — posterior mean as a 2-D heatmap
- `gp_std_heatmap.png` — posterior std heatmap
- `training_coverage.png` — scatter of training points by class
- `gp_mu_phase_profiles.png` — phase profiles at six wavelengths with overlaid data
- `gp_spectra.png` — "snap to nearest real spectrum" panels with all near-simultaneous spec phases overlaid
- `training_residuals.png` — residual histograms by class

### `compare_runs.py`

```text
--tags TAG [TAG ...]   one or more runs to compare
--output DIR           where to write the comparison plots
--runs-dir DIR         parent dir (default: runs)
```

Outputs three PNGs (`compare_mu_heatmaps.png`, `compare_std_heatmaps.png`,
`compare_phase_profiles.png`) and a one-line-per-run `summary.tsv`.

## Output schemas

### `runs/<tag>/predictions.npz`

| Key | Shape | Description |
|---|---|---|
| `X_fill` | (N_pred, 2) | normalized (log-wls, log-phase) prediction grid |
| `mu` | (N_pred,) | posterior mean **after** post-processing |
| `mu_raw` | (N_pred,) | posterior mean **before** post-processing |
| `std` | (N_pred,) | posterior std (sqrt of clipped variance) |
| `point_class_train` | (N,) | per-training-point class (`'phot'` / `'spec'`) |
| `sigma_eff_train` | (N,) | per-training-point effective sigma used in `gp.compute` |
| `mu_train` | (N,) | posterior mean on the training X (only if `--predict-train`) |

### `runs/<tag>/config.json`

A flat JSON object with:

- All CLI flag values (kernel, mean, optimization, post-processing).
- `n_phot`, `n_spec` after classification.
- `log_likelihood_initial`, `log_likelihood_final`, `log_likelihood_at_compute` (the last is the most reliable; the first two are on the optimization subsample).
- `chi2_per_n_total`, `chi2_per_n_phot`, `chi2_per_n_spec`.
- `n_modified_mono`, `n_modified_blue` — number of grid points changed by each constraint.
- `config` — the final `KernelConfig` as a dict (every kernel hyperparameter, in both log and natural units).
- `total_runtime_seconds`.

This is intentionally exhaustive so a single `config.json` is sufficient to reproduce a run.

## Module API

If you import `gp_utils.py` from another script:

```python
import gp_utils as gu

# Classification
point_class = gu.classify_points(X, threshold=50)  # array of 'phot' / 'spec'

# Effective per-point sigma (for gp.compute(X, diag))
diag = gu.compute_diagonal(yerr, point_class, sigma_phot, sigma_spec)

# Build kernel + mean
cfg = gu.KernelConfig(
    name_t="matern52", name_w="matern52",
    additive_t=True, additive_w=True,
    log_amp=..., log_metric_t=..., log_metric_w=...,
    log_metric_t2=..., log_metric_w2=...,
    logit_weight_t=..., logit_weight_w=...,
    log_sigma_phot=..., log_sigma_spec=...,
)
mean_model = gu.build_mean("linear", prior_pts, prior_val, cache_workdir=".")
gp = gu.make_gp(cfg, mean_model)

# Vector <-> KernelConfig conversion (for use with scipy.optimize)
vec, names = cfg.to_vector()              # only the *free* (non-fixed) params
cfg2 = cfg.from_vector(vec)               # round-trip
bounds = cfg.optimization_bounds()        # list of (lo, hi) in vector order
```

`KernelConfig` is a `@dataclass` and is JSON-serialisable via `cfg.as_dict()`.

## Things to refactor before integrating into a larger codebase

These are deliberate placeholders or local-laptop conveniences. None of them
affect correctness for the present run, but you'll likely want to clean them
up when this leaves my workspace.

1. **`grid_norm_info`.** `plot_results.py` plots in the *normalized* coordinate
   frame because `gp_minimal_bundle_meta.json` (with `x_means`, `x_stds`,
   `y_mean`, `y_std`) was not in the bundle. The script has identity-fallback
   `denormalize_*` helpers and a `GRID_NORM_INFO` placeholder dict at module
   scope. Once the meta JSON exists, point them at it and axes will render in
   physical units. Search `plot_results.py` for `GRID_NORM_INFO`.

2. **Hard-coded default paths.** `DEFAULT_BUNDLE = "gp_minimal_bundle.npz"`
   and `DEFAULT_OUTPUT_DIR = "runs"` in both scripts. Replace with config
   loaded from a project-level `paths.py` or env vars.

3. **Mean-interpolator cache location.** `prior_linear_interp.pkl` is created
   in the current working directory by `gp_utils._build_linear_with_nearest_fallback`.
   It's ~85 MB. Plumb the cache directory through as a flag (`--mean-cache-dir`)
   if you don't want it in CWD.

4. **`classify_points` heuristic.** The `n_wls/phase >= 50` rule works for the
   present bundle but is brittle. A real implementation should consume an
   explicit class label from the bundle.

5. **`predict_train` cost.** `run_gp.py` calls `gp.predict(y, X)` to compute
   residuals (~12 s). The same residuals are available for free from the
   already-factored Cholesky as $\Sigma\,(K+\Sigma)^{-1}\,y$. Replace if you
   end up running thousands of configurations.

6. **`PriorMeanModel` cache shape.** The cache pickles a `LinearNDInterpolator`
   and a `NearestNDInterpolator`. SciPy's pickle format for these is not
   guaranteed stable across versions. Bump the cache version key (or just
   `rm prior_linear_interp.pkl`) if you upgrade SciPy.

7. **Subsample seed.** `--seed 0` by default. If you do model selection across
   many configurations, vary the seed and take medians, not maxes.

8. **`enforce_blue_early` is order-dependent.** `np.minimum.accumulate` on a
   wls-sorted column gives a monotonically non-increasing function, but the
   resulting flux values can be slightly below the GP at large wls. If
   downstream code cares about the difference, capture both `mu_raw` (saved)
   and `mu` (post-processed).

9. **No prior on hyperparameters.** L-BFGS-B maximizes the marginal likelihood
   with hard bounds. If you want a Bayesian treatment, wrap `_make_neg_ll` in
   `emcee` --- the param vector / bounds plumbing is already in place via
   `KernelConfig`.

## Extending

Adding a new kernel family:

1. Add a branch in `gp_utils._normalized_axis_kernel` that constructs the
   `george` kernel for the new family on a single axis.
2. Add the family name to `gp_utils.KERNEL_NAMES` (the choices set used by the
   CLI parser).

Adding a new mean function:

1. Implement a `george.modeling.Model` subclass (see `_LinearPriorMeanModel`
   for the template).
2. Add a branch in `gp_utils.build_mean` and add the name to `MEAN_NAMES`.

Adding a new diagnostic figure:

1. Write the function in `plot_results.py` taking `(X_fill, mu, std, X, y, yerr, point_class, ...)`.
2. Call it from `main()` after the existing figures.

## Notes on reproducibility

- `numpy.random.default_rng(seed)` is used for the optimization subsample, so
  the same `--seed` reproduces the subsample exactly.
- L-BFGS-B is deterministic given the same starting vector and bounds.
- Floating-point order may shift the last few digits between runs; the
  reported $\log L$ should agree to $\sim 10^{-2}$.

## Where to read for "why"

- `WRITEUP.pdf` §4 — kernel/mean/jitter rationale.
- `WRITEUP.pdf` §5 — diagnoses of the v3/v4 regressions and why v5 fixes them.
- `WRITEUP.pdf` Appendix A — full iteration log v1 → v5.

## Regenerating the writeup PDF

```bash
pandoc WRITEUP.md -o WRITEUP.pdf --pdf-engine=pdflatex \
    --toc --toc-depth=2 -V colorlinks=true -V linkcolor=blue
```

(Requires `pandoc` and a TeX distribution providing `pdflatex`.)
