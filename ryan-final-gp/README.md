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
| `runs/<tag>/figs/outliers/` | Outlier LC / spectrum plots (`plot_outliers.py`) |
| `runs/<tag>/figs/overview/` | Band-by-band phot + bundle spectra vs GP (`plot_bands_gp_overview.py`) |
| `WRITEUP.pdf` | Change log + rationale |
| `filter_synthesis.py` | Optional: resolve pysynphot bandpasses using TRDS; missing filters reported, not fatal |
| `spectrum_bundles.py` | Optional: cluster spectra by epoch; composite λ grid + small-gap fill |
| `outlier_pipeline.py` | Flag large standardized residuals; write `outliers_iter*.json` and optional `train_include.npz` |
| `plot_outliers.py` | LC + spectrum diagnostic plots for flagged points → `runs/<tag>/figs/outliers/` |
| `bundle_preprocess.py` | After loading data: locate rows, force phot labels, telluric spike repair → new `*.npz` |
| `plot_bands_gp_overview.py` | All phot bands + GP + optional pysynphot synth; spectral bundles + GP slice → `figs/overview/` |
| `iterative_gp.py` | Loop: `run_gp` → `plot_results` → outlier mask → repeat with `--train-include` |
| `iterate_gp_surface_bundle_scale.py` | Outer loop: `run_gp` (with optional warm-start from prior `config.json`) → interpolate GP surface on `X_fill` → WLS linear scale per `spec_bundle_id` → next bundle; JSONL + metrics + optional overview plots |
| `gp_grid_interp.py` | `LinearNDInterpolator` + `NearestNDInterpolator` fallback: latent μ from fill grid onto arbitrary training rows (shared by overview + surface driver) |
| `configs/filter_pipeline.example.yaml` | Example TRDS roots + `band_aliases` for filter synthesis |
| `requirements-preprocess.txt` | Optional deps (`pysynphot`, `PyYAML`) |
| `bundle_scale_pipeline.py` | Time bundles, intra-arm + intra-bundle **relative** scaling, optional **absolute** (phot) anchoring |
| `run_full_pipeline.py` | Preprocess → scaler → final `run_gp` → plots (see docstring for `--skip-global-phot-anchor`) |
| `tests/test_bundle_scale_pipeline.py` | Regression tests for intra-bundle **relative** spectral scaling (gap seams, MST edges, phot anchors). Treat as the lock on scaler geometry: do not weaken gap/edge behavior without updating tests. |

## Installation

Python 3.10+ recommended. Dependencies:

```bash
pip install numpy scipy matplotlib george
```

`george` is the only non-standard dep (Gaussian-process library, MIT-licensed).
There is no `requirements.txt` yet; add one if you want to pin versions.

Synthetic photometry and YAML-driven filter lists (optional):

```bash
pip install -r requirements-preprocess.txt
```

Point `trds_roots` in `configs/filter_pipeline.example.yaml` at your local TRDS tree (directory containing `grp/`). Export `PYSYN_CDBS` is set automatically from the first valid root.

## Quick start

End-to-end, three commands:

```bash
# 1. Train + predict + optional early-time post-process on X_fill μ (defaults on).
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

### Full scaler + GP (relative and absolute scaling)

`run_gp.py` trains on whatever is in the input npz. For production, that bundle should include **intra-bundle relative** scaling and **photometric absolute** anchoring. **`bundle_scale_pipeline.py`** and **`run_full_pipeline.py`** default to **`--global-scale-iters` 1** (phot anchor on). With **enrich + filter YAML** (auto-discovered when present), the scaler uses the **band + inner GP** path; **without enrich**, photometric anchoring still runs via **rough / pooled χ²** on photometry in the training bundle (**no enrich required**). Use **`--skip-global-phot-anchor`** or **`--global-scale-iters 0`** for relative-only.

### Iterative GP surface rescaling (`iterate_gp_surface_bundle_scale.py`)

This driver alternates a full **`run_gp`** fit (re-optimizing length scales and jitters each time, with **`--warm-start-config-json`** from the previous iteration when available) with a **single linear flux multiplier per spectroscopic `spec_bundle_id`**, chosen by WLS against the GP latent surface interpolated from **`predictions.npz`** (`X_fill` and **`mu_raw`** by default via `--surface-mu-key`). Photometry rows are not rescaled. Workspace layout: `iter_00/bundle.npz` (copy of input), `iter_k/bundle.npz`, mirrored `predictions.npz` / `config.json`, `iteration_log.jsonl`, `metrics/*.png`, optional `figs/overview/` (subset via `--diag-bundles`), `scaling/*.csv` and `spec_bundle_*_scale_evolution.png` for highlight bundles (default 3 and 5 if `--diag-bundles` is omitted). Only **`run_gp`** options go after a lone **`--`** (e.g. **`--optimize`**, **`--log-metric-w-min`**); driver flags such as **`--diag-bundles`**, **`--diag-full-overview-interval`**, and **`--plot-results-each-iter`** must appear **before** **`--`**. Summary PNGs under **`metrics/`** are written when the outer loop finishes (each iteration still appends **`iteration_log.jsonl`**). Use **`--max-iters`** / **`--converge-max-log-scale`** / **`--converge-delta-chi2-spec`** for the outer loop.

## CLI reference

### `run_gp.py`

```text
--tag NAME                    output dir is runs/<tag>/ (auto-named from flags if omitted)
--input PATH                  input bundle (default: gp_minimal_bundle.npz)
--train-include PATH          optional npz with bool ``include`` or ``mask`` (length N_train); False rows dropped
--output-dir DIR              parent of runs/

--kernel-time {matern32,matern52,exp_squared,rational_quadratic}
--kernel-wls  {matern32,matern52,exp_squared,rational_quadratic}
--additive-time / --no-additive-time      sum of two same-family kernels on time axis
--additive-wls  / --no-additive-wls       same on wls axis
--mean {none,constant,nearest,linear}     mean function (linear is default)
--phot-spec-threshold INT     min n_wls/phase to call a phase "spec" (default 50)

# Warm starts (all metrics are squared length scales in george's convention)
--warm-start-config-json PATH             seed optimizer state from a prior run's ``config.json``
                                          (inner ``config`` dict: ``log_amp``, ``log_metric_*``, jitters, etc.;
                                          applied after scalar overrides; clipped to L-BFGS-B bounds)
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

# Early-time monotonicity / blue-spectrum constraints (post-processing of X_fill μ only)
--enforce-mono-early / --no-enforce-mono-early    default ON (below --early-time-cutoff, default -4)
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
--no-spec-overlap-scale          turn off overlap-based λ-segment flux rescaling
                                 on `gp_spectra.png` (default: scaling ON)
--spec-segment-gap-factor FLOAT  split spectra into contiguous λ chunks when Δ
                                 exceeds this × median spacing (default 35)
--spec-min-gap-norm FLOAT      minimum Δlog10(λ)_norm forcing a segment break (default 0.003)
--plot-as-phot-indices LIST    comma-separated train rows to plot as photometry
                                 (see `bundle_preprocess.py find`)
```

Always reads `runs/<tag>/predictions.npz` and `runs/<tag>/config.json`,
writes to `runs/<tag>/figs/`. Six figures per run:

- `gp_results_wavelength_slices.pdf` — collaborator-style multi-panel slice plot
- `gp_results_wavelength_slices_linear_phase_linear_flux.pdf` — same on linear axes
- `gp_mu_heatmap.png` — posterior mean as a 2-D heatmap
- `gp_std_heatmap.png` — posterior std heatmap
- `training_coverage.png` — scatter of training points by class
- `gp_mu_phase_profiles.png` — phase profiles at six wavelengths with overlaid data
- `gp_spectra.png` — "snap to nearest real spectrum" panels with all near-simultaneous spec phases overlaid; overlapped λ−segments default to **median flux-ratio alignment** (**plot-only**, not applied to GP training)
- `training_residuals.png` — residual histograms by class

### `plot_outliers.py`

Writes **`runs/<tag>/figs/outliers/`** after loading `predictions.npz` (needs `mu_train`) and `outliers_iter{k}.json`, or recomputes flags with `--recompute-outliers`.

| Figure | Meaning |
|---|---|
| `phot_outliers_*.png` | One panel per **band** (from optional `--enrich` npz `band_name` / `band_id`) or per **pseudo-band** (rounded normalized log wavelength) **only if that band contains a flagged phot point**. Gray points = all phot in bin; **crimson rings** = outliers; blue curve = smooth GP posterior (`mu_train`) in time; optional green curve from enrich **synth** arrays (see below). |
| `spec_outliers_phase_*.png` | For each flagged spectroscopic **exposure phase** (`X[:,1]`): overlap-scaled λ segments for all spec phases within `--near-phase-tol`; **thicker** line = the flagged phase; **crimson rings** = flagged pixels. |

Optional **`--enrich train_enrich.npz`** (same row order as bundle `N`):

- `mjd` — x-axis for photometry LCs instead of phase days.
- `band_name` (UTF object array) or `band_id` — real band grouping for phot plots.
- `synth_times`, `synth_flux` — single synthetic photometry curve (same units as linear flux), plotted as **default** fallback for every band panel that does not define its own curve.
- Per-band: `synth_times_<band>` + `synth_flux_<band>` matching string labels from `band_name`.

With **only** the minimal bundle (no enrich), phot plots use **pseudo-bands** and the smooth line is **not** true synthetic photometry from filter convolution — wire `filter_synthesis.py` + spectra in a future ingest step for that.

If the heuristic labels a **photometry point as spec**, use **`bundle_preprocess.py find`** to print candidate train indices, then either:

- **`plot_outliers.py --plot-as-phot-indices i`** — treat row `i` as photometry **for figures only**, or  
- **`bundle_preprocess.py preprocess ... --phot-indices i`** — write **`train_obs_class`** into a new npz so **`run_gp`** uses true phot/spec labels.

### `bundle_preprocess.py`

Two subcommands:

```text
find    --bundle BUNDLE --norm-phase X2 --log10-wavelength PHYS_LOG10_WL
        [--phase-tolerance] [--log10-wl-tolerance]

preprocess  -i INPUT -o OUTPUT
            [--phot-indices i,j,...]      force rows to photometry (writes train_obs_class)
            [--telluric] [--telluric-phases p,...] | [--telluric-all-spec]
            [--telluric-sigma] [--telluric-dilate] [--median-window]
```

**Telluric repair:** For each requested spectroscopic phase (snapped to the nearest phase present in the bundle), pixels with robust median residuals or extreme spikes are masked, **linear flux interpolated** across λ from good neighbours in latent space re-encoded into `y`, and **`yerr` / `y_compute` set to ~1e30** (finite “infinite” noise). Arrays **`telluric_bad_mask`** and optionally **`train_obs_class`** are added to the output npz.

Re-fit the GP with **`python run_gp.py --input gp_bundle_edited.npz ...`**.

### `plot_bands_gp_overview.py`

High-level photometry and spectroscopy figures (separate from per-run residual tooling).

```bash
python plot_bands_gp_overview.py --bundle gp_minimal_bundle.npz --tag matern52_linear_opt
```

- **Photometry:** one PNG per band — either **`band_name` / `band_id`** from **`--enrich`**, or pseudo-bands (rounded norm log λ). Overlays the GP slice along time: by default **`--phot-lc-time-step-days 0.05`** resamples the latent posterior on a dense time grid (MJD if **`--enrich`** has **`mjd`**, else phase in days) via 2D interpolation, so the blue curve is not just straight chords between sparse epochs. Set **`0`** to fall back to line segments through training-row µ only. Residuals use the same grid when **`--plot-residuals-vs-gp`** is on. With **`--filter-config`** (YAML for TRDS + `band_aliases`), attempts **synthetic photometry** from the nearest spectroscopic exposure at each phot epoch via **`filter_synthesis`**; plotted on a **secondary axis in AB magnitudes** (training flux stays linear on the left — compare qualitatively). Writes **`filter_synth_report.json`** when filters resolve/skip.

- **Spectra:** clusters unique spec phases by **`phase_days`** within **`--bundle-minutes`** (default 5). For each bundle, overlap-scales λ segments (same helper as `plot_results`) and overlays **GP mean ±1σ** taken at the grid phase nearest the bundle median.

Outputs default to **`runs/<tag>/figs/overview/`**.

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
| `y_train` | (N,) | training targets actually used in the fit (matches `mu_train`; supports masked runs) |
| `yerr_train` | (N,) | input `yerr` after any row masking |

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
