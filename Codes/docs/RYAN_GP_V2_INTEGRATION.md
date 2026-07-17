# Ryan v2 GP integration — changes, runbook, and knobs

This document summarizes the **Ryan v2** integration in `PyCoCo_templates`: vendored scripts, parallel **twodim_ryanv2** output tree, optional **iterative GP + bundle scaling**, **identity-only re-mangle** for notebook 7, a **photo–x₁ bucket** diagnostic, and **downstream FINAL path** toggles.

---

## 1. What changed (by area)

### 1.1 Vendored tooling — `Codes/ryan_gp/`

- Ryan’s Python drivers (e.g. `run_gp.py`, `iterate_gp_surface_bundle_scale.py`, `strip_photometry_bands.py`, `bundle_scale_pipeline.py`, `gp_utils.py`, plotting/diagnose helpers) live here.
- **`Codes/ryan_gp/README.md`**: how to run with `cwd`/`PYTHONPATH`.

### 1.2 Path configuration — `Codes/pipeline_config.py`

New constants mirror the existing RJF parallel layout:

- `TWODIM_RYANV2_SUBDIR_ROOT` (`"twodim_ryanv2"`), `LEGACY_TWODIM_RYANV2_DIRNAME`.
- Helpers: `twodim_ryanv2_extended_base`, `twodim_ryanv2_product_dir`, `twodim_ryanv2_final_branch`, `twodim_ryanv2_remangled_dir`.

**`final_spectra_twodim_branch(gp_mode_short, product, *, use_rjf=False, use_ryanv2=False)`**

- **`use_rjf=True`**: branch `FINAL_spectra_2dim/twodim_rjf/<mode>/<product>/…` (unchanged semantics).
- **`use_ryanv2=True`**: branch `FINAL_spectra_2dim/twodim_ryanv2/<mode>/<product>/…`.
- Mutual exclusion: **`ValueError`** if both are true.

### 1.3 Identity re-mangle (no GP warp) — `Codes/remangle_identity.py`

- Implements **`REMANGLE_IDENTITY_ONLY`** behavior for **`7_Rimangle_KN_log_ryanv2.ipynb`**:
  - **Skipped** collaborator **`GP_interpolation_mangle`** (unity mask vs extended flux).
  - Differs from **`REMANGLE_MAX_ITERATIONS=0`**, which still runs **one** GP-based initial mask solve.
- **`identity_no_overlap`**: branch when **no** in-MJD overlapping photometry; copies extended spectrum to mangled/FINAL intermediates and returns success so **`save_FINAL_spectrum`** can run.
- **Overlapping photometry**: **`setup_identity_iteration_zero`** fills **`mangled_spec[0]`** and **`mangling_mask[0]=(1…, 0…)`** without altering the spectral shape; **`mangle_iteration_function`** syncs **`mang_mask`** locals from **`self.mangling_mask[0]`** for gate‑0 diagnostics, still runs **`calculate_ratios4mangling`** for logging, skips refinement loops, and assigns **`mangling_mask_FINAL`** from **`final_flux/ext_flux`** (≈1 over valid pixels). FINAL product directory layout (**as_observed**, **hostnocorr**, etc.) matches non-identity remangle bookkeeping.

### 1.4 Diagnostic — `Codes/diagnostics/trace_phot_x1_buckets.py`

- Maps **photometric** training rows in a bundle **`*.npz`** to **rounded \(x_1\)** buckets and, when **`grid_norm_info`** is present in `*_meta.json`, to **physical \(\log_{10}\lambda\)** / **λ (Å)** (same inversion as `ryan_gp.bundle_scale_pipeline` usage: `log10 λ_phys = x1_mean + x1_std * u`). Phot/spec classification duplicates **`ryan_gp.gp_utils.effective_point_class`** so the script runs with **numpy/scipy only** (no **`george`** import).

### 1.5 Notebooks

| Notebook | Purpose |
|----------|---------|
| **`6_TwoDim_UVExtend_Extrapolate_KN_ryanv2.ipynb`** | Same flow as `*_rjf` but under **`twodim_ryanv2/`**; optional **`iterate_gp_surface_bundle_scale`** cell; trailing optional finalize cell rewires **`save_plots_files`** to the **last** iterate (**`ryan_v2_finalize_iter_surface`**). |
| **`7_Rimangle_KN_log_ryanv2.ipynb`** | Consumes extended spectra from **`twodim_ryanv2`**; sets **`REMANGLE_IDENTITY_ONLY = True`** by default. |

Downstream notebooks now support **`USE_RYANV2_FINAL_SPECTRA`** alongside **`USE_RJF_FINAL_SPECTRA`** when resolving FINAL directories (classic two-way is still “neither ⇒ twodim/ only”):

- `model_comparison.ipynb`
- `bolometric_luminosity.ipynb`
- `plotting-final.ipynb`
- `7.5_spectra.ipynb`
- `7.5_alternate.ipynb`
- `7.5_comparison_check_log.ipynb`

### 1.6 CLI — `Codes/model_chi2.py`

- **`--ryanv2-final`**: resolve FINAL data under **`twodim_ryanv2`** for non-`--spectrum` χ² workflows (orthogonal to **`--no-rjf-final`**; Ryan v2 wins when set).

---

## 2. How to run the new version

### 2.1 Notebook 6 (Ryan v2 tree)

1. Open **`Codes/6_TwoDim_UVExtend_Extrapolate_KN_ryanv2.ipynb`** (kernel with project dependencies).
2. Run cells in order through the collaborator GP cell (`run_2DGP_GRID_rjf`): outputs go to **`Outputs/<SN>/twodim_ryanv2/<extend|extrapolate>/`** (same `save_plot_path` pattern as `*_rjf`).
3. **Minimal export**: ensure **`pipeline_config.GP_EXPORT_MINIMAL = True`** (default) before the GP cell; bundle path is **`<save_plot_path>/gp_minimal_export/gp_minimal_bundle.npz`** plus **`gp_minimal_bundle_meta.json`**. As of the spec-bundle export patch, that NPZ also includes **`spec_bundle_id`** and **`train_obs_class`** (unless **`GP_EXPORT_SPEC_BUNDLE_IDS = False`**) so **`iterate_gp_surface_bundle_scale.py`** can run without a separate **`bundle_scale_pipeline`** pass.
4. **Optional iterative surface scaling**: enable **`RUN_RYAN_ITERATE_GP_SURFACE=True`**. Tune **`pipeline_config`** (**`GP_RYAN_*`**) for Ryan parity (**strip**, **`--optimize`**, **`--surface-mu-key`**, convergence) or **`RUN_RYAN_STRIP_BAD_PHOT_BANDS`**, **`RYAN_GP_TAG_PREFIX`**, **`RYAN_ITER_MAX_ITERS`**. **`GP_RYAN_WRITE_GP_STYLE_CHECKS=True`** writes collaborator PNGs beneath **`ryan_surface_iterations/`** + **`ryan_gp_runs/<TAG>/figs/`** (**`MPLBACKEND=Agg`** in subprocess). **Re-run the GP/export cell** after upgrading schema so **`spec_bundle_id`** exists when needed.
5. **Finalize last iterate for downstream (notebook 7)**: enable **`RUN_FINALIZE_RYAN_ITER_SURFACE = True`** in the **final** cell (below plotting / save is fine). Loads **`ryan_surface_iterations/iter_KK/predictions.npz`** (or a vendor workspace with the same **`iteration_log.jsonl` / `iter_KK`** layout), reapplies **`transform_back_andPlot`** / **`save_plots_files`** so **`_spec_extended.txt`** (same layout as collaborator-only NB6) reflects the **iterated** **`mu`** / **`std`** on **`X_fill`**. Set **`RYAN_VENDOR_SURFACE_WORKSPACE`** in that cell for an out-of-repo tree (**`runs/my_surface_iter`**), or **`spec_class.ryan_surface_workspace`** to override **`save_plot_path/ryan_surface_iterations`**. Helpers: **`ryan_v2_finalize_iter_surface.load_final_surface_arrays`** supports **`iteration=<KK>`** and **`std_key=...`**; the cell can verify **`X_fill`** vs **`gp_minimal_export/gp_minimal_bundle.npz`** (**`VERIFY_VENDOR_X_FILL`**). **Overwrites** earlier collaborator **`save_plots_files`** outputs under **`save_plot_path`**; duplicate the tree before finalizing if you need both surfaces. Prefer **`REMANGLE_IDENTITY_ONLY = True`** + **zero** refinement iterations in notebook **7** when only bookkeeping / FINAL layout is needed.

### 2.2 Notebook 7 (Ryan v2 + identity re-mangle)

1. Produce extended spectra with notebook **`6*_ryanv2`**.
2. Open **`7_Rimangle_KN_log_ryanv2.ipynb`**.
3. Re-mangling paths follow **`twodim_ryanv2`** helpers (RE_mangled, FINAL branches).
4. **`REMANGLE_IDENTITY_ONLY`**: keep **`True`** for **no GP remangle**. With **no** overlapping photometry, **`identity_no_overlap`** runs (**`save_FINAL_spectrum`** succeeds). With **overlapping** photometry, **`setup_identity_iteration_zero`** applies (**mask = 1**); ratios are computed for diagnostics only; **`GP_interpolation_mangle`** never runs; refinement iterations are suppressed. **`False`** restores the original GP mask path (**`GP_interpolation_mangle`**) on the same **ryanv2** directories.

### 2.3 Downstream comparison / plots

In the config cell of each updated notebook:

```python
USE_RJF_FINAL_SPECTRA = True   # or False
USE_RYANV2_FINAL_SPECTRA = False  # set True to read FINAL under twodim_ryanv2/...
```

Only one of **RJF** vs **Ryan v2** should be active for FINAL resolution; both false selects the **classic** `twodim/<mode>/<product>` branch.

### 2.4 `model_chi2.py` example

```bash
cd /path/to/PyCoCo_templates/Codes
python model_chi2.py --ryanv2-final --no-rjf-final ...   # ryanv2 branch for resolved FINAL dir
```

(Adjust other flags as in `--help`.)

### 2.5 Photo–x₁ trace diagnostic

```bash
cd /path/to/PyCoCo_templates/Codes
python diagnostics/trace_phot_x1_buckets.py \
  --bundle ../Outputs/AT2017gfo/twodim_rjf/extrapolate/gp_minimal_export/gp_minimal_bundle.npz \
  --meta ../Outputs/AT2017gfo/twodim_rjf/extrapolate/gp_minimal_export/gp_minimal_bundle_meta.json \
  --bands -0.8767,-0.8217
```

(Use a **`twodim_ryanv2`** bundle path after you have generated one.)

---

## 3. Knobs reference

### 3.1 `pipeline_config.py`

- **`TWODIM_RYANV2_SUBDIR_ROOT`**, **`LEGACY_TWODIM_RYANV2_DIRNAME`**, **`USE_LEGACY_TWODIM_LAYOUT`**: same role as RJF parallel paths.
- **`GP_EXPORT_MINIMAL`**, **`GP_EXPORT_SUBDIR`**: location of bundle for Ryan drivers.
- **`GP_EXPORT_SPEC_BUNDLE_IDS`** (default **`True`**): when True, `gp2dim_export` writes **`spec_bundle_id`** / **`train_obs_class`** using **`ryan_gp/spec_bundle_id_assign.py`** (matches **`bundle_scale_pipeline`** epoch clustering; no flux mutation). Set **`False`** only for legacy minimal NPZ without IDs.
- **`GP_EXPORT_PHOT_SPEC_THRESHOLD`**, **`GP_EXPORT_MAX_BUNDLE_MINUTES`**: passed through to the ID assignment (defaults **50**, **5** — align with **`run_gp`** / **`bundle_scale_pipeline`**).
- **`GP_RYAN_STRIP_PHOT_BANDS_BEFORE_ITERATE`** (default **`False`**): notebook cell also accepts **`RUN_RYAN_STRIP_BAD_PHOT_BANDS`** → writes **`GP_RYAN_STRIP_OUTPUT_NPZ`** beside **`gp_minimal_bundle.npz`** via **`ryan_v2_iterate_prep.strip_minimal_export_bundle`** + **`strip_photometry_bands.py`** (**`GP_RYAN_STRIP_BANDS_ROUNDED_X1`** default **`-0.8767,-0.8217`**).
- **`GP_RYAN_ITERATE_SURFACE_MU_KEY`** (**`mu_raw`** | **`mu`**): surfaced as iterate **`--surface-mu-key`** (**`mu_raw`** matches Ryan **`RUNNING_MY_SURFACE_ITER`**); finalize cell uses **`FINAL_SURFACE_MU_KEY=None`** to mirror this.
- **`GP_RYAN_ITERATE_OPTIMIZE_RUN_GP`**: forwarded as **`run_gp.py --optimize`** (**`True`** by default — align archived runbook).
- **`GP_RYAN_ITERATE_RUN_GP_MAX_ITER`**, **`GP_RYAN_ITERATE_BUNDLE_SCALE_CLIP`**, **`GP_RYAN_ITERATE_CONVERGE_MAX_LOG_SCALE`**, **`GP_RYAN_ITERATE_CONVERGE_DELTA_CHI2_SPEC`**: iterate driver flags before **`--`**.
- **`GP_RYAN_WRITE_GP_STYLE_CHECKS`**: headless collaborator **`plot_results.py`** (**`--heatmap-raw`**) + **`plot_bands_gp_overview.py`** subsets (**`GP_RYAN_GP_STYLE_CHECKS_BANDS_IDS`**) → **`ryan_surface_iterations/<GP_RYAN_GP_STYLE_CHECKS_SUBDIR>/kXX/`** via **`ryan_v2_gp_style_checks.run_gp_style_check_pngs`**.
- **`GP2DIM_Class` export labels** (optional): **`gp_export_train_obs_class`** overrides everything for the saved NPZ. Otherwise **`run_2DGP_GRID_rjf`** passes through **`train_obs_class`** or **`gp_train_obs_class`** when the length matches the training set (after **`gp_2d_anchor_t0`**, missing anchor rows are filled with **`phot`** so pseudo points are not treated as extra spectroscopic epochs).
- **`GP_RJF_KWARGS`**: still applies to **`run_2DGP_GRID_rjf`** in notebook 6 (unchanged).

### 3.2 Notebook 6 optional — `iterate_gp_surface_bundle_scale.py` (via subprocess)

The iterate **code cell** builds most driver flags from **`pipeline_config`** (strip, **`--optimize`**, **`--surface-mu-key`**, convergence, **`--run-gp-max-iter`**). Optionally override **`RUN_RYAN_STRIP_BAD_PHOT_BANDS`** or **`pipeline_config`** before running the cell.

Set in (or upstream of) the optional cell:

- **`RUN_RYAN_ITERATE_GP_SURFACE`**, **`RUN_RYAN_STRIP_BAD_PHOT_BANDS`** (`None` = use **`GP_RYAN_STRIP_PHOT_BANDS_BEFORE_ITERATE`**)
- **`RYAN_GP_TAG_PREFIX`**, **`RYAN_ITER_MAX_ITERS`**

Additional **`run_gp.py`** overrides still go **after** the lone **`--`** if you extend the **`_run_gp_tail`** list in-notebook (**conflicts rarely needed** alongside **`pipeline_config`**).

### 3.3 `ryan_gp/strip_photometry_bands.py`

- **`--bands`**: comma-separated **rounded \(x_1\)** targets (default `-0.8767,-0.8217`). **Manual CLI:** use **`--bands=-0.8767,-0.8217`** (equals form). If the value is passed as a separate token starting with **`-`**, **`argparse`** may treat it as another flag and exit **2** (“expected one argument”).
- **`--round-digits`** (default 4): must match rounding used to define bands.
- **`--phot-spec-threshold`**: passed to **`gp_utils.effective_point_class`** (consistent with collaborator bundle convention).

### 3.4 Notebook 7 — re-mangle globals

- **`REMANGLE_IDENTITY_ONLY`**: **`True`** = no **`GP_interpolation_mangle`**; extended spectrum propagated (with **`remangle_identity`** helpers).
- **`REMANGLE_MAX_ITERATIONS`**, **`REMANGLE_RATIO_TOLERANCE`**, **`REMANGLE_FLUX_FLOOR`**, **`REMANGLE_MASK_DIAG_PLOTS`**: unchanged meaning when **`REMANGLE_IDENTITY_ONLY`** is **`False`**; refinement loop body is skipped when identity mode is on.

---

## 4. Tests

From **`Codes/tests/`**:

```bash
cd /path/to/PyCoCo_templates/Codes
python -m pytest tests/test_twodim_ryanv2_paths.py tests/test_twodim_rjf_paths.py tests/test_pipeline_config_resolve.py tests/test_ryan_v2_finalize_iter_surface.py tests/test_ryan_strip_photometry_bands.py -q
```

Run the full suite if you merge larger changes:

```bash
python -m pytest tests/ -q
```

---

## 5. File checklist

| Path | Role |
|------|------|
| `Codes/ryan_gp/` | Vendored Ryan scripts + README |
| `Codes/ryan_gp/spec_bundle_id_assign.py` | **`spec_bundle_id`** / phot-spec heuristic (no **george**); used by **`gp2dim_export`** |
| `Codes/ryan_v2_finalize_iter_surface.py` | Last-iterate **`predictions.npz`** picker + arrays for **`save_plots_files`** ingest |
| `Codes/ryan_v2_iterate_prep.py` | Thin wrapper → **`strip_photometry_bands.py`** ahead of iterate |
| `Codes/ryan_v2_gp_style_checks.py` | Headless **`plot_results`** + **`plot_bands_gp_overview`** after iterate |
| `Codes/gp2dim_export.py` | Writes minimal NPZ + optional **`spec_bundle_id`** / **`train_obs_class`** |
| `Codes/remangle_identity.py` | Identity re-mangle helpers |
| `Codes/diagnostics/trace_phot_x1_buckets.py` | x₁ bucket → λ diagnostic |
| `Codes/pipeline_config.py` | **`twodim_ryanv2_*`**, **`final_spectra_twodim_branch`**, **`GP_EXPORT_SPEC_BUNDLE_IDS`**, … |
| `Codes/model_chi2.py` | **`--ryanv2-final`** |
| `Codes/6_*_ryanv2.ipynb`, `Codes/7_*_ryanv2.ipynb` | Main workflows |
| `Codes/tests/test_twodim_ryanv2_paths.py` | Path + branch exclusivity tests |

---

## 6. Troubleshooting

### `ERROR: bundle lacks spec_bundle_id` (`iterate_gp_surface_bundle_scale.py`)

**Cause:** The iterate driver needs **`spec_bundle_id`** on each training row to compute per-bundle linear scales vs the GP surface. Bundles produced **before** the export patch, or any export with **`GP_EXPORT_SPEC_BUNDLE_IDS = False`**, omit this key.

**Fix:** Re-run notebook **6** through **`run_2DGP_GRID_rjf`** so **`gp_minimal_export/gp_minimal_bundle.npz`** is regenerated (defaults write **`spec_bundle_id`** and **`train_obs_class`**). Then run the optional iterate cell again.

---

## 7. Collaborator parity (Ryan **`RUNNING_MY_SURFACE_ITER.md`**)

| Requirement | Meaning | PyCoCo default / how to enable |
|-------------|---------|-------------------------------|
| **`gp_*_scaled_*`** bundle | Prior **`bundle_scale_pipeline`** intra-spectrum scaling + optional global phot anchors (mutates training **`y`**) | Not applied in **`gp_minimal_export`**; iterate loop rescales spectroscopy bundles vs successive GP surfaces. Expect closer behavior after stripping + iterations, not a drop-in **`gp_work_scaled`** clone without running that pipeline. |
| **`nophot_m8767_m8217`** | **`strip_photometry_bands`**: removes **phot** rows whose **rounded `X[:,0]`** ∈ **`{-0.8767,-0.8217}`** (default **`--round-digits`** 4) | Set **`pipeline_config.GP_RYAN_STRIP_PHOT_BANDS_BEFORE_ITERATE=True`** or **`RUN_RYAN_STRIP_BAD_PHOT_BANDS=True`** before iterate (**`ryan_v2_iterate_prep`** writes **`gp_minimal_bundle_nophot_m8767_m8217.npz`**). Omitting strip leaves full phot anchors in every **`run_gp`**. |
| **`run_gp --optimize`** | Hyper-parameter optimizer on subsample (archived **`RUNNING_MY_SURFACE_ITER`** CLI) | **`GP_RYAN_ITERATE_OPTIMIZE_RUN_GP=True`** (**default**); toggle in **`pipeline_config`**. |
| **`iterate --surface-mu-key`** | WLS scales vs latent **`mu_raw`** vs **`mu`** on **`X_fill`** | **`GP_RYAN_ITERATE_SURFACE_MU_KEY='mu_raw'`** (**default**); finalize (**`FINAL_SURFACE_MU_KEY=None`**) inherits the same key. |

**Upstream driver parity:** **`Codes/ryan_gp/iterate_gp_surface_bundle_scale.py`** is kept aligned with **`ryan-updates/py_files/`**.

**Ryan-style QA PNGs (no notebook display):** set **`pipeline_config.GP_RYAN_WRITE_GP_STYLE_CHECKS=True`**; iterate cell runs collaborator **`plot_results.py --heatmap-raw`** (figures under **`ryan_gp_runs/<TAG>/figs/`**) and **`plot_bands_gp_overview.py`** (under **`ryan_surface_iterations/<subdir>/kXX/plot_bands_overview/`**). Optional manual: **`python gp_scales.py <TAG>`** with **`cwd=ryan_gp`**.
