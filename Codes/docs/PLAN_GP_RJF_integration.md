# Collaborator GP (`gp_rjf`) integration

This doc describes **what landed in the repo** for parallel “RJF” 2‑D Gaussian-process extrapolation versus the legacy George run in notebook 6 (`run_2DGP_GRID`).

## Why a parallel tree?

- **`twodim/` outputs stay untouched.** With the modern layout, RJF notebooks write under **`Outputs/<SN>/twodim_rjf/<extend|extrapolate>/`** (`spliced/`, `full_gp/`, `diagnostics/` same as before). If you insist on **`USE_LEGACY_TWODIM_LAYOUT=True`**, RJF writes to **`Outputs/<SN>/TwoDextended_spectra_rjf`** so **`TwoDextended_spectra`** is not wiped by notebook 6.
- **Original notebooks 6–7 unchanged.** Use copies:
  - [`6_TwoDim_UVExtend_Extrapolate_KN_rjf.ipynb`](../6_TwoDim_UVExtend_Extrapolate_KN_rjf.ipynb)
  - [`7_Rimangle_KN_log_rjf.ipynb`](../7_Rimangle_KN_log_rjf.ipynb)

## Engineering summary (technical)

| Piece | Role |
| --- | --- |
| [`Codes/gp_collab_rjf/`](../gp_collab_rjf/) | Vendor of collaborator code: [`gp_utils.py`](../gp_collab_rjf/gp_utils.py) + [`run_inference.py`](../gp_collab_rjf/run_inference.py) (`run_gp_from_bundle`, defaults in `DEFAULT_KWARGS`). |
| [`Codes/GP2dim_utils_newlog_rjf.py`](../GP2dim_utils_newlog_rjf.py) | `run_2DGP_GRID_rjf(...)`: same **prediction grid stacking** as [`GP2dim_utils_newlog.run_2DGP_GRID`](./GP2dim_utils_newlog.py) **or** [`GP2dim_utils_newlog_zscore.run_2DGP_GRID`](./GP2dim_utils_newlog_zscore.py), chosen from `spec_class.grid_norm_info` (min–max vs z-score); collaborator fit/predict/mono+blue. |
| [`Codes/pipeline_config.py`](./pipeline_config.py) | Helpers `twodim_rjf_extended_base`, `twodim_rjf_product_dir`, `twodim_rjf_final_branch`, `twodim_rjf_remangled_dir` (re-mangled spectra under `.../<product>/RE_mangled_spectra_2dim`, or legacy `RE_mangled_spectra_2dim_rjf`); flags `GP_RJF_KWARGS`, `GP_RJF_PLOT_RAW_AND_PROCESSED`, cache subdir constant. |

**Post-processing:** Collaborator adjusts **predicted μ** at early phases (smooth monotone rise in phase; “blue” cumulative-min along wavelength) while **keeping σ from the untouched GP predictive variance.** Uncertainty bands are therefore approximate where μ was smoothed.

**Minimal bundle export:** `maybe_save_gp_minimal_export` still runs (notebook `GP_EXPORT_MINIMAL`) with `kernel_layout="collaborator_gp_rjf_matern_additive_opt"`. A trimmed JSON digest is written under `diagnostics/rjf_gp_config.json` after each RJF fit.

## Easy mental model

- You still prepare the spectrophotometric grid exactly as before (`transform2LOG_reshape`, prior surface, kernel scale arguments—the latter only seed the collaborator optimizer warm-start inside the bundle arrays).
- The **shape and ordering** of flux on the extrapolation lattice match the legacy GP so **`save_plots_files`**, **Rimangle**, and downstream FINAL naming stay aligned.
- The **solver** swaps from “fixed hypers + George Matern‑3/2 product kernel” → “optimized hypers + collaborator kernel + jitter + μ rules”.
- **`μ` feeding text spectra for Rimangle:** **after** collaborator rules (**production μ**); **raw μ** exists for optional diagnostic PDFs (`diagnostics/rjf_mu_raw_vs_post/`).

## How to run the RJF pipeline

1. Confirm **`pipeline_config.USE_TWO_D_GP_LINEAR_FLUX = False`** (RJF notebook hard‑errors otherwise).
2. Open **`6_*_rjf.ipynb`**; set **`COCO_PATH`**, **`OUTPUT_DIR`**, **`snname`** as in the classic notebook.
3. **`spec_class.create_extended_spec_folder`** now clears/writes **`twodim_rjf/...`** (via `twodim_rjf_extended_base`).
4. Run through the collaborator GP cell (imports `GP2dim_utils_newlog_rjf`).
5. Open **`7_*_rjf.ipynb`**; extended inputs use **`twodim_rjf/...`**; re-mangled intermediates use **`remangled_spectra_path`** → `twodim_rjf/<mode>/<product>/RE_mangled_spectra_2dim/`; FINAL spectra use **`FINAL_spectra_2dim/twodim_rjf/...`** via `final_spectra_path_2dim`.

## Parameters to tweak

Defined in **[`pipeline_config.py`](pipeline_config.py)** unless noted:

| Flag / dict | Effect |
| --- | --- |
| **`GP_RJF_KWARGS`** | In [`pipeline_config.py`](./pipeline_config.py) defaults preset to **`gp_rjf/WRITEUP.md` §2** (`matern52_addw_addt_linear_opt_v5`): additive λ+time kernels, jitter warm-starts **0.012 / 0.005**, **`log_amp=ln(0.0135)`**, warm metrics/weights from their post-fit table, **`optimize_subsample=2500`**, **`predict_train=True`**. Override entries as needed per SN. Keys mirror collaborator `run_inference.DEFAULT_KWARGS`. |
| **`GP_RJF_PLOT_RAW_AND_PROCESSED`** | Saves `gp_rjf_mu_raw_vs_post_linflux.pdf` with raw vs processed μ mapped to linear flux. |
| **`spec_class.gp_rjf_plot_mu_raw_diagnostic = True`** | Same plot trigger from the notebook object. |
| **`GP_RJF_PRIOR_CACHE_SUBDIR`** | Relative to `save_plot_path`; caches `prior_linear_interp.pkl` for **`mean='linear'`** Gaussian-process mean surface. |

Standard notebook tuning (`gp_predict_n_wavelength`, dense log-phase prediction, anchors, WL caps) still flows through **`spec_class` / pipeline_config exactly as in notebook 6**.

### Min–max vs z-score coordinates (`USE_TWO_D_GP_ZSCORE_COORDS`)

Notebook 6 still imports **`GP2dim_utils_newlog`** or **`GP2dim_utils_newlog_zscore`** for **`transform2LOG_reshape`**; `run_2DGP_GRID_rjf` reads the resulting **`grid_norm_info`** and builds **`X_fill`** in the same normalized coordinates as training.

### `early_time_cutoff` (mono / blue rules)

In **`GP_RJF_KWARGS`**, **`early_time_cutoff`** is compared to **`X_fill[:, 1]`** inside [`gp_collab_rjf/run_inference.py`](../gp_collab_rjf/run_inference.py)—the **same units as your normalized phase column**. Default **`-4`** matches collaborator bundles that used PyCoCo-style **min–max** phase normalization. If you use **z-score** coordinates, that numeric threshold may not correspond to the same physical phases; override **`early_time_cutoff`** (or related mono kwargs) after inspecting log-phase coverage.

## Tests run in CI / locally

- `tests/test_gp_collab_rjf_inference.py` covers inference smoke, **min–max** grid cardinality, and **z-score** `grid_norm_info` (no `KeyError: norm1`). `tests/test_twodim_rjf_paths.py` covers **`twodim_rjf`** path helpers.

Dependencies (minimal for those tests): `numpy`, `scipy`, `george`; importing `GP2dim_utils_newlog_*` pulls `matplotlib` / `pandas` as elsewhere in the codebase.

```bash
cd Codes
python -m unittest tests.test_gp_collab_rjf_inference tests.test_twodim_rjf_paths -v
```

## Checklist comparing classic vs RJF

1. Finish notebook 6 RJF → confirm **`Outputs/<SN>/twodim_rjf/...`** exists alongside classic **`twodim/...`**.
2. Inspect **`gp_results_*`** plots (same filenames as legacy; surface uses **processed μ**).
3. Optionally enable **`GP_RJF_PLOT_RAW_AND_PROCESSED`** and review **`diagnostics/rjf_mu_raw_vs_post/`** for Physics-facing sanity at early phases.
4. Run notebook 7 RJF → confirm extended spectrum glob resolves under **`twodim_rjf/...`** and FINAL branch includes **`twodim_rjf/...`** in the filesystem path helper.
