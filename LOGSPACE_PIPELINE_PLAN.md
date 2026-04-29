# Log-space PyCoCo pipeline (notebooks 5–7 + `GP2dim_utils_newlog`)

Living document: root causes found in debugging and code changes applied.

## 2026-04-17 — Notebook 6 cell 8 kernel crash / overflow / DataFrame fragmentation

### Root cause (overflow)

`5_Mangle_spectra_KN_log.ipynb` **saves mangled spectra as hybrid columns**: `wls` = **linear Å** (from `raw_spec['wls']`), `flux` / `fluxerr` = **log10** (mangled spectrum in log). Notebook 6 newlog assumed **log10(Å)** in `wls` and applied `10**spec['wls']` in `band_flux_modified`, which overflows for typical linear wavelengths (~3000–10000 Å).

### Root cause (wrong grid / lam_eff logic)

The spectral grid was built in **linear Å** while `lam_eff()` returns **log10(λ)** and `GP2dim_utils_newlog` expects training coordinates **log10(wavelength)**. That broke UV row keys, in-range checks (comparing log λ to linear λ), and GP consistency.

### Root cause (fragmentation)

`grid_all_spectraltimeseries` grew a `DataFrame` with `grid_all[str]=...` in a loop; `extend_grid_all_spectraltimeseries` added many rows/columns via chained `.loc` / `df[col]=`. Pandas warns and heavy fragmentation can contribute to memory pressure.

### Fixes applied (cell 8 class in `6_TwoDim_UVExtend_Extrapolate_KN_newlog.ipynb`)

1. **Helpers** `_wls_are_linear_angstrom`, `_spec_wls_linear`, `_log10_flux_to_linear` — detect linear-Å `wls`, convert to linear only when needed; map log10 flux to linear with clipped exponent.
2. **`grid_all_spectraltimeseries`** — common wavelength axis is **log10(Å)** with step derived from `DELTA` (Å) at the mid-wavelength; interpolation uses `log10(λ)` vs log10 flux; **plot** uses linear Å on the x-axis (equivalent display to the original).
3. **`load_phot_for_extention`** — in-range test compares **linear** λ_eff to **linear** file `wls`.
4. **`band_flux_modified`** — uses helpers (no `10**` on linear wavelengths).
5. **`create_extended_spec_folder`** — uses `self.snname` (was `snname`).
6. **UV / extrapolation** — `fill_gaps` guards `n_steps <= 0`; safe linear flux from log10 via `_log10_flux_to_linear`; `phot_perc` avoids divide-by-zero; **defragment** with `grid_notext.copy()` / `grid_notext_err.copy()` before final sort.
7. **DataFrame build** — spectral grid columns built from a `dict` + `pd.DataFrame(...)` once per loop.

### Verification

- Run: `conda run -n SED_clean python -m unittest discover -s Codes/tests -p 'test_*.py' -v`
- Re-run notebook 6 from imports through cell 8; expect **no** `overflow encountered in power` from `wls`.

### Still manual

- Confirm notebook **imports** `GP2dim_utils_newlog` for the log chain.
- Align photometry path (`DATALC_PATH` / `fitted_phot_logspace`) with the LC files you actually produced.

## Second root cause — `GP2dim_utils_newlog.py` (plot / save paths)

Notebook 6 cell 8 was fixed to treat mangled `wls` as **linear Å**, but **`transform_back_andPlot`** and **`save_plots_files`** still used `10**spec['wls']` under the (wrong) assumption that files stored log10(wavelength). That reproduces the same overflow when the GP step loads mangled spectra for `extended_spec.pdf` or writes `*_spec_extended.txt`.

**Fix:** Shared helpers in `GP2dim_utils_newlog.py` — `mangled_wls_max_is_linear_angstrom`, `mangled_wls_linear_angstrom`, `mangled_flux_linear_from_log10` — used in those functions and (optionally) delegated from notebook 6 cell 8 so there is a single source of truth.

**Hygiene:** Removed stale `Codes/_cell8_newlog_class.py` (old `10**wls` copy). Clear old notebook cell **outputs** if tracebacks still show the obsolete line.

## Kernel crash during / after GP (notebook 6)

**Cause:** `run_2DGP_GRID` called `george` with `return_cov=True`, allocating a full **N×N** predictive covariance for each batch of test points (N = N_wavelength × 3 phases per batch). Memory scales as **O(N²)** and often triggers the OS OOM killer; Jupyter may report the death on the **next** cell (e.g. a trivial `print` on `yprior`).

**Fix:** Use `gp.predict(..., return_var=True)` (diagonal only, same mean) and cap the number of log₁₀-λ prediction samples (defaults **300** points and **0.01** dex step, override with `spec_class.gp_predict_n_wavelength` / `spec_class.gp_predict_wl_step`). Optional **`gp_predict_slot_size`** (default3) batches more phases per `predict` call for speed.

**Training / surface plots:** `make_plots` and the 2D surface scatter in `transform_back_andPlot` must use **`offset2 + norm2 * x2_norm`** on the phase axis so the label “log10(phase days)” matches absolute phase, not phase minus the minimum epoch.

**Extrapolated phase range:** The old `extrapolate_spectra` branch trimmed `mjds_extention` to **2 dex** span in log10(phase). That limits **linear** time to a factor of 100 from the earliest extrap point (e.g. ~0.1 d → ~10 d), not “100 calendar days.” Default is now **no chop**; optional `spec_class.extrapolate_log_phase_span_dex` restores a cap. The grid also appends **`max` of LC log-phases** so photometry can reach the last observation.

**Linear training plot:** `make_plots` also saves `data_for2d_interpolation_linear_axes.pdf` (phase in days, λ in Å).

**GP predict:** `run_2DGP_GRID` chunks `gp.predict` batches (`gp_predict_chunk_size`, default 1500) to reduce peak memory.
