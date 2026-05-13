# Minimal 2D GP export for collaborators

After a successful `run_2DGP_GRID` (ln-flux or z-score) or `run_2DGP_GRID_linear` (linear flux), you can save the arrays someone needs to rebuild or modify the George GP.

## Enable export

**Option A — config (recommended)**  
In [`pipeline_config.py`](pipeline_config.py) set:

```python
GP_EXPORT_MINIMAL: bool = True
# optional: GP_EXPORT_SUBDIR = "gp_minimal_export"
```

**Option B — per run in the notebook**  
After you build `spec_class` (e.g. right after `create_extended_spec_folder()`):

```python
spec_class.gp_export_minimal = True
```

You can also set a custom directory:

```python
spec_class.gp_export_dir = "/path/to/folder"
```

(`GP_EXPORT_MINIMAL` can stay `False` if you use this class flag.)

Then run the notebook through the cell that calls `run_2DGP_GRID` / `run_2DGP_GRID_linear` as usual.

## Output location

Files are written under:

`{spec_class.save_plot_path}/{GP_EXPORT_SUBDIR}/`

For the standard dual-product layout, `save_plot_path` is something like  
`Outputs/<SN>/twodim/<extend|extrapolate>/`. Check `spec_class.save_plot_path` in the notebook after folder setup.

You get:

| File | Contents |
|------|-----------|
| `gp_minimal_bundle.npz` | NumPy arrays (see below) |
| `gp_minimal_bundle_meta.json` | `snname`, `mode`, `gp_module`, `kernel_layout`, `grid_norm_info`, paths |

## Arrays in `gp_minimal_bundle.npz`

Rows of `X` and `X_fill`: column 0 = normalized log10(wavelength), column 1 = normalized log10(phase days).  
(See `grid_norm_info` in the JSON for min–max or z-score conventions.)

| Key | Description |
|-----|-------------|
| `X` | Training design matrix `(N_train, 2)`. |
| `y` | Training targets (scaled ln-flux or scaled linear flux — matches the module you ran). |
| `yerr` | Per-point 1σ **before** the final diagonal tweak used in `compute` (floors applied). |
| `y_compute` | **Diagonal passed to `gp.compute`**: ln-flux path uses `sqrt(yerr**2 + 1e-6**2)`; linear-flux path uses `yerr` unchanged. |
| `X_fill` | Full prediction grid `(N_pred, 2)` (Cartesian product of wavelength nodes × phase columns). |
| `kernel_wls_scale`, `kernel_time_scale` | Matern 3/2 length scales (as in the notebook / `run_2DGP_GRID`). |
| `y_var_scale` | `np.var(y)` — multiplies the kernel in your pipeline. |
| `white_noise_variance` | Jitter variance on the class (`gp_white_noise`). |
| `white_noise_log` | `log(white_noise_variance)` when variance &gt; 0; else NaN (George keyword). |
| `prior_used` | 0 or 1. |
| `prior_points`, `prior_values` | Prior grid for the mean model (`nearest` `griddata`); empty if no prior. |

`mean_model` in George is not a single array: it is rebuilt from `prior_points` / `prior_values` when `prior_used` is 1.

### Kernel layout

- **ln-flux / z-score** (`GP2dim_utils_newlog`, `GP2dim_utils_newlog_zscore`): `kernel_layout` = `per_axis_Matern32_product` — product of two axis-aligned `Matern32Kernel`s scaled by `y_var_scale`.
- **Linear flux** (`GP2dim_utils_newlog_linear_flux`): `kernel_layout` = `joint_Matern32_ndim2` — `Matern32Kernel([l_wl, l_time], ndim=2)` scaled by `y_var_scale`.

## Rebuild the GP in George (ln-flux or z-score: per-axis Matern32)

Check `kernel_layout` in `gp_minimal_bundle_meta.json`. For `per_axis_Matern32_product` (modules `GP2dim_utils_newlog` / `GP2dim_utils_newlog_zscore`), use:

```python
import json
import numpy as np
import george
from george.kernels import Matern32Kernel
from george.modeling import Model
from scipy.interpolate import griddata

d = np.load("gp_minimal_bundle.npz", allow_pickle=False)
with open("gp_minimal_bundle_meta.json", encoding="utf-8") as f:
    meta = json.load(f)

X = d["X"]
y = d["y"]
y_compute = d["y_compute"]
X_fill = d["X_fill"]
lw, lt = float(d["kernel_wls_scale"]), float(d["kernel_time_scale"])
yv = float(d["y_var_scale"])
wn = float(d["white_noise_variance"])

prior_used = int(d["prior_used"]) == 1
prior_pts = np.asarray(d["prior_points"], dtype=float)
prior_val = np.asarray(d["prior_values"], dtype=float)

k_wave = Matern32Kernel(metric=lw, ndim=2, axes=1)
k_time = Matern32Kernel(metric=lt, ndim=2, axes=0)
kernel = yv * (k_wave * k_time)

gp_kwargs = {}
if wn > 0.0:
    gp_kwargs["white_noise"] = float(np.log(wn))

# Prior mean (matches run_2DGP_GRID / Model_2dim: nearest griddata, NaN -> 0)
if prior_used and prior_pts.size and prior_val.size:
    class PriorMeanModel(Model):
        parameter_names = ()

        def get_value(self, t):
            pts = np.asarray(prior_pts, dtype=float)
            vals = np.asarray(prior_val, dtype=float)
            pe = np.column_stack((t[:, 0], t[:, 1]))
            z = griddata(pts, vals, pe, method="nearest")
            z = np.where(np.isnan(z), 0.0, z)
            return z

    gp = george.GP(kernel, mean=PriorMeanModel(), **gp_kwargs)
else:
    gp = george.GP(kernel, **gp_kwargs)

gp.compute(X, y_compute)
```

**Linear-flux export** (`kernel_layout` = `joint_Matern32_ndim2`): replace the three kernel lines with:

```python
kernel = yv * Matern32Kernel([lw, lt], ndim=2)
```

and keep the same prior block / `gp.compute` / `predict` pattern (`y_compute` is already `yerr` for that path).

### Prediction on the full `X_fill` at once (no chunking)

The pipeline chunks `predict` only to limit peak memory. If `X_fill` fits in RAM, call:

```python
mu, var = gp.predict(y, X_fill, return_var=True)
std = np.sqrt(np.maximum(var, 0.0))
```

**Memory:** `X_fill` is `(N_pred, 2)` float64; `mu` and `var` add two more length-`N_pred` vectors. Rough order of magnitude: `N_pred × 8 × (2 + 1 + 1)` bytes for those alone, plus George’s internal buffers. For very large grids this can exceed machine RAM—then chunk explicitly:

```python
mu = np.empty(len(X_fill), dtype=float)
var = np.empty(len(X_fill), dtype=float)
chunk = 50_000
for s0 in range(0, len(X_fill), chunk):
    s1 = min(s0 + chunk, len(X_fill))
    m, v = gp.predict(y, X_fill[s0:s1], return_var=True)
    mu[s0:s1] = m
    var[s0:s1] = v
std = np.sqrt(np.maximum(var, 0.0))
```

You can tune `chunk` the same way the notebook uses `gp_predict_chunk_size` (often ~1500–2000+ depending on hardware).

## Source code references

- Prior mean: `GP2dim_utils_newlog.py` — inside `run_2DGP_GRID`, `Model_2dim.get_value` uses `griddata(points, values, ..., method="nearest")` and sets NaNs to 0 (same in `run_2DGP_GRID_linear`).
- Training stack and `compute` (ln-flux): same file — `X = np.vstack(...).T`, `kernel2dim = np.var(y) * ...`, `gp.compute(X, np.sqrt(yerr**2 + 1e-6**2))`.
- Prediction mesh: same file — `X_fill = np.vstack((x1_fill, x2_fill)).T`, `gp.predict(y, X_fill[...], return_var=True)`.
