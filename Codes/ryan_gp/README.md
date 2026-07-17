# Ryan GP tooling (vendored)

Python utilities for collaborator-style 2-D GP bundles (`run_gp.py`, iterative surface rescaling, band stripping diagnostics). Imported into this repo from the Ryan updates tree and kept under **`Codes/ryan_gp/`** so notebooks can call scripts by path relative to `PyCoCo_templates`.

## Layout and `PYTHONPATH`

Internal imports (`import gp_utils`, `import bundle_meta`, …) assume the **current working directory** is **`Codes/ryan_gp/`** when you run a driver (e.g. `python iterate_gp_surface_bundle_scale.py`). The notebooks set `cwd=` accordingly via `subprocess.check_call(..., cwd=_ryan)`.

Alternatively, from anywhere:

```bash
export PYTHONPATH="/path/to/PyCoCo_templates/Codes/ryan_gp:${PYTHONPATH}"
python /path/to/PyCoCo_templates/Codes/ryan_gp/run_gp.py --help
```

## Entry points (high level)

| Script | Role |
|--------|------|
| `run_gp.py` | Fit/predict on a training bundle (`*.npz`); writes `runs/<tag>/`. |
| `iterate_gp_surface_bundle_scale.py` | Outer loop: refit + rescale spectroscopic bundles vs GP surface; delegates to `run_gp.py` after `--`. |
| `bundle_scale_pipeline.py` | Per-bundle linear flux multipliers vs surface. |
| `strip_photometry_bands.py` | Drop photometry rows whose rounded `X[:,0]` matches target pseudo-bands. |
| `bundle_meta.py` | Load/write `*_meta.json` beside bundles (`grid_norm_info`, etc.). |

See each file’s module docstring and `--help` for full flag lists.

## Notebook integration

Notebook **`6_TwoDim_UVExtend_Extrapolate_KN_ryanv2.ipynb`** documents an **optional** cell that calls `iterate_gp_surface_bundle_scale.py` on the notebook 6 export `gp_minimal_export/gp_minimal_bundle.npz`.

Provenance: maintained alongside `PyCoCo_templates`; keep this README when refreshing files from upstream.
