# Phase 1 — Pre-scale spectra + mangling I/O (completed)

Phase 1 of the iterative GP+mangle pipeline. Phases 2–4 remain in the master plan.

## What was built

| Deliverable | Path |
|-------------|------|
| Pre-scale engine | [`spectra_pre_scale.py`](../spectra_pre_scale.py) |
| Pre-scale diagnostics | [`spec_scale_diagnostics.py`](../spec_scale_diagnostics.py) |
| Mangling helpers (I/O, mask, NB5 extract) | [`mangle_spectra_log.py`](../mangle_spectra_log.py) |
| Config / paths | [`pipeline_config.py`](../pipeline_config.py) (`SPEC_SCALE_*`, `USE_PRESCALED_SPECTRA`) |
| Notebook | [`4.5_Scale_spectra_KN.ipynb`](../4.5_Scale_spectra_KN.ipynb) |
| NB5 reads prescaled list | [`5_Mangle_spectra_KN_log.ipynb`](../5_Mangle_spectra_KN_log.ipynb) (`spec_list_path_for_mangling`) |
| Tests | [`tests/test_spectra_pre_scale.py`](../tests/test_spectra_pre_scale.py), [`tests/test_mangle_spectra_log.py`](../tests/test_mangle_spectra_log.py), [`tests/test_pipeline_config_prescale.py`](../tests/test_pipeline_config_prescale.py) |

## Run order (so far)

0.1 → 1 → 2 → 4 → **4.5** → **5** → *(Phase 2+: iterative GP)*

## Notebook 4.5 — quick start

1. Set `COCO_PATH` and `snname` in the config cell (defaults use `pipeline_config.SNNAME_DEFAULT`).
2. Run the template cell: creates `Outputs/<SN>/<SN>_spec_scale_groups.json` from MJD clustering if missing.
3. **Edit the JSON** to confirm XSHooter triplets (`members`, optional `merge_order`: `uvb`, `vis`, `nir`).
4. Run the scaling cell.

### Output modes

- **`scale_only`** (default in `pipeline_config.SPEC_SCALE_OUTPUT_MODE`): align flux; **keep separate files**.
- **`merge_join`**: after scaling, write one merged spectrum per group (global or per-group `"output_mode"` in JSON).

### Outputs

- `Inputs/Spectroscopy/2_spec_prescaled/<SN>/`
- `Inputs/Spectroscopy/2_spec_lists_prescaled/<SN>.list`
- `Outputs/<SN>/<SN>_spec_scale_report.json`
- `Outputs/<SN>/spec_scale_diagnostics/index.html` (when matplotlib works in your kernel)

## Tests

```bash
cd Codes
PYTHONPATH=. python3 -m unittest tests.test_spectra_pre_scale tests.test_mangle_spectra_log tests.test_pipeline_config_prescale -v
```

## Smoke run (CLI)

```bash
cd Codes
PYTHONPATH=. python3 -c "
import pipeline_config as p
from spectra_pre_scale import run_prescale_pipeline
run_prescale_pipeline(
    snname='AT2017gfo',
    coco_path=p.COCO_PATH,
    output_dir=p.COCO_PATH + 'Outputs/',
    diagnostics_dir=p.spec_scale_diagnostics_dir(p.COCO_PATH + 'Outputs/', 'AT2017gfo'),
)
"
```

On AT2017gfo, auto-grouping finds 10 same-time clusters (mostly XSHooter UVB/VIS/NIR + Magellan pairs); 9 ungrouped spectra are copied unchanged.

## Not in Phase 1

- Ryan GP sync (Phase 2)
- Iterative GP+mangle loop (Phase 3)
- 7.5 path toggles for `twodim_iter` (Phase 4)

See master plan: `.cursor/plans/gp_pipeline_review_bf44b5a2.plan.md` (or workspace plan copy).
