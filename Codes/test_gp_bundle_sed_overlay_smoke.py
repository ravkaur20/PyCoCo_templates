#!/usr/bin/env python3
"""Smoke tests: overlap median helper (pure) + GP bundle segments (SciPy/Ryan deps).

Run from Codes with conda/kernel matching ``ryan-updates/py_files``::

  cd Codes && python3 test_gp_bundle_sed_overlay_smoke.py

Use Ryan’s **`gp_scaled_bundle_meta.json`** with the **`bundle.npz`** he paired for that scaling
(e.g. **``gp_work_scaled_*.npz`` vs ``iter_KK/bundle.npz``**); mismatched pairs give wrong latent→linear flux.
"""

import os
import sys

import numpy as np

_CODES = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_CODES)
if _CODES not in sys.path:
    sys.path.insert(0, _CODES)


def test_median_overlap_rescale() -> None:
    from gp_bundle_sed_overlay import median_overlap_rescale_like_plot_native_epoch

    cw = np.linspace(4000.0, 8000.0, 20)
    cf = np.ones_like(cw) * 2.0
    rw = np.linspace(3000.0, 9000.0, 200)
    rf = np.ones_like(rw) * 10.0

    fl, md = median_overlap_rescale_like_plot_native_epoch(cw, cf, rw, rf)
    assert md.get("applied") is True, md
    assert abs(float(md["scale"]) - 5.0) < 1e-9
    assert np.allclose(fl, cf * float(md["scale"]))


def test_observer_window_vs_nearest_cluster() -> None:
    from gp_bundle_sed_overlay import scaled_training_segments_observer_aa

    bundle = os.path.join(
        _ROOT,
        "ryan-updates/my_surface_iter_repro_bundle/runs/my_surface_iter/iter_19/bundle.npz",
    )
    meta = os.path.join(_ROOT, "ryan-updates/py_files/gp_scaled_bundle_meta.json")
    cfg = os.path.join(
        _ROOT,
        "ryan-updates/my_surface_iter_repro_bundle/runs/my_surface_iter/iter_19/config.json",
    )
    assert os.path.isfile(bundle), "missing bundle: %s" % bundle

    _, info_w = scaled_training_segments_observer_aa(
        bundle,
        observer_phase_days=1.44,
        meta_json_path=meta,
        config_json_path=cfg,
        phase_pick="observer_window",
        phase_window_days=0.25,
    )

    _, info_c = scaled_training_segments_observer_aa(
        bundle,
        observer_phase_days=1.44,
        meta_json_path=meta,
        config_json_path=cfg,
        phase_pick="nearest_cluster",
        phase_window_days=None,
        spectrum_tolerance_norm=0.05,
    )

    nw = int(np.asarray(info_w["near_spec_phase_norm_keys"]).size)
    nc = int(np.asarray(info_c["near_spec_phase_norm_keys"]).size)
    print(
        "parity check: observer_window x2-keys=%d | nearest_cluster keys=%d" % (nw, nc)
    )
    assert nw >= nc


def smoke_bundle_segments() -> None:
    from gp_bundle_sed_overlay import scaled_training_segments_observer_aa

    bundle = os.path.join(
        _ROOT,
        "ryan-updates/my_surface_iter_repro_bundle/runs/my_surface_iter/iter_19/bundle.npz",
    )
    meta = os.path.join(_ROOT, "ryan-updates/py_files/gp_scaled_bundle_meta.json")
    cfg = os.path.join(
        _ROOT,
        "ryan-updates/my_surface_iter_repro_bundle/runs/my_surface_iter/iter_19/config.json",
    )

    segments, info = scaled_training_segments_observer_aa(
        bundle,
        observer_phase_days=1.0,
        meta_json_path=meta,
        config_json_path=cfg,
        phase_pick="nearest_cluster",
        spectrum_tolerance_norm=0.08,
        phase_window_days=None,
    )
    assert info.get("n_segments", 0) >= 1, info
    wl0 = segments[0]["wl_aa"]
    assert float(wl0.min()) > 0.0
    print(
        "bundle ok n_segments=%d wl=(%.1f, %.1f) AA anchor=%.4f d"
        % (
            len(segments),
            float(wl0.min()),
            float(wl0.max()),
            float(info["observer_phase_days_snap_anchor_observer"]),
        )
    )


if __name__ == "__main__":
    test_median_overlap_rescale()
    print("median_overlap_rescale ok")
    smoke_bundle_segments()
    test_observer_window_vs_nearest_cluster()
    print("all ok")
