"""Tests for model_chi2 grid scan."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_DIR = os.path.dirname(os.path.abspath(__file__))
_CODES = os.path.dirname(_DIR)
if _CODES not in sys.path:
    sys.path.insert(0, _CODES)

from model_comparison_helpers import scale_bulla_intensity_to_observer, wave_rest_to_observer  # noqa: E402


def test_chi2_red_grid_argmin_planted_cell():
    import model_chi2

    z = 0.01
    d = 40.0
    n_w = 16
    w_rest = np.linspace(4500.0, 6500.0, n_w)
    t_days = np.array([0.5, 1.0, 1.5])
    n_obs = 3
    n_t = len(t_days)
    I = np.full((n_obs, n_t, n_w), 1e-30)
    ib, jb = 1, 2
    I[ib, jb, :] = 1.0
    blob = {
        "I_stokes": I,
        "time_days": t_days,
        "wave_rest": w_rest,
        "lbol": np.zeros((n_obs, n_t)),
    }
    wl_data = wave_rest_to_observer(w_rest, z)
    F_native = I[ib, jb, :].astype(float)
    F_data = scale_bulla_intensity_to_observer(F_native, z, d)
    fe_data = np.full(n_w, 0.1)
    chi2_red, _cos_theta, td = model_chi2.chi2_red_grid(
        blob, wl_data, F_data, fe_data, z=z, d_lum_mpc=d
    )
    assert chi2_red.shape == (n_obs, n_t)
    assert np.allclose(td, t_days)
    flat_i = int(np.nanargmin(chi2_red.ravel()))
    ii, jj = divmod(flat_i, n_t)
    assert (ii, jj) == (ib, jb)
    assert chi2_red[ib, jb] == pytest.approx(0.0, abs=1e-9)


def test_rows_from_grid_length():
    import model_chi2

    rng = np.random.default_rng(0)
    chi2_red = rng.random((2, 4))
    cos_theta = np.linspace(0.0, 1.0, 2)
    t_days = np.arange(4.0)
    rows = model_chi2.rows_from_grid("/dummy/path/model.hdf5", chi2_red, cos_theta, t_days)
    assert len(rows) == 8
    assert rows[0]["model_basename"] == "model.hdf5"


def test_rows_from_grid_extra_fields():
    import model_chi2

    chi2_red = np.ones((1, 1))
    cos_theta = np.array([0.5])
    t_days = np.array([1.0])
    rows = model_chi2.rows_from_grid(
        "/m.hdf5",
        chi2_red,
        cos_theta,
        t_days,
        extra={"target_phase_days": 2.5, "spectrum_fname": "x.txt"},
    )
    assert rows[0]["target_phase_days"] == 2.5
    assert rows[0]["spectrum_fname"] == "x.txt"


def test_default_target_phases():
    import model_chi2

    seq = model_chi2.default_target_phases_sequence()
    assert len(seq) == 11
    assert seq[0] == 0.5 and seq[-1] == 10.5
    assert model_chi2.parse_target_phases(None) == seq
    assert model_chi2.parse_target_phases("1,3") == [1.0, 3.0]


def test_iter_hdf5_paths_empty_tmp(tmp_path):
    import model_chi2

    assert model_chi2.iter_hdf5_paths(str(tmp_path)) == []


def test_plot_summary_bar_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    import pandas as pd

    import model_chi2

    df = pd.DataFrame(
        {
            "model_basename": ["a.hdf5", "b.hdf5", "b.hdf5"],
            "chi2_red": [3.0, 2.0, 1.2],
        }
    )
    out = tmp_path / "summary.png"
    model_chi2.plot_summary_bar(df, str(out))
    assert out.is_file()
