"""Tests for phot dense-curve x1 tracking."""

import numpy as np

import plot_bands_gp_overview as pbo


def test_x1_on_dense_time_grid_interpolates() -> None:
    tt = np.array([0.0, 1.0, 2.0])
    x1 = np.array([-0.5, -0.3, -0.1])
    td = np.array([0.5, 1.5])
    out = pbo._x1_on_dense_time_grid(td, tt, x1)
    np.testing.assert_allclose(out[0], -0.4)
    np.testing.assert_allclose(out[1], -0.2)


def test_x1_duplicate_times_averaged() -> None:
    tt = np.array([1.0, 1.0, 2.0])
    x1 = np.array([0.0, 2.0, 10.0])
    td = np.array([1.0])
    out = pbo._x1_on_dense_time_grid(td, tt, x1)
    assert abs(float(out[0]) - 1.0) < 1e-9


def test_photometry_pseudo_wavelength_groups_rounded_merges() -> None:
    X = np.zeros((4, 2), dtype=float)
    X[:, 0] = [0.12341, 0.12344, 0.20001, 0.20002]
    phot = np.array([0, 1, 2, 3], dtype=int)
    gn = {"x1_mean": 0.0, "x1_std": 1.0, "_normalized_only": False}
    g = pbo.photometry_pseudo_wavelength_groups(
        X,
        phot,
        gn,
        grouping="rounded",
        pseudo_band_digits=4,
        unique_x1_decimals=12,
        max_unique_panels=500,
    )
    assert len(g) == 2
    assert set(g.keys()) == {"log10λ_norm≈0.1234", "log10λ_norm≈0.2000"}
    assert np.array_equal(np.sort(g["log10λ_norm≈0.1234"]), [0, 1])
    assert np.array_equal(np.sort(g["log10λ_norm≈0.2000"]), [2, 3])


def test_photometry_pseudo_wavelength_groups_unique_x1_splits() -> None:
    X = np.zeros((3, 2), dtype=float)
    X[:, 0] = [0.1, 0.2, 0.3]
    phot = np.array([0, 1, 2], dtype=int)
    gn = {"x1_mean": 0.0, "x1_std": 1.0, "_normalized_only": False}
    g = pbo.photometry_pseudo_wavelength_groups(
        X,
        phot,
        gn,
        grouping="unique_x1",
        pseudo_band_digits=4,
        unique_x1_decimals=-1,
        max_unique_panels=0,
    )
    assert len(g) == 3
