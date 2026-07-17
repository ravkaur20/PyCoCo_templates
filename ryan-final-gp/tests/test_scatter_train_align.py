"""Tests for aligning subset GP training outputs to full-bundle row count."""

import numpy as np

from plot_results import scatter_train_vector_to_bundle


def test_scatter_identity_when_lengths_match() -> None:
    class _P:
        files = frozenset()

    v = np.array([1.0, 2.0, 3.0])
    out = scatter_train_vector_to_bundle(_P(), v, 3)
    np.testing.assert_array_equal(out, v)


def test_scatter_maps_subset() -> None:
    class _P:
        files = frozenset({"train_row_index_orig"})

        def __getitem__(self, k: str) -> np.ndarray:
            if k == "train_row_index_orig":
                return np.array([0, 2, 4], dtype=np.int64)
            raise KeyError(k)

    v = np.array([10.0, 20.0, 30.0])
    out = scatter_train_vector_to_bundle(_P(), v, 6)
    assert out is not None
    np.testing.assert_array_equal(out[[0, 2, 4]], v)
    assert np.isnan(out[1]) and np.isnan(out[3]) and np.isnan(out[5])


def test_scatter_returns_none_without_mapping_on_mismatch() -> None:
    class _P:
        files = frozenset()

    v = np.array([1.0, 2.0])
    assert scatter_train_vector_to_bundle(_P(), v, 5) is None
