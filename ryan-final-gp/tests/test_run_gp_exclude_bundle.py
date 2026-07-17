"""Tests for run_gp --exclude-spec-bundle-id row filtering."""

import numpy as np
import pytest

from run_gp import _exclude_rows_by_spec_bundle_id, _parse_exclude_spec_bundle_ids


def test_parse_exclude_ids() -> None:
    assert _parse_exclude_spec_bundle_ids(None) == []
    assert _parse_exclude_spec_bundle_ids("") == []
    assert _parse_exclude_spec_bundle_ids("1, 2 ,3") == [1, 2, 3]


def test_exclude_drops_matching_rows_only() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    yerr = np.ones(6)
    sb = np.array([-1, -1, 1, 1, 2, 2], dtype=np.int32)
    obs = np.array(["phot", "phot", "spec", "spec", "spec", "spec"], dtype=object)
    X2, y2, e2, o2, nd = _exclude_rows_by_spec_bundle_id(X, y, yerr, sb, [1], obs)
    assert nd == 2
    assert X2.shape[0] == 4
    np.testing.assert_array_equal(X2, X[[0, 1, 4, 5]])
    np.testing.assert_array_equal(y2, y[[0, 1, 4, 5]])
    assert list(o2) == ["phot", "phot", "spec", "spec"]


def test_exclude_empty_list_noop() -> None:
    X = np.ones((3, 2))
    y = np.zeros(3)
    yerr = np.ones(3)
    sb = np.array([1, 1, 1], dtype=np.int32)
    X2, y2, e2, o2, nd = _exclude_rows_by_spec_bundle_id(X, y, yerr, sb, [], None)
    assert nd == 0
    assert X2 is X


def test_exclude_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="spec_bundle_id length"):
        _exclude_rows_by_spec_bundle_id(
            np.ones((2, 2)),
            np.ones(2),
            np.ones(2),
            np.ones(3, dtype=np.int32),
            [1],
            None,
        )
