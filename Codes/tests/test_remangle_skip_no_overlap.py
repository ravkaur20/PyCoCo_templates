"""Re-mangle skips when there is no in-MJD ratio point (no out-MJD fallback)."""
import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_ReMangle_class():
    path = os.path.join(_REPO, "Codes", "_cell8_rimangle_extract.py")
    spec = importlib.util.spec_from_file_location("re_cell8_rimangle", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.ReMangle_SingleSpectrumClass


class TestRemangleSkipNoOverlap(unittest.TestCase):
    def test_mangle_skips_and_returns_false_when_no_in_mjd_ratios(self):
        ReM = _load_ReMangle_class()
        obj = ReM.__new__(ReM)
        obj.spec_file = "0.0_dummy_spec_extended_FL.txt"

        empty = np.array([])
        out_wls = np.array([4000.0, 5000.0])
        out_filt = np.array(["Swope_g", "Swope_r"], dtype=object)

        with patch.object(
            ReM,
            "calculate_ratios4mangling",
            return_value=(empty, empty, empty, empty, out_wls, out_filt),
        ) as _cr:
            with patch.object(
                ReM,
                "GP_interpolation_mangle",
                side_effect=AssertionError("GP must not run with zero in-MJD points"),
            ):
                out = ReM.mangle_iteration_function(obj)

        self.assertIs(out, False)
        _cr.assert_called_once()

    def test_iteration_guard_does_not_call_max_on_empty(self):
        """Regression: empty ratios must not be passed to max()."""
        ratios = np.array([])
        should_iterate = bool(
            len(np.asarray(ratios).ravel()) and np.nanmax(np.abs(ratios - 1.0)) > 0.01
        )
        self.assertFalse(should_iterate)


if __name__ == "__main__":
    unittest.main()
