"""Re-mangle skips when there is no in-MJD ratio point (no out-MJD fallback)."""
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

_CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODES not in sys.path:
    sys.path.insert(0, _CODES)
import remangle_identity

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

    def test_identity_iteration_zero_gate0_locals_alias_mangling_mask(self):
        """NB7 uses ``mang_mask, mang_mask_err = self.mangling_mask[0]`` after identity/GP fork."""

        class _Obj:
            def band_flux(self, filt, use_what=0):
                return (0.0, 1.0, 0.1)

        obj = _Obj()
        obj.ext_spec_linear = np.array(
            [(4000.0, 1e-17, 1e-18), (5000.0, 2e-17, 2e-18)],
            dtype=[("wls", "<f8"), ("flux", "<f8"), ("fluxerr", "<f8")],
        )
        remangle_identity.setup_identity_iteration_zero(
            obj,
            flux_floor=0.0,
            wls_eff=np.array([4500.0]),
            used_filters=["Swope_g"],
        )
        mang_mask, mang_mask_err = obj.mangling_mask[0]
        self.assertTupleEqual(mang_mask.shape, obj.ext_spec_linear["flux"].shape)
        np.testing.assert_array_equal(mang_mask, np.ones(obj.ext_spec_linear["flux"].shape[0]))
        np.testing.assert_array_equal(mang_mask_err, np.zeros(obj.ext_spec_linear["flux"].shape[0]))

    def test_wire_final_matches_identity_final_mangled_spec_dtype(self):
        """FINAL-from-extended path: same numeric treatment as iteration-0 identity spec."""

        class _Obj:
            pass

        obj = _Obj()
        obj.ext_spec_linear = np.array(
            [(4000.0, 1e-20, 1e-21), (5000.0, 2e-17, 2e-18)],
            dtype=[("wls", "<f8"), ("flux", "<f8"), ("fluxerr", "<f8")],
        )
        floor = 1e-19
        remangle_identity.wire_final_mangled_spec_from_extended_linear(obj, floor)

        lin = obj.ext_spec_linear
        exp_flux = np.maximum(np.asarray(lin["flux"], float), floor)
        np.testing.assert_allclose(obj.final_mangled_spec["flux"], exp_flux)
        np.testing.assert_allclose(obj.final_mangled_spec["fluxerr"], lin["fluxerr"])
        np.testing.assert_allclose(obj.final_mangled_spec["wls"], lin["wls"])
        self.assertEqual(obj.final_mangled_spec.dtype.names, ("wls", "flux", "fluxerr"))


if __name__ == "__main__":
    unittest.main()
