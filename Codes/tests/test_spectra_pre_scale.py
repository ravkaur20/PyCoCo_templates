import os
import tempfile
import unittest

import numpy as np

from spectra_pre_scale import (
    apply_flux_scale,
    load_spectrum_array,
    merge_spectra_concat,
    overlap_scale_factor_wls,
    save_spectrum_array,
    scale_group_members,
    suggest_scale_groups,
    SpectrumEntry,
)


def _make_spec(wls, flux):
    dt = np.dtype([("wls", "f8"), ("flux", "f8"), ("fluxerr", "f8")])
    out = np.empty(len(wls), dtype=dt)
    out["wls"] = wls
    out["flux"] = flux
    out["fluxerr"] = flux * 0.1
    return out


class TestSpectraPreScale(unittest.TestCase):
    def test_overlap_scale_factor(self):
        w = np.linspace(5000, 7000, 100)
        ref = _make_spec(w, np.ones_like(w) * 2e-15)
        arm = _make_spec(w, np.ones_like(w) * 1e-15)
        m, n = overlap_scale_factor_wls(ref, arm)
        self.assertGreater(n, 10)
        self.assertAlmostEqual(m, 2.0, delta=0.05)

    def test_scale_group_members(self):
        with tempfile.TemporaryDirectory() as td:
            w1 = np.linspace(3000, 6000, 80)
            w2 = np.linspace(5500, 9000, 80)
            p1 = os.path.join(td, "uvb.dat")
            p2 = os.path.join(td, "vis.dat")
            save_spectrum_array(p1, _make_spec(w1, np.ones(len(w1)) * 1e-15))
            save_spectrum_array(p2, _make_spec(w2, np.ones(len(w2)) * 0.5e-15))
            scaled, factors, ref = scale_group_members([p1, p2], merge_order=["uvb", "vis"])
            self.assertIn("uvb.dat", scaled)
            self.assertIn("vis.dat", scaled)
            self.assertAlmostEqual(factors["vis.dat"], 2.0, delta=0.15)

    def test_merge_join(self):
        w1 = np.linspace(3000, 5500, 50)
        w2 = np.linspace(5600, 9000, 50)
        scaled = {
            "a": _make_spec(w1, np.ones(len(w1))),
            "b": _make_spec(w2, np.ones(len(w2)) * 2),
        }
        merged = merge_spectra_concat(scaled, merge_order=["a", "b"])
        self.assertGreater(len(merged), 90)
        self.assertGreater(float(np.min(np.diff(merged["wls"]))), 0.0)

    def test_suggest_groups(self):
        entries = [
            SpectrumEntry(57983.969, 1.0, "/p/a", "a"),
            SpectrumEntry(57983.969, 1.0, "/p/b", "b"),
            SpectrumEntry(57990.0, 8.0, "/p/c", "c"),
        ]
        groups = suggest_scale_groups(entries, same_time_minutes=5.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].members), 2)


if __name__ == "__main__":
    unittest.main()
