"""Resolve FINAL_spectra path: grid stem (new) or legacy spec_mjd filename (via mangle table)."""
import os
import unittest

import numpy as np
import pandas as pd

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(
    CODES, "..", "Outputs", "AT2017gfo", "fitted_phot4mangling_AT2017gfo.dat"
)
FINAL_BASE = os.path.join(
    CODES, "..", "Outputs", "AT2017gfo", "FINAL_spectra_2dim", ""
)

convert2mjd = lambda x: float(
    x.replace("_spec_extended.txt", "")
    .replace("_spec_extended_FL.txt", "")
    .replace("_spec_extended_SMOOTH.txt", "")
)


def _path_FINAL_hostcorr(base, mpath, l):
    p_stem = base + l.replace("spec_extended", "FINAL_spec")
    if os.path.isfile(p_stem):
        return p_stem
    if not os.path.isfile(mpath):
        return p_stem
    phot = pd.read_csv(mpath, sep="\t")
    fk = float(convert2mjd(l))
    n = len(phot)
    m = np.zeros(n, dtype=bool)
    if "ext_grid_phase" in phot.columns:
        v = phot["ext_grid_phase"].values.astype(float)
        m |= np.isclose(v, fk, rtol=0.0, atol=1e-3, equal_nan=True)
        m |= np.isclose(
            np.round(v, 6), np.round(fk, 6), rtol=0.0, atol=0.0, equal_nan=True
        )
    sm = phot["spec_mjd"].values.astype(float)
    m |= sm == fk
    m |= np.isclose(sm, fk, rtol=0.0, atol=1e-3, equal_nan=True)
    if (not m.any()) and "spec_log_phase" in phot.columns:
        slp = phot["spec_log_phase"].values.astype(float)
        m |= np.isclose(slp, fk, rtol=0.0, atol=1e-3, equal_nan=True)
    if m.any():
        smjd = float(phot.loc[m, "spec_mjd"].values[0])
        p_mjd = base + "/%.6f_FINAL_spec.txt" % smjd
        if os.path.isfile(p_mjd):
            return p_mjd
    return p_stem


@unittest.skipUnless(os.path.isfile(OUT), "mangle file missing")
class TestRimangleFinalPath(unittest.TestCase):
    def test_match_0p537_touches_legacy_mjd_file(self):
        """0.537315 stem matches mangle ext_grid ~0.53725 -> MJD 57985.974001 on disk (legacy)."""
        l = "0.537315_spec_extended.txt"
        p = _path_FINAL_hostcorr(FINAL_BASE, OUT, l)
        self.assertTrue(os.path.isfile(p), p)
        self.assertIn("57985.974001_FINAL_spec", p)


if __name__ == "__main__":
    unittest.main()
