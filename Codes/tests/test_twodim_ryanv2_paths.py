import os
import sys
import unittest

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
    sys.path.insert(0, CODES)

import pipeline_config as pconf


class TestTwodimRyanv2Paths(unittest.TestCase):
    def tearDown(self):
        pconf.USE_LEGACY_TWODIM_LAYOUT = False

    def test_ryanv2_roots(self):
        od = "/tmp/out"
        sn = "SNX"
        pconf.USE_LEGACY_TWODIM_LAYOUT = False
        b = pconf.twodim_ryanv2_extended_base(od, sn, "extend_spectra")
        self.assertIn(pconf.TWODIM_RYANV2_SUBDIR_ROOT, b)
        pd = pconf.twodim_ryanv2_product_dir(od, sn, "extend_spectra", pconf.SUBDIR_FULL_GP)
        self.assertIn("full_gp", pd)
        br = pconf.twodim_ryanv2_final_branch("extend", "spliced")
        self.assertTrue(br.startswith(pconf.TWODIM_RYANV2_SUBDIR_ROOT))

    def test_legacy_ryanv2_dirname(self):
        pconf.USE_LEGACY_TWODIM_LAYOUT = True
        b = pconf.twodim_ryanv2_extended_base("/tmp", "SN", "extend_spectra")
        self.assertIn(pconf.LEGACY_TWODIM_RYANV2_DIRNAME, b)

    def test_final_branch_rjf_vs_ryanv2(self):
        a = pconf.final_spectra_twodim_branch(
            "extrapolate", "full_gp", use_rjf=True, use_ryanv2=False
        )
        self.assertIn("twodim_rjf", a)
        b = pconf.final_spectra_twodim_branch(
            "extrapolate", "full_gp", use_rjf=False, use_ryanv2=True
        )
        self.assertIn("twodim_ryanv2", b)
        self.assertNotEqual(a, b)

    def test_final_branch_exclusive(self):
        with self.assertRaises(ValueError):
            pconf.final_spectra_twodim_branch(
                "extend", "spliced", use_rjf=True, use_ryanv2=True
            )


if __name__ == "__main__":
    unittest.main()
