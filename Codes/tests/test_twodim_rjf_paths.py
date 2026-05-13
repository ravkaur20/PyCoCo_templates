import os
import sys
import unittest

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
    sys.path.insert(0, CODES)

import pipeline_config as pconf


class TestTwodimRjfPaths(unittest.TestCase):
    def tearDown(self):
        pconf.USE_LEGACY_TWODIM_LAYOUT = False

    def test_rjf_roots_under_twodim_rjf(self):
        od = "/tmp/out"
        sn = "SNX"
        pconf.USE_LEGACY_TWODIM_LAYOUT = False
        b = pconf.twodim_rjf_extended_base(od, sn, "extend_spectra")
        self.assertIn("twodim_rjf", b)
        pd = pconf.twodim_rjf_product_dir(od, sn, "extend_spectra", pconf.SUBDIR_FULL_GP)
        self.assertIn("full_gp", pd)
        br = pconf.twodim_rjf_final_branch("extend", "spliced")
        self.assertTrue(br.startswith("twodim_rjf"))

    def test_legacy_rjf_uses_parallel_dirname(self):
        pconf.USE_LEGACY_TWODIM_LAYOUT = True
        b = pconf.twodim_rjf_extended_base("/tmp", "SN", "extend_spectra")
        self.assertIn(pconf.LEGACY_TWODIM_RJF_DIRNAME, b)



if __name__ == "__main__":
    unittest.main()
