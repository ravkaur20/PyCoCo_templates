import os
import sys
import unittest

CODES = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODES not in sys.path:
    sys.path.insert(0, CODES)

import comparison_check_log_utils as cc


class TestResolveFinalBranch(unittest.TestCase):
    def test_twodim_branch_joins_path(self):
        coco = "/tmp/coco"
        d = cc.resolve_final_directory(
            coco, "SN", "as_observed", twodim_branch="extend/spliced"
        )
        self.assertIn("extend%s%s" % (os.sep, "spliced"), d)
        self.assertTrue(d.endswith(os.path.join("as_observed")))


if __name__ == "__main__":
    unittest.main()
