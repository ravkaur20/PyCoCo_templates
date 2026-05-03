"""Lightweight tests for native FINAL helpers (no notebook / no matplotlib required)."""

from __future__ import annotations

import os
import unittest

import numpy as np

import pipeline_config as pconf
from comparison_check_log_utils import (
    collect_input_spectra_for_mode,
    index_final_native_files,
    list_final_spectra_native_rows,
    resolve_final_directory,
    twodim_final_branch,
)


class TestNativeCompareHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coco = pconf.COCO_PATH
        cls.sn = pconf.SNNAME_DEFAULT
        cls.datalc = os.path.join(
            cls.coco, "Inputs", "Photometry", "4_LCs_late_extrapolated"
        )
        branch = twodim_final_branch(pconf.MODE_EXTRAPOLATE_SHORT, pconf.SUBDIR_SPLICED)
        cls.fdir = resolve_final_directory(
            cls.coco, cls.sn, "as_observed", twodim_branch=branch
        )

    def test_index_matches_list_count_when_dir_exists(self):
        if not os.path.isdir(self.fdir):
            self.skipTest("FINAL directory missing: %s" % self.fdir)
        idx = index_final_native_files(
            self.fdir,
            self.coco,
            self.sn,
            datalc_path=self.datalc,
            final_suffixes=None,
        )
        rows = list_final_spectra_native_rows(
            self.fdir,
            self.coco,
            self.sn,
            flux_on_disk="auto",
            datalc_path=self.datalc,
            final_suffixes=None,
        )
        self.assertEqual(len(idx), len(rows))
        for a, b in zip(idx, rows):
            self.assertAlmostEqual(a[0], b[0], places=9)
            self.assertAlmostEqual(a[1], b[1], places=9)
            self.assertEqual(a[2], b[4])

    def test_collect_smoothed_nonempty(self):
        paths, mjds, _d, _ref = collect_input_spectra_for_mode(
            "smoothed", None, None, self.sn, self.coco
        )
        self.assertGreater(len(paths), 0)
        self.assertEqual(mjds.size, len(paths))
        self.assertTrue(np.all(np.isfinite(mjds)))


if __name__ == "__main__":
    unittest.main()
