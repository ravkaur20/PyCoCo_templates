import os
import unittest

import pipeline_config as pconf


class TestPrescalePaths(unittest.TestCase):
    def test_prescaled_list_fallback(self):
        coco = pconf.COCO_PATH
        sn = "AT2017gfo"
        smooth = pconf.smoothed_spec_list_path(coco, sn)
        self.assertTrue(os.path.isfile(smooth))
        # prescaled may not exist yet; helper still returns path
        pre = pconf.prescaled_spec_list_path(coco, sn)
        self.assertIn("2_spec_lists_prescaled", pre)

    def test_spec_scale_paths(self):
        od = os.path.join(pconf.COCO_PATH, "Outputs")
        p = pconf.spec_scale_groups_json_path(od, "SN")
        self.assertIn("spec_scale_groups.json", p)


if __name__ == "__main__":
    unittest.main()
