import os
import tempfile
import unittest

import numpy as np

from mangle_spectra_log import (
    apply_mangling_mask_linear,
    demangle_log_spectrum,
    load_mangled_spectrum,
    save_mangled_spectrum,
)

CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestMangleSpectraLog(unittest.TestCase):
    def test_demangle_roundtrip(self):
        wls = np.linspace(4000, 8000, 50)
        flux = np.exp(-0.5 * ((wls - 6000) / 500) ** 2) * 1e-15
        ferr = flux * 0.1
        mask = 0.05 * np.sin(wls / 1000.0)
        _, log_f, log_e = apply_mangling_mask_linear(wls, flux, ferr, mask)
        back = demangle_log_spectrum(log_f, mask)
        np.testing.assert_allclose(back, np.log10(flux), rtol=1e-10)

    def test_four_column_io(self):
        wls = np.array([4000.0, 5000.0, 6000.0])
        log_f = np.array([-15.0, -14.5, -14.0])
        log_e = np.array([0.05, 0.05, 0.05])
        mask = np.array([0.01, 0.02, 0.01])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "spec.txt")
            save_mangled_spectrum(path, wls, log_f, log_e, mask)
            spec, m = load_mangled_spectrum(path)
            self.assertIsNotNone(m)
            np.testing.assert_allclose(spec["wls"], wls)
            np.testing.assert_allclose(spec["flux"], log_f)
            np.testing.assert_allclose(m, mask)


if __name__ == "__main__":
    unittest.main()
