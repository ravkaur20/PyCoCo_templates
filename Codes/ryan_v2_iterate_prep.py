"""Pre-iterate bookkeeping: optional phot-band strip compatible with notebook 6 bundles."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional


def strip_minimal_export_bundle(
	export_dir: str,
	out_name: str = "gp_minimal_bundle_nophot_m8767_m8217.npz",
	*,
	bands_csv: str = "-0.8767,-0.8217",
	round_digits: int = 4,
	phot_spec_threshold: int = 50,
	ryan_gp_dir: Optional[str] = None,
	python_exe: Optional[str] = None,
) -> str:
	"""Copy ``gp_minimal_bundle.npz`` with strips applied; writes ``export_dir/out_name``.

	Delegates to [**``ryan_gp/strip_photometry_bands.py``**](../ryan_gp/strip_photometry_bands.py)
	(subprocess uses ``--bands=…`` so negative CSV values are not misparsed by ``argparse``).

	Returns
	-------
	str
	    Absolute output path ``export_dir/out_name``.
	"""
	inp = os.path.join(os.path.abspath(export_dir), "gp_minimal_bundle.npz")
	out = os.path.join(os.path.abspath(export_dir), out_name)
	here_codes = os.path.dirname(os.path.abspath(__file__))
	_ry = ryan_gp_dir or os.path.join(here_codes, "ryan_gp")
	strip_py = os.path.join(_ry, "strip_photometry_bands.py")
	if not os.path.isfile(inp):
		raise FileNotFoundError("missing minimal bundle before strip: %r" % inp)
	if not os.path.isfile(strip_py):
		raise FileNotFoundError("strip script not found: %r" % strip_py)
	cmd = [
		python_exe or sys.executable,
		strip_py,
		"-i",
		inp,
		"-o",
		out,
		"--bands=%s" % str(bands_csv).strip(),
		"--round-digits=%d" % int(round_digits),
		"--phot-spec-threshold=%d" % int(phot_spec_threshold),
	]
	os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
	r = subprocess.run(cmd, cwd=_ry, capture_output=True, text=True)
	if r.returncode != 0:
		raise subprocess.CalledProcessError(
			r.returncode,
			cmd,
			output=r.stdout,
			stderr=r.stderr,
		)
	return os.path.abspath(out)
