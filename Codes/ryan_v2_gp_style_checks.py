"""Headless Colin-Ryan GP diagnostic PNGs matching ``RUNNING_MY_SURFACE_ITER.md``.

Runs ``ryan_gp/plot_results.py`` and ``ryan_gp/plot_bands_gp_overview.py`` as subprocesses
(so imports resolve with ``cwd=ryan_gp``) and forces ``MPLBACKEND=Agg``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional


def run_gp_style_check_pngs(
	*,
	workspace: str,
	runs_dir: str,
	meta: str,
	gp_tag_prefix: str,
	k_final: int,
	ryan_gp_dir: Optional[str] = None,
	bands_overview_only: str = "3,5",
	python_exe: Optional[str] = None,
	out_subdir_under_workspace: str = "ryan_gp_style_checks",
	mpl_backend: str = "Agg",
) -> tuple[str, str]:
	"""Plot last-iteration collaborator figures into ``workspace/<out_subdir>/kXX/``.

	Uses ``runs_dir/<prefix>_k{KK:02d}/`` for ``plot_results`` (same layout as iterate).

	Returns
	-------
	tag : str
	    ``"{gp_tag_prefix}_k{KK:02d}"``.
	out_root : str
	    Parent of ``plot_bands_overview`` (PNG overlays from ``plot_bands_gp_overview.py``).
	"""
	here = os.path.dirname(os.path.abspath(__file__))
	_ry = ryan_gp_dir or os.path.join(here, "ryan_gp")
	py = python_exe or sys.executable
	ws = os.path.abspath(workspace)
	rd = os.path.abspath(runs_dir)
	mt = os.path.abspath(os.path.expanduser(meta))
	tag = "%s_k%02d" % (str(gp_tag_prefix).strip(), int(k_final))

	bundle_p = os.path.join(ws, "iter_%02d" % int(k_final), "bundle.npz")

	sub_ws = os.path.normpath(str(out_subdir_under_workspace).strip("./"))
	out_root = os.path.join(ws, sub_ws, "k%02d" % int(k_final))
	ov_dir = os.path.join(out_root, "plot_bands_overview")
	os.makedirs(ov_dir, exist_ok=True)

	env = dict(os.environ)
	env["MPLBACKEND"] = mpl_backend

	pr_cmd = [
		py,
		os.path.join(_ry, "plot_results.py"),
		"--tag",
		tag,
		"--bundle",
		bundle_p,
		"--meta",
		mt,
		"--output-dir",
		rd,
		"--heatmap-raw",
	]
	subprocess.run(pr_cmd, cwd=_ry, check=True, env=env)

	pre = os.path.join(ws, "iter_%02d" % int(k_final), "predictions.npz")
	ob_cmd = [
		py,
		os.path.join(_ry, "plot_bands_gp_overview.py"),
		"--bundle",
		bundle_p,
		"--meta",
		mt,
		"--predictions",
		pre,
		"--output-dir",
		ov_dir,
		"--expect-pipeline-bundle",
		"--only-spec-bundle-ids",
		str(bands_overview_only),
	]
	subprocess.run(ob_cmd, cwd=_ry, check=True, env=env)
	return tag, out_root
