"""Resolve last Ryan surface-iteration posterior for NB6 plotting / ``save_plots_files``.

Collaborator subprocess ``iterate_gp_surface_bundle_scale`` writes::

    <save_plot_path>/ryan_surface_iterations/iteration_log.jsonl
    <save_plot_path>/ryan_surface_iterations/iter_KK/{predictions,bundle}.npz

Use :func:`load_final_surface_arrays` to rebuild ``mu_fill`` / ``std_fill`` (latent ln-space),
then run the usual ``GP2dim.transform_back_andPlot`` + ``save_plots_files`` for notebook 7.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np


def default_ryan_surface_workspace(save_plot_path: str) -> str:
	return os.path.join(os.path.normpath(save_plot_path), "ryan_surface_iterations")


def iteration_log_path(workspace: str) -> str:
	return os.path.join(workspace, "iteration_log.jsonl")


def predictions_npz_for_iter(workspace: str, k: int) -> str:
	return os.path.join(workspace, "iter_%02d" % int(k), "predictions.npz")


def last_logged_iteration_record(workspace: str) -> Optional[dict[str, Any]]:
	log_p = iteration_log_path(workspace)
	if not os.path.isfile(log_p):
		return None
	last: Optional[dict[str, Any]] = None
	with open(log_p, encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				rec = json.loads(line)
			except json.JSONDecodeError:
				continue
			if isinstance(rec, dict) and isinstance(rec.get("iteration"), int):
				last = rec
	return last


def last_iteration_from_log(workspace: str) -> Optional[int]:
	rec = last_logged_iteration_record(workspace)
	if rec is None:
		return None
	return int(rec["iteration"])


def last_iteration_from_filesystem(workspace: str) -> Optional[int]:
	"""Highest ``KK`` such that ``iter_KK/predictions.npz`` exists."""
	if not os.path.isdir(workspace):
		return None
	best: Optional[int] = None
	for name in os.listdir(workspace):
		if not name.startswith("iter_"):
			continue
		suf = name[len("iter_") :]
		if not suf.isdigit():
			continue
		k = int(suf)
		if os.path.isfile(predictions_npz_for_iter(workspace, k)):
			if best is None or k > best:
				best = k
	return best


def resolve_last_iteration_index(workspace: str) -> Optional[int]:
	"""Prefer ``iteration_log.jsonl`` tail; fallback to filesystem scan."""
	k_log = last_iteration_from_log(workspace)
	pred_exists = predictions_npz_for_iter
	if k_log is not None and os.path.isfile(pred_exists(workspace, k_log)):
		return k_log
	ch = last_iteration_from_filesystem(workspace)
	if ch is not None:
		return ch
	return k_log


def iteration_record_for_index(workspace: str, iteration: int) -> Optional[dict[str, Any]]:
	"""Return the JSONL record whose ``iteration`` field equals ``iteration`` if any."""
	log_p = iteration_log_path(workspace)
	if not os.path.isfile(log_p):
		return None
	k_target = int(iteration)
	out: Optional[dict[str, Any]] = None
	with open(log_p, encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				rec = json.loads(line)
			except json.JSONDecodeError:
				continue
			if isinstance(rec, dict) and "iteration" in rec and int(rec["iteration"]) == k_target:
				out = rec
	return out


def load_final_surface_arrays(
	workspace: str,
	*,
	mu_key: str = "mu",
	std_key: str = "std",
	iteration: Optional[int] = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]]]:
	"""Return ``(k, x1_fill, x2_fill, mu_fill, std_fill, iteration_record)``.

	Shapes: ``mu_fill``.size == ``std_fill``.size == ``x1_fill``.size == ``x2_fill``.size.
	Arrays are mutable copies suitable for overwriting notebook variables.

	If ``iteration`` is an integer, load that ``iter_KK/predictions.npz`` directly.
	Otherwise resolve ``KK`` from ``iteration_log.jsonl`` tail (if ``predictions`` exists),
	then filesystem max.

	The returned ``iteration_record`` is the JSON line for ``KK`` when present; if
	``iteration`` was ``None``, it falls back to the **last line** of the log (historical behaviour).

	Raises
	------
	FileNotFoundError
	    Missing workspace directory, no iteration resolved, or ``predictions.npz`` absent.
	KeyError / ValueError
	    Malformed collaborator ``predictions.npz``.
	"""
	if not workspace or not os.path.isdir(workspace):
		raise FileNotFoundError("Ryan surface workspace not found: %r" % workspace)

	if iteration is None:
		k = resolve_last_iteration_index(workspace)
		last_rec = last_logged_iteration_record(workspace)
	else:
		k = int(iteration)
		last_rec = iteration_record_for_index(workspace, k)
	if k is None:
		raise FileNotFoundError(
			"No completed iteration found under %r (empty or missing iteration_log / predictions)"
			% workspace
		)

	pred = predictions_npz_for_iter(workspace, k)
	if not os.path.isfile(pred):
		raise FileNotFoundError("predictions.npz missing for iter %02d: %s" % (k, pred))

	z = np.load(pred, allow_pickle=False)
	try:
		if mu_key not in z.files:
			raise KeyError("predictions missing key %r; available: %s" % (mu_key, z.files))
		if std_key not in z.files:
			raise KeyError("predictions missing key %r" % std_key)
		if "X_fill" not in z.files:
			raise KeyError("predictions missing X_fill; available: %s" % (z.files,))

		Xf = np.asarray(z["X_fill"], dtype=np.float64)
		if Xf.ndim != 2 or Xf.shape[1] < 2:
			raise ValueError("X_fill must be (N,2+) array; got %s" % (Xf.shape,))

		mu_fill = np.asarray(z[mu_key], dtype=np.float64).reshape(-1)
		std_fill = np.asarray(z[std_key], dtype=np.float64).reshape(-1)
		n = int(Xf.shape[0])
		if mu_fill.shape[0] != n or std_fill.shape[0] != n:
			raise ValueError(
				"len(mu)=%d len(std)=%d vs X_fill rows %d"
				% (mu_fill.shape[0], std_fill.shape[0], n)
			)

		x1_fill = np.asarray(Xf[:, 0], dtype=np.float64).reshape(-1)
		x2_fill = np.asarray(Xf[:, 1], dtype=np.float64).reshape(-1)
	finally:
		z.close()

	return k, x1_fill.copy(), x2_fill.copy(), mu_fill.copy(), std_fill.copy(), last_rec
