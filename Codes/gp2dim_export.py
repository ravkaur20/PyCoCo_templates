"""Save minimal George 2D-GP training/prediction arrays for collaborators (``.npz`` + small JSON)."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np


def _jsonable(x: Any) -> Any:
	if isinstance(x, Mapping):
		return {str(k): _jsonable(v) for k, v in x.items()}
	if isinstance(x, (list, tuple)):
		return [_jsonable(v) for v in x]
	if isinstance(x, np.ndarray):
		return x.tolist()
	if isinstance(x, (np.floating, np.integer)):
		return float(x) if isinstance(x, np.floating) else int(x)
	if isinstance(x, (float, int, str, bool)) or x is None:
		return x
	return str(x)


def resolve_export_dir(GP2DIM_Class) -> str | None:
	"""Return directory to write under, or ``None`` if export is disabled."""
	try:
		import pipeline_config as _pc
	except ImportError:
		_pc = None
	conf_on = bool(getattr(_pc, "GP_EXPORT_MINIMAL", False)) if _pc is not None else False
	class_on = bool(getattr(GP2DIM_Class, "gp_export_minimal", False))
	if not (conf_on or class_on):
		return None
	sub = str(getattr(_pc, "GP_EXPORT_SUBDIR", "gp_minimal_export")) if _pc is not None else "gp_minimal_export"
	override = getattr(GP2DIM_Class, "gp_export_dir", None)
	base = override or os.path.join(GP2DIM_Class.save_plot_path, sub)
	os.makedirs(base, exist_ok=True)
	return base


def _prior_parts(prior: bool, points: Any, values: Any) -> tuple[bool, np.ndarray, np.ndarray]:
	if (
		prior
		and isinstance(points, np.ndarray)
		and isinstance(values, np.ndarray)
		and points.size
		and values.size
	):
		return True, np.asarray(points, dtype=np.float64), np.asarray(values, dtype=np.float64)
	return False, np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)


def save_gp_minimal_bundle(
	out_dir: str,
	*,
	X: np.ndarray,
	y: np.ndarray,
	yerr: np.ndarray,
	y_compute: np.ndarray,
	X_fill: np.ndarray,
	kernel_wls_scale: float,
	kernel_time_scale: float,
	y_var_scale: float,
	white_noise_variance: float,
	prior: bool,
	prior_points: np.ndarray,
	prior_values: np.ndarray,
	grid_norm_info: Mapping[str, Any],
	gp_module: str,
	snname: str,
	mode: str,
	kernel_layout: str,
) -> tuple[str, str]:
	"""Write ``gp_minimal_bundle.npz`` and ``gp_minimal_bundle_meta.json`` under ``out_dir``."""
	os.makedirs(out_dir, exist_ok=True)
	npz_path = os.path.join(out_dir, "gp_minimal_bundle.npz")
	json_path = os.path.join(out_dir, "gp_minimal_bundle_meta.json")

	wn = float(white_noise_variance)
	wn_log = float(np.log(wn)) if wn > 0.0 else np.nan

	np.savez_compressed(
		npz_path,
		X=np.asarray(X, dtype=np.float64),
		y=np.asarray(y, dtype=np.float64),
		yerr=np.asarray(yerr, dtype=np.float64),
		y_compute=np.asarray(y_compute, dtype=np.float64),
		X_fill=np.asarray(X_fill, dtype=np.float64),
		kernel_wls_scale=np.float64(kernel_wls_scale),
		kernel_time_scale=np.float64(kernel_time_scale),
		y_var_scale=np.float64(y_var_scale),
		white_noise_variance=np.float64(wn),
		white_noise_log=np.float64(wn_log),
		prior_used=np.int32(1 if prior else 0),
		prior_points=np.asarray(prior_points, dtype=np.float64),
		prior_values=np.asarray(prior_values, dtype=np.float64),
	)

	meta = {
		"snname": snname,
		"mode": mode,
		"gp_module": gp_module,
		"kernel_layout": kernel_layout,
		"column_order": "X[:,0]=normalized_log10_wavelength, X[:,1]=normalized_log10_phase_days",
		"prior_used": bool(prior),
		"compute_note": (
			"Use array ``y_compute`` for ``gp.compute``. For ln-flux modules it equals sqrt(yerr**2 + 1e-6**2); "
			"for linear_flux it equals yerr."
		),
		"grid_norm_info": _jsonable(grid_norm_info),
		"files": {"npz": os.path.basename(npz_path), "meta": os.path.basename(json_path)},
	}
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)

	print("[gp2dim_export] wrote %s and %s" % (npz_path, json_path), flush=True)
	return npz_path, json_path


def maybe_save_gp_minimal_export(
	GP2DIM_Class,
	*,
	X: np.ndarray,
	y: np.ndarray,
	yerr: np.ndarray,
	y_compute: np.ndarray,
	x1_fill: np.ndarray,
	x2_fill: np.ndarray,
	kernel_wls_scale: float,
	kernel_time_scale: float,
	prior: bool,
	points: Any,
	values: Any,
	grid_norm_info: Mapping[str, Any],
	gp_module: str,
	kernel_layout: str,
) -> None:
	out_dir = resolve_export_dir(GP2DIM_Class)
	if out_dir is None:
		return
	X_fill = np.vstack((np.asarray(x1_fill, dtype=float), np.asarray(x2_fill, dtype=float))).T
	prior_ok, pp, pv = _prior_parts(prior, points, values)
	wn = float(getattr(GP2DIM_Class, "gp_white_noise", 0.0))
	save_gp_minimal_bundle(
		out_dir,
		X=X,
		y=y,
		yerr=yerr,
		y_compute=y_compute,
		X_fill=X_fill,
		kernel_wls_scale=kernel_wls_scale,
		kernel_time_scale=kernel_time_scale,
		y_var_scale=float(np.var(y)),
		white_noise_variance=wn,
		prior=prior_ok,
		prior_points=pp,
		prior_values=pv,
		grid_norm_info=grid_norm_info,
		gp_module=gp_module,
		snname=str(getattr(GP2DIM_Class, "snname", "")),
		mode=str(getattr(GP2DIM_Class, "mode", "")),
		kernel_layout=kernel_layout,
	)
