"""2D GP on log10(wavelength) x log10(phase) with **scaled linear flux** targets.

Sibling to ``GP2dim_utils_newlog`` (ln-flux GP). Grid cells are still log10(F) and log10(sigma)
from the log pipeline; this module converts to linear F and sigma_F before scaling.

Use ``import GP2dim_utils_newlog_linear_flux as GP2dim`` in notebook 6 and call
``transform2LINEAR_reshape``, ``setPRIOR_linear``, ``run_2DGP_GRID_linear``,
``make_results_plots_linear``, ``transform_back_andPlot_linear``.

``GP_WHITE_NOISE`` on the class is interpreted as variance in **scaled linear flux** squared
(not ln-flux). Tune separately from the log-flux module; see ``pipeline_config.GP_WHITE_NOISE_LINEAR``.
Training ``yerr`` floors: ``pipeline_config.GP_YERR_FLOOR_FRAC`` / ``GP_YERR_ABS_FLOOR`` (same as newlog).
"""

from __future__ import annotations

import os
import time
from itertools import cycle

import george
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from george.kernels import Matern32Kernel
from scipy.interpolate import griddata

from gp2dim_phase_merge import merge_extrap_mjds_dense_log_phase
from gp2dim_export import maybe_save_gp_minimal_export

from GP2dim_utils_newlog import (
    _apply_training_yerr_floors,
    color_dict,
    fill_gaps_phase_logspace,
    gp_dense_matrix_bytes_order_of_magnitude,
    log_prediction_phase_coverage,
    make_plots,
    mangled_flux_linear_from_log10,
    mangled_wls_linear_angstrom,
    mangled_wls_max_is_linear_angstrom,
    mycmap,
    phases_close,
    phase_days_from_norm_x2,
    prepare_grid,
    save_plots_files,
    x2_mask_for_phase,
)

_SCALE_FACTOR_ABS_FLOOR = 1e-8


def scaled_affine_to_physical(scaled_mu, offset, scale_factor):
    """Invert ``transform2LINEAR_reshape``: linear flux from scaled training targets."""
    return np.asarray(scaled_mu, dtype=float) * float(scale_factor) + float(offset)


def transform2LINEAR_reshape(GP2DIM_Class, raw_numbers, raw_numbers_err, off_xa, off_ya):
    """Linear F_lambda from log10 grid; scaled (F - offset) / scale_factor; yerr = sigma_F / scale.

    ``raw_numbers`` / ``raw_numbers_err`` are log10(F) and sigma_log10 (dex), as in NB6.
    """
    LN10 = np.log(10.0)
    data_log10 = raw_numbers.T.reshape(raw_numbers.shape[0] * raw_numbers.shape[1])
    data_log10_err = raw_numbers_err.T.reshape(
        raw_numbers_err.shape[0] * raw_numbers_err.shape[1]
    )

    data_log10 = np.copy(data_log10)
    data_log10_err = np.copy(data_log10_err)
    data_log10[~np.isfinite(data_log10)] = np.nan
    data_log10_err[~np.isfinite(data_log10_err)] = np.nan

    data = np.power(10.0, np.clip(np.asarray(data_log10, dtype=float), -350.0, 300.0))
    sigma_F = np.full_like(data, np.nan, dtype=float)
    m = np.isfinite(data) & np.isfinite(data_log10_err)
    sigma_F[m] = (
        np.abs(data[m])
        * LN10
        * np.maximum(np.asarray(data_log10_err[m], dtype=float), 0.0)
    )

    offset = float(np.nanmin(data[np.isfinite(data)]))
    spread = float(np.nanmedian(data[np.isfinite(data)] - offset))
    scale_factor = max(
        spread,
        _SCALE_FACTOR_ABS_FLOOR * max(abs(offset), 1.0),
        _SCALE_FACTOR_ABS_FLOOR,
    )

    data_scaled = (data - offset) / scale_factor
    data_error_scaled = sigma_F / scale_factor

    resh_wls = []
    for i in range(raw_numbers.shape[1]):
        resh_wls = np.concatenate([resh_wls, off_xa])

    resh_mjd = []
    for i in off_ya:
        resh_mjd = np.concatenate([resh_mjd, np.ones(len(off_xa)) * i])

    NOT_Isnan = (~np.isnan(data_scaled)) & (~np.isnan(data_error_scaled))

    x1_data = resh_wls[NOT_Isnan]
    x2_data = resh_mjd[NOT_Isnan]

    y_data_nonan = np.copy(data_scaled[NOT_Isnan])
    y_data_nonan_err = np.copy(data_error_scaled[NOT_Isnan])
    y_data_nonan_err = _apply_training_yerr_floors(
        GP2DIM_Class, y_data_nonan, y_data_nonan_err, stage="transform"
    )

    norm1 = np.max(x1_data)
    offset2 = np.min(x2_data)
    norm2 = np.max(x2_data - offset2)
    if norm2 <= 0.0 or not np.isfinite(norm2):
        raise ValueError(
            "transform2LINEAR_reshape: invalid norm2 (log-phase range); check grid time columns."
        )

    x1_data_norm = x1_data / norm1
    x2_data_norm = (x2_data - offset2) / norm2

    GP2DIM_Class.grid_norm_info = {
        "offset": offset,
        "scale_factor": scale_factor,
        "norm1": norm1,
        "norm2": norm2,
        "offset2": offset2,
        "flux_parametrization": "linear_scaled",
    }
    return (y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm)


def setPRIOR_linear(GP2DIM_Class, type_=None, PRIOR_file=None, PRIOR_folder=None):
    """Same geometry as ``setPRIOR`` in newlog, but prior values live in scaled **linear** flux space."""
    norm1 = GP2DIM_Class.grid_norm_info["norm1"]
    norm2 = GP2DIM_Class.grid_norm_info["norm2"]
    offset = GP2DIM_Class.grid_norm_info["offset"]
    offset2 = GP2DIM_Class.grid_norm_info["offset2"]
    scale_factor = GP2DIM_Class.grid_norm_info["scale_factor"]
    t0 = GP2DIM_Class.t0_fix

    if not PRIOR_file:
        if type_ in ["II", "IIn", "IIP", "IIL"]:
            PRIOR_file = "/prior_Hrich.txt"
        elif type_ in ["Ib", "Ic", "Ibc", "Ic-BL", "IcBL", "IIb"]:
            PRIOR_file = "/prior_SE.txt"
        else:
            raise ValueError("setPRIOR_linear: specify type_ / PRIOR_file / PRIOR_folder")

    wls_prior, phase_prior, color_prior = np.genfromtxt(
        PRIOR_folder + PRIOR_file, delimiter=",", unpack=True
    )

    original_fit = pd.read_csv(GP2DIM_Class.path_fit_phot, delimiter="\t")
    ref_cols = [c for c in original_fit.columns if c.endswith("_log_flux") and "_err" not in c]
    if not ref_cols:
        raise ValueError("setPRIOR_linear: no *_log_flux columns in %s" % GP2DIM_Class.path_fit_phot)
    counts = {c: np.sum(np.isfinite(original_fit[c].values)) for c in ref_cols}
    best = max(counts, key=counts.get)
    if counts[best] < 2:
        raise ValueError("setPRIOR_linear: insufficient valid points in reference band %s" % best)
    original_fit = original_fit[np.isfinite(original_fit[best].values)].copy()
    Vflux_log = original_fit[best].values
    Vflux = 10 ** Vflux_log
    others = [c for c in ref_cols if c != best]
    if len(others) > 0 and np.any(np.isfinite(original_fit[others[0]].values)):
        BVflux = 10 ** original_fit[best].values + 10 ** original_fit[others[0]].values
    else:
        BVflux = np.copy(Vflux)
    mjd_fit = t0 + 10 ** (original_fit["Log_Phase"].values)
    ok = ~np.isnan(BVflux)
    peak_MJD = mjd_fit[ok][np.argmax(BVflux[ok])]

    absolute_MJD = phase_prior + peak_MJD
    phase_gp = np.log10(np.maximum(absolute_MJD - t0, 1e-12))
    phase_prior_norm = (phase_gp - offset2) / norm2
    wls_prior_norm = np.log10(wls_prior) / norm1

    reshaped_color_prior = color_prior.reshape(
        len(np.unique(wls_prior)), len(np.unique(phase_prior))
    )
    Vflux_phase = np.interp(np.unique(phase_prior), mjd_fit - peak_MJD, Vflux)

    flux_prior = reshaped_color_prior * Vflux_phase
    flux_prior_transform = (flux_prior - offset) / scale_factor

    points = np.array([tup for tup in zip(wls_prior_norm, phase_prior_norm)])
    values = (flux_prior_transform).reshape(
        len(np.unique(phase_prior)) * len(np.unique(wls_prior))
    )
    return points, values


def augment_2dgp_training_t0_anchor_linear(
    x1_data_norm,
    x2_data_norm,
    y_data_nonan,
    y_data_nonan_err,
    grid_norm_info,
    *,
    log_phase_anchor: float,
    log10_flux_cap: float,
    log10_flux_err: float,
):
    """T0 anchor pseudo-points in scaled linear flux space."""
    LN10 = np.log(10.0)
    offset = float(grid_norm_info["offset"])
    sf = float(grid_norm_info["scale_factor"])
    offset2 = float(grid_norm_info["offset2"])
    norm2 = float(grid_norm_info["norm2"])

    F_cap = float(10.0 ** float(log10_flux_cap))
    sigma_F = abs(F_cap) * LN10 * max(float(log10_flux_err), 0.0)
    y_cap = (F_cap - offset) / sf
    sigma = sigma_F / sf
    x2a = (float(log_phase_anchor) - offset2) / norm2
    u1 = np.unique(np.asarray(x1_data_norm, dtype=float))
    if u1.size == 0:
        return x1_data_norm, x2_data_norm, y_data_nonan, y_data_nonan_err
    x1_add = u1.astype(float)
    x2_add = np.full(u1.shape, x2a, dtype=float)
    y_add = np.full(u1.shape, y_cap, dtype=float)
    yerr_add = np.full(u1.shape, sigma, dtype=float)
    return (
        np.concatenate([np.asarray(x1_data_norm, dtype=float), x1_add]),
        np.concatenate([np.asarray(x2_data_norm, dtype=float), x2_add]),
        np.concatenate([np.asarray(y_data_nonan, dtype=float), y_add]),
        np.concatenate([np.asarray(y_data_nonan_err, dtype=float), yerr_add]),
    )


def run_2DGP_GRID_linear(GP2DIM_Class, y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm,\
		kernel_wls_scale, kernel_time_scale, extrap_mjds, prior=False, points=np.nan, values=np.nan):
	
	""" ## for NUV extention:   extrap_mjds = grid_ext_columns
	## for spectra augmentation: 
	extrap_mjds = grid_ext.columns.values
	 if (len(extrap_mjds)>200):
		 extrap_mjds = grid_ext.columns.values[:200]
	 if (max(extrap_mjds-min(extrap_mjds))>200):
		 extrap_mjds = extrap_mjds[extrap_mjds-min(extrap_mjds)<200]
	 
	 tot_iteration = int(len(extrap_mjds)/slot_size+1)
	 print (tot_iteration)"""

	_log_progress = getattr(GP2DIM_Class, "gp_predict_progress", True)

	def _gp_log(msg):
		print(msg, flush=True)

	# TRAINING: X, y, terr
	norm1 = GP2DIM_Class.grid_norm_info['norm1']
	norm2 = GP2DIM_Class.grid_norm_info['norm2']

	if bool(getattr(GP2DIM_Class, "gp_2d_anchor_t0", False)):
		_nu_before = int(np.unique(np.asarray(x1_data_norm, dtype=float)).size)
		x1_data_norm, x2_data_norm, y_data_nonan, y_data_nonan_err = augment_2dgp_training_t0_anchor_linear(
			x1_data_norm,
			x2_data_norm,
			y_data_nonan,
			y_data_nonan_err,
			GP2DIM_Class.grid_norm_info,
			log_phase_anchor=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log_phase", -8.0)),
			log10_flux_cap=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log10_flux_cap", -50.0)),
			log10_flux_err=float(getattr(GP2DIM_Class, "gp_2d_t0_anchor_log10_flux_err", 2.0)),
		)
		if _log_progress:
			_gp_log(
				"[run_2DGP_GRID_linear] 2D t0-anchor training: +%i pseudo points (%i unique x1 nodes)"
				% (_nu_before, _nu_before)
			)

	if prior:
		from george.modeling import Model

		class Model_2dim(Model):
			parameter_names = ()
			def get_value(self, t):
				verbose = getattr(GP2DIM_Class, 'verbose', False)
				if verbose:
					print("t shape:", t.shape)
					print("t contents:", t)
				points_eval = np.array([tup for tup in zip(t[:,0], t[:,1])])
				if points_eval.size == 0 and verbose:
					print("Warning: points_eval is empty!")
				grid_z1 = griddata(points, values, points_eval, method='nearest')
				if verbose:
					print("grid_z1 contains NaN:", np.any(np.isnan(grid_z1)))
				grid_z1[np.isnan(grid_z1)] = 0.
				#plt.plot(t[:,0]*norm1, grid_z1, '-b', label='PRIOR')
				return grid_z1
    	
		mean_model = Model_2dim()

	X = np.vstack((x1_data_norm, x2_data_norm)).T
	y = y_data_nonan
	yerr = _apply_training_yerr_floors(GP2DIM_Class, y, y_data_nonan_err, stage="compute")

	_n_train = len(y)
	_gp_log("[run_2DGP_GRID_linear] starting (prior=%r) N_train=%i" % (bool(prior), _n_train))
	if getattr(GP2DIM_Class, "verbose", False) or getattr(GP2DIM_Class, "gp_print_training_size", True):
		_gp_log(
			"[run_2DGP_GRID_linear] N_train = %i finite training points (scaled linear flux); X.shape=%s dtype=%s"
			% (_n_train, X.shape, X.dtype)
		)
		n2b = gp_dense_matrix_bytes_order_of_magnitude(_n_train)
		_gp_log(
			"[run_2DGP_GRID_linear] rough dense N×N float64 footprint ~ %.2f GB (hint only; George uses factorization ~O(N³) time)"
			% (n2b / (1024.0 ** 3),)
		)
		if _n_train >= 12000:
			_gp_log(
				"[run_2DGP_GRID_linear] WARNING: N_train is very large — expect long runtime and high RAM use. "
				"Reduce training density (e.g. larger DELTA in the grid builder) if the kernel dies."
			)

	kernel_mix = Matern32Kernel([kernel_wls_scale, kernel_time_scale], ndim=2)
	kernel2dim = np.var(y) * kernel_mix
	_gp_wn = float(getattr(GP2DIM_Class, "gp_white_noise", 0.0))
	# George >=0.4: homogeneous jitter via GP(white_noise=ln(variance)); legacy code used
	# kernels.WhiteKernel(c) with c the *variance* added on the diagonal (not log).
	_gp_extra = {}
	if _gp_wn > 0.0:
		_gp_extra["white_noise"] = float(np.log(_gp_wn))

	if prior:
		gp = george.GP(kernel2dim, mean=mean_model, **_gp_extra)
	else:
		gp = george.GP(kernel2dim, **_gp_extra)

	_gp_log("[run_2DGP_GRID_linear] calling gp.compute (no progress inside; may take minutes) …")
	_t0 = time.perf_counter()
	gp.compute(X, yerr)
	_gp_log("[run_2DGP_GRID_linear] gp.compute finished in %.1f s" % (time.perf_counter() - _t0,))
		
	# wls_normed_range = np.sort(np.concatenate(( np.arange(1600.,3000., 40),
	# 										  np.arange(3000.,9000., 10),
	# 										  np.arange(9000.,10350., 40))))/GP2DIM_Class.grid_norm_info['norm1']
	#RAV added this
	# wls_min = np.min(GP2DIM_Class.grids[0])
	# wls_max = np.max(GP2DIM_Class.grids[0])
	# wls_normed_range = np.arange(wls_min, wls_max + 1, 40) / GP2DIM_Class.grid_norm_info['norm1']

	wls_min = float(np.min(GP2DIM_Class.grids[0]))
	wls_max = float(np.max(GP2DIM_Class.grids[0]))
	_wl_min_a = getattr(GP2DIM_Class, "pipeline_wl_min_a", None)
	_wl_max_a = getattr(GP2DIM_Class, "pipeline_wl_max_a", None)
	if _wl_min_a is not None:
		wls_min = min(wls_min, float(np.log10(float(_wl_min_a))))
	if _wl_max_a is not None:
		wls_max = max(wls_max, float(np.log10(float(_wl_max_a))))
	span_wl = float(wls_max - wls_min)
	if span_wl <= 0.0:
		raise ValueError("run_2DGP_GRID_linear: invalid log10(wavelength) span (wls_max <= wls_min).")

	# Prediction grid in log10(lambda): cap count so gp.predict memory stays bounded.
	# Old code used np.arange(..., 0.005) which could create ~10^3 points per phase; combined with
	# return_cov=True that allocated (Ntest x Ntest) doubles per batch and often OOM-killed the kernel.
	# Defaults tuned for kilonova SED speed vs ~0.5–1% sampling in lambda (see notebook / LOGSPACE_PIPELINE_PLAN)
	_gp_n_wl = int(getattr(GP2DIM_Class, "gp_predict_n_wavelength", 300))
	_wl_step = float(getattr(GP2DIM_Class, "gp_predict_wl_step", 0.01))
	n_from_step = int(np.ceil(span_wl / _wl_step)) + 1
	n_wl_use = max(2, min(_gp_n_wl, n_from_step))
	wls_log_grid = np.linspace(wls_min, wls_max, n_wl_use)
	wls_normed_range = wls_log_grid / norm1

	#mu_fill_resh = []
	mu_fill_resh = np.empty((0, 3))
	std_fill_resh = []
	
	slot_size = max(1, int(getattr(GP2DIM_Class, "gp_predict_slot_size", 3)))
	extrap_mjds = np.asarray(extrap_mjds, dtype=float)
	if extrap_mjds.size == 0:
		raise ValueError("run_2DGP_GRID_linear: extrap_mjds is empty (no phase columns to predict).")
	_dense_on = bool(getattr(GP2DIM_Class, "gp_predict_dense_log_phase", False))
	_dense_n = int(getattr(GP2DIM_Class, "gp_predict_dense_log_phase_n", 64))
	if _dense_on:
		_phase_before = extrap_mjds.copy()
		extrap_mjds = merge_extrap_mjds_dense_log_phase(extrap_mjds, _dense_n)
		if _log_progress:
			_gp_log(
				"[run_2DGP_GRID_linear] dense log-phase prediction: %i → %i columns (n_dense=%i)"
				% (len(_phase_before), len(extrap_mjds), _dense_n)
			)
			log_prediction_phase_coverage(_phase_before, label="phase columns (before dense merge)")
			log_prediction_phase_coverage(extrap_mjds, label="phase columns (after dense merge)")
	tot_iteration = max(1, int(len(extrap_mjds) / slot_size + 1))
	frac_tot_iteration = 0
	if _log_progress:
		_gp_log(
			"[run_2DGP_GRID_linear] predict grid: n_wavelength=%i | extrap phase columns=%i | slot_size=%i | outer loops=%i"
			% (n_wl_use, len(extrap_mjds), slot_size, tot_iteration)
		)

	for j in range(tot_iteration):
		mjd_normed_range = ((extrap_mjds[j*slot_size:(j+1)*slot_size])-GP2DIM_Class.grid_norm_info['offset2'])/GP2DIM_Class.grid_norm_info['norm2']
		x1_fill = []#np.random.permutation(np.linspace(0,1., N))
		x2_fill = []#np.random.permutation(np.linspace(0,1., N))
		for i in wls_normed_range:
			for k in mjd_normed_range:
				x1_fill.append(i)
				x2_fill.append(k)
		
		x1_fill=np.array(x1_fill) 
		x2_fill=np.array(x2_fill)
		
		X_fill = np.vstack((x1_fill, x2_fill)).T	
		# return_var=True: diagonal only (avoids Ntest^2 covariance allocation; same mean as return_cov)
		# Chunk test points so each predict() stays small (avoids peak RAM / solver edge cases)
		_chunk = max(200, int(getattr(GP2DIM_Class, "gp_predict_chunk_size", 1500)))
		n_pred = len(X_fill)
		if _log_progress or getattr(GP2DIM_Class, "verbose", False):
			_n_chunk_outer = int(np.ceil(n_pred / float(_chunk)))
			_gp_log(
				"[run_2DGP_GRID_linear] predict slot %i / %i | n_pred=%i | chunk_size=%i (~%i chunks)"
				% (j + 1, tot_iteration, n_pred, _chunk, _n_chunk_outer)
			)
		frac_tot_iteration = int(20.0 * (j + 1) / tot_iteration)
		#print('[','*'*frac_tot_iteration,' '*(20-frac_tot_iteration),']' + ' %i of %i'%(slot_size*(j+1),slot_size*tot_iteration)+' spec extrapolated', end='\r')
		mu_iter = np.empty(n_pred, dtype=float)
		var_iter = np.empty(n_pred, dtype=float)
		_log_chunks = bool(
			getattr(GP2DIM_Class, "gp_predict_log_chunks", False)
			or getattr(GP2DIM_Class, "verbose", False)
		)
		_chunk_idx = 0
		for s0 in range(0, n_pred, _chunk):
			s1 = min(s0 + _chunk, n_pred)
			_tc0 = time.perf_counter()
			m_sub, v_sub = gp.predict(y, X_fill[s0:s1], return_var=True)
			mu_iter[s0:s1] = m_sub
			var_iter[s0:s1] = v_sub
			_chunk_idx += 1
			if _log_chunks:
				_gp_log(
					"  [run_2DGP_GRID_linear] chunk %i rows %i:%i done in %.2f s"
					% (_chunk_idx, s0, s1, time.perf_counter() - _tc0)
				)
		std_iter = np.sqrt(np.maximum(var_iter, 0.0))

		if getattr(GP2DIM_Class, "gp_diagnostic_slices", False) and j == 0:
			_diag_dir = GP2DIM_Class.save_plot_path
			os.makedirs(_diag_dir, exist_ok=True)
			n_phase_plot = min(3, len(mjd_normed_range))
			for pi in range(n_phase_plot):
				mj = float(mjd_normed_range[pi])
				mask = np.isclose(x2_fill, mj, rtol=0.0, atol=1e-12)
				if not np.any(mask):
					continue
				fig_d, ax_d = plt.subplots(figsize=(8, 2))
				ax_d.plot(norm1 * x1_fill[mask], mu_iter[mask], "-k", label="PREDICTION")
				if prior and isinstance(points, np.ndarray) and isinstance(values, np.ndarray) and points.size and values.size:
					pe = np.column_stack((x1_fill[mask], x2_fill[mask]))
					gz = griddata(points, values, pe, method="nearest")
					gz = np.where(np.isnan(gz), 0.0, gz)
					ax_d.plot(norm1 * x1_fill[mask], gz, "-b", label="PRIOR")
				ax_d.set_xlabel("log10(wavelength)")
				ax_d.set_ylabel("scaled linear flux (GP space)")
				ax_d.legend(loc="best", fontsize=8)
				fig_d.savefig(
					os.path.join(_diag_dir, "gp_diag_slot0_phase%i.pdf" % pi),
					bbox_inches="tight",
				)
				plt.close(fig_d)

		mu_resh_iter = mu_iter.reshape(len(wls_normed_range), len(mjd_normed_range))
		std_resh_iter = std_iter.reshape(len(wls_normed_range), len(mjd_normed_range))

		#if mu_fill_resh==[]:
		if mu_fill_resh.size == 0:
			mu_fill_resh = np.copy(mu_resh_iter)
			std_fill_resh = np.copy(std_resh_iter)
		else:
			mu_fill_resh = np.concatenate([mu_fill_resh, mu_resh_iter], axis=1)
			std_fill_resh = np.concatenate([std_fill_resh, std_resh_iter], axis=1)

	_gp_log('[' + '*'*frac_tot_iteration + ' '*(20-frac_tot_iteration) + '] '
		+ '%i of %i' % (min(slot_size * (j + 1), len(extrap_mjds)), len(extrap_mjds)) + ' spec extrapolated')
	mu_fill = mu_fill_resh.reshape(len(wls_normed_range)*len(extrap_mjds))
	std_fill = std_fill_resh.reshape(len(wls_normed_range)*len(extrap_mjds))

	mjd_normed_range = (extrap_mjds-GP2DIM_Class.grid_norm_info['offset2'])/GP2DIM_Class.grid_norm_info['norm2']
	
	x1_fill = []#np.random.permutation(np.linspace(0,1., N))
	x2_fill = []#np.random.permutation(np.linspace(0,1., N))
	for i in wls_normed_range:
		for k in mjd_normed_range:
			x1_fill.append(i)
			x2_fill.append(k)
	
	x1_fill=np.array(x1_fill) 
	x2_fill=np.array(x2_fill)
	
	_gp_log('EXTENDING SPECTRA BETWEEN:')
	_gp_log(
		'log10(wavelength): %s %s'
		% (min(x1_fill * GP2DIM_Class.grid_norm_info['norm1']), max(x1_fill * GP2DIM_Class.grid_norm_info['norm1']))
	)
	_gp_log(
		'log10(phase days): %s %s'
		% (
			min(x2_fill * GP2DIM_Class.grid_norm_info['norm2']) + GP2DIM_Class.grid_norm_info['offset2'],
			max(x2_fill * GP2DIM_Class.grid_norm_info['norm2']) + GP2DIM_Class.grid_norm_info['offset2'],
		)
	)
	_gp_log("[run_2DGP_GRID_linear] done.")

	maybe_save_gp_minimal_export(
		GP2DIM_Class,
		X=X,
		y=y,
		yerr=yerr,
		y_compute=np.asarray(yerr, dtype=float),
		x1_fill=x1_fill,
		x2_fill=x2_fill,
		kernel_wls_scale=kernel_wls_scale,
		kernel_time_scale=kernel_time_scale,
		prior=prior,
		points=points,
		values=values,
		grid_norm_info=GP2DIM_Class.grid_norm_info,
		gp_module="GP2dim_utils_newlog_linear_flux",
		kernel_layout="joint_Matern32_ndim2",
	)

	return (x1_fill, x2_fill, mu_fill, std_fill)

def make_results_plots_linear(GP2DIM_Class, x1_fill, x2_fill, mu_fill, std_fill):
	norm1 = GP2DIM_Class.grid_norm_info['norm1']
	norm2 = GP2DIM_Class.grid_norm_info['norm2']
	offset = GP2DIM_Class.grid_norm_info['offset']
	offset2 = GP2DIM_Class.grid_norm_info['offset2']
	scale_factor = GP2DIM_Class.grid_norm_info['scale_factor']

	#plt.scatter(norm2*x2_fill, norm1*x1_fill, marker='.', c=mu_fill, alpha=1., 
	#		vmin=0., cmap = mycmap)
	##plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	##plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.xlabel('MJD')
	#plt.ylabel('wls')
	#plt.colorbar()
	
	# PLOT xWLS LC and check how smooth the time variation in each single wls is:
	fit_wls = (np.unique(x1_fill)[::10])
	len_wls = len(fit_wls)
	color=cycle(plt.cm.gnuplot(np.linspace(0.05,0.95,len_wls)))
	
	fig = plt.figure(figsize=(10,6))
	plt.subplot(221)
	plt.title('log10(wl): %.3f–%.3f'%(min(fit_wls[:int(len_wls/4)]*norm1),max(fit_wls[:int(len_wls/4)]*norm1)))
	for i in fit_wls[:int(len_wls/4)]:
		mask = x1_fill==i
		plt.plot((x2_fill[mask])*norm2+offset2, scaled_affine_to_physical(mu_fill[mask], offset, scale_factor),
				 lw=3, color=next(color), label='%.3f'%(i*norm1))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(222)
	plt.title('from %.1f to %.1f'%(min(fit_wls[int(len_wls/4):2*int(len_wls/4)]*norm1),max(fit_wls[int(len_wls/4):2*int(len_wls/4)]*norm1)))
	for i in fit_wls[int(len_wls/4):2*int(len_wls/4)]:
		mask = x1_fill==i
		plt.plot((x2_fill[mask])*norm2+offset2, scaled_affine_to_physical(mu_fill[mask], offset, scale_factor),
				 lw=3, color=next(color), label='%.3f'%(i*norm1))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(223)
	plt.title('from %.1f to %.1f'%(min(fit_wls[2*int(len_wls/4):3*int(len_wls/4)]*norm1),max(fit_wls[2*int(len_wls/4):3*int(len_wls/4)]*norm1)))
	for i in fit_wls[2*int(len_wls/4):3*int(len_wls/4)]:
		mask = x1_fill==i
		plt.plot((x2_fill[mask])*norm2+offset2, scaled_affine_to_physical(mu_fill[mask], offset, scale_factor),
				 lw=3, color=next(color), label='%.3f'%(i*norm1))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(224)
	plt.title('from %.1f to %.1f'%(min(fit_wls[3*int(len_wls/4):int(len_wls)]*norm1),max(fit_wls[3*int(len_wls/4):int(len_wls)]*norm1)))
	for i in fit_wls[3*int(len_wls/4):int(len_wls)]:
	
		mask = x1_fill==i
		plt.plot((x2_fill[mask])*norm2+offset2, scaled_affine_to_physical(mu_fill[mask], offset, scale_factor),
				 lw=3, color=next(color), label='%.3f'%(i*norm1))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	fig.savefig(
		os.path.join(GP2DIM_Class.save_plot_path, "gp_results_wavelength_slices.pdf"),
		bbox_inches="tight",
	)
	plt.show()
	plt.close(fig)

	# Linear phase (days) × linear flux (no log y); compare to log-y PDF above if dynamic range is large
	color2 = cycle(plt.cm.gnuplot(np.linspace(0.05, 0.95, len_wls)))
	fig_lin = plt.figure(figsize=(10, 6))
	fig_lin.suptitle(
		'Linear phase (days) and linear flux — y-range may look compressed vs gp_results_wavelength_slices.pdf',
		fontsize=9,
		y=1.02,
	)
	plt.subplot(221)
	plt.title('log10(wl): %.3f–%.3f' % (min(fit_wls[: int(len_wls / 4)] * norm1), max(fit_wls[: int(len_wls / 4)] * norm1)))
	for i in fit_wls[: int(len_wls / 4)]:
		mask = x1_fill == i
		plt.plot(
			phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
			scaled_affine_to_physical(mu_fill[mask], offset, scale_factor),
			lw=3,
			color=next(color2),
			label='%.3f' % (i * norm1),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(222)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)] * norm1),
			max(fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)] * norm1),
		)
	)
	for i in fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)]:
		mask = x1_fill == i
		plt.plot(
			phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
			scaled_affine_to_physical(mu_fill[mask], offset, scale_factor),
			lw=3,
			color=next(color2),
			label='%.3f' % (i * norm1),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(223)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)] * norm1),
			max(fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)] * norm1),
		)
	)
	for i in fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)]:
		mask = x1_fill == i
		plt.plot(
			phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
			scaled_affine_to_physical(mu_fill[mask], offset, scale_factor),
			lw=3,
			color=next(color2),
			label='%.3f' % (i * norm1),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(224)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(fit_wls[3 * int(len_wls / 4) : int(len_wls)] * norm1),
			max(fit_wls[3 * int(len_wls / 4) : int(len_wls)] * norm1),
		)
	)
	for i in fit_wls[3 * int(len_wls / 4) : int(len_wls)]:
		mask = x1_fill == i
		plt.plot(
			phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
			scaled_affine_to_physical(mu_fill[mask], offset, scale_factor),
			lw=3,
			color=next(color2),
			label='%.3f' % (i * norm1),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.tight_layout(rect=[0, 0, 1, 0.92])
	fig_lin.savefig(
		os.path.join(GP2DIM_Class.save_plot_path, "gp_results_wavelength_slices_linear_phase_linear_flux.pdf"),
		bbox_inches="tight",
	)
	plt.show()
	plt.close(fig_lin)


def transform_back_andPlot_linear(GP2DIM_Class, x1_fill, x2_fill, mu_fill, std_fill, y_data_nonan):

	norm1 = GP2DIM_Class.grid_norm_info['norm1']
	norm2 = GP2DIM_Class.grid_norm_info['norm2']
	offset = GP2DIM_Class.grid_norm_info['offset']
	offset2 = GP2DIM_Class.grid_norm_info['offset2']
	scale_factor = GP2DIM_Class.grid_norm_info['scale_factor']

	mu_fill_conv = scaled_affine_to_physical(mu_fill, offset, scale_factor)
	std_fill_conv = np.abs(scale_factor * std_fill)

	y_data_conv = scaled_affine_to_physical(y_data_nonan, offset, scale_factor)
	#else:
	#	mu_fill_conv = (mu_fill*scale_factor + offset)
	#	std_fill_conv = np.abs( scale_factor*std_fill )
	#
	#	y_data_conv =(y_data_nonan*scale_factor + offset)
		
	fig = plt.figure(1, figsize=(10,3))
	x2_fill_phase = offset2 + norm2 * x2_fill
	plt.subplot(121)
	plt.scatter(x2_fill_phase, norm1*x1_fill, marker='s', s=10,  c=mu_fill_conv, alpha=1., 
				vmin=0., cmap = mycmap)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	plt.xlabel('log10(phase days)')
	plt.ylabel('log10(wavelength)')
	plt.colorbar()
	
	plt.subplot(122)
	plt.scatter(x2_fill_phase, norm1*x1_fill, marker='s', s=10,  c=std_fill_conv, alpha=1., 
				vmin=0., cmap = mycmap)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	plt.xlabel('log10(phase days)')
	plt.ylabel('log10(wavelength)')
	plt.colorbar()
	plt.show()
	fig.savefig(GP2DIM_Class.save_plot_path+'/2d_surface.png', bbox_inches='tight')
	plt.close(fig)

	phase_lin_days = np.power(10.0, x2_fill_phase)
	wl_lin_angstrom = np.power(10.0, norm1 * x1_fill)
	fig_lin = plt.figure(figsize=(10, 3))
	plt.subplot(121)
	plt.scatter(phase_lin_days, wl_lin_angstrom, marker='s', s=10, c=mu_fill_conv, alpha=1.,
				vmin=0., cmap=mycmap)
	plt.xlabel('Phase (days)')
	plt.ylabel('Wavelength (Å)')
	cb = plt.colorbar()
	cb.set_label('Linear flux')
	plt.subplot(122)
	plt.scatter(phase_lin_days, wl_lin_angstrom, marker='s', s=10, c=std_fill_conv, alpha=1.,
				vmin=0., cmap=mycmap)
	plt.xlabel('Phase (days)')
	plt.ylabel('Wavelength (Å)')
	cb2 = plt.colorbar()
	cb2.set_label('Std (linear)')
	_splin = GP2DIM_Class.save_plot_path
	fig_lin.savefig(os.path.join(_splin, 'gp_2d_surface_linear_axes.pdf'), bbox_inches='tight')
	fig_lin.savefig(os.path.join(_splin, 'gp_2d_surface_linear_axes.png'), bbox_inches='tight')
	plt.show()
	plt.close(fig_lin)

	fig_mu = plt.figure(figsize=(10, 3))
	plt.subplot(121)
	plt.scatter(phase_lin_days, wl_lin_angstrom, marker='s', s=10, c=mu_fill, alpha=1., cmap=mycmap)
	plt.xlabel('Phase (days)')
	plt.ylabel('Wavelength (Å)')
	cb_m = plt.colorbar()
	cb_m.set_label('Scaled linear flux mean (GP)')
	plt.subplot(122)
	plt.scatter(phase_lin_days, wl_lin_angstrom, marker='s', s=10, c=std_fill, alpha=1., cmap=mycmap)
	plt.xlabel('Phase (days)')
	plt.ylabel('Wavelength (Å)')
	cb_m2 = plt.colorbar()
	cb_m2.set_label('Scaled linear flux std (GP)')
	fig_mu.savefig(
		os.path.join(GP2DIM_Class.save_plot_path, 'gp_2d_surface_linear_axes_scaled_mu.pdf'),
		bbox_inches='tight',
	)
	plt.show()
	plt.close(fig_mu)
	
	max_val = np.max(y_data_conv)
	med_val = np.median(y_data_conv)
	
	#fig = plt.figure(1, figsize=(8,4))
	#spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	#scale = (max_val-med_val)/5.
	#a=0
	#mangled_original_list = GP2DIM_Class.mangledspec_list
	#
	#for j in range(len(GP2DIM_Class.get_spec_mjd())):
	#	mj = spec_mjd_list[j]
	#	spec_file_original = GP2DIM_Class.load_mangledfile(mangled_original_list[j])
	#	a +=1
	#	#mask = x2_fill==(mj-offset2)/norm2
	#	#plt.plot(x1_fill[mask]*norm1, mu_fill_conv[mask], label='%i'%(mj-offset2), lw=0.8, color='r')
	#	#plt.plot(off_xa, grid_ext[mj]+(a-1)*scale, label='Raw spec %i'%(mj-offset2), lw=1.8, color='k')
	#	plt.plot(spec_file_original['wls'], spec_file_original['flux']+(a-1)*scale,
	#			 label='Raw spec %i'%(mj-offset2), lw=1.0, color='k')
	#for b in GP2DIM_Class.avail_filters:
	#	wls, T = GP2DIM_Class.get_filt_transmission(b)
	#	plt.plot(wls, 0.5*T*max_val/max(T), linestyle='-', lw=2, color=PyCoCo_info.color_dict[b])
	#plt.xlim(1600,11000)
	#plt.title(GP2DIM_Class.snname)
	#plt.xlabel('Wavelength')
	#plt.ylabel('Calibrated Flux + offset')
	#fig.savefig(GP2DIM_Class.save_plot_path+'/to_be_extended_spec1.pdf', bbox_inches='tight')
	#plt.show()
	#plt.close(fig)
	
	fig = plt.figure(1, figsize=(8,5))

	spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	scale = (max_val-med_val)/5.
	a=0
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		a +=1
		mask = x2_mask_for_phase(x2_fill, mj, offset2, norm2)
		plt.plot(x1_fill[mask]*norm1, mu_fill_conv[mask]+(a-1)*scale, 
				 label='Extrapolated %.2f'%(mj-offset2), lw=0.8, color='r')
		plt.fill_between(x1_fill[mask]*norm1, (mu_fill_conv[mask]-std_fill_conv[mask])+(a-1)*scale , 
				 (mu_fill_conv[mask]+std_fill_conv[mask])+(a-1)*scale , facecolor='r', alpha=0.3)
	
	#colors_to_replace = plt.cm.viridis(np.linspace(0, 1, len(GP2DIM_Class.avail_filters)))
	#plt.xlim(1600,11000)
	#for i, b in enumerate(GP2DIM_Class.avail_filters):
	#	plt.vlines((GP2DIM_Class.lam_eff(b)), 0, 1., linestyle='--', lw=4, label=b, color=colors_to_replace[b])
	#RAV commented this out
	#plt.xlim(1600,11000)
	for b in GP2DIM_Class.avail_filters:
		plt.vlines((10**GP2DIM_Class.lam_eff(b)), 0, 1., linestyle='--', lw=4, label=b, color=color_dict[b])

	plt.title(GP2DIM_Class.snname)
	plt.xlabel('log10(wavelength)')
	plt.ylabel('Calibrated Flux + offset (linear)')
	fig.savefig(GP2DIM_Class.save_plot_path+'/extended_spec_LOG_SPACE.pdf', bbox_inches='tight')
	plt.show()
	plt.close(fig)
	
	fig = plt.figure(1, figsize=(14,6))
	plt.rc('font', family='serif')
	plt.rc('xtick', labelsize=13)
	plt.rc('ytick', labelsize=13)
	
	spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	scale = (max_val-med_val)/5.
	a=0
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		a +=1
		mask = x2_mask_for_phase(x2_fill, mj, offset2, norm2)
		plt.plot(10**(x1_fill[mask]*norm1), mu_fill_conv[mask]+(a-1)*scale, 
				 label='Extrapolated %.2f'%(mj-offset2), lw=0.8, color='r')
		plt.fill_between(10**(x1_fill[mask]*norm1), (mu_fill_conv[mask]-std_fill_conv[mask])+(a-1)*scale , 
				 (mu_fill_conv[mask]+std_fill_conv[mask])+(a-1)*scale , facecolor='r', alpha=0.3)
	a=0	
	mangled_original_list = GP2DIM_Class.mangledspec_list
	
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		spec_file_original = GP2DIM_Class.load_mangledfile(mangled_original_list[j])
		a +=1
		mask = x2_mask_for_phase(x2_fill, mj, offset2, norm2)
		#plt.plot(x1_fill[mask]*norm1, mu_fill_conv[mask], label='%i'%(mj-offset2), lw=0.8, color='r')
		#plt.plot(off_xa, grid_ext[mj]+(a-1)*scale, label='Raw spec %i'%(mj-offset2), lw=1.8, color='k')
		# Mangled files: linear Å in ``wls``, log10 flux in ``flux`` (see 5_Mangle_spectra_KN_log save)
		wls_lin_m = mangled_wls_linear_angstrom(spec_file_original)
		flx_lin_m = mangled_flux_linear_from_log10(spec_file_original['flux'])
		plt.plot(wls_lin_m, flx_lin_m+(a-1)*scale,
				 label='Raw spec %.2f'%(mj-offset2), lw=1, color='k')
	
	#plt.xlim(1600,11000)
	
	#for b in GP2DIM_Class.avail_filters:
	#	wls, T = GP2DIM_Class.get_filt_transmission(b)
	#	plt.plot(wls, 0.5*T*max_val/max(T), linestyle='-', lw=4, color=PyCoCo_info.color_dict[b])
	plt.title(GP2DIM_Class.snname)
	plt.xlabel('Wavelength')
	plt.ylabel('Calibrated Flux + offset')
	fig.savefig(GP2DIM_Class.save_plot_path+'/extended_spec.pdf', bbox_inches='tight')
	plt.show()
	plt.close(fig)
	
	return (mu_fill_conv, std_fill_conv, y_data_conv)
