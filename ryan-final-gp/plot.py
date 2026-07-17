def make_results_plots(GP2DIM_Class, x1_fill, x2_fill, mu_fill, std_fill):
	gn = GP2DIM_Class.grid_norm_info
	offset = gn["offset"]
	scale_factor = gn["scale_factor"]
	x1m, x1s = float(gn["x1_mean"]), float(gn["x1_std"])
	x2m, x2s = float(gn["x2_mean"]), float(gn["x2_std"])

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
	plt.title('log10(wl): %.3f-%.3f'%(min(x1m + x1s * fit_wls[:int(len_wls/4)]),max(x1m + x1s * fit_wls[:int(len_wls/4)])))
	for i in fit_wls[:int(len_wls/4)]:
		mask = x1_fill==i
		# plt.plot((x2_fill[mask])*norm2+offset2, scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 		 lw=3, color=next(color), label='%.3f'%(i*norm1))
		plt.scatter(x2m + x2s * x2_fill[mask], scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			 color=next(color), s=1, label='%.3f'%(x1m + x1s * i))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(222)
	plt.title('from %.1f to %.1f'%(min(x1m + x1s * fit_wls[int(len_wls/4):2*int(len_wls/4)]),max(x1m + x1s * fit_wls[int(len_wls/4):2*int(len_wls/4)])))
	for i in fit_wls[int(len_wls/4):2*int(len_wls/4)]:
		mask = x1_fill==i
		# plt.plot((x2_fill[mask])*norm2+offset2, scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 		 lw=3, color=next(color), label='%.3f'%(i*norm1))
		plt.scatter(x2m + x2s * x2_fill[mask], scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			 color=next(color), s=1, label='%.3f'%(x1m + x1s * i))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(223)
	plt.title('from %.1f to %.1f'%(min(x1m + x1s * fit_wls[2*int(len_wls/4):3*int(len_wls/4)]),max(x1m + x1s * fit_wls[2*int(len_wls/4):3*int(len_wls/4)])))
	for i in fit_wls[2*int(len_wls/4):3*int(len_wls/4)]:
		mask = x1_fill==i
		# plt.plot((x2_fill[mask])*norm2+offset2, scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 		 lw=3, color=next(color), label='%.3f'%(i*norm1))
		plt.scatter(x2m + x2s * x2_fill[mask], scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			 color=next(color), s=1, label='%.3f'%(x1m + x1s * i))
	plt.xlabel('log10(phase days)')
	plt.ylabel('flux (linear)')
	plt.yscale('log')
	plt.subplot(224)
	plt.title('from %.1f to %.1f'%(min(x1m + x1s * fit_wls[3*int(len_wls/4):int(len_wls)]),max(x1m + x1s * fit_wls[3*int(len_wls/4):int(len_wls)])))
	for i in fit_wls[3*int(len_wls/4):int(len_wls)]:
	
		mask = x1_fill==i
		# plt.plot((x2_fill[mask])*norm2+offset2, scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 		 lw=3, color=next(color), label='%.3f'%(i*norm1))
		plt.scatter(x2m + x2s * x2_fill[mask], scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			 color=next(color), s=1, label='%.3f'%(x1m + x1s * i))
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
		'Linear phase (days) and linear flux - y-range may look compressed vs gp_results_wavelength_slices.pdf',
		fontsize=9,
		y=1.02,
	)
	plt.subplot(221)
	plt.title('log10(wl): %.3f-%.3f' % (min(x1m + x1s * fit_wls[: int(len_wls / 4)]), max(x1m + x1s * fit_wls[: int(len_wls / 4)])))
	for i in fit_wls[: int(len_wls / 4)]:
		mask = x1_fill == i
		# plt.plot(
		# 	phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
		# 	scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 	lw=3,
		# 	color=next(color2),
		# 	label='%.3f' % (i * norm1),
		# )
		plt.scatter(
			phase_days_from_norm_x2(x2_fill[mask], gn),
			scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			s=1,
			color=next(color2),
			label='%.3f' % (x1m + x1s * i),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(222)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(x1m + x1s * fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)]),
			max(x1m + x1s * fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)]),
		)
	)
	for i in fit_wls[int(len_wls / 4) : 2 * int(len_wls / 4)]:
		mask = x1_fill == i
		# plt.plot(
		# 	phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
		# 	scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 	lw=3,
		# 	color=next(color2),
		# 	label='%.3f' % (i * norm1),
		# )
		plt.scatter(
			phase_days_from_norm_x2(x2_fill[mask], gn),
			scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			s=1,
			color=next(color2),
			label='%.3f' % (x1m + x1s * i),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(223)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(x1m + x1s * fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)]),
			max(x1m + x1s * fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)]),
		)
	)
	for i in fit_wls[2 * int(len_wls / 4) : 3 * int(len_wls / 4)]:
		mask = x1_fill == i
		# plt.plot(
		# 	phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
		# 	scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 	lw=3,
		# 	color=next(color2),
		# 	label='%.3f' % (i * norm1),
		# )

		plt.scatter(
			phase_days_from_norm_x2(x2_fill[mask], gn),
			scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			s=1,
			color=next(color2),
			label='%.3f' % (x1m + x1s * i),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.subplot(224)
	plt.title(
		'from %.1f to %.1f'
		% (
			min(x1m + x1s * fit_wls[3 * int(len_wls / 4) : int(len_wls)]),
			max(x1m + x1s * fit_wls[3 * int(len_wls / 4) : int(len_wls)]),
		)
	)
	for i in fit_wls[3 * int(len_wls / 4) : int(len_wls)]:
		mask = x1_fill == i
		# plt.plot(
		# 	phase_days_from_norm_x2(x2_fill[mask], offset2, norm2),
		# 	scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
		# 	lw=3,
		# 	color=next(color2),
		# 	label='%.3f' % (i * norm1),
		# )
		plt.scatter(
			phase_days_from_norm_x2(x2_fill[mask], gn),
			scaled_ln_to_linear(mu_fill[mask], offset, scale_factor),
			s=1,
			color=next(color2),
			label='%.3f' % (x1m + x1s * i),
		)
	plt.xlabel('Phase (days)')
	plt.ylabel('flux (linear)')
	plt.tight_layout(rect=[0, 0, 1, 0.92])
	fig_lin.savefig(
		os.path.join(GP2DIM_Class.save_plot_path, "gp_results_wavelength_slices_linear_phase_linear_flux.pdf"),
		bbox_inches="tight",
	)
	plt.show()
