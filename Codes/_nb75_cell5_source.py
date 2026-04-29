from textwrap import dedent

CELL5 = dedent(
    r"""
    from comparison_check_log_utils import (
        create_lookup_table as _create_lookup_table_impl,
        lookup_index_is_mjd,
    )


    def create_lookup_table(data_dir, wavelength_range=None, wavelength_bins=10000):
        return _create_lookup_table_impl(
            data_dir,
            COCO_PATH,
            SNNAME,
            flux_on_disk=FINAL_FLUX_ON_DISK,
            datalc_path=DATALC_PATH,
            wavelength_range=wavelength_range,
            wavelength_bins=wavelength_bins,
        )


    def _flux_at_observing_mjd(mjd_query, lookup_table, mjd0=None):
        """Closest table row to ``mjd_query``; returns (MJD at that row, F_lambda array)."""
        idx = np.asarray(lookup_table.index, dtype=float)
        if lookup_index_is_mjd(lookup_table):
            key = float(idx[np.argmin(np.abs(idx - mjd_query))])
            return key, lookup_table.loc[key].values.astype(float)
        if mjd0 is None:
            raise ValueError(
                "mjd0 is required when the lookup table index is days from a reference, not MJD."
            )
        rel = mjd_query - mjd0
        key = float(idx[np.argmin(np.abs(idx - rel))])
        return mjd0 + key, lookup_table.loc[key].values.astype(float)


    def synthetic_lightcurve(filter_file, system="abmag",
                             lookup_table=None, common_wavelengths=None,
                             spectra_list=None):
        """
        Compute a synthetic light curve by integrating spectra through a filter.

        Inputs (provide ONE of the two data pathways)
        --------------------------------------------
        - spectra_list: list of (time, wavelength, flux) tuples.
            * time: MJD (float) after TwoD ``create_lookup_table``, or days-from-start for legacy tables
            * wavelength: Å array (monotonic)
            * flux: F_lambda array in erg s^-1 cm^-2 Å^-1
        - OR
        - lookup_table + common_wavelengths:
            * lookup_table: DataFrame [times x wavelengths] of F_lambda
            * common_wavelengths: 1D Å array

        Parameters
        ----------
        filter_file : str
            Two-column file (wavelength[Å], transmission[0–1]).
        system : str
            "flam", "abmag", "stmag", "vegamag", or "countrate".

        Returns
        -------
        times : np.ndarray
            Synthetic times (same convention as the input table / spectra_list).
        lc : np.ndarray
            Synthetic values in the requested system.
        """
        filt_wave, filt_thru = np.loadtxt(filter_file, unpack=True)
        bandpass = SpectralElement(Empirical1D, points=filt_wave * u.AA, lookup_table=filt_thru)

        times, lc = [], []

        if spectra_list is not None:
            for (t, wl, fl) in spectra_list:
                m = np.isfinite(fl)
                if m.sum() == 0:
                    times.append(t); lc.append(np.nan); continue
                src = SourceSpectrum(Empirical1D, points=wl[m] * u.AA,
                                     lookup_table=fl[m] * u.erg/u.s/(u.cm**2)/u.AA)
                obs = Observation(src, bandpass, force="taper")
                val = obs.countrate().value if system == "countrate" else obs.effstim(system).value
                times.append(t); lc.append(val)

        elif lookup_table is not None and common_wavelengths is not None:
            for i, t in enumerate(lookup_table.index):
                fl = lookup_table.iloc[i].values.astype(float)
                m = np.isfinite(fl)
                if m.sum() == 0:
                    times.append(t); lc.append(np.nan); continue
                src = SourceSpectrum(Empirical1D, points=common_wavelengths[m] * u.AA,
                                     lookup_table=fl[m] * u.erg/u.s/(u.cm**2)/u.AA)
                obs = Observation(src, bandpass, force="taper")
                val = obs.countrate().value if system == "countrate" else obs.effstim(system).value
                times.append(t); lc.append(val)
        else:
            raise ValueError("Provide either spectra_list OR (lookup_table + common_wavelengths).")

        return np.array(times), np.array(lc)


    def _auto_align_times(syn_times, obs_mjd, reference_mjd=None, syn_times_are_mjd=None):
        """
        Align observed MJD and synthetic times onto a common 'days since t0' axis.

        Heuristics:
          - If syn_times_are_mjd is provided, use it.
          - Else treat syn times as MJD if median(syn_times) > 10000.
          - If syn times are relative (days), set t0 = obs_mjd.min().
          - If syn times are absolute MJD, set t0 = min(obs_mjd.min(), syn_times.min()).
          - If reference_mjd is given, use that as t0.
        """
        syn_times = np.asarray(syn_times, dtype=float)
        obs_mjd = np.asarray(obs_mjd, dtype=float)

        if reference_mjd is not None:
            t0 = float(reference_mjd)
            if syn_times_are_mjd is None:
                syn_times_are_mjd = (np.nanmedian(syn_times) > 10000)
        else:
            if syn_times_are_mjd is None:
                syn_times_are_mjd = (np.nanmedian(syn_times) > 10000)

            if syn_times_are_mjd:
                t0 = np.nanmin([np.nanmin(obs_mjd), np.nanmin(syn_times)])
            else:
                t0 = np.nanmin(obs_mjd)

        obs_time = obs_mjd - t0
        syn_time = syn_times - (t0 if syn_times_are_mjd else 0.0)
        return syn_time, obs_time, t0

    def compare_lightcurves_mag(filter_file, data_file,
                                lookup_table=None, common_wavelengths=None,
                                spectra_list=None, system="abmag", mjd0=None,
                                syn_times_are_mjd=None):
        """
        Compare synthetic lightcurve in AB mag to observed photometry (in mags).

        For TwoD FINAL tables built with ``create_lookup_table`` in this notebook, synthetic
        times are absolute MJD; they are plotted as days since ``mjd0``.
        """
        syn_times, syn_mags = synthetic_lightcurve(
            filter_file, system=system,
            lookup_table=lookup_table,
            common_wavelengths=common_wavelengths,
            spectra_list=spectra_list
        )

        df = pd.read_csv(data_file)

        bandname = os.path.basename(filter_file).replace(".dat", "")

        obs = df[df["band"] == bandname].copy()
        if obs.empty:
            raise ValueError("No photometry found for band '%s' in %s" % (bandname, data_file))

        if mjd0 is None:
            mjd0 = obs["MJD"].min()

        if syn_times_are_mjd is None:
            syn_times_are_mjd = float(np.nanmax(np.abs(syn_times))) > 40000.0

        obs_time = obs["MJD"].values - mjd0
        syn_time = (syn_times - mjd0) if syn_times_are_mjd else syn_times

        plt.figure(figsize=(7, 5))
        plt.errorbar(obs_time, obs["Mag"], yerr=obs["Mag_err"],
                     fmt="o", label="Observed", color="tab:blue")
        plt.plot(syn_time, syn_mags, "-", label="Synthetic", color="tab:orange")

        plt.xlabel("Time since %.5f MJD (days)" % mjd0)
        plt.ylabel("AB mag")
        plt.title("Lightcurve in %s" % bandname)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_spectrum(time_value, lookup_table, wavelengths, mjd0=None):
        """``time_value`` is interpreted as MJD when the lookup table index is MJD."""
        t_mjd, flux = _flux_at_observing_mjd(time_value, lookup_table, mjd0=mjd0)
        closest_time = float(np.asarray(lookup_table.index, dtype=float)[
            np.argmin(np.abs(np.asarray(lookup_table.index, dtype=float) - (
                time_value if lookup_index_is_mjd(lookup_table) else (time_value - (mjd0 or 0.0))
            )))
        ])

        plt.figure(figsize=(8, 5))
        plt.plot(wavelengths, flux, label="Time index ≈ %.5f (row MJD ≈ %.5f)" % (closest_time, t_mjd))
        plt.xlabel("Wavelength (Å)")
        plt.ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
        plt.title("Spectrum at closest grid epoch")
        plt.legend()
        plt.grid()
        plt.show()


    def plot_lightcurve(wavelength, lookup_table):
        closest_wavelength = lookup_table.columns[np.argmin(np.abs(lookup_table.columns - wavelength))]
        flux = lookup_table[closest_wavelength]

        plt.figure(figsize=(8, 5))
        plt.scatter(lookup_table.index, flux, label="Wavelength = %.2f Å" % closest_wavelength)
        plt.xlabel("MJD" if lookup_index_is_mjd(lookup_table) else "Time (days from start)")
        plt.ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
        plt.title("Lightcurve at Wavelength %.2f Å" % closest_wavelength)
        plt.legend()
        plt.grid()
        plt.show()


    def plot_sed_evolution(
        lookup_table,
        common_wavelengths,
        spec_mjds=None,
        vmin=None,
        vmax=None
    ):
        flux_values = lookup_table.values
        spec_mjds = np.asarray(lookup_table.index, dtype=float) if spec_mjds is None else np.asarray(spec_mjds, dtype=float)

        if vmin is None:
            pos = flux_values[flux_values > 0]
            vmin = float(np.nanmin(pos)) if pos.size else 1e-30
        if vmax is None:
            vmax = float(np.nanmax(flux_values))

        is_mjd = lookup_index_is_mjd(lookup_table)
        x_lab = "MJD" if is_mjd else "Days from first detection"

        plt.figure(figsize=(10, 6))
        plt.imshow(
            flux_values.T,
            aspect='auto',
            extent=[spec_mjds.min(), spec_mjds.max(),
                    common_wavelengths.min(), common_wavelengths.max()],
            origin='lower',
            cmap='viridis',
            norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        plt.colorbar(label=r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
        plt.xlabel(x_lab)
        plt.ylabel("Wavelength (Å)")
        plt.title("SED Evolution")
        plt.show()

    def compare_sed_and_original_spectrum(
        mjd_query,
        lookup_table,
        wavelengths,
        mjd0,
        list_file,
        original_spec_dir=None,
        ax=None,
        mode="original",
        time_window=0.,
        z=0.00984,
        sed_scale=1.0,
    ):
        """
        Plot SED spectrum and original or smoothed input spectrum at the closest time
        to mjd_query. If time_window > 0, plot all spectra within ±time_window of mjd_query.
        """
        import matplotlib.pyplot as plt
        import os

        sed_mjd, sed_flux = _flux_at_observing_mjd(mjd_query, lookup_table, mjd0=mjd0)

        if mode == "smoothed":
            list_file = list_file.replace("1_spec_lists_original", "2_spec_lists_smoothed")
            if original_spec_dir is None:
                original_spec_dir = COCO_PATH + "Inputs/Spectroscopy/2_spec_smoothed"
        else:
            if original_spec_dir is None:
                original_spec_dir = COCO_PATH + "Inputs/Spectroscopy/1_spec_original"

        orig_paths, orig_mjds = [], []
        with open(list_file) as f:
            for line in f:
                if line.strip() == "" or line.startswith("#"):
                    continue
                parts = line.split()
                if mode == "smoothed":
                    if len(parts) < 3:
                        continue
                    try:
                        orig_mjds.append(float(parts[0]))
                        orig_paths.append(parts[2])
                    except Exception:
                        continue
                else:
                    if len(parts) < 3:
                        continue
                    try:
                        orig_paths.append(parts[0])
                        orig_mjds.append(float(parts[-1]))
                    except Exception:
                        continue
        if not orig_paths or not orig_mjds:
            print("No valid spectra found in %s" % list_file)
            return
        orig_mjds = np.array(orig_mjds)

        if time_window > 0:
            idxs = np.where(np.abs(orig_mjds - mjd_query) <= time_window)[0]
            if len(idxs) == 0:
                idxs = [np.argmin(np.abs(orig_mjds - mjd_query))]
        else:
            idxs = [np.argmin(np.abs(orig_mjds - mjd_query))]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            wavelengths, sed_scale * sed_flux,
            label="SED (t=%.6f MJD)" % sed_mjd,
            color="black",
            alpha=0.75
        )

        label = "Smoothed" if mode == "smoothed" else "Original"

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for j, idx in enumerate(idxs):
            orig_path = orig_paths[idx]
            orig_mjd = orig_mjds[idx]

            if orig_path.startswith("/data/1_spec_original/") or orig_path.startswith("/data/2_spec_smoothed/"):
                if mode == "smoothed":
                    local_base = COCO_PATH + "Inputs/Spectroscopy/2_spec_smoothed"
                    orig_path = os.path.join(local_base, orig_path.replace("/data/2_spec_smoothed/", ""))
                else:
                    local_base = COCO_PATH + "Inputs/Spectroscopy/1_spec_original"
                    orig_path = os.path.join(local_base, orig_path.replace("/data/1_spec_original/", ""))
            elif not orig_path.startswith("/"):
                orig_path = os.path.join(original_spec_dir, orig_path)
            orig_path = os.path.expanduser(orig_path)

            try:
                orig_data = np.loadtxt(orig_path)
                orig_wl = orig_data[:, 0]
                orig_flux = orig_data[:, 1]
            except Exception as e:
                print("Could not load spectrum: %s\n%s" % (orig_path, e))
                continue

            if z > 0.0:
                orig_wl = orig_wl / (1 + z)
                orig_flux = orig_flux * (1 + z)

            ax.plot(
                orig_wl, orig_flux,
                label="%s (t=%.6f MJD)" % (label, orig_mjd),
                color=color_cycle[j % len(color_cycle)],
                alpha=0.8
            )

        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
        ax.set_title("SED vs %s Spectrum(s) near %.2f MJD" % (label, mjd_query))
        ax.legend()
        ax.grid()
        plt.show()
    """
).strip()
