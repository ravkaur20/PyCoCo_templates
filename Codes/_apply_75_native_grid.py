"""Apply native-grid comparison patches to 7.5_comparison_check_log.ipynb."""
from __future__ import annotations

import json
import re

path = "/Users/ravkaur/Desktop/research/kilonova-SED/PyCoCo_templates/Codes/7.5_comparison_check_log.ipynb"


def inject_compare_sed_native_kw(src: str) -> str:
    """Insert use_native_final_grid=True before closing paren of compare_sed_and_original_spectrum(...)."""
    if "compare_sed_and_original_spectrum(" not in src:
        return src
    if "def compare_sed_and_original_spectrum" in src:
        return src
    if "use_native_final_grid=True" in src:
        return src
    needle = "compare_sed_and_original_spectrum("
    out = []
    i = 0
    while i < len(src):
        j = src.find(needle, i)
        if j < 0:
            out.append(src[i:])
            break
        out.append(src[i:j])
        k = j + len(needle)
        depth = 1
        start = j
        while k < len(src) and depth > 0:
            if src[k] == "(":
                depth += 1
            elif src[k] == ")":
                depth -= 1
            k += 1
        block = src[start:k]
        if "use_native_final_grid" not in block:
            block = block[:-1].rstrip() + ",\n    use_native_final_grid=True,\n)"
        out.append(block)
        i = k
    return "".join(out)


CELL6_NEW = '''def compare_sed_spectrum_test(
    mjd_query,
    lookup_table,
    wavelengths,
    mjd0,
    list_file=None,
    original_spec_dir=None,
    ax=None,
    mode="original",
    time_window=0.0,
    z=0.00984,
    sed_scale=1.0,
    snname=None,
    flux_on_disk=None,
    use_native_final_grid=False,
    final_data_dir=None,
):
    """
    Same as compare_sed_and_original_spectrum but kept for older notebooks; uses shared helpers.

    mode: "original", "smoothed", or "mangled"
    """
    _fod = FINAL_FLUX_ON_DISK if flux_on_disk is None else flux_on_disk
    _compare_sed_vs_input_spectra(
        mjd_query,
        lookup_table,
        wavelengths,
        mjd0,
        list_file=list_file,
        original_spec_dir=original_spec_dir,
        ax=ax,
        mode=mode,
        time_window=time_window,
        z=z,
        sed_scale=sed_scale,
        snname=snname,
        flux_on_disk=_fod,
        use_native_final_grid=use_native_final_grid,
        final_data_dir=final_data_dir,
    )
'''


def main() -> None:
    with open(path) as f:
        nb = json.load(f)

    c6 = next(
        c
        for c in nb["cells"]
        if c.get("cell_type") == "code"
        and "def compare_sed_spectrum_test(" in "".join(c.get("source", []))
    )
    s6 = "".join(c6["source"])
    if "use_native_final_grid" not in s6:
        m = re.search(r"def compare_sed_spectrum_test\([\s\S]*?\n\)", s6)
        if not m:
            raise RuntimeError("Could not find compare_sed_spectrum_test")
        s6 = s6[: m.start()] + CELL6_NEW.rstrip() + s6[m.end() :]
    c6["source"] = [s6]

    # Cell 42: compare_sed_and_spectra
    c42 = next(
        c
        for c in nb["cells"]
        if c.get("cell_type") == "code"
        and "def compare_sed_and_spectra(" in "".join(c.get("source", []))
    )
    s42 = "".join(c42["source"])
    head_old = """def compare_sed_and_spectra(
    mjd_query,
    lookup_table,
    wavelengths,
    mjd0,
    list_file=None,
    original_spec_dir=None,
    ax=None,
    mode="original",
    time_window=0.,
    z=0.00984
):
"""
    head_new = """def compare_sed_and_spectra(
    mjd_query,
    lookup_table,
    wavelengths,
    mjd0,
    list_file=None,
    original_spec_dir=None,
    ax=None,
    mode="original",
    time_window=0.,
    z=0.00984,
    use_native_final_grid=False,
    final_data_dir=None,
    sed_scale=1.0,
    snname=None,
    flux_on_disk=None,
):
"""
    if head_old in s42:
        s42 = s42.replace(head_old, head_new, 1)

    ins = """    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # --- SED spectrum ---
"""
    deleg = """    import matplotlib.pyplot as plt
    import os
    import numpy as np

    if use_native_final_grid:
        _fod = FINAL_FLUX_ON_DISK if flux_on_disk is None else flux_on_disk
        _sn = snname if snname is not None else SNNAME
        return compare_sed_and_original_spectrum(
            mjd_query,
            lookup_table,
            wavelengths,
            mjd0,
            list_file=list_file,
            original_spec_dir=original_spec_dir,
            ax=ax,
            mode=mode,
            time_window=time_window,
            z=z,
            sed_scale=sed_scale,
            snname=_sn,
            flux_on_disk=_fod,
            use_native_final_grid=True,
            final_data_dir=final_data_dir,
        )

    # --- SED spectrum ---
"""
    if ins in s42 and "if use_native_final_grid:" not in s42:
        s42 = s42.replace(ins, deleg, 1)

    c42["source"] = [s42]

    # Cell 44: plot_spectral_series
    c44 = next(
        c
        for c in nb["cells"]
        if c.get("cell_type") == "code"
        and "def plot_spectral_series(" in "".join(c.get("source", []))
    )
    s44 = "".join(c44["source"])
    sig_old = """def plot_spectral_series(
    lookup_table,
    wavelengths,
    mjd0,
    time_interval=1.0,
    offset=True,
    overplot_mode="none",
    list_file=None,
    original_spec_dir=None,
    colors=None,
    z=0.00984,
    time_window=0.5,
    spec_dir=None,
    first_spec_mjd=None,
    max_plots=None,
    min_sep=0.05
):
"""
    sig_new = """def plot_spectral_series(
    lookup_table,
    wavelengths,
    mjd0,
    time_interval=1.0,
    offset=True,
    overplot_mode="none",
    list_file=None,
    original_spec_dir=None,
    colors=None,
    z=0.00984,
    time_window=0.5,
    spec_dir=None,
    first_spec_mjd=None,
    max_plots=None,
    min_sep=0.05,
    use_native_final_grid=False,
    final_data_dir=None,
    snname=None,
):
"""
    if sig_old in s44 and "use_native_final_grid" not in sig_old:
        s44 = s44.replace(sig_old, sig_new, 1)

    s44 = s44.replace(
        """    # absolute MJDs of SED spectra
    sed_abs_mjds = lookup_table.index.to_numpy() + mjd_ref

    # build desired MJD grid starting at mjd0
""",
        """    # absolute MJDs of SED spectra
    if hasattr(lookup_table, "attrs") and "spec_mjd" in lookup_table.attrs:
        sed_abs_mjds = np.asarray(lookup_table.attrs["spec_mjd"], dtype=float)
    else:
        sed_abs_mjds = lookup_table.index.to_numpy(dtype=float) + mjd_ref

    # build desired MJD grid starting at mjd0
""",
        1,
    )

    old_off = """    # compute per-spectrum offsets if requested
    if offset:
        # build normalized spectra (safe against zeros/NaNs)
        norm_spectra = []
        for i in chosen_idxs:
            f = lookup_table.iloc[i].values.astype(float)
            mx = np.nanmax(f)
            if not np.isfinite(mx) or mx == 0:
                norm_spectra.append(np.zeros_like(f))
            else:
                norm_spectra.append(f / mx)

        # iterative offsets: bottom spectrum = 0; each next one is moved up so that
        # min( (cur + off_cur) - (prev + off_prev) ) >= min_sep
        offsets = np.zeros(len(norm_spectra), dtype=float)
        for k in range(1, len(norm_spectra)):
            prev = norm_spectra[k - 1]
            cur = norm_spectra[k]
            diff = cur - prev
            min_diff = np.nanmin(diff)
            # required increment to guarantee separation at all wavelengths
            inc = max(min_sep - min_diff, 0.0)
            offsets[k] = offsets[k - 1] + inc
    else:
        offsets = None
        norm_spectra = None
"""

    new_off = """    # compute per-spectrum offsets if requested
    raw_wl_list = None
    raw_fl_list = None
    if use_native_final_grid:
        fdir = final_data_dir or spec_dir or globals().get("data_dir")
        _sn = snname if snname is not None else globals().get("SNNAME", None)
        if not fdir or _sn is None:
            raise ValueError(
                "use_native_final_grid=True needs final_data_dir/spec_dir/data_dir and SNNAME in globals or snname="
            )
        raw_wl_list, raw_fl_list = [], []
        for ii in chosen_idxs:
            sm = float(sed_abs_mjds[ii])
            _, wl_n, fl_n, _ = nearest_final_spectrum_native(
                fdir,
                sm,
                globals()["COCO_PATH"],
                _sn,
                flux_on_disk=globals().get("FINAL_FLUX_ON_DISK", "auto"),
                datalc_path=globals().get("DATALC_PATH"),
                final_suffixes=globals().get("FINAL_SUFFIXES_TO_LOAD"),
            )
            raw_wl_list.append(wl_n)
            raw_fl_list.append(fl_n.astype(float))

    if offset:
        norm_spectra = []
        if use_native_final_grid:
            for fl in raw_fl_list:
                mx = np.nanmax(fl)
                if not np.isfinite(mx) or mx == 0:
                    norm_spectra.append(np.zeros_like(fl))
                else:
                    norm_spectra.append(fl / mx)
            offsets = np.arange(len(norm_spectra), dtype=float) * (1.0 + min_sep)
        else:
            # build normalized spectra (safe against zeros/NaNs)
            for i in chosen_idxs:
                f = lookup_table.iloc[i].values.astype(float)
                mx = np.nanmax(f)
                if not np.isfinite(mx) or mx == 0:
                    norm_spectra.append(np.zeros_like(f))
                else:
                    norm_spectra.append(f / mx)

            # iterative offsets: bottom spectrum = 0; each next one is moved up so that
            # min( (cur + off_cur) - (prev + off_prev) ) >= min_sep
            offsets = np.zeros(len(norm_spectra), dtype=float)
            for k in range(1, len(norm_spectra)):
                prev = norm_spectra[k - 1]
                cur = norm_spectra[k]
                diff = cur - prev
                min_diff = np.nanmin(diff)
                inc = max(min_sep - min_diff, 0.0)
                offsets[k] = offsets[k - 1] + inc
    else:
        offsets = None
        norm_spectra = None
"""

    if old_off in s44 and "use_native_final_grid" not in old_off:
        s44 = s44.replace(old_off, new_off, 1)

    old_loop = """    for j, idx in enumerate(chosen_idxs):
        sed_flux = lookup_table.iloc[idx].values
        sed_mjd = sed_abs_mjds[idx]            # actual MJD of this spectrum
        phase_days = sed_mjd - mjd0
        color = color_cycle[j % len(color_cycle)]
        if offset:
            y_offset = offsets[j]
            plot_flux = norm_spectra[j] + y_offset
        else:
            y_offset = 0.0
            plot_flux = sed_flux

        # plot SED spectrum (no legend entry)
        ax.plot(wavelengths, plot_flux, color=color, linewidth=1.5)

        # label with actual MJD (using plotted value)
        ax.text(
            wavelengths[-1] + (wavelengths[-1] - wavelengths[0]) * 0.005,
            plot_flux[-1],
            f"{sed_mjd:.6f}",  # actual MJD with reasonable precision
            color=color,
            fontsize=9,
            verticalalignment="center"
        )
"""

    new_loop = """    for j, idx in enumerate(chosen_idxs):
        sed_mjd = float(sed_abs_mjds[idx])
        phase_days = sed_mjd - mjd0
        color = color_cycle[j % len(color_cycle)]
        if use_native_final_grid:
            wl_n = raw_wl_list[j]
            fl_n = raw_fl_list[j]
            if offset:
                y_offset = offsets[j]
                plot_flux = norm_spectra[j] + y_offset
            else:
                y_offset = 0.0
                plot_flux = fl_n
            ax.plot(wl_n, plot_flux, color=color, linewidth=1.5)
            ax.text(
                wl_n[-1] + max((wl_n[-1] - wl_n[0]) * 0.005, 1.0),
                plot_flux[-1],
                f"{sed_mjd:.6f}",
                color=color,
                fontsize=9,
                verticalalignment="center"
            )
        else:
            sed_flux = lookup_table.iloc[idx].values
            if offset:
                y_offset = offsets[j]
                plot_flux = norm_spectra[j] + y_offset
            else:
                y_offset = 0.0
                plot_flux = sed_flux

            # plot SED spectrum (no legend entry)
            ax.plot(wavelengths, plot_flux, color=color, linewidth=1.5)

            ax.text(
                wavelengths[-1] + (wavelengths[-1] - wavelengths[0]) * 0.005,
                plot_flux[-1],
                f"{sed_mjd:.6f}",
                color=color,
                fontsize=9,
                verticalalignment="center"
            )
"""

    if old_loop in s44:
        s44 = s44.replace(old_loop, new_loop, 1)

    c44["source"] = [s44]

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        t = "".join(cell["source"])
        if "compare_sed_and_original_spectrum(" not in t:
            continue
        t2 = inject_compare_sed_native_kw(t)
        cell["source"] = [t2]

    # Notebook note: redshift / native FINAL grid
    md_text = (
        "**Spectrum comparison (`compare_sed_and_original_spectrum`, …):** with `use_native_final_grid=True`, "
        "the black SED curve is the nearest FINAL spectrum **read from disk** on its native λ grid. "
        "With `z>0`, **only input spectra** are shifted to rest frame (λ/(1+z), F×(1+z)); the FINAL curve "
        "stays observer-frame as stored. Use `z=0` or apply the same transform to both sides if you need "
        "them in the same frame."
    )
    insert_after_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        t = "".join(cell["source"])
        if "def compare_sed_and_original_spectrum(" in t:
            insert_after_idx = i
            break
    if insert_after_idx is not None:
        next_i = insert_after_idx + 1
        already = False
        if next_i < len(nb["cells"]):
            cnext = nb["cells"][next_i]
            if cnext.get("cell_type") == "markdown":
                already = md_text in "".join(cnext.get("source", []))
        if not already:
            nb["cells"].insert(
                next_i,
                {
                    "cell_type": "markdown",
                    "id": "native-grid-sed-note",
                    "metadata": {},
                    "source": [md_text + "\n"],
                },
            )

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("apply_75_native_grid: ok")


if __name__ == "__main__":
    main()
