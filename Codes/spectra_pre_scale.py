"""Pre-scale spectroscopic arms / same-time epochs before mangling (notebook 4.5).

Default mode ``scale_only`` applies flux multipliers and keeps separate files.
Optional ``merge_join`` concatenates group members after scaling.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

OutputMode = Literal["scale_only", "merge_join"]
MergeGapPolicy = Literal["linear_bridge", "nan_gap"]

_ARM_ORDER_DEFAULT = ("uvb", "vis", "nir", "blue", "red")


@dataclass
class SpectrumEntry:
    mjd: float
    phase: float
    path: str
    basename: str

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)


@dataclass
class ScaleGroup:
    id: str
    members: list[str]
    output_mode: OutputMode = "scale_only"
    merge_order: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ScaleReport:
    snname: str
    default_output_mode: OutputMode
    groups: list[dict[str, Any]] = field(default_factory=list)
    ungrouped: list[str] = field(default_factory=list)


def load_spec_list(list_path: str) -> list[SpectrumEntry]:
    rows = np.genfromtxt(list_path, dtype=None, encoding="utf-8")
    if rows.size == 0:
        return []
    if rows.ndim == 0:
        rows = np.array([rows])
    out: list[SpectrumEntry] = []
    for row in rows:
        out.append(
            SpectrumEntry(
                mjd=float(row["f0"]),
                phase=float(row["f1"]),
                path=str(row["f2"]).strip(),
                basename=os.path.basename(str(row["f2"]).strip()),
            )
        )
    return out


def write_spec_list(list_path: str, entries: list[SpectrumEntry]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(list_path)), exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write("%.6f\t%.6f\t%s\n" % (e.mjd, e.phase, e.path))


def load_scale_groups_json(path: str) -> tuple[OutputMode, list[ScaleGroup]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    default_mode: OutputMode = data.get("default_output_mode", "scale_only")
    groups: list[ScaleGroup] = []
    for g in data.get("groups", []):
        members = [str(m) for m in g.get("members", [])]
        if not members:
            continue
        groups.append(
            ScaleGroup(
                id=str(g.get("id", "group_%i" % len(groups))),
                members=members,
                output_mode=g.get("output_mode", default_mode),
                merge_order=[str(x).lower() for x in g.get("merge_order", [])],
                reason=str(g.get("reason", "")),
            )
        )
    return default_mode, groups


def write_scale_groups_template(path: str, groups: list[ScaleGroup], default_mode: OutputMode = "scale_only") -> None:
    payload = {
        "default_output_mode": default_mode,
        "groups": [
            {
                "id": g.id,
                "members": g.members,
                "output_mode": g.output_mode,
                "merge_order": g.merge_order,
                "reason": g.reason,
            }
            for g in groups
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _basename_key(name: str) -> str:
    return os.path.basename(name)


def _arm_sort_key(filename: str, merge_order: list[str]) -> tuple[int, float]:
    low = filename.lower()
    for i, tag in enumerate(merge_order or list(_ARM_ORDER_DEFAULT)):
        if tag in low:
            return (i, 0.0)
    return (len(_ARM_ORDER_DEFAULT), 0.0)


def load_spectrum_array(path: str) -> np.ndarray:
    raw = np.genfromtxt(
        path, dtype=None, encoding="utf-8", names=["wls", "flux", "fluxerr"]
    )
    mask = (
        np.isfinite(raw["wls"])
        & np.isfinite(raw["flux"])
        & np.isfinite(raw["fluxerr"])
        & (raw["flux"] > 0.0)
    )
    return raw[mask]


def save_spectrum_array(path: str, spec: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#wls\tflux\tfluxerr\n")
        for row in spec:
            fh.write("%E\t%E\t%E\n" % (row["wls"], row["flux"], row["fluxerr"]))


def overlap_scale_factor_wls(
    ref: np.ndarray,
    arm: np.ndarray,
    *,
    wl_tol_a: float = 1.0,
) -> tuple[float, int]:
    """WLS linear scale ``m`` so ``m * arm ≈ ref`` on overlapping valid pixels."""
    w_ref = ref["wls"]
    w_arm = arm["wls"]
    lo = max(float(np.min(w_ref)), float(np.min(w_arm)))
    hi = min(float(np.max(w_ref)), float(np.max(w_arm)))
    if hi <= lo:
        return 1.0, 0

    grid = np.linspace(lo, hi, max(50, min(len(ref), len(arm))))
    f_ref = np.interp(grid, w_ref, ref["flux"])
    e_ref = np.interp(grid, w_ref, ref["fluxerr"])
    f_arm = np.interp(grid, w_arm, arm["flux"])
    e_arm = np.interp(grid, w_arm, arm["fluxerr"])

    good = (
        np.isfinite(f_ref)
        & np.isfinite(f_arm)
        & (f_ref > 0)
        & (f_arm > 0)
        & np.isfinite(e_ref)
        & np.isfinite(e_arm)
    )
    if not np.any(good):
        return 1.0, 0

    w = 1.0 / np.maximum(e_arm[good] ** 2, 1e-60)
    num = float(np.sum(w * f_ref[good] * f_arm[good]))
    den = float(np.sum(w * f_arm[good] ** 2))
    if den <= 0.0:
        return 1.0, int(good.sum())
    return num / den, int(good.sum())


def apply_flux_scale(spec: np.ndarray, factor: float) -> np.ndarray:
    out = spec.copy()
    out["flux"] = out["flux"] * factor
    out["fluxerr"] = out["fluxerr"] * abs(factor)
    return out


def median_snr(spec: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = spec["flux"] / np.maximum(spec["fluxerr"], 1e-99)
    snr = snr[np.isfinite(snr) & (snr > 0)]
    if snr.size == 0:
        return 0.0
    return float(np.median(snr))


def scale_group_members(
    member_paths: list[str],
    *,
    merge_order: Optional[list[str]] = None,
    wl_tol_a: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, float], str]:
    """Return scaled spectra dict, per-file cumulative factors, reference basename."""
    loaded = {os.path.basename(p): load_spectrum_array(p) for p in member_paths}
    if not loaded:
        return {}, {}, ""

    order = sorted(
        loaded.keys(),
        key=lambda n: _arm_sort_key(n, merge_order or list(_ARM_ORDER_DEFAULT)),
    )
    ref_name = max(order, key=lambda n: median_snr(loaded[n]))
    factors: dict[str, float] = {ref_name: 1.0}
    scaled: dict[str, np.ndarray] = {ref_name: loaded[ref_name].copy()}

    for name in order:
        if name == ref_name:
            continue
        m, _n = overlap_scale_factor_wls(scaled[ref_name], loaded[name], wl_tol_a=wl_tol_a)
        factors[name] = m
        scaled[name] = apply_flux_scale(loaded[name], m)

    return scaled, factors, ref_name


def merge_spectra_concat(
    scaled: dict[str, np.ndarray],
    *,
    merge_order: Optional[list[str]] = None,
    gap_policy: MergeGapPolicy = "linear_bridge",
    gap_log10: float = 0.005,
) -> np.ndarray:
    names = sorted(
        scaled.keys(),
        key=lambda n: _arm_sort_key(n, merge_order or list(_ARM_ORDER_DEFAULT)),
    )
    parts = [scaled[n] for n in names]
    merged = np.concatenate(parts)
    order = np.argsort(merged["wls"])
    merged = merged[order]

    # dedupe near-equal wavelength (keep first)
    if len(merged) > 1:
        dw = np.diff(merged["wls"])
        keep = np.ones(len(merged), dtype=bool)
        keep[1:] = dw > 1e-3
        merged = merged[keep]

    if gap_policy == "nan_gap" or len(merged) < 2:
        return merged

    # optional small-gap linear bridge (in log10 lambda space)
    w = merged["wls"].astype(float)
    logw = np.log10(w)
    for i in range(len(merged) - 1):
        dlog = logw[i + 1] - logw[i]
        if 0.0 < dlog <= gap_log10:
            n_insert = max(2, int(dlog / gap_log10 * 5))
            mid_log = np.linspace(logw[i], logw[i + 1], n_insert + 2)[1:-1]
            mid_w = 10 ** mid_log
            t = (mid_log - logw[i]) / dlog
            f = (1 - t) * merged["flux"][i] + t * merged["flux"][i + 1]
            e = (1 - t) * merged["fluxerr"][i] + t * merged["fluxerr"][i + 1]
            # append — caller may re-sort; for simplicity rebuild once
            extra = np.array(list(zip(mid_w, f, e)), dtype=merged.dtype)
            merged = np.concatenate([merged, extra])
    order = np.argsort(merged["wls"])
    return merged[order]


def suggest_scale_groups(
    entries: list[SpectrumEntry],
    *,
    same_time_minutes: float = 5.0,
) -> list[ScaleGroup]:
    """Cluster list rows by MJD proximity (assist only)."""
    if not entries:
        return []
    dt_days = same_time_minutes / (24.0 * 60.0)
    used = set()
    groups: list[ScaleGroup] = []
    sorted_e = sorted(entries, key=lambda e: e.mjd)
    gid = 0
    for i, e in enumerate(sorted_e):
        if i in used:
            continue
        cluster = [e]
        used.add(i)
        for j in range(i + 1, len(sorted_e)):
            if j in used:
                continue
            if abs(sorted_e[j].mjd - e.mjd) <= dt_days:
                cluster.append(sorted_e[j])
                used.add(j)
        if len(cluster) > 1:
            gid += 1
            groups.append(
                ScaleGroup(
                    id="suggested_%03i_mjd_%.4f" % (gid, e.mjd),
                    members=[c.basename for c in cluster],
                    output_mode="scale_only",
                    merge_order=_infer_merge_order([c.basename for c in cluster]),
                    reason="auto: |ΔMJD| <= %.2f min" % same_time_minutes,
                )
            )
    return groups


def _infer_merge_order(filenames: list[str]) -> list[str]:
    tags = []
    for tag in _ARM_ORDER_DEFAULT:
        if any(tag in f.lower() for f in filenames):
            tags.append(tag)
    return tags


def resolve_member_path(basename: str, entries: list[SpectrumEntry]) -> str:
    for e in entries:
        if e.basename == basename or basename in e.path:
            return e.path
    raise FileNotFoundError("List has no spectrum matching %r" % basename)


def run_prescale_pipeline(
    *,
    snname: str,
    coco_path: str,
    output_dir: str,
    groups_json: Optional[str] = None,
    default_output_mode: OutputMode = "scale_only",
    same_time_minutes: float = 5.0,
    wl_tol_a: float = 1.0,
    gap_log10: float = 0.005,
    merge_gap_policy: MergeGapPolicy = "linear_bridge",
    write_diagnostics: bool = True,
    diagnostics_dir: Optional[str] = None,
) -> ScaleReport:
    import pipeline_config as pconf

    list_path = pconf.smoothed_spec_list_path(coco_path, snname)
    entries = load_spec_list(list_path)
    if not entries:
        raise FileNotFoundError("No spectra in %s" % list_path)

    groups_path = groups_json or pconf.spec_scale_groups_json_path(output_dir, snname)
    if os.path.isfile(groups_path):
        default_output_mode, groups = load_scale_groups_json(groups_path)
    else:
        groups = suggest_scale_groups(entries, same_time_minutes=same_time_minutes)
        write_scale_groups_template(groups_path, groups, default_output_mode)

    out_spec_dir = pconf.prescaled_spec_dir(coco_path, snname)
    os.makedirs(out_spec_dir, exist_ok=True)

    grouped_basenames: set[str] = set()
    for g in groups:
        grouped_basenames.update(_basename_key(m) for m in g.members)

    report = ScaleReport(snname=snname, default_output_mode=default_output_mode)
    new_entries: list[SpectrumEntry] = []

    if write_diagnostics and diagnostics_dir:
        from spec_scale_diagnostics import diagnostics_available, save_group_diagnostics

        if not diagnostics_available():
            import warnings

            warnings.warn(
                "matplotlib unavailable; spec scale figures skipped (scaling outputs still written)",
                stacklevel=1,
            )

    for g in groups:
        mode = g.output_mode or default_output_mode
        paths = [resolve_member_path(_basename_key(m), entries) for m in g.members]
        scaled, factors, ref_name = scale_group_members(
            paths, merge_order=g.merge_order or None, wl_tol_a=wl_tol_a
        )
        group_rec: dict[str, Any] = {
            "id": g.id,
            "output_mode": mode,
            "reference": ref_name,
            "scale_factors": factors,
            "members": [],
        }
        merged_arr: Optional[np.ndarray] = None

        if mode == "merge_join":
            merged_arr = merge_spectra_concat(
                scaled,
                merge_order=g.merge_order or None,
                gap_policy=merge_gap_policy,
                gap_log10=gap_log10,
            )
            rep_entry = next(e for e in entries if e.basename in scaled)
            out_name = "%.6f_merged_%s.dat" % (rep_entry.mjd, re.sub(r"[^\w]+", "_", g.id))
            out_path = os.path.join(out_spec_dir, out_name)
            save_spectrum_array(out_path, merged_arr)
            new_entries.append(
                SpectrumEntry(mjd=rep_entry.mjd, phase=rep_entry.phase, path=out_path, basename=out_name)
            )
            group_rec["merged_file"] = out_name
            for bn, spec in scaled.items():
                group_rec["members"].append({"file": bn, "factor": factors.get(bn, 1.0)})
        else:
            for bn, spec in scaled.items():
                src_entry = next(e for e in entries if e.basename == bn)
                out_path = os.path.join(out_spec_dir, bn)
                save_spectrum_array(out_path, spec)
                new_entries.append(
                    SpectrumEntry(
                        mjd=src_entry.mjd,
                        phase=src_entry.phase,
                        path=out_path,
                        basename=bn,
                    )
                )
                group_rec["members"].append({"file": bn, "factor": factors.get(bn, 1.0), "output": bn})

        report.groups.append(group_rec)

        if write_diagnostics and diagnostics_dir and diagnostics_available():
            before = {
                bn: load_spectrum_array(resolve_member_path(bn, entries)) for bn in scaled
            }
            save_group_diagnostics(
                diagnostics_dir,
                g.id,
                before=before,
                after=scaled,
                factors=factors,
                merged=merged_arr,
            )

    # ungrouped: copy as-is
    for e in entries:
        if e.basename in grouped_basenames:
            continue
        out_path = os.path.join(out_spec_dir, e.basename)
        if os.path.abspath(e.path) != os.path.abspath(out_path):
            shutil.copy2(e.path, out_path)
        new_entries.append(
            SpectrumEntry(mjd=e.mjd, phase=e.phase, path=out_path, basename=e.basename)
        )
        report.ungrouped.append(e.basename)

    list_out = pconf.prescaled_spec_list_path(coco_path, snname)
    write_spec_list(list_out, new_entries)

    report_path = pconf.spec_scale_report_json_path(output_dir, snname)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "snname": report.snname,
                "default_output_mode": report.default_output_mode,
                "groups": report.groups,
                "ungrouped": report.ungrouped,
                "prescaled_list": list_out,
            },
            fh,
            indent=2,
        )

    if write_diagnostics and diagnostics_dir:
        from spec_scale_diagnostics import write_diagnostics_index

        write_diagnostics_index(diagnostics_dir, report.groups)

    return report
