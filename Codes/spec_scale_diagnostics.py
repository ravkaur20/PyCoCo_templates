"""Saved diagnostic figures for notebook 4.5 pre-scaling (Agg backend, no inline flood)."""

from __future__ import annotations

import html
import os
from typing import Any, Mapping, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover — ImportError, NumPy ABI mismatch, etc.
    _HAS_MPL = False
    plt = None  # type: ignore


def diagnostics_available() -> bool:
    return _HAS_MPL


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_group_diagnostics(
    diagnostics_dir: str,
    group_id: str,
    *,
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    factors: Mapping[str, float],
    merged: Optional[np.ndarray] = None,
) -> None:
    if not _HAS_MPL:
        return
    _ensure_dir(diagnostics_dir)
    safe = group_id.replace("/", "_").replace(" ", "_")

    # before/after overlay
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(before), 1)))
    for i, (name, spec) in enumerate(sorted(before.items())):
        c = colors[i % len(colors)]
        ax.plot(spec["wls"], spec["flux"], color=c, alpha=0.35, lw=0.8, label="%s raw" % name)
        if name in after:
            ax.plot(
                after[name]["wls"],
                after[name]["flux"],
                color=c,
                lw=1.0,
                label="%s scaled (×%.3f)" % (name, factors.get(name, 1.0)),
            )
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.set_title("Pre-scale group %s: before vs after" % group_id)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(diagnostics_dir, "group_%s_before_after.pdf" % safe), bbox_inches="tight")
    plt.close(fig)

    # scale factor bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(factors.keys())
    vals = [factors[n] for n in names]
    ax.bar(range(len(names)), vals, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Linear scale factor")
    ax.set_title("Group %s scale factors" % group_id)
    fig.tight_layout()
    fig.savefig(os.path.join(diagnostics_dir, "group_%s_scale_factors.png" % safe), dpi=120)
    plt.close(fig)

    # overlap zoom between reference and each other arm
    if len(before) >= 2:
        ref_name = max(before.keys(), key=lambda n: _median_snr(before[n]))
        ref = before[ref_name]
        fig, axes = plt.subplots(
            max(1, len(before) - 1),
            1,
            figsize=(10, 3 * max(1, len(before) - 1)),
            squeeze=False,
        )
        ax_i = 0
        for name, spec in sorted(before.items()):
            if name == ref_name:
                continue
            ax = axes[ax_i, 0]
            ax_i += 1
            lo = max(float(np.min(ref["wls"])), float(np.min(spec["wls"])))
            hi = min(float(np.max(ref["wls"])), float(np.max(spec["wls"])))
            if hi > lo:
                m = (ref["wls"] >= lo) & (ref["wls"] <= hi)
                n = (spec["wls"] >= lo) & (spec["wls"] <= hi)
                ax.plot(ref["wls"][m], ref["flux"][m], label=ref_name, lw=1)
                ax.plot(spec["wls"][n], spec["flux"][n], label=name, lw=1, alpha=0.7)
                if name in after:
                    a = after[name]
                    am = (a["wls"] >= lo) & (a["wls"] <= hi)
                    ax.plot(a["wls"][am], a["flux"][am], "--", label="%s scaled" % name, lw=1)
            ax.set_xlim(lo, hi) if hi > lo else None
            ax.set_title("Overlap %s vs %s (×%.3f)" % (ref_name, name, factors.get(name, 1.0)))
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(diagnostics_dir, "group_%s_overlap_zoom.pdf" % safe), bbox_inches="tight")
        plt.close(fig)

    if merged is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, spec in before.items():
            ax.plot(spec["wls"], spec["flux"], alpha=0.25, lw=0.6, label=name)
        ax.plot(merged["wls"], merged["flux"], "k-", lw=1.2, label="merged")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.set_title("Group %s merged vs arms" % group_id)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(diagnostics_dir, "group_%s_merged_vs_arms.pdf" % safe), bbox_inches="tight")
        plt.close(fig)


def _median_snr(spec: np.ndarray) -> float:
    snr = spec["flux"] / np.maximum(spec["fluxerr"], 1e-99)
    snr = snr[np.isfinite(snr) & (snr > 0)]
    return float(np.median(snr)) if snr.size else 0.0


def write_diagnostics_index(diagnostics_dir: str, groups: list[dict[str, Any]]) -> None:
    _ensure_dir(diagnostics_dir)
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Spec pre-scale diagnostics</title></head><body>",
        "<h1>Pre-scale diagnostics</h1><ul>",
    ]
    for g in groups:
        gid = html.escape(str(g.get("id", "")))
        safe = gid.replace("/", "_")
        lines.append("<li><b>%s</b> (mode=%s)<ul>" % (gid, html.escape(str(g.get("output_mode", "")))))
        for stem, ext in [
            ("before_after", "pdf"),
            ("scale_factors", "png"),
            ("overlap_zoom", "pdf"),
            ("merged_vs_arms", "pdf"),
        ]:
            fn = "group_%s_%s.%s" % (safe, stem, ext)
            fp = os.path.join(diagnostics_dir, fn)
            if os.path.isfile(fp):
                lines.append("<li><a href='%s'>%s</a></li>" % (html.escape(fn), html.escape(fn)))
        lines.append("</ul></li>")
    lines.append("</ul></body></html>")
    with open(os.path.join(diagnostics_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
