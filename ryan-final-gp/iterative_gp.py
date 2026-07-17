"""Orchestrate GP fit + plots + outlier pass over multiple iterations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag-base", default="iter_gp", help="run tag prefix; iteration k -> {base}_k")
    p.add_argument("--max-iter", type=int, default=3)
    p.add_argument("--input", default=os.path.join(HERE, "gp_minimal_bundle.npz"))
    p.add_argument("--output-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--bundle", default=None, help="bundle for outlier_pipeline y alignment (default: --input)")
    p.add_argument("--z-threshold", type=float, default=5.0)
    p.add_argument(
        "--gp-extra",
        default="",
        help="extra args passed to run_gp.py as a single string (quoted)",
    )
    p.add_argument("--skip-outliers", action="store_true", help="only run_gp + plot_results each iter")
    args = p.parse_args(argv)

    bundle = args.bundle or args.input
    exe = sys.executable
    for k in range(args.max_iter):
        tag = f"{args.tag_base}_{k}"
        run_dir = os.path.join(args.output_dir, tag)
        gp_cmd = [exe, os.path.join(HERE, "run_gp.py"), "--tag", tag, "--input", args.input, "--output-dir", args.output_dir]
        if k > 0 and not args.skip_outliers:
            mask_path = os.path.join(args.output_dir, f"{args.tag_base}_{k - 1}", "train_include.npz")
            if os.path.isfile(mask_path):
                gp_cmd.extend(["--train-include", mask_path])
        if args.gp_extra.strip():
            import shlex

            gp_cmd.extend(shlex.split(args.gp_extra))

        print(f"[iterative_gp] === iteration {k} tag={tag} ===")
        print("[iterative_gp] running:", " ".join(gp_cmd))
        subprocess.check_call(gp_cmd, cwd=HERE)

        plot_cmd = [exe, os.path.join(HERE, "plot_results.py"), "--tag", tag, "--output-dir", args.output_dir, "--bundle", bundle]
        print("[iterative_gp] running:", " ".join(plot_cmd))
        subprocess.check_call(plot_cmd, cwd=HERE)

        if args.skip_outliers:
            continue

        from outlier_pipeline import build_include_mask, flag_outliers, save_outlier_report, save_train_mask

        pred = os.path.join(run_dir, "predictions.npz")
        summ = flag_outliers(pred, bundle_npz=bundle, z_threshold=args.z_threshold, iteration=k)
        save_outlier_report(run_dir, summ)
        mask_path = os.path.join(run_dir, "train_include.npz")
        save_train_mask(mask_path, build_include_mask(summ.n_total, summ))
        print(f"[iterative_gp] outliers: {summ.n_flagged} flagged -> mask {mask_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
