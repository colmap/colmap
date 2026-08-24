# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Compare the reports of two evaluate.py runs (A vs B).

With one report per variant, this prints the result tables for A, B and A - B.

With several reports per variant -- one per random seed, produced by an
evaluate.py --seeds/--num_seeds run -- it instead prints mean +/- std over the
seeds. The std is the run-to-run spread; use enough seeds that it is small
relative to the A-B effect you care about. Seeds are shared between the
variants, so A - B is computed per seed before averaging (a paired difference).
Reports should be generated with --threads_per_scene 1 so that a fixed seed is
deterministic.

Pass --inference to compute simultaneous confidence intervals within each
dataset/category for macro scene-average A-B differences with a paired,
crossed scene x seed bootstrap. This treats the observed scenes as a sample
from the dataset's scene population. An optional noise-ceiling artifact adds
an equivalence check against a per-scene synthetic or calibrated ceiling. Such
artifacts are produced by synthetic_pose_noise.py for any supported dataset.

To produce such reports for two colmap binaries, run evaluate.py once per
binary with the same seeds and --run_name -- so that they share the scene
workspaces -- and a different --report_name, which becomes the report path
prefix passed here. Pass --overwrite_two_view_geometries to both, otherwise
they reuse the two-view geometries cached in the shared database and a change
to geometric verification has no effect on A - B.

Each variant is given either as a single report, or as a report path prefix,
in which case its <prefix>_s<seed>.pkl reports are discovered and matched by
seed against the other variant's. A prefix resolves to <prefix>.pkl when the
runs were made without seeds. Prefixed variants may live in different run
directories.

Examples:
  # Two individual reports:
  python compare.py --report_a_path runs/X/base.pkl \\
    --report_b_path runs/X/msac.pkl --labels base msac

  # Every seed shared by two multi-seed runs:
  python compare.py --report_a_path_prefix runs/X/base \\
    --report_b_path_prefix runs/Y/msac

  # ... restricted to a subset of the seeds:
  python compare.py --report_a_path_prefix runs/X/base \\
    --report_b_path_prefix runs/Y/msac --seeds 0 1 2 7 9

  # Paired inference plus a ceiling check for candidate B:
  python compare.py --report_a_path_prefix runs/X/base \\
    --report_b_path_prefix runs/Y/candidate --inference \\
    --noise_ceiling runs/ceilings/dataset.npz --ceiling_variant b

The same prefix invocation also covers unseeded runs, where it resolves to
runs/X/base.pkl and runs/Y/msac.pkl.
"""

import argparse
from pathlib import Path

from evaluation.utils import collect_reports, compare_reports, pair_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report_a_path",
        type=Path,
        default=None,
        metavar="PKL",
        help="Variant A report.",
    )
    parser.add_argument(
        "--report_b_path",
        type=Path,
        default=None,
        metavar="PKL",
        help="Variant B report, compared against --report_a_path.",
    )
    parser.add_argument(
        "--report_a_path_prefix",
        type=Path,
        default=None,
        metavar="PREFIX",
        help="Variant A given as a report path without the _s<seed>.pkl "
        "suffix, e.g. runs/X/base. Its seeds are discovered from the "
        "filenames and matched against variant B's, which may live in a "
        "different run directory.",
    )
    parser.add_argument(
        "--report_b_path_prefix",
        type=Path,
        default=None,
        metavar="PREFIX",
        help="Variant B report path prefix, paired by seed with "
        "--report_a_path_prefix.",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=None,
        metavar=("A", "B"),
        help="Variant labels used in the table titles. Defaults to the "
        "prefix names, or to A and B for explicit report paths.",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Compare only these seeds of the prefixed reports, e.g. --seeds "
        "0 1 2 7 9. Defaults to every seed the two variants share.",
    )
    seed_group.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Shorthand for --seeds 0 1 ... N-1.",
    )
    parser.add_argument(
        "--inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute paired, simultaneous bootstrap intervals over scenes "
        "and seeds.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=10_000,
        help="Number of bootstrap samples for statistical inference.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Simultaneous confidence level across metric thresholds.",
    )
    parser.add_argument(
        "--minimum_effect",
        type=float,
        default=0.5,
        help="Smallest meaningful A-B effect in score percentage points.",
    )
    parser.add_argument(
        "--inference_seed",
        type=int,
        default=0,
        help="Random seed for reproducible bootstrap resampling.",
    )
    parser.add_argument(
        "--noise_ceiling",
        type=Path,
        default=None,
        metavar="NPZ",
        help="Optional per-scene noise-ceiling artifact generated by "
        "synthetic_pose_noise.py.",
    )
    parser.add_argument(
        "--ceiling_variant",
        choices=["a", "b"],
        default="b",
        help="Variant to compare with --noise_ceiling.",
    )
    parser.add_argument(
        "--ceiling_margin",
        type=float,
        default=0.5,
        help="Maximum ceiling-minus-run gap considered ceiling-limited, in "
        "score percentage points.",
    )
    args = parser.parse_args()

    # Each variant is given as a report or as a prefix, independently: a
    # single report may be compared against every seed of a prefixed one.
    for side in ("a", "b"):
        path = getattr(args, f"report_{side}_path")
        prefix = getattr(args, f"report_{side}_path_prefix")
        if (path is None) == (prefix is None):
            parser.error(
                f"provide exactly one of --report_{side}_path and "
                f"--report_{side}_path_prefix"
            )
    args.use_prefixes = (
        args.report_a_path_prefix is not None
        or args.report_b_path_prefix is not None
    )
    if args.num_seeds is not None:
        if args.num_seeds <= 0:
            parser.error("--num_seeds must be > 0")
        args.seeds = list(range(args.num_seeds))
    if args.seeds is not None and not args.use_prefixes:
        parser.error(
            "--seeds/--num_seeds select among prefixed reports; they do not "
            "apply to two individual reports"
        )
    if args.bootstrap_samples < 2:
        parser.error("--bootstrap_samples must be at least 2")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    if args.minimum_effect < 0:
        parser.error("--minimum_effect must be non-negative")
    if args.ceiling_margin < 0:
        parser.error("--ceiling_margin must be non-negative")
    if args.noise_ceiling is not None and not args.noise_ceiling.exists():
        parser.error(f"no such noise ceiling: {args.noise_ceiling}")
    return args


def main() -> None:
    args = parse_args()

    labels = args.labels or [
        (path or prefix).name.removesuffix(".pkl")
        for path, prefix in (
            (args.report_a_path, args.report_a_path_prefix),
            (args.report_b_path, args.report_b_path_prefix),
        )
    ]
    reports_a = collect_reports(args.report_a_path, args.report_a_path_prefix)
    reports_b = collect_reports(args.report_b_path, args.report_b_path_prefix)
    report_a_paths, report_b_paths, seeds = pair_reports(
        reports_a, reports_b, labels, args.seeds
    )

    compare_reports(
        report_a_paths,
        report_b_paths,
        labels,
        seeds,
        run_inference=args.inference,
        num_bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        minimum_effect=args.minimum_effect,
        inference_seed=args.inference_seed,
        ceiling_path=args.noise_ceiling,
        ceiling_variant=args.ceiling_variant,
        ceiling_margin=args.ceiling_margin,
    )


if __name__ == "__main__":
    main()
