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

"""Multi-seed comparison of two variants (A vs B), like compare.py but over
several seeds.

Given N reports for variant A and N reports for variant B produced on the same
N seeds (report i of A and report i of B share seed i), this prints three tables
-- A, B, and the per-seed difference A - B -- each as mean +/- std over the
seeds. The std is the run-to-run spread; use enough seeds that it is small
relative to the A-B effect you care about. Seeds are shared so A - B is computed
per seed before averaging (a paired difference); reports should be generated
with --threads_per_scene 1 so a fixed seed is deterministic.

Usage (common: point at a run dir; paths are built from <label>_s<seed>.pkl):
  python compare_multi_seeded.py runs/X --labels base msac --num_seeds 5
  python compare_multi_seeded.py runs/X --labels base msac --seeds 0 1 2 7 9

Advanced (arbitrary paths, paired by position):
  python compare_multi_seeded.py \
    --report_a_paths runs/X/base_s0.pkl runs/X/base_s1.pkl \
    --report_b_paths runs/X/msac_s0.pkl runs/X/msac_s1.pkl --labels base msac
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from evaluation.utils import get_scores

import pycolmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory holding reports named <label>_s<seed>.pkl. Combined "
        "with --labels and --seeds/--num-seeds, the paired report paths are "
        "built for you (the usual way to call this).",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=["base", "msac"],
        metavar=("A", "B"),
        help="Variant labels; also the filename stems in run_dir mode.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Seed list, e.g. --seeds 0 1 2 7 9. In run_dir mode these pick "
        "the files; otherwise they are display labels for the given paths.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Shorthand for --seeds 0 1 ... N-1 (run_dir mode).",
    )
    parser.add_argument(
        "--report_a_paths",
        type=Path,
        nargs="+",
        default=None,
        help="Advanced: explicit A report paths (skips run_dir path building).",
    )
    parser.add_argument(
        "--report_b_paths",
        type=Path,
        nargs="+",
        default=None,
        help="Advanced: explicit B report paths, paired by position with A.",
    )
    return parser.parse_args()


def resolve_report_paths(
    args: argparse.Namespace,
) -> tuple[list[Path], list[Path], list[str]]:
    """Return (a_paths, b_paths, seed_labels). Either explicit --report_*_paths
    or run_dir + labels + seeds/--num-seeds. In run_dir mode, seeds whose A or B
    report is missing (e.g. a failed run leg) are skipped with a warning."""
    if args.report_a_paths is not None or args.report_b_paths is not None:
        if args.report_a_paths is None or args.report_b_paths is None:
            raise SystemExit(
                "provide BOTH --report_a_paths and --report_b_paths"
            )
        return args.report_a_paths, args.report_b_paths, args.seeds

    if args.run_dir is None:
        raise SystemExit(
            "give a run_dir (+ --labels + --seeds/--num-seeds), or explicit "
            "--report_a_paths/--report_b_paths"
        )
    seeds = args.seeds
    if args.num_seeds is not None:
        seeds = [str(i) for i in range(args.num_seeds)]
    if not seeds:
        raise SystemExit("run_dir mode needs --seeds or --num-seeds")

    a_paths: list[Path] = []
    b_paths: list[Path] = []
    kept: list[str] = []
    for seed in seeds:
        pa = args.run_dir / f"{args.labels[0]}_s{seed}.pkl"
        pb = args.run_dir / f"{args.labels[1]}_s{seed}.pkl"
        missing = [str(p) for p in (pa, pb) if not p.exists()]
        if missing:
            pycolmap.logging.warning(
                f"seed {seed}: skipping (missing {', '.join(missing)})"
            )
            continue
        a_paths.append(pa)
        b_paths.append(pb)
        kept.append(str(seed))
    if not a_paths:
        raise SystemExit(
            f"no seed has both {args.labels[0]}_s*.pkl and "
            f"{args.labels[1]}_s*.pkl in {args.run_dir}"
        )
    return a_paths, b_paths, kept


def load_reports(paths: list[Path]) -> list[dict]:
    reports = []
    for path in paths:
        with open(path, "rb") as report_file:
            reports.append(pickle.load(report_file))
    return reports


def common_scene_keys(reports: list[dict]) -> list[tuple[str, str, str]]:
    """(dataset, category, scene) keys present in every report, in the order
    they appear in the first report."""
    key_sets = []
    for report in reports:
        keys = set()
        for dataset, cat_metrics in report.items():
            for category, scene_metrics in cat_metrics.items():
                for scene in scene_metrics:
                    keys.add((dataset, category, scene))
        key_sets.append(keys)
    shared = set.intersection(*key_sets) if key_sets else set()
    ordered = []
    first = reports[0]
    for dataset, cat_metrics in first.items():
        for category, scene_metrics in cat_metrics.items():
            for scene in scene_metrics:
                if (dataset, category, scene) in shared:
                    ordered.append((dataset, category, scene))
    return ordered


def is_summary(scene: str) -> bool:
    return scene.startswith("__") and scene.endswith("__")


def stack_scores(
    reports: list[dict], key: tuple[str, str, str], error_type: str
) -> np.ndarray:
    """Return array of shape (n_reports, n_thresholds) of scores for one key."""
    dataset, category, scene = key
    return np.array(
        [
            get_scores(error_type, reports[i][dataset][category][scene])
            for i in range(len(reports))
        ]
    )


def render_meanstd(
    title: str,
    per_key_stack: dict[tuple[str, str, str], np.ndarray],
    keys: list[tuple[str, str, str]],
    error_type: str,
    thresholds: np.ndarray,
    n: int,
    signed: bool = False,
) -> str:
    """Render one table with mean ± std (over seeds) per scene x threshold.
    per_key_stack maps each key to a (n_seeds, n_thresholds) score array."""
    is_relative = error_type.startswith("relative")
    thr_disp = thresholds if is_relative else 100 * thresholds
    size_scene = max(8, max(len(s) for _, _, s in keys))
    cell = 12  # "+dd.dd±dd.dd"
    fmt = "{:+6.2f}±{:5.2f}" if signed else "{:6.2f}±{:5.2f}"
    header = f"{'scene':<{size_scene}} " + " ".join(
        f"{'@' + str(t).rstrip('.'):^{cell}}" for t in thr_disp
    )
    lines = [title, header, "-" * len(header)]
    ordered = sorted(keys, key=lambda k: (is_summary(k[2]), k))
    prev_summary = False
    for key in ordered:
        _, _, scene = key
        if is_summary(scene) and not prev_summary:
            lines.append("-" * len(header))
            prev_summary = True
        arr = per_key_stack[key]  # (n_seeds, n_thresholds)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
        label = {"__avg__": "average", "__all__": "overall"}.get(scene, scene)
        cells = " ".join(fmt.format(m, s) for m, s in zip(mean, std))
        lines.append(f"{label:<{size_scene}} {cells}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    a_paths, b_paths, seeds = resolve_report_paths(args)
    if len(a_paths) != len(b_paths):
        raise SystemExit(
            "A and B must have the same number of reports (paired by seed): "
            f"{len(a_paths)} vs {len(b_paths)}"
        )
    n = len(a_paths)
    if n < 2:
        pycolmap.logging.warning(
            "Only one seed provided; std will be zero. Provide multiple seeds "
            "to measure run-to-run spread."
        )
    if seeds is not None and len(seeds) != n:
        raise SystemExit(f"seed count ({len(seeds)}) != report count ({n})")

    reports_a = load_reports(a_paths)
    reports_b = load_reports(b_paths)
    keys = common_scene_keys(reports_a + reports_b)
    if not keys:
        raise SystemExit("No scenes shared across all reports.")

    ref = reports_a[0]
    first_scene = next(
        iter(next(iter(next(iter(ref.values())).values())).values())
    )
    error_type = first_scene.error_type
    thresholds = np.asarray(first_scene.error_thresholds)
    score = "AUC" if error_type.endswith("auc") else "Recall"

    label_a, label_b = args.labels
    seeds_label = " ".join(map(str, seeds)) if seeds is not None else f"{n}"
    n_scenes = len([k for k in keys if not is_summary(k[2])])
    pycolmap.logging.info(
        f"{label_a} vs {label_b}: {n} seeds x 2 variants = {2 * n} "
        f"reconstruction runs over {n_scenes} scenes; seeds: [{seeds_label}]"
    )

    # Per (scene, threshold) score arrays of shape (n_seeds, n_thresholds).
    stacks_a = {k: stack_scores(reports_a, k, error_type) for k in keys}
    stacks_b = {k: stack_scores(reports_b, k, error_type) for k in keys}
    stacks_d = {k: stacks_a[k] - stacks_b[k] for k in keys}

    common = (keys, error_type, thresholds, n)
    pycolmap.logging.info(
        "\n"
        + render_meanstd(
            f"A = {label_a}  ({score} mean ± std over {n} seeds)",
            stacks_a,
            *common,
        )
    )
    pycolmap.logging.info(
        "\n"
        + render_meanstd(
            f"B = {label_b}  ({score} mean ± std over {n} seeds)",
            stacks_b,
            *common,
        )
    )
    pycolmap.logging.info(
        "\n"
        + render_meanstd(
            f"A - B = {label_a} - {label_b}  ({score} mean ± std over {n} "
            "seeds)",
            stacks_d,
            *common,
            signed=True,
        )
    )


if __name__ == "__main__":
    main()
