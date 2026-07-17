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

"""Run the reconstruction benchmark across several random seeds, for one or
two variants (colmap binaries), then optionally compare with
compare_multi_seeded.py.

Each (seed, variant) is one evaluate.py run writing <label>_s<seed>.pkl into
the run directory. Runs are single-threaded per scene by default
(--threads_per_scene 1) so a fixed seed is deterministic -- RANSAC seeds per
thread as random_seed + omp_get_thread_num(), so only single-threaded scenes
reproduce. Use --num_parallel_scenes (forwarded to evaluate.py) for throughput
instead: cross-scene parallelism does not touch any one scene's RNG stream.

All evaluate.py flags are accepted and forwarded verbatim (e.g. --data_path,
--datasets, --categories, --scenes, --mapper, --num_parallel_scenes,
--gpu_index, --overwrite_matches). This script only injects --colmap_path,
--random_seed, --report_name and --threads_per_scene, so those must NOT be
passed here.

Examples:
  # One variant across 5 seeds (pure run-to-run variance):
  python run_multi_seeded.py \
    --colmap_path_a /path/colmap --label_a base --num_seeds 5 \
    --data_path data --datasets eth3d --categories dslr --scenes "meadow" \
    --run_path runs --run_name variance-meadow --overwrite_matches

  # Two variants (A/B) on shared seeds, then compare:
  python run_multi_seeded.py \
    --colmap_path_a /path/base/colmap  --label_a base \
    --colmap_path_b /path/msac/colmap  --label_b msac \
    --num_seeds 5 --compare \
    --data_path data --datasets eth3d --categories dslr \
    --scenes "door pipes" --num_parallel_scenes 2 \
    --run_path runs --run_name base-vs-msac --overwrite_matches
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pycolmap

HERE = Path(__file__).resolve().parent

# Flags this script owns/injects; passing them through would double-inject.
_RESERVED = {
    "--colmap_path",
    "--random_seed",
    "--report_name",
    "--threads_per_scene",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--colmap_path_a", required=True, help="Variant A colmap binary."
    )
    parser.add_argument(
        "--label_a", default="a", help="Variant A label / stem."
    )
    parser.add_argument(
        "--colmap_path_b",
        default=None,
        help="Variant B colmap binary (optional; omit for a single variant).",
    )
    parser.add_argument(
        "--label_b", default="b", help="Variant B label / stem."
    )
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seeds", nargs="+", help="Explicit seed list, e.g. --seeds 0 1 2 7 9."
    )
    seed_group.add_argument(
        "--num_seeds", type=int, help="Shorthand for seeds 0 1 ... N-1."
    )
    parser.add_argument(
        "--threads_per_scene",
        type=int,
        default=1,
        help="Threads within each scene (default 1 => deterministic at a fixed "
        "seed). Increase only if you accept nondeterminism.",
    )
    parser.add_argument(
        "--run_path",
        type=Path,
        required=True,
        help="Forwarded to evaluate.py; reports go under run_path/run_name.",
    )
    parser.add_argument(
        "--run_name", required=True, help="Forwarded to evaluate.py."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run compare_multi_seeded.py at the end (needs --colmap_path_b).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for the evaluate.py subprocesses.",
    )
    args, passthrough = parser.parse_known_args()

    reserved_hit = [a for a in passthrough if a.split("=")[0] in _RESERVED]
    if reserved_hit:
        parser.error(
            "these flags are injected per run and must not be passed: "
            + ", ".join(sorted(set(reserved_hit)))
        )
    if args.compare and args.colmap_path_b is None:
        parser.error("--compare requires --colmap_path_b (needs two variants)")
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()

    seeds = (
        args.seeds
        if args.seeds is not None
        else [str(i) for i in range(args.num_seeds)]
    )
    variants = [(args.label_a, args.colmap_path_a)]
    if args.colmap_path_b is not None:
        variants.append((args.label_b, args.colmap_path_b))

    if args.threads_per_scene != 1:
        pycolmap.logging.warning(
            f"--threads_per_scene={args.threads_per_scene} != 1: a fixed seed "
            "is NOT deterministic (RANSAC seeds per thread). Use 1 for "
            "reproducible / paired runs; --num_parallel_scenes for throughput."
        )

    run_dir = args.run_path / args.run_name
    labels = " vs ".join(label for label, _ in variants)
    n_runs = len(seeds) * len(variants)
    pycolmap.logging.info(
        f"run_multi_seeded: {labels}; {len(seeds)} seeds {seeds} x "
        f"{len(variants)} variants = {n_runs} runs -> {run_dir}"
    )

    forwarded_run = [
        "--run_path",
        str(args.run_path),
        "--run_name",
        args.run_name,
    ]
    failures = 0
    for seed in seeds:
        for label, colmap_path in variants:
            pycolmap.logging.info(f"---- seed={seed} variant={label} ----")
            cmd = [
                args.python,
                str(HERE / "evaluate.py"),
                *passthrough,
                *forwarded_run,
                "--colmap_path",
                colmap_path,
                "--threads_per_scene",
                str(args.threads_per_scene),
                "--random_seed",
                str(seed),
                "--report_name",
                f"{label}_s{seed}",
            ]
            if subprocess.call(cmd, cwd=HERE) != 0:
                failures += 1
                pycolmap.logging.warning(
                    f"seed={seed} variant={label} FAILED (report will be "
                    "missing; compare_multi_seeded skips it)"
                )

    if failures:
        pycolmap.logging.warning(f"{failures} run(s) failed.")

    if args.compare:
        pycolmap.logging.info("Running compare_multi_seeded.py ...")
        subprocess.call(
            [
                args.python,
                str(HERE / "compare_multi_seeded.py"),
                str(run_dir),
                "--labels",
                args.label_a,
                args.label_b,
                "--seeds",
                *seeds,
            ],
            cwd=HERE,
        )


if __name__ == "__main__":
    main()
