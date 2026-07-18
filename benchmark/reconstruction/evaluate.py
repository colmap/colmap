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

"""Run the reconstruction benchmark and report pose accuracy metrics.

A plain run evaluates one colmap binary once and writes a single report.
--seeds/--num_seeds instead runs it once per random seed, writing one
<report_name>_s<seed>.pkl report per seed, to measure run-to-run spread. The
scene workspaces are shared across those runs, so features and raw matches are
computed once and reused.

Reports are compared with compare.py.

Reproducibility caveat: RANSAC seeds per thread as random_seed +
omp_get_thread_num(), so only single-threaded scenes are deterministic at a
fixed seed. Pass --threads_per_scene 1 for reproducible runs, and use
--num_parallel_scenes for throughput instead -- parallelism across scenes does
not touch any one scene's random number stream.

Note that a seed reseeds both geometric verification and the mapper, but only
--overwrite_reconstruction is forced between the runs; pass
--overwrite_two_view_geometries as well to also redo verification.

Example:
  # One binary across 5 seeds (run-to-run variance):
  python evaluate.py --colmap_path /path/colmap --num_seeds 5 \\
    --threads_per_scene 1 --overwrite_two_view_geometries \\
    --data_path data --datasets eth3d --categories dslr --scenes meadow \\
    --run_path runs --run_name variance-meadow
"""

import argparse
import pickle

from evaluation.blended_mvs import DatasetBlendedMVS
from evaluation.eth3d import DatasetETH3D
from evaluation.imc import DatasetIMC2023, DatasetIMC2024
from evaluation.utils import (
    Dataset,
    MetricsByDatasetByCatByScene,
    create_result_table,
    filter_smallest_scenes_per_category,
    parse_args,
    process_scenes,
)

import pycolmap


def run_once(args: argparse.Namespace) -> MetricsByDatasetByCatByScene | None:
    """Evaluates all datasets once and writes args.report_name.

    Returns None if a dataset is unknown or no scenes matched.
    """
    datasets: dict[str, type[Dataset]] = {
        "eth3d": DatasetETH3D,
        "blended-mvs": DatasetBlendedMVS,
        "imc2023": DatasetIMC2023,
        "imc2024": DatasetIMC2024,
    }

    metrics: MetricsByDatasetByCatByScene = {}
    for dataset_name in args.datasets:
        if dataset_name not in datasets:
            pycolmap.logging.error(f"Unknown dataset: {dataset_name}")
            return None

        pycolmap.logging.info(f"Evaluating dataset: {dataset_name}")

        dataset = datasets[dataset_name](
            data_path=args.data_path,
            categories=args.categories,
            scenes=args.scenes,
            run_path=args.run_path,
            run_name=args.run_name,
        )

        scene_infos = dataset.list_scenes()

        if args.fast:
            scene_infos = filter_smallest_scenes_per_category(
                scene_infos, args.fast_num_scenes
            )

        if not scene_infos:
            pycolmap.logging.warning("No scenes found")
            return None

        metrics[dataset_name] = process_scenes(
            args=args,
            scene_infos=scene_infos,
            prepare_scene=dataset.prepare_scene,
            position_accuracy_gt=dataset.position_accuracy_gt,
        )

    pycolmap.logging.info("Results:\n" + create_result_table(metrics))

    report_path = args.run_path / args.run_name / (args.report_name + ".pkl")
    pycolmap.logging.info(f"Saving report to: {report_path}")
    with open(report_path, "wb") as report_file:
        pickle.dump(metrics, report_file)

    return metrics


def run_seeds(args: argparse.Namespace) -> None:
    """Evaluates once per seed, writing <report_name>_s<seed>.pkl each time.

    A failing seed is reported and skipped, so the remaining ones still run.
    """
    # The seeds share one workspace per scene, so without this every run after
    # the first would silently re-evaluate the first one's reconstruction.
    args.overwrite_reconstruction = True
    args.overwrite_alignment = True

    report_name = args.report_name
    pycolmap.logging.info(
        f"{report_name}: {len(args.seeds)} seeds {args.seeds} "
        f"-> {args.run_path / args.run_name}"
    )

    failures = []
    for seed in args.seeds:
        pycolmap.logging.info(f"---- seed={seed} ----")
        args.random_seed = seed
        args.report_name = f"{report_name}_s{seed}"
        try:
            reason = "" if run_once(args) is not None else ": no scenes"
        except Exception as error:
            reason = f": {error!r}"
        if reason:
            failures.append(seed)
            pycolmap.logging.warning(
                f"seed={seed} FAILED{reason} (its report will be missing and "
                "a comparison skips it)"
            )

    if failures:
        pycolmap.logging.warning(
            f"{len(failures)} of {len(args.seeds)} run(s) failed: {failures}"
        )
        raise SystemExit(1)


def main() -> None:
    args = parse_args(__doc__)
    if args.seeds is not None:
        run_seeds(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
