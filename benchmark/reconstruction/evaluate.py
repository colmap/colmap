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

import pickle

from evaluation.blended_mvs import DatasetBlendedMVS
from evaluation.eth3d import DatasetETH3D
from evaluation.imc import DatasetIMC2023, DatasetIMC2024
from evaluation.utils import create_result_table, parse_args, process_scenes

import pycolmap


def main() -> None:
    args = parse_args()

    datasets = {
        "eth3d": DatasetETH3D,
        "blended-mvs": DatasetBlendedMVS,
        "imc2023": DatasetIMC2023,
        "imc2024": DatasetIMC2024,
    }

    metrics = {}
    for dataset_name in args.datasets:
        if dataset_name not in datasets:
            pycolmap.logging.error(f"Unknown dataset: {dataset_name}")
            return

        pycolmap.logging.info(f"Evaluating dataset: {dataset_name}")

        dataset = datasets[dataset_name](
            data_path=args.data_path,
            categories=args.categories,
            scenes=args.scenes,
            run_path=args.run_path,
            run_name=args.run_name,
        )

        scene_infos = dataset.list_scenes()

        if not scene_infos:
            pycolmap.logging.warning("No scenes found")
            return

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


if __name__ == "__main__":
    main()
