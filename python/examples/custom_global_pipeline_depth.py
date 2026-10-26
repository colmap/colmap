# SPDX-License-Identifier: BSD-3-Clause

"""Add measured depth to the ETH3D-SLAM RGB-camera reconstruction.

The depth NPZ maps image IDs to Nx2 arrays of depth/stddev in metres,
with rows in keypoint order and NaN for missing depths.
"""

import argparse
import copy
from pathlib import Path

import custom_global_pipeline as pipeline
import numpy as np

import pycolmap

depth_priors = {}
loss = pycolmap.create_ceres_loss_function(
    pycolmap.LossFunctionType.HUBER, 1.345
)
gp_builder = pipeline.build_global_positioner
ba_builder = pipeline.build_bundle_adjuster
original_normalize = pycolmap.Reconstruction.normalize


def append_depths(owner, reconstruction, depths, loss=None):
    if isinstance(owner, pycolmap.GlobalPositioner):
        append = pycolmap.append_gp_depth_constraints
        options = pycolmap.GPDepthConstraintOptions()
    else:
        append = pycolmap.append_ba_depth_constraints
        options = pycolmap.BADepthConstraintOptions(
            residual_type=pycolmap.DepthResidualType.LINEAR
        )
    log_depth_scale = np.zeros(1)
    count = sum(
        append(
            owner,
            reconstruction,
            image_id,
            values,
            log_depth_scale,
            options=options,
            loss_function=loss,
        )
        for image_id, values in depths.items()
        if image_id in reconstruction.images
    )
    assert count > 0, "No usable depth constraints"
    owner.problem.set_parameter_block_constant(log_depth_scale)
    if isinstance(owner, pycolmap.GlobalPositioner):
        owner.extend_parameter_block_ordering([(log_depth_scale, 2)])


def read_depth_archive(path):
    with np.load(path) as archive:
        depths = {}
        for key in archive.files:
            values = archive[key]
            valid = ~np.isnan(values[:, 0])
            depths[int(key)] = dict(
                zip(
                    np.flatnonzero(valid).tolist(),
                    map(tuple, values[valid].tolist()),
                    strict=True,
                )
            )
    return depths


def build_global_positioner(gp_options, pose_graph, reconstruction):
    depth_options = copy.deepcopy(gp_options)
    depth_options.fix_first_scale = False
    depth_options.use_parameter_block_ordering = True
    positioner = gp_builder(depth_options, pose_graph, reconstruction)
    append_depths(positioner, reconstruction, depth_priors, loss)
    return positioner


def build_bundle_adjuster(ba_options, config, reconstruction):
    if not ba_options.refine_rig_from_world:
        return ba_builder(ba_options, config, reconstruction)
    depth_config = copy.deepcopy(config)
    depth_config.fix_gauge(pycolmap.BundleAdjustmentGauge.UNSPECIFIED)
    depth_config.set_constant_rig_from_world_pose(
        reconstruction.images[min(config.images)].frame_id
    )
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        ba_options, depth_config, reconstruction
    )
    append_depths(adjuster, reconstruction, depth_priors, loss)
    return adjuster


def normalize(reconstruction, *args, **kwargs):
    kwargs["fixed_scale"] = True
    return original_normalize(reconstruction, *args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database_path", type=Path, required=True)
    parser.add_argument("--image_path", type=Path)
    parser.add_argument("--depth_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    options = pycolmap.GlobalPipelineOptions()
    options.mapper.global_positioning.experimental_observation_stddev = 1.0
    depth_priors = read_depth_archive(args.depth_path)
    pipeline.build_global_positioner = build_global_positioner
    pipeline.build_bundle_adjuster = build_bundle_adjuster
    pycolmap.Reconstruction.normalize = normalize
    pipeline.main(
        args.database_path, args.image_path, args.output_path, options
    )
