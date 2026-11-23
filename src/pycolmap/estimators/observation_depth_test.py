# SPDX-License-Identifier: BSD-3-Clause

import gc
import weakref

import numpy as np
import pytest

import pycolmap


@pytest.fixture
def depth_problem():
    reconstruction = pycolmap.synthesize_dataset(
        pycolmap.SyntheticDatasetOptions(num_points3D=5)
    )
    image_id = next(iter(reconstruction.images))
    config = pycolmap.BundleAdjustmentConfig()
    config.add_image(image_id)
    owner = pycolmap.create_default_ceres_bundle_adjuster(
        pycolmap.BundleAdjustmentOptions(
            refine_rig_from_world=False,
            refine_sensor_from_rig=False,
            refine_focal_length=False,
            refine_extra_params=False,
        ),
        config,
        reconstruction,
    )
    image = reconstruction.images[image_id]
    for idx in image.get_observation_point2D_idxs():
        point = reconstruction.points3D[image.points2D[idx].point3D_id]
        owner.problem.set_parameter_block_constant(point.xyz)
    return owner, reconstruction, image_id, pycolmap.append_ba_depth_constraints


def test_caller_owned_scale(depth_problem):
    owner, reconstruction, image_id, append = depth_problem
    image = reconstruction.images[image_id]
    indices = np.asarray(image.get_observation_point2D_idxs())
    priors = {}
    for idx in indices:
        point = reconstruction.points3D[image.points2D[idx].point3D_id]
        priors[idx] = (image.cam_from_world() * point.xyz)[2] / 2, 0.05
    log_depth_scale = np.array([-1.0])
    options = pycolmap.BADepthConstraintOptions()
    # Both batches share the scale and retain their losses.
    for group, loss_type in zip(
        np.array_split(indices, 2),
        [pycolmap.LossFunctionType.CAUCHY, pycolmap.LossFunctionType.HUBER],
        strict=True,
    ):
        loss = pycolmap.create_ceres_loss_function(loss_type, 1)
        assert append(
            owner,
            reconstruction,
            image_id,
            {int(idx): priors[idx] for idx in group},
            log_depth_scale,
            options,
            loss,
        ) == len(group)
    ref = weakref.ref(log_depth_scale)
    del log_depth_scale, loss
    gc.collect()
    assert ref() is not None
    summary = owner.solve()
    assert summary.is_solution_usable()
    np.testing.assert_allclose(ref()[0], np.log(2), rtol=1e-5)


def test_scale_array_is_not_silently_copied(depth_problem):
    owner, reconstruction, image_id, append = depth_problem
    readonly = np.ones(1)
    readonly.flags.writeable = False
    invalid_scales = [
        None,
        [1.0],
        np.ones(1, dtype=np.float32),
        readonly,
        np.ones(2),
    ]
    for invalid in invalid_scales:
        with pytest.raises((TypeError, ValueError)):
            append(
                owner,
                reconstruction,
                image_id,
                {},
                invalid,
            )
