# SPDX-License-Identifier: BSD-3-Clause

import custom_global_pipeline as pipeline
import custom_global_pipeline_depth as depth
import numpy as np

import pycolmap


def test_metric_depth_pipeline(tmp_path, monkeypatch):
    database_path = tmp_path / "database.db"
    depths = {}
    pycolmap.set_random_seed(0)
    with pycolmap.Database.open(database_path) as database:
        truth = pycolmap.synthesize_dataset(
            pycolmap.SyntheticDatasetOptions(
                num_rigs=1,
                num_cameras_per_rig=1,
                num_frames_per_rig=5,
                num_points3D=40,
                two_view_geometry_has_relative_pose=True,
            ),
            database,
        )
        for image in truth.images.values():
            priors = {}
            for i, point in enumerate(image.points2D):
                if point.has_point3D():
                    priors[i] = (
                        (
                            image.cam_from_world()
                            * truth.points3D[point.point3D_id].xyz
                        )[2],
                        0.01,
                    )
            priors.pop(0, None)
            depths[image.image_id] = priors

    options = pycolmap.GlobalPipelineOptions(num_threads=1)
    options.mapper.refine_sensor_from_rig = False
    options.mapper.global_positioning.use_gpu = False
    options.mapper.global_positioning.experimental_observation_stddev = 1.0
    options.mapper.global_positioning.random_seed = 0
    options.mapper.rotation_averaging.random_seed = 0
    options.mapper.bundle_adjustment.ceres.use_gpu = False
    options.mapper.bundle_adjustment.refine_focal_length = False
    options.mapper.bundle_adjustment.refine_extra_params = False
    monkeypatch.setattr(depth, "depth_priors", depths)
    monkeypatch.setattr(
        pipeline, "build_global_positioner", depth.build_global_positioner
    )
    monkeypatch.setattr(
        pipeline, "build_bundle_adjuster", depth.build_bundle_adjuster
    )
    monkeypatch.setattr(pycolmap.Reconstruction, "normalize", depth.normalize)
    models = pipeline.main(database_path, None, tmp_path / "sparse", options)
    assert len(models) == 1
    result = models[0]
    assert result.num_reg_images() == truth.num_reg_images()
    ids = result.reg_image_ids()
    estimated = np.array([result.images[i].projection_center() for i in ids])
    expected = np.array([truth.images[i].projection_center() for i in ids])
    distances = np.linalg.norm(estimated[:, None] - estimated[None], axis=2)
    reference = np.linalg.norm(expected[:, None] - expected[None], axis=2)
    np.testing.assert_allclose(distances, reference, atol=0.03, rtol=0.01)
