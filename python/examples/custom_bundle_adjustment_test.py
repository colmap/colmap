# SPDX-License-Identifier: BSD-3-Clause

import custom_bundle_adjustment

import pycolmap


def test_local_bundle_completes_tracks_from_requested_image() -> None:
    pycolmap.set_random_seed(0)
    with pycolmap.Database.open(":memory:") as database:
        reconstruction = pycolmap.synthesize_dataset(
            pycolmap.SyntheticDatasetOptions(
                num_rigs=1,
                num_frames_per_rig=4,
                num_points3D=30,
                num_points2D_without_point3D=0,
            ),
            database,
        )
        cache = pycolmap.DatabaseCache.create(
            database, pycolmap.DatabaseCacheOptions()
        )

    mapper = pycolmap.IncrementalMapper(cache)
    mapper.begin_reconstruction(reconstruction)
    point3D_id, point3D = next(iter(reconstruction.points3D.items()))
    observations = {
        element.image_id: element.point2D_idx
        for element in point3D.track.elements
    }
    # Image 1 still observes this track. Image 4 cannot complete it after its
    # own observation is removed, so completing the wrong image loses data.
    for image_id in (4, 2):
        mapper.observation_manager.delete_observation(
            image_id, observations[image_id]
        )

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.ceres.solver_options.num_threads = 1
    try:
        report = custom_bundle_adjustment.adjust_local_bundle(
            mapper,
            pycolmap.IncrementalMapperOptions(),
            ba_options,
            pycolmap.IncrementalTriangulatorOptions(),
            image_id=1,
            point3D_ids=set(),
        )

        assert report.num_completed_observations == 2
        for image_id in (2, 4):
            point2D = reconstruction.image(image_id).points2D[
                observations[image_id]
            ]
            assert point2D.point3D_id == point3D_id
        assert reconstruction.point3D(point3D_id).track.length() == 4
    finally:
        mapper.end_reconstruction(discard=False)
