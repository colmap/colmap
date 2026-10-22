# SPDX-License-Identifier: BSD-3-Clause

import copy

import custom_global_pipeline as custom
import numpy as np
import pytest

import pycolmap


def create_database(
    path, cameras=1, components=False, unknown=False, noise=False, relative=True
):
    pycolmap.set_random_seed(2)
    database = pycolmap.Database.open(path)
    truth = pycolmap.synthesize_dataset(
        pycolmap.SyntheticDatasetOptions(
            num_rigs=2,
            num_cameras_per_rig=cameras,
            num_frames_per_rig=5,
            num_points3D=40,
            two_view_geometry_has_relative_pose=relative,
        ),
        database,
    )
    if components:
        ids, _ = database.read_two_view_geometries()
        for pair_id in ids:
            i, j = pycolmap.pair_id_to_image_pair(pair_id)
            if truth.images[i].frame.rig_id != truth.images[j].frame.rig_id:
                database.delete_two_view_geometry(i, j)
    if unknown:
        for rig in database.read_all_rigs():
            for sensor in rig.non_ref_sensors:
                rig.reset_sensor_from_rig(sensor)
            database.update_rig(rig)
    if noise:
        pycolmap.synthesize_noise(
            pycolmap.SyntheticNoiseOptions(point2D_stddev=0.25), truth, database
        )
    return database


def run(database, custom_run, options, stop=False):
    pycolmap.set_random_seed(42)
    options = copy.deepcopy(options)
    updates = []

    def on_update(reconstruction):
        updates.append(reconstruction.num_reg_frames())
        return stop

    if custom_run:
        return custom.main_global_mapper(database, options, on_update), updates
    manager = pycolmap.ReconstructionManager()
    controller = pycolmap.GlobalPipeline(options, database, manager)
    controller.add_callback(
        pycolmap.GlobalPipelineCallback.MODEL_UPDATE_CALLBACK,
        lambda: on_update(manager.get(manager.size() - 1)),
    )
    controller.run()
    return [manager.get(i) for i in range(manager.size())], updates


def compare(native, python):
    assert len(native) == len(python)
    assert all(r.num_reg_frames() > 0 and r.num_points3D() > 0 for r in python)
    native = {frozenset(r.reg_image_ids()): r for r in native}
    for result in python:
        reference = native[frozenset(result.reg_image_ids())]
        assert result.num_reg_frames() == reference.num_reg_frames()
        assert result.num_points3D() == reference.num_points3D()
        assert (
            result.compute_num_observations()
            == reference.compute_num_observations()
        )
        assert result.num_images() == reference.num_images()
        assert result.num_cameras() == reference.num_cameras()
        assert result.num_frames() == reference.num_frames()
        assert result.num_rigs() == reference.num_rigs()
        for frame_id in result.reg_frame_ids():
            frame, expected = (
                result.frames[frame_id],
                reference.frames[frame_id],
            )
            assert frame.rig_id == expected.rig_id
            np.testing.assert_allclose(
                frame.rig_from_world.matrix(),
                expected.rig_from_world.matrix(),
                atol=1e-7,
                rtol=1e-7,
            )
        for rig_id, rig in result.rigs.items():
            expected = reference.rigs[rig_id]
            assert rig.ref_sensor_id == expected.ref_sensor_id
            assert set(rig.non_ref_sensors) == set(expected.non_ref_sensors)
            for sensor in rig.non_ref_sensors:
                np.testing.assert_allclose(
                    rig.sensor_from_rig(sensor).matrix(),
                    expected.sensor_from_rig(sensor).matrix(),
                    atol=1e-7,
                    rtol=1e-7,
                )
        for camera_id, camera in result.cameras.items():
            np.testing.assert_allclose(
                camera.params,
                reference.cameras[camera_id].params,
                atol=1e-7,
                rtol=1e-7,
            )
        for image_id in result.reg_image_ids():
            np.testing.assert_allclose(
                result.images[image_id].cam_from_world().matrix(),
                reference.images[image_id].cam_from_world().matrix(),
                atol=1e-7,
                rtol=1e-7,
            )
        for point_id, point in result.points3D.items():
            expected = reference.points3D[point_id]
            assert {
                (e.image_id, e.point2D_idx) for e in point.track.elements
            } == {(e.image_id, e.point2D_idx) for e in expected.track.elements}
            np.testing.assert_array_equal(point.color, expected.color)
            np.testing.assert_allclose(
                point.error, expected.error, atol=1e-7, rtol=1e-7
            )
            np.testing.assert_allclose(
                point.xyz,
                reference.points3D[point_id].xyz,
                atol=1e-7,
                rtol=1e-7,
            )


@pytest.mark.parametrize(
    "case, expected_models",
    [
        ("mono", 1),
        ("rig", 1),
        ("unknown", 1),
        ("noise", 1),
        ("small_noise", 2),
        ("caspar_fixed", 1),
        ("skip_track_establishment", 0),
        ("skip_global_positioning", 1),
        ("ceres", 1),
        ("decompose", 1),
        ("ba_stop", 1),
        ("skip_ba", 1),
        ("skip_retriangulation", 1),
        ("components", 2),
        ("largest", 1),
        ("empty", 0),
        ("small", 0),
    ],
)
def test_matches_native(tmp_path, case, expected_models):
    database = create_database(
        tmp_path / "database.db",
        cameras=2 if case in ("rig", "unknown") else 1,
        components=case in ("components", "largest", "small_noise"),
        unknown=case == "unknown",
        noise=case in ("noise", "small_noise", "ba_stop"),
        relative=case != "decompose",
    )
    if case == "empty":
        database.clear_two_view_geometries()
    options = pycolmap.GlobalPipelineOptions(num_threads=1, random_seed=42)
    options.mapper.bundle_adjustment.ceres.use_gpu = False
    options.mapper.global_positioning.use_gpu = False
    options.multiple_models = case != "largest"
    if case == "ceres":
        options.mapper.rotation_averaging.backend = (
            pycolmap.RotationAveragingBackend.CERES
        )
    if case == "caspar_fixed":
        options.mapper.bundle_adjustment.backend = (
            pycolmap.BundleAdjustmentBackend.CASPAR
        )
        options.mapper.ba_skip_joint_optimization_stage = True
        options.mapper.skip_retriangulation = True
    if case in ("skip_track_establishment", "skip_global_positioning"):
        setattr(options.mapper, case, True)
    if case == "skip_global_positioning":
        options.mapper.skip_bundle_adjustment = True
        options.mapper.skip_retriangulation = True
    if case == "ba_stop":
        options.mapper.bundle_adjustment.check_if_stopped = lambda: True
    if case == "skip_ba":
        options.mapper.skip_bundle_adjustment = True
    if case == "skip_retriangulation":
        options.mapper.skip_retriangulation = True
    if case == "small":
        options.min_model_size = 100
    native, native_updates = run(database, False, options)
    python, python_updates = run(database, True, options)
    compare(native, python)
    assert native_updates == python_updates
    assert len(python) == expected_models

    if case == "ba_stop":
        options.mapper.bundle_adjustment.check_if_stopped = None
        baseline, _ = run(database, False, options)
        assert (
            abs(
                baseline[0].compute_mean_reprojection_error()
                - native[0].compute_mean_reprojection_error()
            )
            > 1e-6
        )


def test_skip_rotation_averaging_without_poses(tmp_path):
    database = create_database(tmp_path / "database.db")
    options = pycolmap.GlobalPipelineOptions(num_threads=1, random_seed=42)
    options.mapper.bundle_adjustment.ceres.use_gpu = False
    options.mapper.global_positioning.use_gpu = False
    options.mapper.skip_rotation_averaging = True
    # Neither pipeline can establish tracks without initialized poses.
    for custom_run in (False, True):
        with pytest.raises(ValueError, match="Missing keypoints"):
            run(database, custom_run, options)


def test_replaceable_builders(tmp_path, monkeypatch):
    database = create_database(tmp_path / "database.db")
    options = pycolmap.GlobalPipelineOptions(num_threads=1, random_seed=42)
    options.mapper.bundle_adjustment.ceres.use_gpu = False
    options.mapper.global_positioning.use_gpu = False
    gp_builder = custom.build_global_positioner
    ba_builder = custom.build_bundle_adjuster
    calls = {"gp": 0, "ba": 0, "refinement": 0}

    def gp(*args):
        calls["gp"] += 1
        return gp_builder(*args)

    def ba(options, config, reconstruction):
        calls["ba"] += 1
        if not options.refine_rig_from_world:
            calls["refinement"] += 1
        return ba_builder(options, config, reconstruction)

    monkeypatch.setattr(custom, "build_global_positioner", gp)
    monkeypatch.setattr(custom, "build_bundle_adjuster", ba)
    result, _ = run(database, True, options)
    assert len(result) == 1
    assert calls["gp"] == 1
    assert calls["ba"] >= 5
    assert calls["refinement"] >= 1


def test_stop_after_global_positioning(tmp_path):
    database = create_database(tmp_path / "database.db", components=True)
    options = pycolmap.GlobalPipelineOptions(num_threads=1, random_seed=42)
    models, updates = run(database, True, options, stop=True)
    assert len(models) == len(updates) == 1
    assert models[0].num_reg_frames() == 5
    assert models[0].num_points3D() > 0


@pytest.mark.parametrize("stop_after", [None, 0, 1, 2, 3])
def test_solve_callback(tmp_path, stop_after):
    database = create_database(tmp_path / "database.db", noise=True)
    cache = pycolmap.DatabaseCache.create(
        database, pycolmap.DatabaseCacheOptions()
    )
    mapper = pycolmap.GlobalMapper(cache)
    reconstruction = pycolmap.Reconstruction()
    mapper.begin_reconstruction(reconstruction)
    options = pycolmap.GlobalMapperOptions(num_threads=1, random_seed=42)
    options.bundle_adjustment.ceres.use_gpu = False
    options.global_positioning.use_gpu = False
    calls = 0

    def on_progress():
        nonlocal calls
        calls += 1
        expected = copy.deepcopy(reconstruction)
        expected.update_point_3d_errors()
        for point_id, point in reconstruction.points3D.items():
            np.testing.assert_allclose(
                point.error,
                expected.points3D[point_id].error,
                atol=1e-10,
                rtol=1e-10,
            )
        return calls == stop_after

    callback = None if stop_after is None else on_progress
    assert custom.solve(mapper, cache, options, callback)
    assert reconstruction.num_points3D() > 0
    if stop_after is None:
        assert calls == 0
    elif stop_after:
        # A stop during BA exits its loop; retriangulation still reports.
        assert calls == (3 if stop_after == 2 else stop_after)
    else:
        assert calls >= 3


def test_cli_preserves_model_order(tmp_path, monkeypatch):
    models = [
        pycolmap.synthesize_dataset(
            pycolmap.SyntheticDatasetOptions(
                num_rigs=1,
                num_frames_per_rig=frames,
                num_points3D=points,
            )
        )
        for frames, points in [(6, 10), (3, 20)]
    ]
    database_path = tmp_path / "database.db"
    create_database(database_path).close()
    monkeypatch.setattr(custom, "main_global_mapper", lambda *args: models)
    custom.main(database_path, None, tmp_path / "output")
    for i, model in enumerate(models):
        saved = pycolmap.Reconstruction(tmp_path / "output" / str(i))
        assert saved.num_reg_frames() == model.num_reg_frames()
        assert saved.num_points3D() == model.num_points3D()
