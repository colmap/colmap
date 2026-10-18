# SPDX-License-Identifier: BSD-3-Clause

"""
An example for iterative VI optimization with IMU preintegration factors.
Data was initialized and rectified with the online MPS service from
Project Aria 1: https://facebookresearch.github.io/projectaria_tools/docs/intro
"""

import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pyceres
import wget

import pycolmap
import pycolmap.inertial
from pycolmap import logging


class ImuReintegrationCallback(pyceres.IterationCallback):
    """Ceres iteration callback that reintegrates IMU preintegration data
    when the optimized biases have drifted beyond the linearization point."""

    def __init__(
        self, options: pycolmap.inertial.ImuReintegrationOptions
    ) -> None:
        pyceres.IterationCallback.__init__(self)
        self.options = options
        self.edges: list[
            tuple[
                pycolmap.inertial.ImuPreintegrator,
                pycolmap.inertial.PreintegratedImuData,
                pycolmap.ImuState,
            ]
        ] = []

    def add_edge(
        self,
        integrator: pycolmap.inertial.ImuPreintegrator,
        data: pycolmap.inertial.PreintegratedImuData,
        imu_state: pycolmap.ImuState,
    ) -> None:
        self.edges.append((integrator, data, imu_state))

    def _should_reintegrate(
        self, data: pycolmap.inertial.PreintegratedImuData, biases: np.ndarray
    ) -> bool:
        diff = biases - data.biases
        delta_t = data.delta_t
        if (
            np.linalg.norm(diff[:3]) * delta_t
            > self.options.reintegrate_angle_norm_thres
        ):
            return True
        return bool(
            np.linalg.norm(diff[3:]) * delta_t
            > self.options.reintegrate_vel_norm_thres
        )

    def __call__(
        self, summary: pyceres.IterationSummary
    ) -> pyceres.CallbackReturnType:
        for integrator, data, imu_state in self.edges:
            biases = imu_state.params
            if self._should_reintegrate(data, biases):
                integrator.reintegrate(biases)
                integrator.update(data)
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


def add_imu_residuals(
    prob: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    imu_data: dict[int, pycolmap.inertial.PreintegratedImuData],
    variables: dict[str, Any],
    optimize_scale: bool = True,
    optimize_gravity: bool = True,
    optimize_cam_from_imu: bool = True,
    optimize_bias: bool = True,
) -> pyceres.Problem:
    loss = pyceres.TrivialLoss()
    for image_id, integrated_m in imu_data.items():
        image_i = reconstruction.images[image_id]
        image_j = reconstruction.images[image_id + 1]
        assert image_i.frame is not None
        assert image_i.frame.rig is not None
        assert image_i.frame.rig.ref_sensor_id.type == pycolmap.SensorType.IMU
        assert image_j.frame is not None
        assert image_j.frame.rig is not None
        assert image_j.frame.rig.ref_sensor_id.type == pycolmap.SensorType.IMU
        i_from_world = image_i.frame.rig_from_world
        j_from_world = image_j.frame.rig_from_world
        assert i_from_world is not None
        assert j_from_world is not None
        assert image_i.frame.has_velocity()
        assert image_j.frame.has_velocity()

        prob.add_residual_block(
            pycolmap.inertial.AnalyticalVisualCentricImuPreintegrationCost(
                integrated_m
            ),
            loss,
            [
                variables["log_scale"],
                variables["gravity"],
                i_from_world.params,
                image_i.frame.velocity_in_world,
                variables["imu_states"][image_id].params,
                j_from_world.params,
                image_j.frame.velocity_in_world,
                variables["imu_states"][image_id + 1].params,
            ],
        )
    prob.set_manifold(variables["gravity"], pyceres.SphereManifold(3))
    # [Optional] fix variables.
    if not optimize_scale:
        prob.set_parameter_block_constant(variables["log_scale"])
    if not optimize_gravity:
        prob.set_parameter_block_constant(variables["gravity"])
    for rig in reconstruction.rigs.values():
        for sensor_id in rig.non_ref_sensors:
            sensor_from_rig = rig.sensor_from_rig(sensor_id)
            assert sensor_from_rig is not None
            if prob.has_parameter_block(sensor_from_rig.params):
                if optimize_cam_from_imu:
                    prob.set_parameter_block_variable(sensor_from_rig.params)
                else:
                    prob.set_parameter_block_constant(sensor_from_rig.params)
    if not optimize_bias:
        for state in variables["imu_states"].values():
            if prob.has_parameter_block(state.params):
                prob.set_parameter_block_constant(state.params)
    return prob


def solve_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    ba_options: pycolmap.BundleAdjustmentOptions,
    ba_config: pycolmap.BundleAdjustmentConfig,
    integrators: dict[int, pycolmap.inertial.ImuPreintegrator],
    imu_data: dict[int, pycolmap.inertial.PreintegratedImuData],
    variables: dict[str, Any],
) -> pyceres.SolverSummary:
    bundle_adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        ba_options, ba_config, reconstruction
    )
    problem = bundle_adjuster.problem  # type: ignore[attr-defined]
    add_imu_residuals(problem, reconstruction, imu_data, variables)
    solver_options = pyceres.SolverOptions(
        ba_options.ceres.create_solver_options(ba_config, problem)
    )
    solver_options.minimizer_progress_to_stdout = True
    # Set up reintegration callback to update preintegrated data when
    # biases drift beyond the linearization point.
    callback = ImuReintegrationCallback(
        pycolmap.inertial.ImuReintegrationOptions()
    )
    for image_id in integrators:
        callback.add_edge(
            integrators[image_id],
            imu_data[image_id],
            variables["imu_states"][image_id],
        )
    # Assign the list. The callbacks getter returns a copy, so appending to it
    # would not register the callback.
    solver_options.callbacks = [callback]
    solver_options.update_state_every_iteration = True
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    print(summary.BriefReport())
    return summary


def adjust_global_bundle(
    mapper: pycolmap.IncrementalMapper,
    mapper_options: pycolmap.IncrementalMapperOptions,
    ba_options: pycolmap.BundleAdjustmentOptions,
    integrators: dict[int, pycolmap.inertial.ImuPreintegrator],
    imu_data: dict[int, pycolmap.inertial.PreintegratedImuData],
    variables: dict[str, Any],
) -> None:
    reconstruction = mapper.reconstruction

    # Avoid degeneracies in bundle adjustment.
    mapper.observation_manager.filter_observations_with_negative_depth()

    # Configure bundle adjustment.
    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        ba_config.add_image(image_id)

    # Run bundle adjustment.
    summary = solve_bundle_adjustment(
        reconstruction,
        ba_options,
        ba_config,
        integrators,
        imu_data,
        variables,
    )
    logging.info("Global Bundle Adjustment")
    logging.info(summary.BriefReport())


def run_iterative(
    mapper: pycolmap.IncrementalMapper,
    max_num_refinements: int,
    max_refinement_change: float,
    mapper_options: pycolmap.IncrementalMapperOptions,
    ba_options: pycolmap.BundleAdjustmentOptions,
    tri_options: pycolmap.IncrementalTriangulatorOptions,
    integrators: dict[int, pycolmap.inertial.ImuPreintegrator],
    imu_data: dict[int, pycolmap.inertial.PreintegratedImuData],
    variables: dict[str, Any],
    normalize_reconstruction: bool = True,
) -> None:
    """Equivalent to mapper.iterative_global_refinement(...)."""
    reconstruction = mapper.reconstruction
    mapper.complete_and_merge_tracks(tri_options)
    num_retriangulated_observations = mapper.retriangulate(tri_options)
    logging.verbose(
        1, f"=> Retriangulated observations: {num_retriangulated_observations}"
    )
    for _ in range(max_num_refinements):
        num_observations = reconstruction.compute_num_observations()
        for image_id in reconstruction.reg_image_ids():
            img = reconstruction.images[image_id]
            if img.frame is not None and not img.frame.has_velocity():
                img.frame.velocity_in_world = np.zeros(3)
            if image_id not in variables["imu_states"]:
                variables["imu_states"][image_id] = pycolmap.ImuState(
                    np.zeros(3), np.zeros(3)
                )
        adjust_global_bundle(
            mapper,
            mapper_options,
            ba_options,
            integrators,
            imu_data,
            variables,
        )
        if normalize_reconstruction:
            reconstruction.normalize()
        num_changed_observations = mapper.complete_and_merge_tracks(tri_options)
        num_changed_observations += mapper.filter_points(mapper_options)
        changed = (
            num_changed_observations / num_observations
            if num_observations > 0
            else 0
        )
        logging.verbose(1, f"=> Changed observations: {changed:.6f}")
        if changed < max_refinement_change:
            break


def iterative_global_refinement(
    options: pycolmap.IncrementalPipelineOptions,
    mapper_options: pycolmap.IncrementalMapperOptions,
    mapper: pycolmap.IncrementalMapper,
    integrators: dict[int, pycolmap.inertial.ImuPreintegrator],
    imu_data: dict[int, pycolmap.inertial.PreintegratedImuData],
    variables: dict[str, Any],
) -> None:
    ba_options = options.get_global_bundle_adjustment()
    ba_options.print_summary = True
    ba_options.refine_focal_length = True
    ba_options.refine_extra_params = True
    run_iterative(
        mapper,
        options.ba_global_max_refinements,
        options.ba_global_max_refinement_change,
        mapper_options,
        ba_options,
        options.get_triangulation(),
        integrators,
        imu_data,
        variables,
        normalize_reconstruction=False,
    )
    mapper.filter_frames(mapper_options)


def iterative_refine(
    database_path: str,
    recon: pycolmap.Reconstruction,
    integrators: dict[int, pycolmap.inertial.ImuPreintegrator],
    imu_data: dict[int, pycolmap.inertial.PreintegratedImuData],
    variables: dict[str, Any],
) -> pycolmap.Reconstruction:
    with pycolmap.Database.open(database_path) as database:
        image_names = []
        for image_id in recon.reg_image_ids():
            image_names.append(recon.images[image_id].name)
        cache_options = pycolmap.DatabaseCacheOptions()
        database_cache = pycolmap.DatabaseCache.create(database, cache_options)
    mapper = pycolmap.IncrementalMapper(database_cache)
    mapper.begin_reconstruction(recon)

    orig_cam_from_world = {
        image_id: recon.images[image_id].cam_from_world()
        for image_id in recon.reg_image_ids()
    }
    imu = pycolmap.Imu()
    imu.imu_id = 1
    for rig_id in list(recon.rigs.keys()):
        camera = recon.cameras[1]
        rig = pycolmap.Rig()
        rig.rig_id = rig_id
        rig.add_ref_sensor(imu.sensor_id)
        rig.add_sensor(camera.sensor_id, pycolmap.Rigid3d())
        recon.rigs[rig_id] = rig

    for image_id, cam_from_world in orig_cam_from_world.items():
        image = recon.images[image_id]
        if image.frame is not None:
            image.frame.set_cam_from_world(image.camera_id, cam_from_world)
            if (
                "initial_velocities" in variables
                and image_id in variables["initial_velocities"]
            ):
                image.frame.velocity_in_world = variables["initial_velocities"][
                    image_id
                ]
            elif not image.frame.has_velocity():
                image.frame.velocity_in_world = np.zeros(3)

    options = pycolmap.IncrementalPipelineOptions()
    options.fix_existing_frames = False

    # Warm-start IMU states, gravity, scale, and cam_from_imu with 3D points
    # fixed.
    init_ba_options = options.get_global_bundle_adjustment()
    init_ba_options.refine_points3D = False
    init_ba_options.refine_focal_length = False
    init_ba_options.refine_extra_params = False
    init_ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in recon.reg_image_ids():
        init_ba_config.add_image(image_id)
    solve_bundle_adjustment(
        recon, init_ba_options, init_ba_config, integrators, imu_data, variables
    )
    for image_id, cam_from_world in orig_cam_from_world.items():
        image = recon.images[image_id]
        if image.frame is not None:
            image.frame.set_cam_from_world(image.camera_id, cam_from_world)

    iterative_global_refinement(
        options,
        options.get_mapper(),
        mapper,
        integrators,
        imu_data,
        variables,
    )
    return mapper.reconstruction


def run_visual_inertial_optimization(
    sfm_path: str,
    database_path: str,
    output_folder: str,
    integrators: dict[int, pycolmap.inertial.ImuPreintegrator],
    imu_data: dict[int, pycolmap.inertial.PreintegratedImuData],
    variables: dict[str, Any],
) -> pycolmap.Reconstruction:
    rec = pycolmap.Reconstruction(sfm_path)
    os.makedirs(output_folder, exist_ok=True)
    rec_optimized = iterative_refine(
        database_path, rec, integrators, imu_data, variables
    )
    rec_optimized.write(output_folder)
    return rec_optimized


def download_data() -> None:
    data_url = "https://polybox.ethz.ch/index.php/s/SHcCxyecDsz7MzS/download"
    data_path = Path("sample_data")
    if not data_path.exists():
        logging.info("Downloading the data.")
        zip_path = "sample_data.zip"
        wget.download(data_url, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as fid:
            fid.extractall()
        logging.info(f"Data extracted to {data_path}.")


def run() -> None:
    pycolmap.set_random_seed(0)
    if not Path("sample_data").exists():
        download_data()
    sfm_path = "./sample_data/mps_triangulation"
    database_path = "./sample_data/database.db"
    image_timestamps_data = "./sample_data/image_timestamps.npy"
    imu_data_path = "./sample_data/rectified_imu_measurements.npy"
    output_path = "./visual_inertial_optimization_output"
    image_timestamps: dict[int, int] = np.load(
        image_timestamps_data, allow_pickle=True
    ).item()
    reconstruction = pycolmap.Reconstruction(sfm_path)
    num_images = len(reconstruction.images)
    raw_imu_data = np.load(imu_data_path, allow_pickle=True)
    imu_measurements = pycolmap.ImuMeasurements()
    for row in raw_imu_data:
        imu_measurements.insert(
            pycolmap.ImuMeasurement(
                timestamp=int(row[0]),
                accel=np.array(row[1:4]),
                gyro=np.array(row[4:7]),
            )
        )

    # IMU preintegration.
    options = pycolmap.inertial.ImuPreintegrationOptions()
    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = 9.81
    # [Reference]
    # https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/imu_noise_model
    imu_calib.accel_noise_density = 0.8e-4 * 9.81
    imu_calib.gyro_noise_density = 1e-2 * (np.pi / 180.0)
    imu_calib.bias_accel_random_walk_sigma = 3.5e-5 * 9.81 * np.sqrt(353)
    imu_calib.bias_gyro_random_walk_sigma = (
        1.3e-3 * (np.pi / 180.0) * np.sqrt(116)
    )
    imu_calib.accel_saturation_max = 8.0 * imu_calib.gravity_magnitude
    imu_calib.gyro_saturation_max = 1000.0
    imu_calib.imu_rate = 1000.0

    # Preintegrate IMU measurements between consecutive images.
    # Keep both integrators (for reintegration) and extracted data (for cost
    # functions). The cost function holds a pointer to the data, and
    # reintegration updates the data in place.
    integrators: dict[int, pycolmap.inertial.ImuPreintegrator] = {}
    preintegrated: dict[int, pycolmap.inertial.PreintegratedImuData] = {}
    for i in range(1, num_images):
        t1, t2 = image_timestamps[i], image_timestamps[i + 1]
        # Timestamps are in nanoseconds (int64).
        ms = imu_measurements.extract_measurements_in_range(t1, t2)
        if len(ms) == 0:
            continue
        integrators[i] = pycolmap.inertial.ImuPreintegrator(
            options, imu_calib, t1, t2
        )
        integrators[i].integrate(ms)
        preintegrated[i] = integrators[i].extract()

    # Set up variables.
    variables: dict[str, Any] = {}
    variables["gravity"] = np.array([0.0, 0.0, -1.0])
    variables["log_scale"] = np.array([0.0])
    variables["imu_states"] = {}
    initial_velocities: dict[int, np.ndarray] = {}
    for i in range(1, num_images + 1):
        # Finite-difference velocity, backward for the last image.
        j, k = (i, i + 1) if i < num_images else (i - 1, i)
        dt = pycolmap.timestamp_diff_seconds(
            image_timestamps[k], image_timestamps[j]
        )
        pj = reconstruction.images[j].cam_from_world().inverse().translation
        pk = reconstruction.images[k].cam_from_world().inverse().translation
        initial_velocities[i] = (pk - pj) / dt
        variables["imu_states"][i] = pycolmap.ImuState(np.zeros(3), np.zeros(3))
    variables["initial_velocities"] = initial_velocities

    # Iterative optimization.
    rec_optimized = run_visual_inertial_optimization(
        sfm_path,
        database_path,
        output_path,
        integrators,
        preintegrated,
        variables,
    )

    # Eval.
    rig = rec_optimized.rigs[1]
    camera = rec_optimized.cameras[1]
    cam_from_imu = rig.sensor_from_rig(camera.sensor_id)
    assert cam_from_imu is not None
    imu_from_cam = cam_from_imu.inverse()
    gravity = variables["gravity"]
    log_scale = variables["log_scale"]
    imu_states = variables["imu_states"]
    logging.info("Values after Optimization")
    logging.info(f"cam_from_imu = {cam_from_imu}")
    logging.info(f"imu_from_cam = {imu_from_cam}")
    logging.info(f"gravity = {gravity}")
    logging.info(
        f"scale = exp({log_scale[0]:.6f}) = {np.exp(log_scale[0]):.6f}"
    )
    for image_id in range(1, min(len(rec_optimized.images) + 1, 402), 50):
        if image_id in imu_states:
            logging.info(f"imu_states[{image_id}] = {imu_states[image_id]}")
        frame = rec_optimized.images[image_id].frame
        if frame is not None and frame.has_velocity():
            logging.info(
                f"frame[{image_id}].velocity = {frame.velocity_in_world}"
            )


if __name__ == "__main__":
    logging.verbose_level = 2
    run()
