"""
An example for iterative VI optimization with IMU preintegration factors.
Data was initialized and rectified with online MPS service from Project Aria.
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import pyceres
import pycolmap.cost_functions
import wget

import pycolmap
from pycolmap import logging


def add_imu_residuals(
    prob: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    preintegrated_measurements: dict[int, pycolmap.PreintegratedImuMeasurement],
    variables: dict[str, object],
    optimize_scale: bool = True,
    optimize_gravity: bool = True,
    optimize_imu_from_cam: bool = True,
    optimize_bias: bool = True,
) -> pyceres.Problem:
    loss = pyceres.TrivialLoss()
    for image_id, integrated_m in preintegrated_measurements.items():
        image_i = reconstruction.images[image_id]
        image_j = reconstruction.images[image_id + 1]
        assert len(image_i.frame.rig.non_ref_sensors) == 0, (
            "IMU cost function requires trivial frame (no rig)"
        )
        assert len(image_j.frame.rig.non_ref_sensors) == 0, (
            "IMU cost function requires trivial frame (no rig)"
        )
        i_from_world = image_i.frame.rig_from_world
        j_from_world = image_j.frame.rig_from_world

        prob.add_residual_block(
            pycolmap.cost_functions.PreintegratedImuMeasurementCost(
                integrated_m
            ),
            loss,
            [
                variables["imu_from_cam"].rotation.quat,
                variables["imu_from_cam"].translation,
                variables["log_scale"],
                variables["gravity"],
                i_from_world.rotation.quat,
                i_from_world.translation,
                variables["imu_states"][image_id].data,
                j_from_world.rotation.quat,
                j_from_world.translation,
                variables["imu_states"][image_id + 1].data,
            ],
        )
    prob.set_manifold(variables["gravity"], pyceres.SphereManifold(3))
    prob.set_manifold(
        variables["imu_from_cam"].rotation.quat,
        pyceres.EigenQuaternionManifold(),
    )
    # [Optional] fix variables.
    if not optimize_scale:
        prob.set_parameter_block_constant(variables["log_scale"])
    if not optimize_gravity:
        prob.set_parameter_block_constant(variables["gravity"])
    if not optimize_imu_from_cam:
        prob.set_parameter_block_constant(
            variables["imu_from_cam"].rotation.quat
        )
        prob.set_parameter_block_constant(variables["imu_from_cam"].translation)
    if not optimize_bias:
        constant_idxs = np.arange(3, 9)
        for image_id, _ in variables["imu_states"].items():
            prob.set_manifold(
                variables["imu_states"][image_id].data,
                pyceres.SubsetManifold(9, constant_idxs),
            )
    return prob


def solve_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    ba_options: pycolmap.BundleAdjustmentOptions,
    ba_config: pycolmap.BundleAdjustmentConfig,
    preintegrated_measurements: dict[int, pycolmap.PreintegratedImuMeasurement],
    variables: dict[str, object],
) -> pyceres.SolverSummary:
    bundle_adjuster = pycolmap.create_default_bundle_adjuster(
        ba_options, ba_config, reconstruction
    )
    problem = bundle_adjuster.problem
    problem = add_imu_residuals(
        problem, reconstruction, preintegrated_measurements, variables
    )
    solver_options = ba_options.create_solver_options(ba_config, problem)
    solver_options.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    print(summary.BriefReport())
    return summary


def adjust_global_bundle(
    mapper: pycolmap.IncrementalMapper,
    mapper_options: pycolmap.IncrementalMapperOptions,
    ba_options: pycolmap.BundleAdjustmentOptions,
    preintegrated_measurements: dict[int, pycolmap.PreintegratedImuMeasurement],
    variables: dict[str, object],
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
        preintegrated_measurements,
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
    preintegrated_measurements: dict[int, pycolmap.PreintegratedImuMeasurement],
    variables: dict[str, object],
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
        adjust_global_bundle(
            mapper,
            mapper_options,
            ba_options,
            preintegrated_measurements,
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
    preintegrated_measurements: dict[int, pycolmap.PreintegratedImuMeasurement],
    variables: dict[str, object],
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
        preintegrated_measurements,
        variables,
        normalize_reconstruction=False,
    )
    mapper.filter_frames(mapper_options)


def iterative_refine(
    database_path: str,
    recon: pycolmap.Reconstruction,
    preintegrated_measurements: dict[int, pycolmap.PreintegratedImuMeasurement],
    variables: dict[str, object],
) -> pycolmap.Reconstruction:
    database = pycolmap.Database(database_path)
    image_names = []
    for image_id in recon.reg_image_ids():
        image_names.append(recon.images[image_id].name)
    database_cache = pycolmap.DatabaseCache.create(
        database, 15, False, set(image_names)
    )
    mapper = pycolmap.IncrementalMapper(database_cache)
    mapper.begin_reconstruction(recon)
    options = pycolmap.IncrementalPipelineOptions()
    options.fix_existing_frames = False
    iterative_global_refinement(
        options,
        options.get_mapper(),
        mapper,
        preintegrated_measurements,
        variables,
    )
    return mapper.reconstruction


def run_vi_optimization(
    sfm_path: str,
    database_path: str,
    output_folder: str,
    preintegrated_measurements: dict[int, pycolmap.PreintegratedImuMeasurement],
    variables: dict[str, object],
) -> None:
    rec = pycolmap.Reconstruction(sfm_path)
    os.makedirs(output_folder, exist_ok=True)
    rec_optimized = iterative_refine(
        database_path, rec, preintegrated_measurements, variables
    )
    rec_optimized.write(output_folder)


def download_data() -> None:
    data_url = "https://polybox.ethz.ch/index.php/s/NS6ozZswc90hzt4/download"
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
    output_path = "./vi_optimization_output"
    image_timestamps: dict[int, int] = np.load(
        image_timestamps_data, allow_pickle=True
    ).item()
    reconstruction = pycolmap.Reconstruction(sfm_path)
    num_images = len(reconstruction.images)
    imu_data = np.load(imu_data_path, allow_pickle=True)
    imu_measurements = pycolmap.ImuMeasurements()
    for row in imu_data:
        imu_measurements.append(
            pycolmap.ImuMeasurement(
                int(row[0]),
                np.array(row[1:4]),
                np.array(row[4:7]),
            )
        )

    # IMU preintegration.
    options = pycolmap.ImuPreintegrationOptions()
    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = 9.81
    # [Reference]
    # https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/imu_noise_model
    imu_calib.acc_noise_density = 0.8e-4 * 9.81
    imu_calib.gyro_noise_density = 1e-2 * (np.pi / 180.0)
    imu_calib.acc_bias_random_walk_sigma = 3.5e-5 * 9.81 * np.sqrt(353)
    imu_calib.gyro_bias_random_walk_sigma = (
        1.3e-3 * (np.pi / 180.0) * np.sqrt(116)
    )
    imu_calib.acc_saturation_max = 8.0 * imu_calib.gravity_magnitude
    imu_calib.gyro_saturation_max = 1000.0
    imu_calib.imu_rate = 1000.0

    preintegrated_measurements: dict[
        int, pycolmap.PreintegratedImuMeasurement
    ] = {}
    for i in np.arange(1, num_images - 1):
        t1, t2 = image_timestamps[i], image_timestamps[i + 1]
        # Timestamps are in nanoseconds (int64).
        ms = pycolmap.get_measurements_contain_edge(imu_measurements, t1, t2)
        if len(ms) == 0:
            continue
        integrated_m = pycolmap.PreintegratedImuMeasurement(
            options, imu_calib, t1, t2
        )
        integrated_m.add_measurements(ms)
        preintegrated_measurements[i] = integrated_m

    # Set up variables.
    variables: dict[str, object] = {}
    variables["imu_from_cam"] = pycolmap.Rigid3d()
    variables["gravity"] = np.array([0.0, 0.0, -1.0])
    variables["log_scale"] = np.array([0.0])
    variables["imu_states"] = {}
    for i in np.arange(1, num_images):
        variables["imu_states"][i] = pycolmap.ImuState()
        dt = pycolmap.timestamp_diff_seconds(
            image_timestamps[i + 1], image_timestamps[i]
        )
        pi = reconstruction.images[i].cam_from_world().inverse().translation
        pj = reconstruction.images[i + 1].cam_from_world().inverse().translation
        vel = (pj - pi) / dt
        variables["imu_states"][i].set_velocity(vel)

    # Iterative optimization.
    run_vi_optimization(
        sfm_path,
        database_path,
        output_path,
        preintegrated_measurements,
        variables,
    )

    # Eval.
    imu_from_cam = variables["imu_from_cam"]
    gravity = variables["gravity"]
    log_scale = variables["log_scale"]
    imu_states = variables["imu_states"]
    logging.info("Values after Optimization")
    logging.info(f"imu_from_cam = {imu_from_cam}")
    logging.info(f"gravity = {gravity}")
    logging.info(f"scale = exp({log_scale}) = {np.exp(log_scale)}")
    for image_id in np.arange(1, 402, 50).tolist():
        logging.info(f"imu_states[{image_id}] = {imu_states[image_id]}")


if __name__ == "__main__":
    logging.verbose_level = 2
    run()
