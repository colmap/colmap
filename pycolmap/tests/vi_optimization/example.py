"""
An example for iterative VI optimization with IMU preintegration factors.
Data was initialized and rectified with online MPS service from Project Aria.
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import pyceres
import wget

import pycolmap
from pycolmap import logging


def add_imu_residuals(
    prob,
    reconstruction,
    preintegrated_measurements,
    variables,
    optimize_scale=True,
    optimize_gravity=True,
    optimize_imu_from_cam=True,
    optimize_bias=True,
):
    loss = pyceres.TrivialLoss()
    for image_id, integrated_m in preintegrated_measurements.items():
        i_from_world = reconstruction.images[image_id].cam_from_world
        j_from_world = reconstruction.images[image_id + 1].cam_from_world

        prob.add_residual_block(
            pycolmap.PreintegratedImuMeasurementCost(integrated_m),
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
        variables["imu_from_cam"].rotation.quat, pyceres.QuaternionManifold()
    )
    # [Optional] fix variables
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
    reconstruction, ba_options, ba_config, preintegrated_measurements, variables
):
    bundle_adjuster = pycolmap.BundleAdjuster(ba_options, ba_config)
    bundle_adjuster.set_up_problem(
        reconstruction, ba_options.create_loss_function()
    )
    solver_options = bundle_adjuster.set_up_solver_options(
        bundle_adjuster.problem, ba_options.solver_options
    )
    problem = bundle_adjuster.problem
    problem = add_imu_residuals(
        problem, reconstruction, preintegrated_measurements, variables
    )
    solver_options.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    print(summary.BriefReport())
    return summary


def adjust_global_bundle(
    mapper, mapper_options, ba_options, preintegrated_measurements, variables
):
    reconstruction = mapper.reconstruction

    # Avoid degeneracies in bundle adjustment
    mapper.observation_manager.filter_observations_with_negative_depth()

    # Configure bundle adjustment
    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        ba_config.add_image(image_id)

    # Run bundle adjustment
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
    mapper,
    max_num_refinements,
    max_refinement_change,
    mapper_options,
    ba_options,
    tri_options,
    preintegrated_measurements,
    variables,
    normalize_reconstruction=True,
):
    """Equivalent to mapper.iterative_global_refinement(...)"""
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
    options, mapper_options, mapper, preintegrated_measurements, variables
):
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
    mapper.filter_images(mapper_options)


def iterative_refine(
    database_path, recon, preintegrated_measurements, variables
):
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
    options.fix_existing_images = False
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
    preintegrated_measurements: dict,
    variables: dict,
):
    rec = pycolmap.Reconstruction(sfm_path)
    os.makedirs(output_folder, exist_ok=True)
    rec_optimized = iterative_refine(
        database_path, rec, preintegrated_measurements, variables
    )
    rec_optimized.write(output_folder)


def download_data():
    data_url = "https://polybox.ethz.ch/index.php/s/NS6ozZswc90hzt4/download"
    data_path = Path("sample_data")
    if not data_path.exists():
        logging.info("Downloading the data.")
        zip_path = "sample_data.zip"
        wget.download(data_url, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as fid:
            fid.extractall()
        logging.info(f"Data extracted to {data_path}.")


def run():
    pycolmap.set_random_seed(0)
    if not Path("sample_data").exists():
        download_data()
    sfm_path = "./sample_data/mps_triangulation"
    database_path = "./sample_data/database.db"
    image_timestamps_data = "./sample_data/image_timestamps.npy"
    imu_data_path = "./sample_data/rectified_imu_measurements.npy"
    output_path = "./vi_optimization_output"
    image_timestamps = np.load(image_timestamps_data, allow_pickle=True).item()
    reconstruction = pycolmap.Reconstruction(sfm_path)
    num_images = len(reconstruction.images)
    imu_data = np.load(imu_data_path, allow_pickle=True)
    imu_measurements = pycolmap.ImuMeasurements(imu_data.tolist())

    # IMU preintegration
    options = pycolmap.ImuPreintegrationOptions()
    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = 9.81
    # [Reference] https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/imu_noise_model
    imu_calib.acc_noise_density = 0.8e-4 * 9.81
    imu_calib.gyro_noise_density = 1e-2 * (np.pi / 180.0)
    imu_calib.acc_bias_random_walk_sigma = 3.5e-5 * 9.81 * np.sqrt(353)
    imu_calib.gyro_bias_random_walk_sigma = (
        1.3e-3 * (np.pi / 180.0) * np.sqrt(116)
    )
    imu_calib.acc_saturation_max = 8.0 * imu_calib.gravity_magnitude
    imu_calib.gyro_saturation_max = 1000.0
    imu_calib.imu_rate = 1000.0

    preintegrated_measurements = {}
    for i in np.arange(1, num_images - 1):
        t1, t2 = image_timestamps[i], image_timestamps[i + 1]
        # convert to seconds
        t1, t2 = t1 / 1e9, t2 / 1e9
        ms = imu_measurements.get_measurements_contain_edge(t1, t2)
        if len(ms) == 0:
            continue
        integrated_m = pycolmap.PreintegratedImuMeasurement(
            options, imu_calib, t1, t2
        )
        integrated_m.add_measurements(ms)
        preintegrated_measurements[i] = integrated_m

    # Set up variables
    variables = {}
    variables["imu_from_cam"] = pycolmap.Rigid3d()
    variables["gravity"] = np.array([0.0, 0.0, -1.0])
    variables["log_scale"] = np.array([0.0])
    variables["imu_states"] = {}
    for i in np.arange(1, num_images):
        variables["imu_states"][i] = pycolmap.ImuState()
        t1, t2 = image_timestamps[i] / 1e9, image_timestamps[i + 1] / 1e9
        pi = reconstruction.images[i].cam_from_world.inverse().translation
        pj = reconstruction.images[i + 1].cam_from_world.inverse().translation
        vel = (pj - pi) / (t2 - t1)
        variables["imu_states"][i].set_velocity(vel)

    # Iterative optimization
    run_vi_optimization(
        sfm_path,
        database_path,
        output_path,
        preintegrated_measurements,
        variables,
    )

    # Eval
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
    run()
