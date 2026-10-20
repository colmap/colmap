# SPDX-License-Identifier: BSD-3-Clause

"""
Python reimplementation of the C++ global mapper with equivalent logic.
"""

import argparse
import copy
from pathlib import Path

import pycolmap
from pycolmap import logging


def build_global_positioner(options, pose_graph, reconstruction):
    return pycolmap.create_default_global_positioner(
        options, pose_graph, reconstruction
    )


def build_bundle_adjuster(options, config, reconstruction):
    return pycolmap.create_default_bundle_adjuster(
        options, config, reconstruction
    )


def run_bundle_adjustment(options, reconstruction):
    if reconstruction.num_images() == 0 or reconstruction.num_points3D() == 0:
        return False
    config = pycolmap.BundleAdjustmentConfig()
    for image_id, image in reconstruction.images.items():
        if image.has_pose:
            config.add_image(image_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    return (
        build_bundle_adjuster(options, config, reconstruction)
        .solve()
        .is_solution_usable()
    )


def refinement_bundle_adjustment_options(options):
    options = copy.deepcopy(options)
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    options.refine_rig_from_world = False
    options.print_summary = False
    options.min_track_length = 0
    if options.ceres is not None:
        options.ceres.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
        solver = options.ceres.solver_options
        solver.function_tolerance = 0.0
        solver.gradient_tolerance = 1.0
        solver.parameter_tolerance = 0.0
        solver.max_num_iterations = 50
        solver.max_linear_solver_iterations = 100
    return options


def filter_reprojection(reconstruction, max_error):
    return pycolmap.ObservationManager(
        reconstruction
    ).filter_points3D_with_large_reprojection_error(
        max_error,
        reconstruction.point3D_ids(),
        pycolmap.ReprojectionErrorType.NORMALIZED,
    )


def filter_final(reconstruction, max_error, min_angle):
    filter_reprojection(reconstruction, max_error)
    pycolmap.ObservationManager(
        reconstruction
    ).filter_points3D_with_small_triangulation_angle(
        min_angle, reconstruction.point3D_ids()
    )


def global_positioning(mapper, options):
    reconstruction = mapper.reconstruction
    if reconstruction.num_images() == 0 or reconstruction.num_points3D() == 0:
        return False
    positioner = build_global_positioner(
        options.get_global_positioning(), mapper.pose_graph, reconstruction
    )
    if not positioner.solve().IsSolutionUsable():
        return False
    mapper.filter_after_global_positioning(
        options.max_angular_reproj_error_deg,
        options.max_normalized_reproj_error,
        options.min_tri_angle_deg,
    )
    reconstruction.normalize()
    return True


def iterative_bundle_adjustment(reconstruction, options, on_progress):
    ba_options = options.get_bundle_adjustment()
    iteration = 0
    while iteration < options.ba_num_iterations:
        if not options.ba_skip_fixed_rotation_stage:
            position_options = copy.deepcopy(ba_options)
            position_options.constant_rig_from_world_rotation = True
            if (
                position_options.backend
                == pycolmap.BundleAdjustmentBackend.CASPAR
            ):
                position_options.backend = (
                    pycolmap.BundleAdjustmentBackend.CERES
                )
            if not run_bundle_adjustment(position_options, reconstruction):
                return False
        if (
            not options.ba_skip_joint_optimization_stage
            and not run_bundle_adjustment(ba_options, reconstruction)
        ):
            return False
        reconstruction.normalize()
        if on_progress():
            break
        filtered = 0
        status = True
        while status and iteration < options.ba_num_iterations:
            filtered += filter_reprojection(
                reconstruction,
                max(3 - iteration, 1) * options.max_normalized_reproj_error,
            )
            if filtered > 1e-3 * reconstruction.num_points3D():
                status = False
            else:
                iteration += 1
        if status:
            break
        iteration += 1
    filter_final(
        reconstruction,
        options.max_normalized_reproj_error,
        options.min_tri_angle_deg,
    )
    return True


def iterative_retriangulate_and_refine(mapper, database_cache, options):
    reconstruction = mapper.reconstruction
    tri_options = options.get_retriangulation()
    ba_options = options.get_bundle_adjustment()
    reconstruction.delete_all_points2D_and_points3D()
    incremental = pycolmap.IncrementalMapper(database_cache)
    incremental.begin_reconstruction(reconstruction)
    for image_id in reconstruction.reg_image_ids():
        incremental.triangulate_image(tri_options, image_id)

    # Equivalent to IncrementalMapper::IterativeGlobalRefinement with the
    # default mapper options and fixed-pose BA used by the global mapper.
    mapper_options = pycolmap.IncrementalMapperOptions(
        random_seed=tri_options.random_seed
    )
    refinement = refinement_bundle_adjustment_options(ba_options)
    if not (refinement.check_if_stopped and refinement.check_if_stopped()):
        incremental.complete_and_merge_tracks(tri_options)
        incremental.retriangulate(tri_options)
        for _ in range(5):
            if refinement.check_if_stopped and refinement.check_if_stopped():
                break
            observations = reconstruction.compute_num_observations()
            incremental.observation_manager.filter_observations_with_negative_depth()
            if reconstruction.num_reg_images() >= 2:
                run_bundle_adjustment(refinement, reconstruction)
            if refinement.check_if_stopped and refinement.check_if_stopped():
                break
            changed = incremental.complete_and_merge_tracks(tri_options)
            changed += incremental.filter_points(mapper_options)
            if (changed / observations if observations else 0) < 0.0005:
                break
        incremental.clear_modified_points3D()
    incremental.end_reconstruction(discard=False)

    for _ in range(2):
        filter_reprojection(reconstruction, options.max_normalized_reproj_error)
        if not run_bundle_adjustment(ba_options, reconstruction):
            return False
    reconstruction.normalize()
    filter_final(
        reconstruction,
        options.max_normalized_reproj_error,
        options.min_tri_angle_deg,
    )
    return True


def solve(mapper, database_cache, options, on_progress):
    """Equivalent to GlobalMapper::Solve."""
    if mapper.pose_graph.empty:
        return False
    if not options.skip_rotation_averaging and not mapper.rotation_averaging(
        options.get_rotation_averaging()
    ):
        return False
    if not options.skip_track_establishment:
        mapper.establish_tracks(options)
    if not options.skip_global_positioning:
        if not global_positioning(mapper, options):
            return False
        if on_progress():
            return True
    if not options.skip_bundle_adjustment and not iterative_bundle_adjustment(
        mapper.reconstruction, options, on_progress
    ):
        return False
    if not options.skip_retriangulation:
        if not iterative_retriangulate_and_refine(
            mapper, database_cache, options
        ):
            return False
        if on_progress():
            return True
    mapper.reconstruction.update_point_3d_errors()
    return True


def num_frames_for_images(reconstruction, image_ids):
    return len({reconstruction.images[i].frame_id for i in image_ids})


def compute_components(options, pose_graph, base, pose_priors, min_model_size):
    components = []
    failed = too_small = 0
    for image_ids in pose_graph.connected_image_ids_for_frame_components(
        base, filter_unregistered=False
    ):
        if num_frames_for_images(base, image_ids) < min_model_size:
            too_small += 1
            continue
        reconstruction = copy.deepcopy(base)
        graph = pycolmap.PoseGraph(pose_graph)
        graph.invalidate_pairs_outside_active_image_ids(image_ids)
        ra_options = copy.deepcopy(options)
        ra_options.filter_unregistered = False
        if not pycolmap.run_rotation_averaging_on_component(
            ra_options, graph, image_ids, reconstruction, pose_priors
        ):
            failed += 1
            continue
        if ra_options.max_rotation_error_deg > 0:
            pycolmap.filter_edges_by_relative_rotation(
                graph, reconstruction, ra_options.max_rotation_error_deg
            )
        components.extend(
            graph.connected_image_ids_for_frame_components(
                reconstruction, filter_unregistered=True
            )
        )
    return components, failed, too_small


def reconstruct_single_component(cache, options, on_progress):
    reconstruction = pycolmap.Reconstruction()
    mapper = pycolmap.GlobalMapper(cache)
    mapper.begin_reconstruction(reconstruction)

    def report_progress():
        reconstruction.update_point_3d_errors()
        return on_progress(reconstruction)

    if not solve(mapper, cache, options, report_progress):
        return None
    pycolmap.align_reconstruction_to_orig_rig_scales(cache.rigs, reconstruction)
    return reconstruction


def main_global_mapper(database, options, on_progress=None):
    """Run mapping; on_progress(reconstruction) returns True to stop."""
    if options.min_model_size < 0:
        raise ValueError("min_model_size must be nonnegative")
    cache = pycolmap.DatabaseCache.create(
        database,
        pycolmap.DatabaseCacheOptions(
            min_num_matches=options.min_num_matches,
            ignore_watermarks=options.ignore_watermarks,
            image_names=set(options.image_names),
        ),
    )
    if options.decompose_relative_pose:
        pycolmap.maybe_decompose_relative_poses(cache)
    mapper_options = copy.deepcopy(options.mapper)
    mapper_options.image_path = options.image_path
    mapper_options.num_threads = options.num_threads
    mapper_options.random_seed = options.random_seed
    models = []
    stopped = False
    failed = too_small = 0
    insufficient_focals = (
        sum(camera.has_prior_focal_length for camera in cache.cameras.values())
        < 0.5 * cache.num_cameras()
    )
    warning = (
        "Less than 50% of cameras have prior focal lengths. "
        "Consider view graph calibration."
    )
    if insufficient_focals:
        logging.warning(warning)

    def report_progress(reconstruction):
        nonlocal stopped
        stopped = bool(on_progress and on_progress(reconstruction)) or stopped
        return stopped

    def reconstruct(component_cache):
        nonlocal failed, too_small
        result = reconstruct_single_component(
            component_cache, mapper_options, report_progress
        )
        if result is None:
            failed += 1
        elif result.num_reg_frames() < options.min_model_size:
            too_small += 1
        else:
            models.append(result)

    if options.multiple_models:
        base = pycolmap.Reconstruction()
        base.load(cache)
        graph = pycolmap.PoseGraph()
        graph.load(cache.correspondence_graph)
        if not graph.empty:
            components, failed, too_small = compute_components(
                mapper_options.get_rotation_averaging(),
                graph,
                base,
                cache.pose_priors,
                options.min_model_size,
            )
            for image_ids in components:
                if stopped:
                    break
                if (
                    num_frames_for_images(base, image_ids)
                    < options.min_model_size
                ):
                    too_small += 1
                    continue
                component_cache = pycolmap.DatabaseCache.create_from_cache(
                    cache,
                    pycolmap.DatabaseCacheOptions(
                        image_names={base.images[i].name for i in image_ids}
                    ),
                )
                reconstruct(component_cache)
    else:
        reconstruct(cache)

    models.sort(
        key=lambda reconstruction: reconstruction.num_reg_frames(), reverse=True
    )
    for reconstruction in models:
        if str(options.image_path) not in ("", "."):
            reconstruction.extract_colors_for_all_images(
                options.image_path, options.num_threads
            )
    logging.info(
        f"Kept {len(models)} reconstruction(s), discarded {too_small} small "
        f"components, and failed to reconstruct {failed}"
    )
    if insufficient_focals:
        logging.warning(warning)
    return models


def main(database_path, image_path, output_path, options=None):
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    if options is None:
        options = pycolmap.GlobalPipelineOptions()
    if image_path is not None:
        options.image_path = image_path.resolve()
    with pycolmap.Database.open(database_path) as database:
        models = main_global_mapper(database, options)
    output_path.mkdir(parents=True, exist_ok=True)
    for i, reconstruction in enumerate(
        sorted(models, key=lambda model: model.num_points3D(), reverse=True)
    ):
        model_path = output_path / str(i)
        model_path.mkdir(exist_ok=True)
        reconstruction.write(model_path)
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database_path", type=Path, required=True)
    parser.add_argument("--image_path", type=Path)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    main(
        args.database_path,
        args.image_path,
        args.output_path,
    )
