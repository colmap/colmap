"""
Python reimplementation of the bundle adjustment for the incremental mapper of C++ with equivalent logic.
As a result, one can add customized residuals on top of the exposed ceres problem from conventional bundle adjustment.
"""
import pyceres
import pycolmap
from pycolmap import logging
import copy

def solve_bundle_adjustment(reconstruction, ba_options, ba_config):
    bundle_adjuster = pycolmap.BundleAdjuster(ba_options, ba_config)
    bundle_adjuster.solve(reconstruction)
    return bundle_adjuster.summary
"""
    problem = bundle_adjuster.set_up_problem(reconstruction, pyceres.TrivialLoss()) # TODO
    if problem.num_residuals() == 0:
        return None
    solver_options = bundle_adjuster.set_up_solver_options(problem, bundle_adjuster.options.solver_options)
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    return summary
"""

def adjust_global_bundle(mapper, mapper_options, ba_options):
    """Equivalent to mapper.adjust_global_bundle(...)"""
    reconstruction = mapper.get_reconstruction()
    assert reconstruction is not None
    reg_image_ids = reconstruction.reg_image_ids()
    if len(reg_image_ids) < 2:
        logging.fatal("At least two images must be registered for global bundle-adjustment")
    ba_options_tmp = copy.deepcopy(ba_options)

    # Use stricter convergence criteria for first registered images
    if len(reg_image_ids) < 10: # kMinNumRegImagesForFastBA = 10
        ba_options_tmp.solver_options.function_tolerance /= 10;
        ba_options_tmp.solver_options.gradient_tolerance /= 10;
        ba_options_tmp.solver_options.parameter_tolerance /= 10;
        ba_options_tmp.solver_options.max_num_iterations *= 2;
        ba_options_tmp.solver_options.max_linear_solver_iterations = 200;

    # Avoid degeneracies in bundle adjustment
    reconstruction.filter_observations_with_negative_depth()

    # Configure bundle adjustment
    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in reg_image_ids:
        ba_config.add_image(image_id)

    # Fix the existing images, if option specified
    if mapper_options.fix_existing_images:
        for image_id in reg_image_ids:
            if image_id in mapper.existing_image_ids:
                ba_config.set_constant_cam_pose(image_id)

    # Fix 7-DOFs of the bundle adjustment problem
    ba_config.set_constant_cam_pose(reg_image_ids[0])
    if (not mapper_options.fix_existing_images) or (reg_image_ids[1] not in mapper.existing_image_ids):
        ba_config.set_constant_cam_positions(reg_image_ids[1], [0])

    # Run bundle adjustment
    summary = solve_bundle_adjustment(reconstruction, ba_options_tmp, ba_config)
    logging.info("Global Bundle Adjustment")
    logging.info(summary.BriefReport())
    return True

def iterative_global_refinement(mapper, max_num_refinements, max_refinement_change, mapper_options, ba_options, tri_options, normalize_reconstruction = True):
    """Equivalent to mapper.iterative_global_refinement(...)"""
    reconstruction = mapper.get_reconstruction()
    mapper.complete_and_merge_tracks(tri_options)
    logging.verbose(1, "=> Retriangulated observations: {0}".format(mapper.retriangulate(tri_options)))
    for i in range(max_num_refinements):
        num_observations = reconstruction.compute_num_observations()
        # mapper.adjust_global_bundle(mapper_options, ba_options)
        adjust_global_bundle(mapper, mapper_options, ba_options)
        if normalize_reconstruction:
            reconstruction.normalize()
        num_changed_observations = mapper.complete_and_merge_tracks(tri_options)
        num_changed_observations += mapper.filter_points(mapper_options)
        changed = 0 if num_observations == 0 else num_changed_observations / num_observations
        logging.verbose(1, "=> Changed observations: {0:.6f}".format(changed))
        if changed < max_refinement_change:
            break

def adjust_local_bundle(mapper, mapper_options, ba_options, tri_options, image_id, point3D_ids):
    """Equivalent to mapper.adjust_local_bundle(...)"""
    reconstruction = mapper.get_reconstruction()
    assert reconstruction is not None
    report = pycolmap.LocalBundleAdjustmentReport()

    # Find images that have most 3D points with given image in common
    local_bundle = mapper.find_local_bundle(mapper_options, image_id)

    # Do the bundle adjustment only if there is any connected images
    if len(local_bundle) > 0:
        ba_config = pycolmap.BundleAdjustmentConfig()
        ba_config.add_image(image_id)
        for local_image_id in local_bundle:
            ba_config.add_image(local_image_id)

        # Fix the existing images, if options specified
        if mapper_options.fix_existing_images:
            for local_image_id in local_bundle:
                if local_image_id in mapper.existing_image_ids:
                    ba_config.set_constant_cam_pose(local_image_id)

        # Determine which cameras to fix, when not all the registered images are within the current local bundle.
        num_images_per_camera = {}
        for image_id in ba_config.image_ids:
            image = reconstruction.images[image_id]
            if image.camera_id not in num_images_per_camera:
                num_images_per_camera[image.camera_id] = 0
            num_images_per_camera[image.camera_id] += 1
        for camera_id, num_images_local in num_images_per_camera.items():
            if num_images_local < mapper.num_reg_images_per_camera[camera_id]:
                ba_config.set_constant_cam_intrinsics(camera_id)

        # Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment
        if len(local_bundle) == 1:
            ba_config.set_constant_cam_pose(local_bundle[0])
            ba_config.set_constant_cam_positions(image_id, [0])
        elif len(local_bundle) > 1:
            image_id1, image_id2 = local_bundle[-1], local_bundle[-2]
            ba_config.set_constant_cam_pose(image_id1)
            if (not mapper_options.fix_existing_images) or (image_id2 not in mapper.existing_image_ids):
                ba_config.set_constant_cam_positions(image_id2, [0])

        # Make sure, we refine all new and short-track 3D points, no matter if
        # they are fully contained in the local image set or not. Do not include
        # long track 3D points as they are usually already very stable and adding
        # to them to bundle adjustment and track merging/completion would slow
        # down the local bundle adjustment significantly.
        variable_point3D_ids = set()
        for point3D_id in list(point3D_ids):
            point3D = reconstruction.point3D(point3D_id)
            kMaxTrackLength = 15
            if (point3D.error == -1.) or point3D.track.length() <= kMaxTrackLength:
                ba_config.add_variable_point(point3D_id)
                variable_point3D_ids.add(point3D_id)

        # Adjust the local bundle
        summary = solve_bundle_adjustment(mapper.get_reconstruction(), ba_options, ba_config)
        logging.info("Local Bundle Adjustment")
        logging.info(summary.BriefReport())

        report.num_adjusted_observations = int(summary.num_residuals / 2)
        # Merge refined tracks with other existing points
        report.num_merged_observations = mapper.get_triangulator().merge_tracks(tri_options, variable_point3D_ids)
        # Complete tracks that may have failed to triangulate before refinement
        # of camera pose and calibration in bundle adjustment. This may avoid that some points are filtered and it helps for subsequent image registrations
        report.num_completed_observations = mapper.get_triangulator().complete_tracks(tri_options, variable_point3D_ids)
        report.num_completed_observations += mapper.get_triangulator().complete_image(tri_options, image_id)

    filter_image_ids = set()
    filter_image_ids.add(image_id)
    filter_image_ids.update(local_bundle)
    report.num_filtered_observations = reconstruction.filter_points3D_in_images(mapper_options.filter_max_reproj_error, mapper_options.filter_min_tri_angle, filter_image_ids)
    report.num_filtered_observations += reconstruction.filter_points3D(mapper_options.filter_max_reproj_error, mapper_options.filter_min_tri_angle, point3D_ids)
    return report

def iterative_local_refinement(mapper, max_num_refinements, max_refinement_change, mapper_options, ba_options, tri_options, image_id):
    """Equivalent to mapper.iterative_local_refinement(...)"""
    ba_options_tmp = copy.deepcopy(ba_options)
    for i in range(max_num_refinements):
        # report = mapper.adjust_local_bundle(mapper_options, ba_options_tmp, tri_options, image_id, mapper.get_modified_points3D())
        report = adjust_local_bundle(mapper, mapper_options, ba_options_tmp, tri_options, image_id, mapper.get_modified_points3D())
        logging.verbose(1, "=> Merged observations: {0}".format(report.num_merged_observations))
        logging.verbose(1, "=> Completed observations: {0}".format(report.num_completed_observations))
        logging.verbose(1, "=> Filtered observations: {0}".format(report.num_filtered_observations))
        changed = 0
        if report.num_adjusted_observations != 0:
            changed = (report.num_merged_observations + report.num_completed_observations + report.num_filtered_observations) / report.num_adjusted_observations
        logging.verbose(1, "=> Changed observations: {0:.6f}".format(changed))
        if changed < max_refinement_change:
            break

        # Only use robust cost function for first iteration
        ba_options_tmp.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
    mapper.clear_modified_points3D()

