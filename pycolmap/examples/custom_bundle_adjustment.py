"""
Python reimplementation of the bundle adjustment for the incremental mapper of
C++ with equivalent logic. As a result, one can add customized residuals on top
of the exposed ceres problem from conventional bundle adjustment.
pyceres is needed as a dependency for this file.
"""

import copy

import pyceres

import pycolmap
from pycolmap import logging


class PyBundleAdjuster:
    # Python implementation of COLMAP bundle adjuster with pyceres
    def __init__(
        self,
        options: pycolmap.BundleAdjustmentOptions,
        config: pycolmap.BundleAdjustmentConfig,
    ):
        self.options = options
        self.config = config
        self.problem = pyceres.Problem()
        self.summary = pyceres.SolverSummary()
        self.camera_ids = set()
        self.point3D_num_observations = dict()

    def solve(self, reconstruction: pycolmap.Reconstruction):
        loss = self.options.create_loss_function()
        self.set_up_problem(reconstruction, loss)
        if self.problem.num_residuals() == 0:
            return False
        solver_options = self.set_up_solver_options(
            self.problem, self.options.solver_options
        )
        pyceres.solve(solver_options, self.problem, self.summary)
        return True

    def set_up_problem(
        self,
        reconstruction: pycolmap.Reconstruction,
        loss: pyceres.LossFunction,
    ):
        assert reconstruction is not None
        self.problem = pyceres.Problem()
        for image_id in self.config.image_ids:
            self.add_image_to_problem(image_id, reconstruction, loss)
        for point3D_id in self.config.variable_point3D_ids:
            self.add_point_to_problem(point3D_id, reconstruction, loss)
        for point3D_id in self.config.constant_point3D_ids:
            self.add_point_to_problem(point3D_id, reconstruction, loss)
        self.parameterize_cameras(reconstruction)
        self.parameterize_points(reconstruction)
        return self.problem

    def set_up_solver_options(
        self, problem: pyceres.Problem, solver_options: pyceres.SolverOptions
    ):
        bundle_adjuster = pycolmap.BundleAdjuster(self.options, self.config)
        return bundle_adjuster.set_up_solver_options(problem, solver_options)

    def add_image_to_problem(
        self,
        image_id: int,
        reconstruction: pycolmap.Reconstruction,
        loss: pyceres.LossFunction,
    ):
        image = reconstruction.images[image_id]
        pose = image.cam_from_world
        camera = reconstruction.cameras[image.camera_id]
        constant_cam_pose = (
            not self.options.refine_extrinsics
        ) or self.config.has_constant_cam_pose(image.image_id)
        num_observations = 0
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            num_observations += 1
            if point2D.point3D_id not in self.point3D_num_observations:
                self.point3D_num_observations[point2D.point3D_id] = 0
            self.point3D_num_observations[point2D.point3D_id] += 1
            point3D = reconstruction.points3D[point2D.point3D_id]
            assert point3D.track.length() > 1
            if constant_cam_pose:
                cost = pycolmap.cost_functions.ReprojErrorCost(
                    camera.model, pose, point2D.xy
                )
                self.problem.add_residual_block(
                    cost, loss, [point3D.xyz, camera.params]
                )
            else:
                cost = pycolmap.cost_functions.ReprojErrorCost(
                    camera.model, point2D.xy
                )
                self.problem.add_residual_block(
                    cost,
                    loss,
                    [
                        pose.rotation.quat,
                        pose.translation,
                        point3D.xyz,
                        camera.params,
                    ],
                )
        if num_observations > 0:
            self.camera_ids.add(image.camera_id)
            # Set pose parameterization
            if not constant_cam_pose:
                self.problem.set_manifold(
                    pose.rotation.quat, pyceres.QuaternionManifold()
                )
                if self.config.has_constant_cam_positions(image_id):
                    constant_position_idxs = self.config.constant_cam_positions(
                        image_id
                    )
                    self.problem.set_manifold(
                        pose.translation,
                        pyceres.SubsetManifold(3, constant_position_idxs),
                    )

    def add_point_to_problem(
        self,
        point3D_id: int,
        reconstruction: pycolmap.Reconstruction,
        loss: pyceres.LossFunction,
    ):
        point3D = reconstruction.points3D[point3D_id]
        if point3D_id in self.point3D_num_observations:
            if (
                self.point3D_num_observations[point3D_id]
                == point3D.track.length()
            ):
                return
        else:
            self.point3D_num_observations[point3D_id] = 0
        for track_el in point3D.track.elements:
            if self.config.has_image(track_el.image_id):
                continue
            self.point3D_num_observations[point3D_id] += 1
            image = reconstruction.images[track_el.image_id]
            camera = reconstruction.cameras[image.camera_id]
            point2D = image.point2D(track_el.point2D_idx)
            if image.camera_id not in self.camera_ids:
                self.camera_ids.add(image.camera_id)
                self.config.set_constant_cam_intrinsics(image.camera_id)
            cost = pycolmap.cost_functions.ReprojErrorCost(
                camera.model, image.cam_from_world, point2D.xy
            )
            self.problem.add_residual_block(
                cost, loss, [point3D.xyz, camera.params]
            )

    def parameterize_cameras(self, reconstruction: pycolmap.Reconstruction):
        constant_camera = (
            (not self.options.refine_focal_length)
            and (not self.options.refine_principal_point)
            and (not self.options.refine_extra_params)
        )
        for camera_id in self.camera_ids:
            camera = reconstruction.cameras[camera_id]
            if constant_camera or self.config.has_constant_cam_intrinsics(
                camera_id
            ):
                self.problem.set_parameter_block_constant(camera.params)
                continue
            const_camera_params = []
            if not self.options.refine_focal_length:
                const_camera_params.extend(camera.focal_length_idxs())
            if not self.options.refine_principal_point:
                const_camera_params.extend(camera.principal_point_idxs())
            if not self.options.refine_extra_params:
                const_camera_params.extend(camera.extra_point_idxs())
            if len(const_camera_params) > 0:
                self.problem.set_manifold(
                    camera.params,
                    pyceres.SubsetManifold(
                        len(camera.params), const_camera_params
                    ),
                )

    def parameterize_points(self, reconstruction: pycolmap.Reconstruction):
        for (
            point3D_id,
            num_observations,
        ) in self.point3D_num_observations.items():
            point3D = reconstruction.points3D[point3D_id]
            if point3D.track.length() > num_observations:
                self.problem.set_parameter_block_constant(point3D.xyz)
        for point3D_id in self.config.constant_point3D_ids:
            point3D = reconstruction.points3D[point3D_id]
            self.problem.set_parameter_block_constant(point3D.xyz)


def solve_bundle_adjustment(reconstruction, ba_options, ba_config):
    bundle_adjuster = pycolmap.BundleAdjuster(ba_options, ba_config)
    # alternative equivalent python-based bundle adjustment (slower):
    # bundle_adjuster = PyBundleAdjuster(ba_options, ba_config)
    bundle_adjuster.set_up_problem(
        reconstruction, ba_options.create_loss_function()
    )
    solver_options = bundle_adjuster.set_up_solver_options(
        bundle_adjuster.problem, ba_options.solver_options
    )
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, bundle_adjuster.problem, summary)
    return summary


def adjust_global_bundle(mapper, mapper_options, ba_options):
    """Equivalent to mapper.adjust_global_bundle(...)"""
    reconstruction = mapper.reconstruction
    assert reconstruction is not None
    reg_image_ids = reconstruction.reg_image_ids()
    if len(reg_image_ids) < 2:
        logging.fatal("At least two images must be registered for global BA")
    ba_options_tmp = copy.deepcopy(ba_options)

    # Use stricter convergence criteria for first registered images
    if len(reg_image_ids) < 10:  # kMinNumRegImagesForFastBA = 10
        ba_options_tmp.solver_options.function_tolerance /= 10
        ba_options_tmp.solver_options.gradient_tolerance /= 10
        ba_options_tmp.solver_options.parameter_tolerance /= 10
        ba_options_tmp.solver_options.max_num_iterations *= 2
        ba_options_tmp.solver_options.max_linear_solver_iterations = 200

    # Avoid degeneracies in bundle adjustment
    mapper.observation_manager.filter_observations_with_negative_depth()

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
    if (not mapper_options.fix_existing_images) or (
        reg_image_ids[1] not in mapper.existing_image_ids
    ):
        ba_config.set_constant_cam_positions(reg_image_ids[1], [0])

    # Run bundle adjustment
    summary = solve_bundle_adjustment(reconstruction, ba_options_tmp, ba_config)
    logging.info("Global Bundle Adjustment")
    logging.info(summary.BriefReport())


def iterative_global_refinement(
    mapper,
    max_num_refinements,
    max_refinement_change,
    mapper_options,
    ba_options,
    tri_options,
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
        # mapper.adjust_global_bundle(mapper_options, ba_options)
        adjust_global_bundle(mapper, mapper_options, ba_options)
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


def adjust_local_bundle(
    mapper, mapper_options, ba_options, tri_options, image_id, point3D_ids
):
    """Equivalent to mapper.adjust_local_bundle(...)"""
    reconstruction = mapper.reconstruction
    assert reconstruction is not None
    report = pycolmap.LocalBundleAdjustmentReport()

    # Find images that have most 3D points with given image in common
    local_bundle = mapper.find_local_bundle(mapper_options, image_id)

    # Do the bundle adjustment only if there is any connected images
    if local_bundle:
        ba_config = pycolmap.BundleAdjustmentConfig()
        ba_config.add_image(image_id)
        for local_image_id in local_bundle:
            ba_config.add_image(local_image_id)

        # Fix the existing images, if options specified
        if mapper_options.fix_existing_images:
            for local_image_id in local_bundle:
                if local_image_id in mapper.existing_image_ids:
                    ba_config.set_constant_cam_pose(local_image_id)

        # Determine which cameras to fix, when not all the registered images
        # are within the current local bundle.
        num_images_per_camera = {}
        for image_id in ba_config.image_ids:
            image = reconstruction.images[image_id]
            if image.camera_id not in num_images_per_camera:
                num_images_per_camera[image.camera_id] = 0
            num_images_per_camera[image.camera_id] += 1
        for camera_id, num_images_local in num_images_per_camera.items():
            if num_images_local < mapper.num_reg_images_per_camera[camera_id]:
                ba_config.set_constant_cam_intrinsics(camera_id)

        # Fix 7 DOF to avoid scale/rotation/translation drift in BA.
        if len(local_bundle) == 1:
            ba_config.set_constant_cam_pose(local_bundle[0])
            ba_config.set_constant_cam_positions(image_id, [0])
        elif len(local_bundle) > 1:
            image_id1, image_id2 = local_bundle[-1], local_bundle[-2]
            ba_config.set_constant_cam_pose(image_id1)
            if (not mapper_options.fix_existing_images) or (
                image_id2 not in mapper.existing_image_ids
            ):
                ba_config.set_constant_cam_positions(image_id2, [0])

        # Make sure, we refine all new and short-track 3D points, no matter if
        # they are fully contained in the local image set or not. Do not include
        # long track 3D points as they are usually already very stable and
        # adding to them to bundle adjustment and track merging/completion would
        # slow down the local bundle adjustment significantly.
        variable_point3D_ids = set()
        for point3D_id in list(point3D_ids):
            point3D = reconstruction.point3D(point3D_id)
            kMaxTrackLength = 15
            if (
                point3D.error == -1.0
            ) or point3D.track.length() <= kMaxTrackLength:
                ba_config.add_variable_point(point3D_id)
                variable_point3D_ids.add(point3D_id)

        # Adjust the local bundle
        summary = solve_bundle_adjustment(
            mapper.reconstruction, ba_options, ba_config
        )
        logging.info("Local Bundle Adjustment")
        logging.info(summary.BriefReport())

        report.num_adjusted_observations = int(summary.num_residuals / 2)
        # Merge refined tracks with other existing points
        report.num_merged_observations = mapper.triangulator.merge_tracks(
            tri_options, variable_point3D_ids
        )
        # Complete tracks that may have failed to triangulate before refinement
        # of camera pose and calibration in bundle adjustment. This may avoid
        # that some points are filtered and helps for subsequent image
        # registrations.
        report.num_completed_observations = mapper.triangulator.complete_tracks(
            tri_options, variable_point3D_ids
        )
        report.num_completed_observations += mapper.triangulator.complete_image(
            tri_options, image_id
        )

    filter_image_ids = {image_id, *local_bundle}
    report.num_filtered_observations = (
        mapper.observation_manager.filter_points3D_in_images(
            mapper_options.filter_max_reproj_error,
            mapper_options.filter_min_tri_angle,
            filter_image_ids,
        )
    )
    report.num_filtered_observations += (
        mapper.observation_manager.filter_points3D(
            mapper_options.filter_max_reproj_error,
            mapper_options.filter_min_tri_angle,
            point3D_ids,
        )
    )
    return report


def iterative_local_refinement(
    mapper,
    max_num_refinements,
    max_refinement_change,
    mapper_options,
    ba_options,
    tri_options,
    image_id,
):
    """Equivalent to mapper.iterative_local_refinement(...)"""
    ba_options_tmp = copy.deepcopy(ba_options)
    for _ in range(max_num_refinements):
        # report = mapper.adjust_local_bundle(
        #     mapper_options,
        #     ba_options_tmp,
        #     tri_options,
        #     image_id,
        #     mapper.get_modified_points3D(),
        # )
        report = adjust_local_bundle(
            mapper,
            mapper_options,
            ba_options_tmp,
            tri_options,
            image_id,
            mapper.get_modified_points3D(),
        )
        logging.verbose(
            1, f"=> Merged observations: {report.num_merged_observations}"
        )
        logging.verbose(
            1, f"=> Completed observations: {report.num_completed_observations}"
        )
        logging.verbose(
            1, f"=> Filtered observations: {report.num_filtered_observations}"
        )
        changed = 0
        if report.num_adjusted_observations > 0:
            changed = (
                report.num_merged_observations
                + report.num_completed_observations
                + report.num_filtered_observations
            ) / report.num_adjusted_observations
        logging.verbose(1, f"=> Changed observations: {changed:.6f}")
        if changed < max_refinement_change:
            break

        # Only use robust cost function for first iteration
        ba_options_tmp.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
    mapper.clear_modified_points3D()
