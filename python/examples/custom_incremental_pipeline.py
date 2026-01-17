"""
Python reimplementation of the C++ incremental mapper with equivalent logic.
"""

import argparse
import time
from pathlib import Path

import custom_bundle_adjustment
import enlighten

import pycolmap
from pycolmap import (
    IncrementalMapper,
    IncrementalMapperOptions,
    IncrementalPipeline,
    IncrementalPipelineCallback,
    IncrementalPipelineOptions,
    IncrementalPipelineStatus,
    Reconstruction,
    ReconstructionManager,
    logging,
)


def write_snapshot(reconstruction: Reconstruction, snapshot_path: Path) -> None:
    logging.info("Creating snapshot")
    timestamp = time.time() * 1000
    path = snapshot_path / f"{timestamp:010d}"
    path.mkdir(exist_ok=True, parents=True)
    logging.verbose(1, f"=> Writing to {path}")
    reconstruction.write(path)


def has_unknown_sensor_from_rig(
    reconstruction: Reconstruction,
) -> bool:
    parameterized_rig_ids = set()
    for image in reconstruction.images.values():
        parameterized_rig_ids.add(image.frame.rig_id)
    for rig_id in parameterized_rig_ids:
        rig = reconstruction.rig(rig_id)
        for sensor_id, sensor_from_rig in rig.non_ref_sensors.items():
            if (
                sensor_id.type == pycolmap.SensorType.CAMERA
                and sensor_from_rig is None
            ):
                return True
    return False


def iterative_global_refinement(
    options: IncrementalPipelineOptions,
    mapper_options: IncrementalMapperOptions,
    mapper: IncrementalMapper,
) -> None:
    logging.info("Retriangulation and Global bundle adjustment")
    # The following is equivalent to mapper.iterative_global_refinement(...)
    custom_bundle_adjustment.iterative_global_refinement(
        mapper,
        options.ba_global_max_refinements,
        options.ba_global_max_refinement_change,
        mapper_options,
        options.get_global_bundle_adjustment(),
        options.get_triangulation(),
    )
    mapper.filter_frames(mapper_options)


def initialize_reconstruction(
    controller: IncrementalPipeline,
    mapper: IncrementalMapper,
    mapper_options: IncrementalMapperOptions,
    reconstruction: Reconstruction,
) -> IncrementalPipelineStatus:
    """Equivalent to IncrementalPipeline.initialize_reconstruction(...)"""
    options = controller.options
    init_pair = (options.init_image_id1, options.init_image_id2)

    # Try to find good initial pair
    if not options.is_initial_pair_provided():
        logging.info("Finding good initial image pair")
        ret = mapper.find_initial_image_pair(mapper_options, *init_pair)
        if ret is None:
            logging.info("No good initial image pair found.")
            return IncrementalPipelineStatus.NO_INITIAL_PAIR
        init_pair, init_cam2_from_cam1 = ret
    else:
        if not all(reconstruction.exists_image(i) for i in init_pair):
            logging.info(f"=> Initial image pair {init_pair} does not exist.")
            return IncrementalPipelineStatus.NO_INITIAL_PAIR
        maybe_init_cam2_from_cam1 = mapper.estimate_initial_two_view_geometry(
            mapper_options, *init_pair
        )
        if maybe_init_cam2_from_cam1 is None:
            logging.info("Provided pair is unsuitable for initialization")
            return IncrementalPipelineStatus.BAD_INITIAL_PAIR
        init_cam2_from_cam1 = maybe_init_cam2_from_cam1
    logging.info(
        f"Registering initial image pair #{init_pair[0]} and #{init_pair[1]}"
    )
    mapper.register_initial_image_pair(
        mapper_options, *init_pair, init_cam2_from_cam1
    )

    tri_options = options.get_triangulation()
    tri_options.min_angle = mapper_options.init_min_tri_angle
    for image_id in init_pair:
        image = reconstruction.images[image_id]
        assert image.frame is not None
        for data_id in image.frame.image_ids:
            mapper.triangulate_image(tri_options, data_id.id)

    logging.info("Global bundle adjustment")
    # The following is equivalent to: mapper.adjust_global_bundle(...)
    custom_bundle_adjustment.adjust_global_bundle(
        mapper, mapper_options, options.get_global_bundle_adjustment()
    )
    reconstruction.normalize()
    mapper.filter_points(mapper_options)
    mapper.filter_frames(mapper_options)

    # Initial image pair failed to register
    if (
        reconstruction.num_reg_frames() == 0
        or reconstruction.num_points3D() == 0
    ):
        return IncrementalPipelineStatus.BAD_INITIAL_PAIR
    if options.extract_colors:
        reconstruction.extract_colors_for_all_images(options.image_path)
    return IncrementalPipelineStatus.SUCCESS


def reconstruct_sub_model(
    controller: IncrementalPipeline,
    mapper: IncrementalMapper,
    mapper_options: IncrementalMapperOptions,
    reconstruction: Reconstruction,
) -> IncrementalPipelineStatus:
    """Equivalent to IncrementalPipeline.reconstruct_sub_model(...)"""
    mapper.begin_reconstruction(reconstruction)

    if has_unknown_sensor_from_rig(reconstruction):
        return IncrementalPipelineStatus.UNKNOWN_SENSOR_FROM_RIG

    if reconstruction.num_reg_frames() == 0:
        init_status = initialize_reconstruction(
            controller, mapper, mapper_options, reconstruction
        )
        if init_status != IncrementalPipelineStatus.SUCCESS:
            return init_status
    controller.callback(
        IncrementalPipelineCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK
    )

    options = controller.options

    structure_less_flags = []
    if options.structure_less_registration_only:
        structure_less_flags = [True]
    else:
        if options.structure_less_registration_fallback:
            structure_less_flags = [False, True]
        else:
            structure_less_flags = [False]

    snapshot_prev_num_reg_frames = reconstruction.num_reg_frames()
    ba_prev_num_reg_frames = reconstruction.num_reg_frames()
    ba_prev_num_points = reconstruction.num_points3D()
    reg_next_success, prev_reg_next_success = True, True
    while True:
        if not (reg_next_success or prev_reg_next_success):
            break
        if controller.check_reached_max_runtime():
            break
        prev_reg_next_success = reg_next_success
        reg_next_success = False
        next_image_id = None
        for structure_less in structure_less_flags:
            next_images = mapper.find_next_images(
                mapper_options, structure_less=structure_less
            )
            for reg_trial, next_image_id in enumerate(next_images):
                logging.info(
                    f"Registering image #{next_image_id} "
                    f"(num_reg_frames={reconstruction.num_reg_frames()})"
                )
                if structure_less:
                    logging.info(
                        "Registering image with structure-less fallback"
                    )
                    num_vis = (
                        mapper.observation_manager.num_visible_correspondences(
                            next_image_id
                        )
                    )
                    num_corrs = mapper.observation_manager.num_correspondences(
                        next_image_id
                    )
                    logging.info(
                        f"=> Image sees {num_vis} / {num_corrs} correspondences"
                    )
                    reg_next_success = (
                        mapper.register_next_structure_less_image(
                            mapper_options, next_image_id
                        )
                    )
                else:
                    num_vis = mapper.observation_manager.num_visible_points3D(
                        next_image_id
                    )
                    num_obs = mapper.observation_manager.num_observations(
                        next_image_id
                    )
                    logging.info(f"=> Image sees {num_vis} / {num_obs} points")
                    reg_next_success = mapper.register_next_image(
                        mapper_options, next_image_id
                    )
                if reg_next_success:
                    break
                else:
                    logging.info("=> Could not register, trying another image.")
                # If initial pair fails to continue for some time,
                # abort and try different initial pair.
                kMinNumInitialRegTrials = 30
                if (
                    reg_trial >= kMinNumInitialRegTrials
                    and reconstruction.num_reg_images() < options.min_model_size
                ):
                    break
            if reg_next_success:
                break
        if reg_next_success and next_image_id is not None:
            image = reconstruction.images[next_image_id]
            assert image.frame is not None
            for data_id in image.frame.image_ids:
                mapper.triangulate_image(
                    options.get_triangulation(), data_id.id
                )
            # This is equivalent to mapper.iterative_local_refinement(...)
            custom_bundle_adjustment.iterative_local_refinement(
                mapper,
                options.ba_local_max_refinements,
                options.ba_local_max_refinement_change,
                mapper_options,
                options.get_local_bundle_adjustment(),
                options.get_triangulation(),
                next_image_id,
            )
            if controller.check_run_global_refinement(
                reconstruction, ba_prev_num_reg_frames, ba_prev_num_points
            ):
                iterative_global_refinement(options, mapper_options, mapper)
                ba_prev_num_points = reconstruction.num_points3D()
                ba_prev_num_reg_frames = reconstruction.num_reg_frames()
            if options.extract_colors:
                for data_id in image.frame.image_ids:
                    if not reconstruction.extract_colors_for_image(
                        data_id.id, options.image_path
                    ):
                        logging.warning(
                            f"Could not read image "
                            f"{reconstruction.images[data_id.id].name} "
                            f"at path {options.image_path}"
                        )
            if (
                options.snapshot_frames_freq > 0
                and reconstruction.num_reg_frames()
                >= options.snapshot_frames_freq + snapshot_prev_num_reg_frames
            ):
                snapshot_prev_num_reg_frames = reconstruction.num_reg_frames()
                write_snapshot(reconstruction, Path(options.snapshot_path))
            controller.callback(
                IncrementalPipelineCallback.NEXT_IMAGE_REG_CALLBACK
            )
        if mapper.num_shared_reg_images() >= int(options.max_model_overlap):
            break
        if (not reg_next_success) and prev_reg_next_success:
            iterative_global_refinement(options, mapper_options, mapper)

    if controller.check_reached_max_runtime():
        return pycolmap.IncrementalPipelineStatus.INTERRUPTED

    # Only run final global BA, if last incremental BA was not global
    if (
        reconstruction.num_reg_frames() > 0
        and reconstruction.num_reg_frames() != ba_prev_num_reg_frames
        and reconstruction.num_points3D() != ba_prev_num_points
    ):
        iterative_global_refinement(options, mapper_options, mapper)
    return IncrementalPipelineStatus.SUCCESS


def reconstruct(
    controller: IncrementalPipeline,
    mapper: IncrementalMapper,
    mapper_options: IncrementalMapperOptions,
    continue_reconstruction: bool,
) -> IncrementalPipelineStatus:
    """Equivalent to IncrementalPipeline.reconstruct(...)"""
    options = controller.options

    database_cache = controller.database_cache
    reconstruction_manager = controller.reconstruction_manager

    for num_trials in range(options.init_num_trials):
        if controller.check_reached_max_runtime():
            break
        if not continue_reconstruction or num_trials > 0:
            reconstruction_idx = reconstruction_manager.add()
        else:
            reconstruction_idx = 0

        reconstruction = reconstruction_manager.get(reconstruction_idx)
        status = reconstruct_sub_model(
            controller, mapper, mapper_options, reconstruction
        )
        if status == IncrementalPipelineStatus.INTERRUPTED:
            reconstruction.update_point_3d_errors()
            logging.info("Keeping reconstruction due to interrupt")
            mapper.end_reconstruction(False)
            pycolmap.align_reconstruction_to_orig_rig_scales(
                database_cache.rigs, reconstruction
            )
        elif status == IncrementalPipelineStatus.UNKNOWN_SENSOR_FROM_RIG:
            logging.error(
                "Discarding reconstruction due to unknown sensor_from_rig "
                "poses. Either explicitly define the poses by configuring the "
                "rigs or first run reconstruction without configured rigs and "
                "then derive the poses from the initial reconstruction for a "
                "subsequent reconstruction with rig constraints. See "
                "documentation for detailed instructions."
            )
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            return IncrementalPipelineStatus.STOP
        elif status == IncrementalPipelineStatus.BAD_INITIAL_PAIR:
            logging.info("Disacarding reconstruction due to bad initial pair")
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
        elif status == IncrementalPipelineStatus.NO_INITIAL_PAIR:
            logging.info("Disacarding reconstruction due to no initial pair")
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            return IncrementalPipelineStatus.CONTINUE
        elif status == IncrementalPipelineStatus.SUCCESS:
            num_reg_images = reconstruction.num_reg_images()
            total_num_reg_images = mapper.num_total_reg_images()
            if (
                options.multiple_models
                and reconstruction_manager.size() > 1
                and num_reg_images < options.min_model_size
            ) or num_reg_images == 0:
                logging.info(
                    "Discarding reconstruction due to insufficient size"
                )
                mapper.end_reconstruction(True)
                reconstruction_manager.delete(reconstruction_idx)
            else:
                reconstruction.update_point_3d_errors()
                logging.info("Keeping successful reconstruction")
                mapper.end_reconstruction(False)
                pycolmap.align_reconstruction_to_orig_rig_scales(
                    database_cache.rigs, reconstruction
                )
            controller.callback(
                IncrementalPipelineCallback.LAST_IMAGE_REG_CALLBACK
            )
            if (
                not options.multiple_models
                or reconstruction_manager.size() >= options.max_num_models
                or total_num_reg_images >= database_cache.num_images() - 1
            ):
                return IncrementalPipelineStatus.STOP
        else:
            logging.fatal(f"Unknown reconstruction status: {status}")

    return IncrementalPipelineStatus.CONTINUE


def main_incremental_mapper(controller: IncrementalPipeline) -> None:
    """Equivalent to IncrementalPipeline.run()"""
    timer = pycolmap.Timer()
    timer.start()

    database_cache = controller.database_cache

    if database_cache.num_images() == 0:
        logging.warning("No images with matches found in the database")
        return

    if (
        controller.options.use_prior_position
        and database_cache.num_pose_priors() == 0
    ):
        logging.warning("No pose priors")
        return

    reconstruction_manager = controller.reconstruction_manager

    continue_reconstruction = reconstruction_manager.size() > 0
    if reconstruction_manager.size() > 1:
        logging.fatal(
            "Can only resume from a single reconstruction, "
            "but multiple are given"
        )

    num_images = database_cache.num_images()
    mapper = IncrementalMapper(database_cache)
    mapper_options = controller.options.get_mapper()
    if (
        reconstruct(controller, mapper, mapper_options, continue_reconstruction)
        == IncrementalPipelineStatus.STOP
    ):
        return

    def should_stop():
        return (
            mapper.num_total_reg_images() == num_images
            or controller.check_reached_max_runtime()
        )

    for _ in range(2):  # number of relaxations
        if should_stop():
            break

        logging.info("=> Relaxing the initialization constraints")
        mapper_options.init_min_num_inliers = int(
            mapper_options.init_min_num_inliers / 2
        )
        mapper.reset_initialization_stats()
        if (
            reconstruct(
                controller,
                mapper,
                mapper_options,
                continue_reconstruction=False,
            )
            == IncrementalPipelineStatus.STOP
        ):
            return

        if should_stop():
            break

        logging.info("=> Relaxing the initialization constraints")
        mapper_options.init_min_tri_angle /= 2
        mapper.reset_initialization_stats()
        if (
            reconstruct(
                controller,
                mapper,
                mapper_options,
                continue_reconstruction=False,
            )
            == IncrementalPipelineStatus.STOP
        ):
            return
    timer.print_minutes()


def main(
    database_path: Path,
    image_path: Path,
    output_path: Path,
    options: IncrementalPipelineOptions | None = None,
    input_path: Path | None = None,
) -> dict[int, Reconstruction]:
    if options is None:
        options = IncrementalPipelineOptions()
    options.image_path = image_path
    if not database_path.exists():
        logging.fatal(f"Database path does not exist: {database_path}")
    if not image_path.exists():
        logging.fatal(f"Image path does not exist: {image_path}")
    output_path.mkdir(exist_ok=True, parents=True)
    reconstruction_manager = ReconstructionManager()
    if input_path:
        reconstruction_manager.read(input_path)

    with pycolmap.Database.open(database_path) as database:
        mapper = IncrementalPipeline(options, database, reconstruction_manager)
        num_images = database.num_images()
        with enlighten.Manager() as manager:
            with manager.counter(
                total=num_images, desc="Images registered:"
            ) as pbar:
                pbar.update(0, force=True)
                mapper.add_callback(
                    IncrementalPipelineCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK,
                    lambda: pbar.update(2),
                )
                mapper.add_callback(
                    IncrementalPipelineCallback.NEXT_IMAGE_REG_CALLBACK,
                    lambda: pbar.update(1),
                )
                main_incremental_mapper(mapper)

    # write and output
    reconstruction_manager.write(output_path)
    reconstructions = {}
    for i in range(reconstruction_manager.size()):
        reconstructions[i] = reconstruction_manager.get(i)
    return reconstructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        database_path=Path(args.database_path),
        image_path=Path(args.image_path),
        input_path=Path(args.input_path) if args.input_path else None,
        output_path=Path(args.output_path),
    )
