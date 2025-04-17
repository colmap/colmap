"""
Python reimplementation of the C++ incremental mapper with equivalent logic.
"""

import argparse
import time
from pathlib import Path

import custom_bundle_adjustment
import enlighten

import pycolmap
from pycolmap import logging


def write_snapshot(reconstruction, snapshot_path):
    logging.info("Creating snapshot")
    timestamp = time.time() * 1000
    path = snapshot_path / f"{timestamp:010d}"
    path.mkdir(exist_ok=True, parents=True)
    logging.verbose(1, f"=> Writing to {path}")
    reconstruction.write(path)


def iterative_global_refinement(options, mapper_options, mapper):
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
    controller, mapper, mapper_options, reconstruction
):
    """Equivalent to IncrementalPipeline.initialize_reconstruction(...)"""
    options = controller.options
    init_pair = (options.init_image_id1, options.init_image_id2)

    # Try to find good initial pair
    if not options.is_initial_pair_provided():
        logging.info("Finding good initial image pair")
        ret = mapper.find_initial_image_pair(mapper_options, *init_pair)
        if ret is None:
            logging.info("No good initial image pair found.")
            return pycolmap.IncrementalMapperStatus.NO_INITIAL_PAIR
        init_pair, two_view_geometry = ret
    else:
        if not all(reconstruction.exists_image(i) for i in init_pair):
            logging.info(f"=> Initial image pair {init_pair} does not exist.")
            return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR
        two_view_geometry = mapper.estimate_initial_two_view_geometry(
            mapper_options, *init_pair
        )
        if two_view_geometry is None:
            logging.info("Provided pair is insuitable for initialization")
            return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR

    logging.info(
        f"Registering initial image pair #{init_pair[0]} and #{init_pair[1]}"
    )
    mapper.register_initial_image_pair(
        mapper_options, two_view_geometry, *init_pair
    )
    for image_id in init_pair:
        for data_id in reconstruction.images[image_id].frame.data_ids:
            if data_id.sensor_id.type == pycolmap.SensorType.CAMERA:
                mapper.triangulate_image(
                    options.get_triangulation(), data_id.id
                )

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
        return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR
    if options.extract_colors:
        reconstruction.extract_colors_for_all_images(controller.image_path)
    return pycolmap.IncrementalMapperStatus.SUCCESS


def reconstruct_sub_model(controller, mapper, mapper_options, reconstruction):
    """Equivalent to IncrementalPipeline.reconstruct_sub_model(...)"""
    # register initial pair
    mapper.begin_reconstruction(reconstruction)
    if reconstruction.num_reg_frames() == 0:
        init_status = initialize_reconstruction(
            controller, mapper, mapper_options, reconstruction
        )
        if init_status != pycolmap.IncrementalMapperStatus.SUCCESS:
            return init_status
        controller.callback(
            pycolmap.IncrementalMapperCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK
        )

    # incremental mapping
    options = controller.options
    snapshot_prev_num_reg_frames = reconstruction.num_reg_frames()
    ba_prev_num_reg_frames = reconstruction.num_reg_frames()
    ba_prev_num_points = reconstruction.num_points3D()
    reg_next_success, prev_reg_next_success = True, True
    while True:
        if not (reg_next_success or prev_reg_next_success):
            break
        prev_reg_next_success = reg_next_success
        reg_next_success = False
        next_images = mapper.find_next_images(mapper_options)
        if len(next_images) == 0:
            break
        for reg_trial in range(len(next_images)):
            next_image_id = next_images[reg_trial]
            logging.info(
                f"Registering image #{next_image_id} "
                f"(num_reg_frames={reconstruction.num_reg_frames() + 1})"
            )
            num_vis = mapper.observation_manager.num_visible_points3D(
                next_image_id
            )
            num_obs = mapper.observation_manager.num_observations(next_image_id)
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
                and reconstruction.num_reg_frames() < options.min_model_size
            ):
                break
        if reg_next_success:
            for data_id in reconstruction.images[next_image_id].frame.data_ids:
                if data_id.sensor_id.type == pycolmap.SensorType.CAMERA:
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
            if (
                options.extract_colors
                and not reconstruction.extract_colors_for_image(
                    next_image_id, controller.image_path
                )
            ):
                logging.warning(
                    f"Could not read image {next_image_id} "
                    f"at path {controller.image_path}"
                )
            if (
                options.snapshot_frames_freq > 0
                and reconstruction.num_reg_frames()
                >= options.snapshot_frames_freq + snapshot_prev_num_reg_frames
            ):
                snapshot_prev_num_reg_frames = reconstruction.num_reg_frames()
                write_snapshot(reconstruction, Path(options.snapshot_path))
            controller.callback(
                pycolmap.IncrementalMapperCallback.NEXT_IMAGE_REG_CALLBACK
            )
        if mapper.num_shared_reg_images() >= int(options.max_model_overlap):
            break
        if (not reg_next_success) and prev_reg_next_success:
            iterative_global_refinement(options, mapper_options, mapper)
    if (
        reconstruction.num_reg_frames() >= 2
        and reconstruction.num_reg_frames() != ba_prev_num_reg_frames
        and reconstruction.num_points3D != ba_prev_num_points
    ):
        iterative_global_refinement(options, mapper_options, mapper)
    return pycolmap.IncrementalMapperStatus.SUCCESS


def reconstruct(controller, mapper, mapper_options, continue_reconstruction):
    """Equivalent to IncrementalPipeline.reconstruct(...)"""
    options = controller.options

    database_cache = controller.database_cache
    reconstruction_manager = controller.reconstruction_manager

    for num_trials in range(options.init_num_trials):
        if not continue_reconstruction or num_trials > 0:
            reconstruction_idx = reconstruction_manager.add()
        else:
            reconstruction_idx = 0

        reconstruction = reconstruction_manager.get(reconstruction_idx)
        status = reconstruct_sub_model(
            controller, mapper, mapper_options, reconstruction
        )
        if status == pycolmap.IncrementalMapperStatus.INTERRUPTED:
            logging.info("Keeping reconstruction due to interrupt")
            mapper.end_reconstruction(False)
        elif status == pycolmap.IncrementalMapperStatus.NO_INITIAL_PAIR:
            logging.info("Disacarding reconstruction due to no initial pair")
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            return
        elif status == pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR:
            logging.info("Disacarding reconstruction due to bad initial pair")
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
        elif status == pycolmap.IncrementalMapperStatus.SUCCESS:
            total_num_reg_frames = mapper.num_total_reg_images()
            min_model_size = min(
                0.8 * database_cache.num_images(), options.min_model_size
            )
            if (
                options.multiple_models
                and reconstruction_manager.size() > 1
                and (
                    reconstruction.num_reg_frames() < min_model_size
                    or reconstruction.num_reg_frames() == 0
                )
            ):
                logging.info(
                    "Discarding reconstruction due to insufficient size"
                )
                mapper.end_reconstruction(True)
                reconstruction_manager.delete(reconstruction_idx)
            else:
                logging.info("Keeping successful reconstruction")
                mapper.end_reconstruction(False)
            controller.callback(
                pycolmap.IncrementalMapperCallback.LAST_IMAGE_REG_CALLBACK
            )
            if (
                not options.multiple_models
                or reconstruction_manager.size() >= options.max_num_models
                or total_num_reg_frames >= database_cache.num_images() - 1
            ):
                return
        else:
            logging.fatal(f"Unknown reconstruction status: {status}")


def main_incremental_mapper(controller):
    """Equivalent to IncrementalPipeline.run()"""
    timer = pycolmap.Timer()
    timer.start()
    if not controller.load_database():
        return

    reconstruction_manager = controller.reconstruction_manager

    continue_reconstruction = reconstruction_manager.size() > 0
    if reconstruction_manager.size() > 1:
        logging.fatal(
            "Can only resume from a single reconstruction, "
            "but multiple are given"
        )

    database_cache = controller.database_cache
    mapper = pycolmap.IncrementalMapper(database_cache)
    mapper_options = controller.options.get_mapper()
    reconstruct(controller, mapper, mapper_options, continue_reconstruction)

    for _ in range(2):  # number of relaxations
        if mapper.num_total_reg_images() == database_cache.num_images():
            break

        logging.info("=> Relaxing the initialization constraints")
        mapper_options.init_min_num_inliers = int(
            mapper_options.init_min_num_inliers / 2
        )
        mapper.reset_initialization_stats()
        reconstruct(
            controller, mapper, mapper_options, continue_reconstruction=False
        )

        if mapper.num_total_reg_images() == database_cache.num_images():
            break

        logging.info("=> Relaxing the initialization constraints")
        mapper_options.init_min_tri_angle /= 2
        mapper.reset_initialization_stats()
        reconstruct(
            controller, mapper, mapper_options, continue_reconstruction=False
        )
    timer.print_minutes()


def main(
    database_path,
    image_path,
    output_path,
    options=None,
    input_path=None,
):
    if options is None:
        options = pycolmap.IncrementalPipelineOptions()
    if not database_path.exists():
        logging.fatal(f"Database path does not exist: {database_path}")
    if not image_path.exists():
        logging.fatal(f"Image path does not exist: {image_path}")
    output_path.mkdir(exist_ok=True, parents=True)
    reconstruction_manager = pycolmap.ReconstructionManager()
    if input_path is not None and input_path != "":
        reconstruction_manager.read(input_path)
    mapper = pycolmap.IncrementalPipeline(
        options, image_path, database_path, reconstruction_manager
    )

    # main runner
    num_images = pycolmap.Database(database_path).num_images
    with enlighten.Manager() as manager:
        with manager.counter(
            total=num_images, desc="Images registered:"
        ) as pbar:
            pbar.update(0, force=True)
            mapper.add_callback(
                pycolmap.IncrementalMapperCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK,
                lambda: pbar.update(2),
            )
            mapper.add_callback(
                pycolmap.IncrementalMapperCallback.NEXT_IMAGE_REG_CALLBACK,
                lambda: pbar.update(1),
            )
            main_incremental_mapper(mapper)

    # write and output
    reconstruction_manager.write(output_path)
    reconstructions = {}
    for i in range(reconstruction_manager.size()):
        reconstructions[i] = reconstruction_manager.get(i)
    return reconstructions


def parse_args():
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
