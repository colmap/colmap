"""
Python reimplementation of the C++ incremental mapper with equivalent logic.
"""

import time
from pathlib import Path

import enlighten

import pycolmap
from pycolmap import logging
import custom_bundle_adjustment


def extract_colors(image_path, image_id, reconstruction):
    if not reconstruction.extract_colors_for_image(image_id, image_path):
        logging.warning(f"Could not read image {image_id} at path {image_path}")


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
    mapper.filter_images(mapper_options)


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
    logging.info(f"Initializing with image pair {init_pair}")
    mapper.register_initial_image_pair(
        mapper_options, two_view_geometry, *init_pair
    )
    logging.info("Global bundle adjustment")
    # The following is equivalent to: mapper.adjust_global_bundle(...)
    custom_bundle_adjustment.adjust_global_bundle(
        mapper, mapper_options, options.get_global_bundle_adjustment()
    )
    reconstruction.normalize()
    mapper.filter_points(mapper_options)
    mapper.filter_images(mapper_options)

    # Initial image pair failed to register
    if (
        reconstruction.num_reg_images() == 0
        or reconstruction.num_points3D() == 0
    ):
        return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR
    if options.extract_colors:
        extract_colors(controller.image_path, init_pair[0], reconstruction)
    return pycolmap.IncrementalMapperStatus.SUCCESS


def reconstruct_sub_model(controller, mapper, mapper_options, reconstruction):
    """Equivalent to IncrementalPipeline.reconstruct_sub_model(...)"""
    # register initial pair
    mapper.begin_reconstruction(reconstruction)
    if reconstruction.num_reg_images() == 0:
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
    snapshot_prev_num_reg_images = reconstruction.num_reg_images()
    ba_prev_num_reg_images = reconstruction.num_reg_images()
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
                f"({reconstruction.num_reg_images() + 1})"
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
                and reconstruction.num_reg_images() < options.min_model_size
            ):
                break
        if reg_next_success:
            mapper.triangulate_image(options.get_triangulation(), next_image_id)
            # The following is equivalent to mapper.iterative_local_refinement(...)
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
                reconstruction, ba_prev_num_reg_images, ba_prev_num_points
            ):
                iterative_global_refinement(options, mapper_options, mapper)
                ba_prev_num_points = reconstruction.num_points3D()
                ba_prev_num_reg_images = reconstruction.num_reg_images()
            if options.extract_colors:
                extract_colors(
                    controller.image_path, next_image_id, reconstruction
                )
            if (
                options.snapshot_images_freq > 0
                and reconstruction.num_reg_images()
                >= options.snapshot_images_freq + snapshot_prev_num_reg_images
            ):
                snapshot_prev_num_reg_images = reconstruction.num_reg_images()
                write_snapshot(reconstruction, Path(options.snapshot_path))
            controller.callback(
                pycolmap.IncrementalMapperCallback.NEXT_IMAGE_REG_CALLBACK
            )
        if mapper.num_shared_reg_images() >= int(options.max_model_overlap):
            break
        if (not reg_next_success) and prev_reg_next_success:
            iterative_global_refinement(options, mapper_options, mapper)
    if (
        reconstruction.num_reg_images() >= 2
        and reconstruction.num_reg_images() != ba_prev_num_reg_images
        and reconstruction.num_points3D != ba_prev_num_points
    ):
        iterative_global_refinement(options, mapper_options, mapper)
    return pycolmap.IncrementalMapperStatus.SUCCESS


def reconstruct(controller, mapper_options):
    """Equivalent to IncrementalPipeline.reconstruct(...)"""
    options = controller.options
    reconstruction_manager = controller.reconstruction_manager
    database_cache = controller.database_cache
    mapper = pycolmap.IncrementalMapper(database_cache)
    initial_reconstruction_given = reconstruction_manager.size() > 0
    if reconstruction_manager.size() > 1:
        logging.fatal(
            "Can only resume from a single reconstruction, but multiple are given"
        )
    for num_trials in range(options.init_num_trials):
        if (not initial_reconstruction_given) or num_trials > 0:
            reconstruction_idx = reconstruction_manager.add()
        else:
            reconstruction_idx = 0
        reconstruction = reconstruction_manager.get(reconstruction_idx)
        status = reconstruct_sub_model(
            controller, mapper, mapper_options, reconstruction
        )
        if status == pycolmap.IncrementalMapperStatus.INTERRUPTED:
            mapper.end_reconstruction(False)
        elif status in (
            pycolmap.IncrementalMapperStatus.NO_INITIAL_PAIR,
            pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR,
        ):
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            if options.is_initial_pair_provided():
                return
        elif status == pycolmap.IncrementalMapperStatus.SUCCESS:
            total_num_reg_images = mapper.num_total_reg_images()
            min_model_size = min(
                0.8 * database_cache.num_images(), options.min_model_size
            )
            if (
                options.multiple_models
                and reconstruction_manager.size() > 1
                and (
                    reconstruction.num_reg_images() < min_model_size
                    or reconstruction.num_reg_images() == 0
                )
            ):
                mapper.end_reconstruction(True)
                reconstruction_manager.delete(reconstruction_idx)
            else:
                mapper.end_reconstruction(False)
            controller.callback(
                pycolmap.IncrementalMapperCallback.LAST_IMAGE_REG_CALLBACK
            )
            if (
                initial_reconstruction_given
                or (not options.multiple_models)
                or reconstruction_manager.size() >= options.max_num_models
                or total_num_reg_images >= database_cache.num_images() - 1
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
    init_mapper_options = controller.options.get_mapper()
    reconstruct(controller, init_mapper_options)

    for i in range(2):  # number of relaxations
        if controller.reconstruction_manager.size() > 0:
            break
        logging.info("=> Relaxing the initialization constraints")
        init_mapper_options.init_min_num_inliers = int(
            init_mapper_options.init_min_num_inliers / 2
        )
        reconstruct(controller, init_mapper_options)
        if controller.reconstruction_manager.size() > 0:
            break
        logging.info("=> Relaxing the initialization constraints")
        init_mapper_options.init_min_tri_angle /= 2
        reconstruct(controller, init_mapper_options)
    timer.print_minutes()


def main(
    database_path,
    image_path,
    output_path,
    options=pycolmap.IncrementalPipelineOptions(),
    input_path=None,
):
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
