import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import enlighten

import pycolmap
from pycolmap import logging


def extract_colors(image_path, image_id, reconstruction):
    if not reconstruction.extract_colors_for_image(image_id, image_path):
        logging.warning(
            "Could not read image {0} at path {1}".format(image_id, image_path)
        )


def write_snapshot(reconstruction, snapshot_path):
    logging.info("Creating snapshot")
    timestamp = time.time() * 1000
    path = os.path.join(snapshot_path, "{0:010d}".format(timestamp))
    if not os.path.exists(path):
        os.makedirs(path)
    logging.verbose("=> Writing to {0}".format(path))
    reconstruction.write(path)


def iterative_global_refinement(options, mapper_options, core_mapper):
    logging.info("Retriangulation and Global bundle adjustment")
    core_mapper.iterative_global_refinement(
        options.ba_global_max_refinements,
        options.ba_global_max_refinement_change,
        mapper_options,
        options.get_global_bundle_adjustment(),
        options.get_triangulation(),
    )
    core_mapper.filter_images(mapper_options)


def initialize_reconstruction(
    mapper, core_mapper, mapper_options, reconstruction
):
    # Following the implementation of src/colmap/controllers/incremental_mapper.cc
    # Equivalent to:
    # return mapper.initialize_reconstruction(core, mapper_options, reconstruction)
    options = mapper.options
    image_id1, image_id2 = options.init_image_id1, options.init_image_id2

    # Try to find good initial pair
    if not options.is_initial_pair_provided():
        logging.info("Finding good initial image pair")
        init_res = core_mapper.find_initial_image_pair(
            mapper_options, image_id1, image_id2
        )
        find_init_success, image_id1, image_id2, two_view_geometry = (
            init_res[0],
            init_res[1][0][0],
            init_res[1][0][1],
            init_res[1][1],
        )
        if not find_init_success:
            logging.info("No good initial image pair found.")
            return pycolmap.IncrementalMapperStatus.NO_INITIAL_PAIR
    else:
        if (not reconstruction.exists_image(image_id1)) or (
            not reconstruction.exists_image(image_id2)
        ):
            logging.info(
                "=> Initial image pair #{0} and #{1} do not exist.".format(
                    image_id1, image_id2
                )
            )
            return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR
        two_view_geometry = core_mapper.estimate_initial_two_view_geometry(
            mapper_options, image_id1, image_id2
        )
        if two_view_geometry is None:
            logging.info("Provided pair is insuitable for initialization")
            return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR
    logging.info(
        "Initializing with image pair #{0} and #{1}".format(
            image_id1, image_id2
        )
    )
    core_mapper.register_initial_image_pair(
        mapper_options, two_view_geometry, image_id1, image_id2
    )
    logging.info("Global bundle adjustment")
    core_mapper.adjust_global_bundle(
        mapper_options, options.get_global_bundle_adjustment()
    )
    reconstruction.normalize()
    core_mapper.filter_points(mapper_options)
    core_mapper.filter_images(mapper_options)

    # Initial image pair failed to register
    if (
        reconstruction.num_reg_images() == 0
        or reconstruction.num_points3D() == 0
    ):
        return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR
    if options.extract_colors:
        extract_colors(mapper.image_path, image_id1, reconstruction)
    return pycolmap.IncrementalMapperStatus.SUCCESS


def main_reconstruct_sub_model(
    mapper, core_mapper, mapper_options, reconstruction
):
    # Following the implementation of src/colmap/controllers/incremental_mapper.cc
    # Equivalent to:
    # return mapper.reconstruct_sub_model(core_mapper, mapper_options, reconstruction)

    # register initial pair
    core_mapper.begin_reconstruction(reconstruction)
    if reconstruction.num_reg_images() == 0:
        init_status = initialize_reconstruction(
            mapper, core_mapper, mapper_options, reconstruction
        )
        if init_status != pycolmap.IncrementalMapperStatus.SUCCESS:
            return init_status
        mapper.callback(
            pycolmap.IncrementalMapperCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK
        )

    # incremental mapping
    options = mapper.options
    snapshot_prev_num_reg_images = reconstruction.num_reg_images()
    ba_prev_num_reg_images = reconstruction.num_reg_images()
    ba_prev_num_points = reconstruction.num_points3D()
    reg_next_success, prev_reg_next_success = True, True
    while True:
        if not (reg_next_success or prev_reg_next_success):
            break
        prev_reg_next_success = reg_next_success
        reg_next_success = False
        next_images = core_mapper.find_next_images(mapper_options)
        if len(next_images) == 0:
            break
        for reg_trial in range(len(next_images)):
            next_image_id = next_images[reg_trial]
            next_image = reconstruction.images[next_image_id]
            logging.info(
                "Registering image #{0} ({1})".format(
                    next_image_id, reconstruction.num_reg_images() + 1
                )
            )
            logging.info(
                "=> Image sees {0} / {1} points".format(
                    next_image.num_visible_points3D(),
                    next_image.num_observations,
                )
            )
            reg_next_success = core_mapper.register_next_image(
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
            core_mapper.triangulate_image(
                options.get_triangulation(), next_image_id
            )
            core_mapper.iterative_local_refinement(
                options.ba_local_max_refinements,
                options.ba_local_max_refinement_change,
                mapper_options,
                options.get_local_bundle_adjustment(),
                options.get_triangulation(),
                next_image_id,
            )
            if mapper.check_run_global_refinement(
                reconstruction, ba_prev_num_reg_images, ba_prev_num_points
            ):
                iterative_global_refinement(
                    options, mapper_options, core_mapper
                )
                ba_prev_num_points = reconstruction.num_points3D()
                ba_prev_num_reg_images = reconstruction.num_reg_images()
            if options.extract_colors:
                extract_colors(mapper.image_path, next_image_id, reconstruction)
            if (
                options.snapshot_images_freq > 0
                and reconstruction.num_reg_images()
                >= options.snapshot_images_freq + snapshot_prev_num_reg_images
            ):
                snapshot_prev_num_reg_images = reconstruction.num_reg_images()
                write_snapshot(reconstruction, options.snapshot_path)
            mapper.callback(
                pycolmap.IncrementalMapperCallback.NEXT_IMAGE_REG_CALLBACK
            )
        if core_mapper.num_shared_reg_images() >= int(
            options.max_model_overlap
        ):
            break
        if (not reg_next_success) and prev_reg_next_success:
            iterative_global_refinement(options, mapper_options, core_mapper)
    if (
        reconstruction.num_reg_images() >= 2
        and reconstruction.num_reg_images() != ba_prev_num_reg_images
        and reconstruciton.num_points3D != ba_prev_num_points
    ):
        iterative_global_refinement(options, mapper_options, core_mapper)
    return pycolmap.IncrementalMapperStatus.SUCCESS


def main_reconstruct(mapper, mapper_options):
    # Following the implementation of src/colmap/controllers/incremental_mapper.cc
    # Equivalent to:
    # mapper.reconstruct(mapper_options)
    options = mapper.options
    reconstruction_manager = mapper.reconstruction_manager
    database_cache = mapper.database_cache
    core_mapper = pycolmap.IncrementalMapper(database_cache)
    initial_reconstruction_given = reconstruction_manager.size() > 0
    if reconstruction_manager.size() > 1:
        logging.error(
            "Can only resume from a single reconstruction, but multiple are given"
        )
    for num_trials in range(options.init_num_trials):
        if (not initial_reconstruction_given) or num_trials > 0:
            reconstruction_idx = reconstruction_manager.add()
        else:
            reconstruction_idx = 0
        reconstruction = reconstruction_manager.get(reconstruction_idx)
        status = main_reconstruct_sub_model(
            mapper, core_mapper, mapper_options, reconstruction
        )
        if status == pycolmap.IncrementalMapperStatus.INTERRUPTED:
            core_mapper.end_reconstruction(False)
        elif (
            status == pycolmap.IncrementalMapperStatus.NO_INITIAL_PAIR
            or status == pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR
        ):
            core_mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            if options.is_initial_pair_provided():
                return
        elif status == pycolmap.IncrementalMapperStatus.SUCCESS:
            total_num_reg_images = core_mapper.num_total_reg_images()
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
                core_mapper.end_reconstruction(True)
                reconstruction_manager.delete(reconstruction_idx)
            else:
                core_mapper.end_reconstruction(False)
            mapper.callback(
                pycolmap.IncrementalMapperCallback.LAST_IMAGE_REG_CALLBACK
            )
            if (
                initial_reconstruction_given
                or options.multiple_models
                or reconstruction_manager.size() >= options.max_num_models
                or total_num_reg_images >= database_cache.num_images() - 1
            ):
                return
        else:
            logging.fatal("Unknown reconstruction status")


def main_incremental_mapper(mapper):
    # Following the implementation of src/colmap/controllers/incremental_mapper.cc
    # Equivalent to:
    # mapper.run()
    timer = pycolmap.Timer()
    timer.start()
    if not mapper.load_database():
        return
    init_mapper_options = mapper.options.get_mapper()
    main_reconstruct(mapper, init_mapper_options)

    kNumInitRelaxations = 2
    for i in range(2):
        if mapper.reconstruction_manager.size() > 0:
            break
        logging.info("=> Relaxing the initialization constraints")
        init_mapper_options.init_min_num_inliers /= 2
        main_reconstruct(mapper, init_mapper_options)
        if mapper.reconstruction_manager.size() > 0:
            break
        logging.info("=> Relaxing the initialization constraints")
        init_mapper_options.init_min_tri_angle /= 2
        main_reconstruct(mapper, init_mapper_options)
    timer.print_minutes()


def incremental_mapping(
    database_path,
    image_path,
    output_path,
    options=pycolmap.IncrementalPipelineOptions(),
    input_path=None,
):
    # Following the implementation of src/pycolmap/pipeline/sfm.cc
    if not os.path.exists(database_path):
        logging.error(
            "Error! Database path does not exist: {0}".format(database_path)
        )
    if not os.path.exists(image_path):
        logging.error(
            "Error! Image path does not exist: {0}".format(image_path)
        )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    reconstruction_manager = pycolmap.ReconstructionManager()
    if (input_path is not None) and input_path != "":
        reconstruction_manager.read(input_path)
    mapper = pycolmap.IncrementalMapperController(
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


def incremental_mapping_cpp(database_path, image_path, sfm_path):
    num_images = pycolmap.Database(database_path).num_images
    with enlighten.Manager() as manager:
        with manager.counter(
            total=num_images, desc="Images registered:"
        ) as pbar:
            pbar.update(0, force=True)
            recs = pycolmap.incremental_mapping(
                database_path,
                image_path,
                sfm_path,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
            )
    return recs


def run():
    output_path = Path("example/")
    image_path = output_path / "Fountain/images"
    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"

    output_path.mkdir(exist_ok=True)
    logging.set_log_destination(
        logging.INFO, output_path / "INFO.log."
    )  # + time

    data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/Strecha-Fountain.zip"
    if not image_path.exists():
        logging.info("Downloading the data.")
        zip_path = output_path / "data.zip"
        urllib.request.urlretrieve(data_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as fid:
            fid.extractall(output_path)
        logging.info(f"Data extracted to {output_path}.")

    if database_path.exists():
        database_path.unlink()
    pycolmap.set_random_seed(0)
    pycolmap.extract_features(database_path, image_path)
    pycolmap.match_exhaustive(database_path)

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    recs = incremental_mapping(database_path, image_path, sfm_path)

    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")


if __name__ == "__main__":
    run()
