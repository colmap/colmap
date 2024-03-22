import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import enlighten

import pycolmap
from pycolmap import logging

def main_reconstruct_sub_model(mapper, core_mapper, mapper_options, reconstruction):
    # Following the implementation of src/colmap/controllers/incremental_mapper.cc
    # Equivalent to:
    return mapper.ReconstructSubModel(core_mapper, mapper_options, reconstruction)

def main_reconstruct(mapper, mapper_options):
    # Following the implementation of src/colmap/controllers/incremental_mapper.cc
    # Equivalent to:
    # mapper.Reconstruct(mapper_options)
    options = mapper.get_options()
    reconstruction_manager = mapper.get_reconstruction_manager()
    database_cache = mapper.get_database_cache()
    core_mapper = pycolmap.IncrementalMapper(database_cache)
    initial_reconstruction_given = reconstruction_manager.size() > 0
    if reconstruction_manager.size() > 1:
        logging.error("Can only resume from a single reconstruction, but multiple are given")
    for num_trials in range(options.init_num_trials):
        if (not initial_reconstruction_given) or num_trials > 0:
            reconstruction_idx = reconstruction_manager.add()
        else:
            reconstruction_idx = 0
        reconstruction = reconstruction_manager.get(reconstruction_idx)
        status = main_reconstruct_sub_model(mapper, core_mapper, mapper_options, reconstruction)
        if status == pycolmap.IncrementalMapperStatus.INTERRUPTED:
            core_mapper.end_reconstruction(False)
        elif status == pycolmap.IncrementalMapperStatus.NO_INITIAL_PAIR or status == pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR:
            core_mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            if options.IsInitialPairProvided():
                return;
        elif status == pycolmap.IncrementalMapperStatus.SUCCESS:
            total_num_reg_images = core_mapper.num_total_reg_images()
            min_model_size = min(0.8 * database_cache.num_images(), options.min_model_size)
            if options.multiple_models and reconstruction_manager.size() > 1 and (reconstruction.num_reg_images() < min_model_size or reconstruction.num_reg_images() == 0):
                core_mapper.end_reconstruction(True)
                reconstruction_manager.delete(reconstruction_idx)
            else:
                core_mapper.end_reconstruction(False)
            mapper.callback(pycolmap.IncrementalMapperCallback.LAST_IMAGE_REG_CALLBACK)
            if initial_reconstruction_given or options.multiple_models or reconstruction_manager.size() >= options.max_num_models or total_num_reg_images >= database_cache.num_images() - 1:
                return;
        else:
            logging.error("Unknown reconstruction status")

def main_incremental_mapper(mapper):
    # Following the implementation of src/colmap/controllers/incremental_mapper.cc
    # Equivalent to:
    # mapper.Run()
    timer = pycolmap.Timer()
    timer.Start()
    if not mapper.load_database():
        return
    init_mapper_options = mapper.get_options().get_mapper()
    main_reconstruct(mapper, init_mapper_options)

    kNumInitRelaxations = 2
    for i in range(2):
        if mapper.get_reconstruction_manager().size() > 0:
            break
        logging.info("=> Relaxing the initialization constraints")
        init_mapper_options.init_min_num_inliers /= 2
        main_reconstruct(mapper, init_mapper_options)
        if mapper.get_reconstruction_manager().size() > 0:
            break
        logging.info("=> Relaxing the initialization constraints")
        init_mapper_options.init_min_tri_angle /= 2
        main_reconstruct(mapper, init_mapper_options)
    timer.PrintMinutes()

def incremental_mapping(database_path, image_path, output_path,
                        options=pycolmap.IncrementalPipelineOptions(),
                        input_path=None):
    # Following the implementation of src/pycolmap/pipeline/sfm.cc
    if not os.path.exists(database_path):
        logging.error("Error! Database path does not exist: {0}".format(database_path))
    if not os.path.exists(image_path):
        logging.error("Error! Image path does not exist: {0}".format(image_path))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    reconstruction_manager = pycolmap.ReconstructionManager()
    if (input_path is not None) and input_path != "":
        reconstruction_manager.read(input_path)
    mapper = pycolmap.IncrementalMapperController(options, image_path, database_path, reconstruction_manager)

    # main runner
    num_images = pycolmap.Database(database_path).num_images
    with enlighten.Manager() as manager:
        with manager.counter(
            total=num_images, desc="Images registered:"
        ) as pbar:
            pbar.update(0, force=True)
            mapper.add_callback(pycolmap.IncrementalMapperCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK, lambda: pbar.update(2))
            mapper.add_callback(pycolmap.IncrementalMapperCallback.NEXT_IMAGE_REG_CALLBACK, lambda: pbar.update(1))
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

