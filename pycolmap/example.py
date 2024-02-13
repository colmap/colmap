import shutil
import urllib.request
import zipfile
from pathlib import Path

import enlighten

import pycolmap
from pycolmap import logging


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
    num_images = pycolmap.Database(database_path).num_images

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

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
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")


if __name__ == "__main__":
    run()
