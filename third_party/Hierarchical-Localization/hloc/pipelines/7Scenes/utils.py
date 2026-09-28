import logging

import numpy as np

from hloc.utils.read_write_model import read_model, write_model

logger = logging.getLogger(__name__)


def create_reference_sfm(full_model, ref_model, blacklist=None, ext=".bin"):
    """Create a new COLMAP model with only training images."""
    logger.info("Creating the reference model.")
    ref_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model, ext)

    if blacklist is not None:
        with open(blacklist, "r") as f:
            blacklist = f.read().rstrip().split("\n")

    images_ref = dict()
    for id_, image in images.items():
        if blacklist and image.name in blacklist:
            continue
        images_ref[id_] = image

    points3D_ref = dict()
    for id_, point3D in points3D.items():
        ref_ids = [i for i in point3D.image_ids if i in images_ref]
        if len(ref_ids) == 0:
            continue
        points3D_ref[id_] = point3D._replace(image_ids=np.array(ref_ids))

    write_model(cameras, images_ref, points3D_ref, ref_model, ".bin")
    logger.info(f"Kept {len(images_ref)} images out of {len(images)}.")
