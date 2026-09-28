import argparse
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...colmap_from_nvm import (
    camera_center_to_translation,
    recover_database_images_and_ids,
)
from ...utils.read_write_model import (
    CAMERA_MODEL_IDS,
    Camera,
    Image,
    Point3D,
    write_model,
)

logger = logging.getLogger(__name__)


def read_nvm_model(nvm_path, database_path, image_ids, camera_ids, skip_points=False):
    # Extract the intrinsics from the db file instead of the NVM model
    db = sqlite3.connect(str(database_path))
    ret = db.execute("SELECT camera_id, model, width, height, params FROM cameras;")
    cameras = {}
    for camera_id, camera_model, width, height, params in ret:
        params = np.fromstring(params, dtype=np.double).reshape(-1)
        camera_model = CAMERA_MODEL_IDS[camera_model]
        assert len(params) == camera_model.num_params, (
            len(params),
            camera_model.num_params,
        )
        camera = Camera(
            id=camera_id,
            model=camera_model.model_name,
            width=int(width),
            height=int(height),
            params=params,
        )
        cameras[camera_id] = camera

    nvm_f = open(nvm_path, "r")
    line = nvm_f.readline()
    while line == "\n" or line.startswith("NVM_V3"):
        line = nvm_f.readline()
    num_images = int(line)
    # assert num_images == len(cameras), (num_images, len(cameras))

    logger.info(f"Reading {num_images} images...")
    image_idx_to_db_image_id = []
    image_data = []
    i = 0
    while i < num_images:
        line = nvm_f.readline()
        if line == "\n":
            continue
        data = line.strip("\n").lstrip("./").split(" ")
        image_data.append(data)
        image_idx_to_db_image_id.append(image_ids[data[0]])
        i += 1

    line = nvm_f.readline()
    while line == "\n":
        line = nvm_f.readline()
    num_points = int(line)

    if skip_points:
        logger.info(f"Skipping {num_points} points.")
        num_points = 0
    else:
        logger.info(f"Reading {num_points} points...")
    points3D = {}
    image_idx_to_keypoints = defaultdict(list)
    i = 0
    pbar = tqdm(total=num_points, unit="pts")
    while i < num_points:
        line = nvm_f.readline()
        if line == "\n":
            continue

        data = line.strip("\n").split(" ")
        x, y, z, r, g, b, num_observations = data[:7]
        obs_image_ids, point2D_idxs = [], []
        for j in range(int(num_observations)):
            s = 7 + 4 * j
            img_index, kp_index, kx, ky = data[s : s + 4]
            image_idx_to_keypoints[int(img_index)].append(
                (int(kp_index), float(kx), float(ky), i)
            )
            db_image_id = image_idx_to_db_image_id[int(img_index)]
            obs_image_ids.append(db_image_id)
            point2D_idxs.append(kp_index)

        point = Point3D(
            id=i,
            xyz=np.array([x, y, z], float),
            rgb=np.array([r, g, b], int),
            error=1.0,  # fake
            image_ids=np.array(obs_image_ids, int),
            point2D_idxs=np.array(point2D_idxs, int),
        )
        points3D[i] = point

        i += 1
        pbar.update(1)
    pbar.close()

    logger.info("Parsing image data...")
    images = {}
    for i, data in enumerate(image_data):
        # Skip the focal length. Skip the distortion and terminal 0.
        name, _, qw, qx, qy, qz, cx, cy, cz, _, _ = data
        qvec = np.array([qw, qx, qy, qz], float)
        c = np.array([cx, cy, cz], float)
        t = camera_center_to_translation(c, qvec)

        if i in image_idx_to_keypoints:
            # NVM only stores triangulated 2D keypoints: add dummy ones
            keypoints = image_idx_to_keypoints[i]
            point2D_idxs = np.array([d[0] for d in keypoints])
            tri_xys = np.array([[x, y] for _, x, y, _ in keypoints])
            tri_ids = np.array([i for _, _, _, i in keypoints])

            num_2Dpoints = max(point2D_idxs) + 1
            xys = np.zeros((num_2Dpoints, 2), float)
            point3D_ids = np.full(num_2Dpoints, -1, int)
            xys[point2D_idxs] = tri_xys
            point3D_ids[point2D_idxs] = tri_ids
        else:
            xys = np.zeros((0, 2), float)
            point3D_ids = np.full(0, -1, int)

        image_id = image_ids[name]
        image = Image(
            id=image_id,
            qvec=qvec,
            tvec=t,
            camera_id=camera_ids[name],
            name=name.replace("png", "jpg"),  # some hack required for RobotCar
            xys=xys,
            point3D_ids=point3D_ids,
        )
        images[image_id] = image

    return cameras, images, points3D


def main(nvm, database, output, skip_points=False):
    assert nvm.exists(), nvm
    assert database.exists(), database

    image_ids, camera_ids = recover_database_images_and_ids(database)

    logger.info("Reading the NVM model...")
    model = read_nvm_model(
        nvm, database, image_ids, camera_ids, skip_points=skip_points
    )

    logger.info("Writing the COLMAP model...")
    output.mkdir(exist_ok=True, parents=True)
    write_model(*model, path=str(output), ext=".bin")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvm", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip_points", action="store_true")
    args = parser.parse_args()
    main(**args.__dict__)
