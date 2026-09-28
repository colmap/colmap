import logging

import cv2
import numpy as np

from hloc.utils.read_write_model import (
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_model,
    write_model,
)

logger = logging.getLogger(__name__)


def scale_sfm_images(full_model, scaled_model, image_dir):
    """Duplicate the provided model and scale the camera intrinsics so that
    they match the original image resolution - makes everything easier.
    """
    logger.info("Scaling the COLMAP model to the original image size.")
    scaled_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model)

    scaled_cameras = {}
    for id_, image in images.items():
        name = image.name
        img = cv2.imread(str(image_dir / name))
        assert img is not None, image_dir / name
        h, w = img.shape[:2]

        cam_id = image.camera_id
        if cam_id in scaled_cameras:
            assert scaled_cameras[cam_id].width == w
            assert scaled_cameras[cam_id].height == h
            continue

        camera = cameras[cam_id]
        assert camera.model == "SIMPLE_RADIAL"
        sx = w / camera.width
        sy = h / camera.height
        assert sx == sy, (sx, sy)
        scaled_cameras[cam_id] = camera._replace(
            width=w, height=h, params=camera.params * np.array([sx, sx, sy, 1.0])
        )

    write_model(scaled_cameras, images, points3D, scaled_model)


def create_query_list_with_intrinsics(
    model, out, list_file=None, ext=".bin", image_dir=None
):
    """Create a list of query images with intrinsics from the colmap model."""
    if ext == ".bin":
        images = read_images_binary(model / "images.bin")
        cameras = read_cameras_binary(model / "cameras.bin")
    else:
        images = read_images_text(model / "images.txt")
        cameras = read_cameras_text(model / "cameras.txt")

    name2id = {image.name: i for i, image in images.items()}
    if list_file is None:
        names = list(name2id)
    else:
        with open(list_file, "r") as f:
            names = f.read().rstrip().split("\n")
    data = []
    for name in names:
        image = images[name2id[name]]
        camera = cameras[image.camera_id]
        w, h, params = camera.width, camera.height, camera.params

        if image_dir is not None:
            # Check the original image size and rescale the camera intrinsics
            img = cv2.imread(str(image_dir / name))
            assert img is not None, image_dir / name
            h_orig, w_orig = img.shape[:2]
            assert camera.model == "SIMPLE_RADIAL"
            sx = w_orig / w
            sy = h_orig / h
            assert sx == sy, (sx, sy)
            w, h = w_orig, h_orig
            params = params * np.array([sx, sx, sy, 1.0])

        p = [name, camera.model, w, h] + params.tolist()
        data.append(" ".join(map(str, p)))
    with open(out, "w") as f:
        f.write("\n".join(data))


def evaluate(model, results, list_file=None, ext=".bin", only_localized=False):
    predictions = {}
    with open(results, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            predictions[name] = (qvec2rotmat(q), t)
    if ext == ".bin":
        images = read_images_binary(model / "images.bin")
    else:
        images = read_images_text(model / "images.txt")
    name2id = {image.name: i for i, image in images.items()}

    if list_file is None:
        test_names = list(name2id)
    else:
        with open(list_file, "r") as f:
            test_names = f.read().rstrip().split("\n")

    errors_t = []
    errors_R = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
        else:
            image = images[name2id[name]]
            R_gt, t_gt = image.qvec2rotmat(), image.tvec
            R, t = predictions[name]
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
        errors_t.append(e_t)
        errors_R.append(e_R)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f"Results for file {results.name}:"
    out += f"\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg"

    out += "\nPercentage of test images localized within:"
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%"
    logger.info(out)
