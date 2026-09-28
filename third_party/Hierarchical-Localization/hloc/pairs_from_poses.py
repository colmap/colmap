import argparse
from pathlib import Path

import numpy as np
import scipy.spatial

from . import logger
from .pairs_from_retrieval import pairs_from_score_matrix
from .utils.read_write_model import read_images_binary

DEFAULT_ROT_THRESH = 30  # in degrees


def get_pairwise_distances(images):
    ids = np.array(list(images.keys()))
    Rs = []
    ts = []
    for id_ in ids:
        image = images[id_]
        R = image.qvec2rotmat()
        t = image.tvec
        Rs.append(R)
        ts.append(t)
    Rs = np.stack(Rs, 0)
    ts = np.stack(ts, 0)

    # Invert the poses from world-to-camera to camera-to-world.
    Rs = Rs.transpose(0, 2, 1)
    ts = -(Rs @ ts[:, :, None])[:, :, 0]

    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ts))

    # Instead of computing the angle between two camera orientations,
    # we compute the angle between the principal axes, as two images rotated
    # around their principal axis still observe the same scene.
    axes = Rs[:, :, -1]
    dots = np.einsum("mi,ni->mn", axes, axes, optimize=True)
    dR = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))

    return ids, dist, dR


def main(model, output, num_matched, rotation_threshold=DEFAULT_ROT_THRESH):
    logger.info("Reading the COLMAP model...")
    images = read_images_binary(model / "images.bin")

    logger.info(f"Obtaining pairwise distances between {len(images)} images...")
    ids, dist, dR = get_pairwise_distances(images)
    scores = -dist

    invalid = dR >= rotation_threshold
    np.fill_diagonal(invalid, True)
    pairs = pairs_from_score_matrix(scores, invalid, num_matched)
    pairs = [(images[ids[i]].name, images[ids[j]].name) for i, j in pairs]

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join(p) for p in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_matched", required=True, type=int)
    parser.add_argument("--rotation_threshold", default=DEFAULT_ROT_THRESH, type=float)
    args = parser.parse_args()
    main(**args.__dict__)
