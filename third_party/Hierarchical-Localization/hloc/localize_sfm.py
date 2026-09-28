import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from tqdm import tqdm

from . import logger
from .utils.io import get_keypoints, get_matches, write_poses
from .utils.parsers import parse_image_lists, parse_retrieval


def do_covisibility_clustering(frame_ids: List[int], reconstruction):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed
                if p2D.has_point3D()
                for obs in reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}

    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        if points2D.shape[0] == 0:
            return None
        import pycolmap

        ret = pycolmap.estimate_and_refine_absolute_pose(
            points2D,
            points3D,
            query_camera,
            estimation_options=self.config.get("estimation", {}),
            refinement_options=self.config.get("refinement", {}),
        )
        return ret


def pose_from_cluster(
    localizer: QueryLocalizer,
    qname: str,
    query_camera,
    db_ids: List[int],
    features_path: Path,
    matches_path: Path,
    **kwargs,
):
    kpq = get_keypoints(features_path, qname)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D == 0:
            logger.debug(f"No 3D points found for {image.name}.")
            continue
        points3D_ids = np.array(
            [p.point3D_id if p.has_point3D() else -1 for p in image.points2D]
        )

        matches, _ = get_matches(matches_path, qname, image.name)
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
    if ret is not None:
        ret["camera"] = query_camera

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [
        (j, kp_idx_to_3D_to_db[i][j]) for i in idxs for j in kp_idx_to_3D[i]
    ]
    log = {
        "db": db_ids,
        "PnP_ret": ret,
        "keypoints_query": kpq[mkp_idxs],
        "points3D_ids": mp3d_ids,
        "points3D_xyz": None,  # we don't log xyz anymore because of file size
        "num_matches": num_matches,
        "keypoint_index_to_db": (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log


def main(
    reference_sfm: Path,
    queries: Path,
    retrieval: Path,
    features: Path,
    matches: Path,
    results: Path,
    image_dir: Path,
    database_path: Path = None,
    ransac_thresh: int = 12,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
):
    """Pose each query with this repository's ``colmap mapper``.

    ``reference_sfm`` is the triangulated map. The database next to it already
    holds the map images, their keypoints, and the verified match pairs. Query
    images are appended and registered by P3P, with the map frames held fixed.
    ``ransac_thresh`` is the mapper's ``abs_pose_max_error`` default of 12 px;
    the binary invocation in ``colmap_backend.register_queries`` uses that value.
    """
    del ransac_thresh, covisibility_clustering, prepend_camera_name, config
    from .colmap_backend import (
        add_query_images,
        read_images_txt,
        register_queries,
        write_keypoints,
        write_query_geometries,
    )
    from .colmap_backend import model_to_txt

    reference_sfm = Path(reference_sfm)
    if database_path is None:
        database_path = reference_sfm.parent / "database.db"
    database_path = Path(database_path)
    query_names = []
    for line in Path(queries).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            query_names.append(line.split()[0])

    import sqlite3

    connection = sqlite3.connect(database_path)
    name_to_id = dict(connection.execute("SELECT name, image_id FROM images"))
    connection.close()
    missing = [name for name in query_names if name not in name_to_id]
    name_to_id.update(add_query_images(database_path, missing))
    write_keypoints(database_path, {name: name_to_id[name] for name in missing}, features)
    write_query_geometries(database_path, name_to_id, retrieval, matches)

    localized = reference_sfm.parent / "localized"
    register_queries(database_path, image_dir, reference_sfm, localized)
    txt_dir = localized / "txt"
    model_to_txt(localized, txt_dir)
    posed = read_images_txt(txt_dir / "images.txt")
    with open(results, "w") as f:
        for name in query_names:
            if name not in posed:
                logger.warning("Query %s was not registered.", name)
                continue
            image = posed[name]
            qvec = " ".join(f"{v:.17g}" for v in image["qvec"])
            tvec = " ".join(f"{v:.17g}" for v in image["tvec"])
            f.write(f"{name} {qvec} {tvec}\n")
    logger.info("Wrote %s", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_sfm", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--database_path", type=Path)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--prepend_camera_name", action="store_true")
    args = parser.parse_args()
    main(**args.__dict__)
