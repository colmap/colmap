# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import dataclasses
import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt

import pycolmap

from .geometry import vec_angular_dist_deg


@dataclasses.dataclass
class Frustum:
    # Sampled 3D points on the frustum in world coordinates.
    points: npt.NDArray[np.floating]
    # Near/far depth bounds (in camera coordinates) for the frustum.
    depth_range: tuple[float, float]


def _estimate_depth_ranges(
    sparse_gt: pycolmap.Reconstruction,
    percentile_near: float = 2.0,
    percentile_far: float = 98.0,
    min_num_points: int = 10,
) -> dict[int, tuple[float, float]]:
    """Estimate per-image near/far depth range from GT 3D points.

    For each image, computes depths of the 3D points visible in that image
    and returns percentile-based near/far bounds. Falls back to (0.1, 100.0)
    for images with insufficient depth data.
    """
    default_range = (0.1, 100.0)
    depth_ranges: dict[int, tuple[float, float]] = {}

    for image_id, image in sparse_gt.images.items():
        valid = [
            p.point3D_id
            for p in image.points2D
            if p.point3D_id != pycolmap.INVALID_POINT3D_ID
        ]
        if len(valid) < min_num_points:
            depth_ranges[image_id] = default_range
            continue

        points_xyz = np.array([sparse_gt.points3D[pid].xyz for pid in valid])
        cam_from_world = image.cam_from_world()
        points_in_cam = (
            cam_from_world.rotation.matrix() @ points_xyz.T
            + cam_from_world.translation[:, np.newaxis]
        )
        depths = points_in_cam[2, :]
        pos_depths = depths[depths > 0]

        if len(pos_depths) < min_num_points:
            depth_ranges[image_id] = default_range
            continue

        near, far = np.percentile(pos_depths, [percentile_near, percentile_far])
        depth_ranges[image_id] = (float(near), float(far))

    return depth_ranges


def _sample_frustum_points(
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    near: float,
    far: float,
    num_steps: int = 5,
) -> npt.NDArray[np.floating]:
    """Sample points on a camera viewing frustum in world coordinates.

    Samples a grid of points on the image plane (corners, edges, and interior)
    at multiple depths between near and far, then transforms them to world
    space for more accurate overlap checks.
    """
    w, h = camera.width, camera.height
    us = np.linspace(0, w, num_steps)
    vs = np.linspace(0, h, num_steps)
    grid_x, grid_y = np.meshgrid(us, vs)
    pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    cam_points = camera.cam_from_img(pixels)
    assert cam_points is not None
    cam_rays = np.column_stack([cam_points, np.ones(len(cam_points))])

    depths = np.linspace(near, far, num_steps)
    points_in_cam = np.vstack([cam_rays * d for d in depths])

    world_from_cam = image.cam_from_world().inverse()
    points_in_world = (
        world_from_cam.rotation.matrix() @ points_in_cam.T
        + world_from_cam.translation[:, np.newaxis]
    ).T
    return points_in_world


def _build_image_point3D_sets(
    sparse_gt: pycolmap.Reconstruction,
) -> dict[int, set[int]]:
    """Build a mapping from image ID to the set of observed 3D point IDs."""
    image_points: dict[int, set[int]] = {}
    for image_id, image in sparse_gt.images.items():
        image_points[image_id] = {
            p.point3D_id
            for p in image.points2D
            if p.point3D_id != pycolmap.INVALID_POINT3D_ID
        }
    return image_points


def _compute_frustums_for_all_images(
    sparse_gt: pycolmap.Reconstruction,
    frustum_near: float | None,
    frustum_far: float | None,
) -> dict[int, Frustum]:
    depth_ranges: dict[int, tuple[float, float]] = {}
    if frustum_near is None or frustum_far is None:
        depth_ranges = _estimate_depth_ranges(sparse_gt)

    frustums: dict[int, Frustum] = {}
    for image_id, image_gt in sparse_gt.images.items():
        camera_gt = sparse_gt.cameras[image_gt.camera_id]
        est_near, est_far = depth_ranges.get(image_id, (0.1, 100.0))
        near = frustum_near if frustum_near is not None else est_near
        far = frustum_far if frustum_far is not None else est_far
        frustums[image_id] = Frustum(
            points=_sample_frustum_points(image_gt, camera_gt, near, far),
            depth_range=(near, far),
        )
    return frustums


def _is_pair_covisible_by_tracks(
    image1: pycolmap.Image,
    image2: pycolmap.Image,
    image_point3D_sets: dict[int, set[int]],
    min_shared_points: int,
) -> bool:
    shared = len(
        image_point3D_sets[image1.image_id]
        & image_point3D_sets[image2.image_id]
    )
    return shared >= min_shared_points


def _is_pair_covisible_by_frustum(
    image1: pycolmap.Image,
    image2: pycolmap.Image,
    sparse_gt: pycolmap.Reconstruction,
    frustums: dict[int, Frustum],
    max_viewing_angle_deg: float,
) -> bool:
    """Check whether two cameras have overlapping viewing frustums.

    Uses a three-stage test that short-circuits on the first definitive result:
    1. Viewing angle: reject if angle between viewing directions exceeds the
       threshold.
    2. Vertex projection: accept if any of A's frustum vertices projects into
       B's image (or vice versa) with positive depth within the target
       image's expected GT depth range.
    """
    if (
        vec_angular_dist_deg(
            image1.viewing_direction(), image2.viewing_direction()
        )
        > max_viewing_angle_deg
    ):
        return False

    camera1 = sparse_gt.cameras[image1.camera_id]
    camera2 = sparse_gt.cameras[image2.camera_id]

    def project_and_check(
        frustum: Frustum,
        image: pycolmap.Image,
        camera: pycolmap.Camera,
    ) -> bool:
        near, far = frustums[image.image_id].depth_range
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        points_in_cam = (R @ frustum.points.T + t[:, np.newaxis]).T
        depths = points_in_cam[:, 2]
        mask = (depths >= near) & (depths <= far)
        if not np.any(mask):
            return False
        img_points = camera.img_from_cam(points_in_cam[mask])
        if img_points is None:
            return False
        in_bounds = (
            (img_points[:, 0] >= 0)
            & (img_points[:, 0] <= camera.width)
            & (img_points[:, 1] >= 0)
            & (img_points[:, 1] <= camera.height)
        )
        return bool(np.any(in_bounds))

    return project_and_check(
        frustums[image1.image_id], image2, camera2
    ) or project_and_check(frustums[image2.image_id], image1, camera1)


def filter_covisibility(
    database_path: Path,
    sparse_gt: pycolmap.Reconstruction,
    covisibility_frustum_near: float | None,
    covisibility_frustum_far: float | None,
    max_viewing_angle_deg: float,
    min_shared_points: int = 0,
) -> None:
    """Filter non-covisible image pairs from the database.

    If the GT reconstruction contains 3D point tracks and min_shared_points > 0,
    uses track-based covisibility: two images are covisible if they share at
    least min_shared_points common 3D points. Otherwise, falls back to
    frustum-based overlap checking using GT camera poses and intrinsics.
    """
    pycolmap.logging.info("Filtering non-covisible image pairs")

    use_tracks = min_shared_points > 0 and sparse_gt.num_points3D() > 0

    if use_tracks:
        pycolmap.logging.info(
            f"Using track-based covisibility "
            f"(min_shared_points={min_shared_points})"
        )
        image_point3D_sets = _build_image_point3D_sets(sparse_gt)
        is_covisible = functools.partial(
            _is_pair_covisible_by_tracks,
            image_point3D_sets=image_point3D_sets,
            min_shared_points=min_shared_points,
        )
    else:
        if min_shared_points > 0:
            pycolmap.logging.warning(
                "No GT 3D points available for track-based covisibility, "
                "falling back to frustum-based check"
            )
        frustums = _compute_frustums_for_all_images(
            sparse_gt, covisibility_frustum_near, covisibility_frustum_far
        )
        is_covisible = functools.partial(
            _is_pair_covisible_by_frustum,
            sparse_gt=sparse_gt,
            frustums=frustums,
            max_viewing_angle_deg=max_viewing_angle_deg,
        )

    images_gt_by_name: dict[str, pycolmap.Image] = {
        image_gt.name: image_gt for image_gt in sparse_gt.images.values()
    }

    with pycolmap.Database.open(str(database_path)) as database:
        db_id_to_name: dict[int, str] = {
            db_image.image_id: db_image.name
            for db_image in database.read_all_images()
        }

        pair_ids, _ = database.read_two_view_geometry_num_inliers()
        total_pairs = len(pair_ids)
        filtered_count = 0

        for pair_id in pair_ids:
            image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
            name1 = db_id_to_name.get(image_id1)
            name2 = db_id_to_name.get(image_id2)

            if name1 is None or name2 is None:
                continue

            if not is_covisible(
                images_gt_by_name[name1], images_gt_by_name[name2]
            ):
                # Only delete the two-view geometry, so it will be ignored
                # during reconstruction but keep the raw matches, so upon
                # re-running the pipeline, the matches will be re-computed,
                # if not filtering by covisibility.
                database.delete_two_view_geometry(image_id1, image_id2)
                filtered_count += 1

        pycolmap.logging.info(
            f"Co-visibility filtering: {filtered_count}/{total_pairs} pairs "
            f"removed, {total_pairs - filtered_count} kept"
        )
