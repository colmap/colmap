import argparse
from pathlib import Path

import numpy as np
import poselib
from tqdm import tqdm

import pycolmap


def image_ids_for_frame(frame: pycolmap.Frame) -> list[int]:
    return sorted(
        data.id
        for data in frame.data_ids
        if data.sensor_id.type == pycolmap.SensorType.CAMERA
    )


def get_rig_matches(db, image_ids1, image_ids2, min_num_matches=20):
    rig_matches = {}
    for image_id1 in image_ids1:
        for image_id2 in image_ids2:
            if not db.exists_matches(image_id1, image_id2):
                continue
            matches = db.read_matches(image_id1, image_id2)
            if len(matches) < min_num_matches:
                continue
            rig_matches[image_id1, image_id2] = matches
    return rig_matches


def generalized_relative_pose(
    frame1: pycolmap.Frame,
    frame2: pycolmap.Frame,
    keypoints1,
    keypoints2,
    matches,
    cameras,
    rigs,
    max_error=6,
):
    # Defined an ordering
    data_ids_1 = [
        data
        for data in frame1.data_ids
        if data.sensor_id.type == pycolmap.SensorType.CAMERA
    ]
    data_ids_2 = [
        data
        for data in frame2.data_ids
        if data.sensor_id.type == pycolmap.SensorType.CAMERA
    ]
    image_id_to_idx_1 = {data.id: idx for idx, data in enumerate(data_ids_1)}
    image_id_to_idx_2 = {data.id: idx for idx, data in enumerate(data_ids_2)}

    def _extrinsics_from_frame(data_ids, rig) -> list[poselib.CameraPose]:
        extrinsics = []
        for data_id in data_ids:
            ex = poselib.CameraPose()
            if not rig.is_ref_sensor(data_id.sensor_id):
                cam_from_rig = rig.sensor_from_rig(data_id.sensor_id)
                ex.t = cam_from_rig.translation
                ex.R = cam_from_rig.rotation.matrix()
            extrinsics.append(ex)
        return extrinsics

    extrinsics1 = _extrinsics_from_frame(data_ids_1, rigs[frame1.rig_id])
    extrinsics2 = _extrinsics_from_frame(data_ids_2, rigs[frame2.rig_id])

    def _cam_dict_from_id(camera_id) -> dict:
        camera = cameras[camera_id]
        return camera.todict() | {"model": camera.model.name}

    cam_dicts1 = [
        _cam_dict_from_id(data_id.sensor_id.id) for data_id in data_ids_1
    ]
    cam_dicts2 = [
        _cam_dict_from_id(data_id.sensor_id.id) for data_id in data_ids_2
    ]

    poselib_matches = []
    for (id1, id2), m in matches.items():
        pm = poselib.PairwiseMatches()
        pm.cam_id1 = image_id_to_idx_1[id1]
        pm.cam_id2 = image_id_to_idx_2[id2]
        pm.x1 = keypoints1[id1][:, :2][m[:, 0]]
        pm.x2 = keypoints2[id2][:, :2][m[:, 1]]
        poselib_matches.append(pm)

    frame2_from_frame1_, out = poselib.estimate_generalized_relative_pose(
        poselib_matches,
        extrinsics1,
        cam_dicts1,
        extrinsics2,
        cam_dicts2,
        {"max_epipolar_error": max_error},
    )

    frame2_from_frame1 = pycolmap.Rigid3d(
        pycolmap.Rotation3d(frame2_from_frame1_.R), frame2_from_frame1_.t
    )

    inlier_matches = {}
    for pm, inlier_mask in zip(poselib_matches, out["inliers"]):
        image_id1 = data_ids_1[pm.cam_id1].id
        image_id2 = data_ids_2[pm.cam_id2].id
        inlier_matches[image_id1, image_id2] = matches[image_id1, image_id2][
            inlier_mask
        ]

    return frame2_from_frame1, inlier_matches


def get_camera_from_rig(rig: pycolmap.Rig, camera_id):
    sensor_id = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=camera_id)
    if rig.is_ref_sensor(sensor_id):
        return pycolmap.Rigid3d()
    return rig.sensor_from_rig(sensor_id)


def compute_triangulation_angles(
    frame1,
    frame2,
    frame1_from_frame2,
    keypoints1,
    keypoints2,
    inlier_matches,
    cameras,
    images,
    rigs,
):
    tri_angles = {}
    for (i, j), m in inlier_matches.items():
        if not len(m):
            continue
        camera_id_1 = images[i].camera_id
        camera_id_2 = images[j].camera_id
        points1 = cameras[camera_id_1].cam_from_img(
            keypoints1[i][m[:, 0]][:, :2]
        )
        points2 = cameras[camera_id_2].cam_from_img(
            keypoints2[j][m[:, 1]][:, :2]
        )
        proj1 = np.eye(3, 4)
        cam1_from_frame1 = get_camera_from_rig(rigs[frame1.rig_id], camera_id_1)
        cam2_from_frame2 = get_camera_from_rig(rigs[frame2.rig_id], camera_id_2)
        cam2_from_cam1 = (
            cam1_from_frame1 * frame1_from_frame2 * cam2_from_frame2.inverse()
        )
        proj2 = cam2_from_cam1.matrix()
        angles = []
        for p1, p2 in zip(points1, points2):
            point3d = pycolmap.TriangulatePoint(proj1, proj2, p1, p2)
            if point3d is not None:
                angles.append(
                    pycolmap.CalculateTriangulationAngle(
                        proj1[:3, 3], proj2[:3, 3], point3d
                    )
                )
        tri_angles[i, j] = np.asarray(angles)
    tri_angle_rig = np.median(np.concatenate(list(tri_angles.values())))
    tri_angle_per_pair = {k: np.median(v) for k, v in tri_angles.items()}
    return tri_angle_rig, tri_angle_per_pair


def write_two_view_geometries(
    db,
    frame1,
    frame2,
    inliers_matches,
    tri_angles,
    frame2_from_frame1,
    images,
    rigs,
):
    for (image_id1, image_id2), matches_pair in inliers_matches.items():
        if not len(matches_pair):
            continue
        tvg = pycolmap.TwoViewGeometry()
        tvg.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
        cam1_from_frame1 = get_camera_from_rig(
            rigs[frame1.rig_id], images[image_id1].camera_id
        )
        cam2_from_frame2 = get_camera_from_rig(
            rigs[frame2.rig_id], images[image_id2].camera_id
        )
        tvg.cam2_from_cam1 = (
            cam2_from_frame2 * frame2_from_frame1 * cam1_from_frame1.inverse()
        )
        tvg.inlier_matches = matches_pair
        tvg.tri_angle = tri_angles[image_id1, image_id2]
        db.write_two_view_geometry(image_id1, image_id2, tvg)


def run_panorama_geometric_verification(
    db: pycolmap.Database, min_num_inliers: int = 10
):
    images = db.read_all_images()
    images = {image.image_id: image for image in images}
    frames = db.read_all_frames()
    rigs = db.read_all_rigs()
    rigs = {rig.rig_id: rig for rig in rigs}
    cameras = db.read_all_cameras()
    cameras = {camera.camera_id: camera for camera in cameras}

    num_frames = len(frames)
    pbar = tqdm(total=(num_frames * (num_frames - 1)) // 2)
    db.clear_two_view_geometries()
    for frame_idx1, frame1 in enumerate(frames):
        frame1_ids = image_ids_for_frame(frame1)
        keypoints1 = {i: db.read_keypoints(i) for i in frame1_ids}
        for frame_idx2 in range(frame_idx1 + 1, num_frames):
            pbar.update(1)
            frame2 = frames[frame_idx2]
            frame2_ids = image_ids_for_frame(frame2)
            keypoints2 = {i: db.read_keypoints(i) for i in frame2_ids}
            rig_matches = get_rig_matches(db, frame1_ids, frame2_ids)
            num_matches = sum(map(len, rig_matches.values()))
            if num_matches < min_num_inliers:
                continue
            frame1_from_frame2, inlier_matches = generalized_relative_pose(
                frame1,
                frame2,
                keypoints1,
                keypoints2,
                rig_matches,
                cameras,
                rigs,
            )
            num_inliers = sum(map(len, inlier_matches.values()))
            if num_inliers < min_num_inliers:
                continue
            tri_angle_rig, tri_angle_per_pair = compute_triangulation_angles(
                frame1,
                frame2,
                frame1_from_frame2,
                keypoints1,
                keypoints2,
                inlier_matches,
                cameras,
                images,
                rigs,
            )
            with pycolmap.DatabaseTransaction(db):
                write_two_view_geometries(
                    db,
                    frame1,
                    frame2,
                    inlier_matches,
                    tri_angle_per_pair,
                    frame1_from_frame2,
                    images,
                    rigs,
                )

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=Path, required=True)
    with pycolmap.Database(parser.parse_args().database_path) as db_:
        run_panorama_geometric_verification(db_)
