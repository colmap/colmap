# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

import pycolmap

from .utils import Dataset, SceneInfo

# The 55 training sequences evaluated in GLOMAP, Table S6.
SCENES = [
    "cables_1",
    "cables_2",
    "cables_3",
    "camera_shake_1",
    "camera_shake_2",
    "camera_shake_3",
    "ceiling_1",
    "ceiling_2",
    "desk_3",
    "desk_changing_1",
    "einstein_1",
    "einstein_2",
    "einstein_dark",
    "einstein_flashlight",
    "einstein_global_light_changes_1",
    "einstein_global_light_changes_2",
    "einstein_global_light_changes_3",
    "kidnap_1",
    "large_loop_1",
    "mannequin_1",
    "mannequin_3",
    "mannequin_4",
    "mannequin_5",
    "mannequin_7",
    "mannequin_face_1",
    "mannequin_face_2",
    "mannequin_face_3",
    "mannequin_head",
    "motion_1",
    "planar_2",
    "planar_3",
    "plant_1",
    "plant_2",
    "plant_3",
    "plant_4",
    "plant_5",
    "plant_scene_1",
    "plant_scene_2",
    "plant_scene_3",
    "reflective_1",
    "repetitive",
    "sfm_bench",
    "sfm_garden",
    "sfm_house_loop",
    "sfm_lab_room_1",
    "sfm_lab_room_2",
    "sofa_1",
    "sofa_2",
    "sofa_3",
    "sofa_4",
    "table_3",
    "table_4",
    "table_7",
    "vicon_light_1",
    "vicon_light_2",
]


def associate_poses(
    timestamps: Iterable[float],
    ground_truth: np.ndarray,
    max_interpolation_timespan: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Interpolate GT poses as in ETH3D's ComputePosesAtTimestamps."""
    rotations = Rotation.from_quat(ground_truth[:, 4:8])
    slerp = Slerp(ground_truth[:, 0], rotations)
    poses = {}
    for index, timestamp in enumerate(timestamps):
        upper = np.searchsorted(ground_truth[:, 0], timestamp, side="right")
        if upper == 0 or upper == len(ground_truth):
            continue
        lower = upper - 1
        timespan = ground_truth[upper, 0] - ground_truth[lower, 0]
        if timespan > max_interpolation_timespan:
            continue
        alpha = (timestamp - ground_truth[lower, 0]) / timespan
        position = (1 - alpha) * ground_truth[
            lower, 1:4
        ] + alpha * ground_truth[upper, 1:4]
        poses[index] = (slerp(timestamp).as_quat(), position)
    return poses


class DatasetETH3DSLAM(Dataset):
    @property
    def position_accuracy_gt(self) -> float:
        return 0.001

    @property
    def supports_covisibility_filtering(self) -> bool:
        return False

    def list_scenes(self) -> list[SceneInfo]:
        if self.categories and "training" not in self.categories:
            return []
        scene_infos: list[SceneInfo] = []
        for scene in SCENES:
            if self.scenes and scene not in self.scenes:
                continue
            scene_path = self.data_path / "eth3d-slam" / scene
            if not (scene_path / "rgb.txt").exists():
                continue
            frames = np.loadtxt(scene_path / "rgb.txt", dtype=str, ndmin=2)
            workspace_path = (
                self.run_path
                / self.run_name
                / "eth3d-slam"
                / "training"
                / scene
            )
            scene_infos.append(
                SceneInfo(
                    dataset="eth3d-slam",
                    category="training",
                    scene=scene,
                    num_images=len(frames),
                    workspace_path=workspace_path,
                    image_path=scene_path / "rgb",
                    sparse_gt_path=scene_path / "sparse_gt",
                    has_camera_priors=True,
                    colmap_extra_args=[
                        "--data_type",
                        "video",
                        "--single_camera",
                        "1",
                    ],
                )
            )
        return scene_infos

    def prepare_scene(self, scene_info: SceneInfo) -> None:
        scene_path = scene_info.image_path.parent
        frames = np.loadtxt(scene_path / "rgb.txt", dtype=str, ndmin=2)
        gt = np.loadtxt(scene_path / "groundtruth.txt", ndmin=2)
        # Limits used by ETH3D's evaluate_eth3d_slam_results.py.
        max_timespan = (
            1 / 11.25 if scene_info.scene.startswith("sfm_") else 1 / 75
        )
        poses = associate_poses(frames[:, 0].astype(float), gt, max_timespan)
        scene_info.images_without_gt = {
            Path(name).name
            for index, (_, name) in enumerate(frames)
            if index not in poses
        }
        if scene_info.sparse_gt_path.exists():
            return

        params = np.loadtxt(scene_path / "calibration.txt")
        params[2:] += 0.5  # COLMAP uses pixel-corner coordinates.
        with Image.open(scene_path / frames[0, 1]) as image:
            width, height = image.size
        camera = pycolmap.Camera(
            camera_id=1,
            model=pycolmap.CameraModelId.PINHOLE,
            width=width,
            height=height,
            params=params,
        )
        reconstruction = pycolmap.Reconstruction()
        reconstruction.add_camera_with_trivial_rig(camera)
        for index, (quaternion, position) in poses.items():
            image = pycolmap.Image(
                image_id=index + 1,
                camera_id=camera.camera_id,
                name=Path(frames[index, 1]).name,
            )
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(quaternion), position
            ).inverse()
            reconstruction.add_image_with_trivial_frame(image, cam_from_world)
        scene_info.sparse_gt_path.mkdir(parents=True)
        reconstruction.write(scene_info.sparse_gt_path)
