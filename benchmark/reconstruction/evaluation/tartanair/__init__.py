# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.

import json

import numpy as np

import pycolmap

from ..utils import Dataset, SceneInfo

# Precomputed TartanAir panoramas have a -90 degree longitude shift: panorama
# +X is NED forward, +Y is down, and +Z is left.
NED_FROM_COLMAP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def tartanair_world_from_camera(
    translation: np.ndarray, quaternion_xyzw: np.ndarray
) -> pycolmap.Rigid3d:
    """Convert a TartanAir NED camera pose to COLMAP camera coordinates."""
    world_from_ned = pycolmap.Rigid3d(
        pycolmap.Rotation3d(quaternion_xyzw), translation
    )
    ned_from_colmap = pycolmap.Rigid3d(
        pycolmap.Rotation3d(NED_FROM_COLMAP), np.zeros(3)
    )
    return world_from_ned * ned_from_colmap


class DatasetTartanAir(Dataset):
    dataset_name = ""
    render_type = ""

    @property
    def position_accuracy_gt(self) -> float:
        return 0.001

    def list_scenes(self) -> list[SceneInfo]:
        scene_infos = []
        dataset_path = self.data_path / "tartanair-v2"
        if not dataset_path.exists():
            return scene_infos

        for category_path in sorted(dataset_path.iterdir()):
            if not category_path.is_dir() or (
                self.categories and category_path.name not in self.categories
            ):
                continue
            category = category_path.name
            for scene_path in sorted(category_path.iterdir()):
                if not scene_path.is_dir():
                    continue
                scene = scene_path.name
                if self.scenes and scene not in self.scenes:
                    continue

                metadata_path = scene_path / "scene.json"
                if not metadata_path.exists():
                    continue
                metadata = json.loads(metadata_path.read_text())
                workspace_path = (
                    self.run_path
                    / self.run_name
                    / self.dataset_name
                    / category
                    / scene
                )
                reconstruction_subdir = (
                    "sparse_equirectangular"
                    if self.render_type == "perspective_overlapping"
                    else "sparse"
                )
                scene_infos.append(
                    SceneInfo(
                        dataset=self.dataset_name,
                        category=category,
                        scene=scene,
                        num_images=len(metadata["frames"]),
                        workspace_path=workspace_path,
                        image_path=scene_path / "images",
                        sparse_gt_path=scene_path / "sparse_gt",
                        has_camera_priors=False,
                        colmap_extra_args=[],
                        reconstruction_backend=f"panorama-{self.render_type}",
                        reconstruction_subdir=reconstruction_subdir,
                    )
                )
        return scene_infos

    def prepare_scene(self, scene_info: SceneInfo) -> None:
        if scene_info.sparse_gt_path.exists():
            return

        scene_path = scene_info.image_path.parent
        metadata = json.loads((scene_path / "scene.json").read_text())
        pose_by_frame = {}
        with open(scene_path / "poses.txt", encoding="ascii") as fid:
            for line in fid:
                values = line.split()
                if not values:
                    continue
                frame_id = int(values[0])
                pose_by_frame[frame_id] = np.array(values[1:8], dtype=float)

        width, height = metadata["image_size"]
        reconstruction = pycolmap.Reconstruction()
        camera = pycolmap.Camera(
            camera_id=1,
            model=pycolmap.CameraModelId.EQUIRECTANGULAR,
            width=width,
            height=height,
            params=[width, height],
        )
        rig = pycolmap.Rig(rig_id=1)
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_camera(camera)
        reconstruction.add_rig(rig)

        for image_id, frame_info in enumerate(metadata["frames"], start=1):
            source_frame = frame_info["source_frame"]
            pose = pose_by_frame[source_frame]
            image = pycolmap.Image(
                image_id=image_id,
                camera_id=camera.camera_id,
                name=frame_info["image_name"],
            )
            image.frame_id = image_id
            frame = pycolmap.Frame(frame_id=image_id)
            frame.rig_id = rig.rig_id
            frame.add_data_id(image.data_id)
            world_from_camera = tartanair_world_from_camera(pose[:3], pose[3:])
            frame.rig_from_world = world_from_camera.inverse()
            reconstruction.add_frame(frame)
            reconstruction.add_image(image)

        scene_info.sparse_gt_path.mkdir(parents=True)
        reconstruction.write(scene_info.sparse_gt_path)


class DatasetTartanAirPerspective(DatasetTartanAir):
    dataset_name = "tartanair-v2-perspective"
    render_type = "perspective_overlapping"


class DatasetTartanAirSpherical(DatasetTartanAir):
    dataset_name = "tartanair-v2-spherical"
    render_type = "spherical"
