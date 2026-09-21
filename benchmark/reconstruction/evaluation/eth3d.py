# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pycolmap

from .utils import Dataset, SceneInfo

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class _DatasetETH3D(Dataset):
    """Shared ETH3D scene discovery.

    Subclasses supply `dataset_name` and `calibration_glob` to select one of
    the published ETH3D variants. The image folders under images/ are named
    per capture device (dslr_images_undistorted for the DSLR, one
    images_rig_cam<i>_undistorted per camera for the rig), so they are not
    matched by name.
    """

    dataset_name = ""
    calibration_glob = ""

    @property
    def position_accuracy_gt(self) -> float:
        return 0.001

    @property
    def supports_covisibility_filtering(self) -> bool:
        return True

    def list_scenes(self) -> list[SceneInfo]:
        scene_infos = []
        dataset_path = self.data_path / self.dataset_name
        category_paths = dataset_path.iterdir() if dataset_path.is_dir() else []
        for category_path in category_paths:
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

                image_path = scene_path / "images"
                sparse_gt_paths = sorted(scene_path.glob(self.calibration_glob))
                if not image_path.is_dir() or not sparse_gt_paths:
                    pycolmap.logging.warning(
                        f"Skipping {self.dataset_name} scene "
                        f"{category}/{scene}: requires "
                        f"images/ and {self.calibration_glob}"
                    )
                    continue

                workspace_path = (
                    self.run_path
                    / self.run_name
                    / self.dataset_name
                    / category
                    / scene
                )
                # Keep the shared images root: ETH3D GT image names include the
                # image subdirectory prefix, such as dslr_images/DSC_0286.JPG.
                sparse_gt_path = sparse_gt_paths[0]

                colmap_extra_args: list[str | Path] = []
                if category == "dslr":
                    colmap_extra_args.extend(["--data_type", "individual"])
                elif category == "rig":
                    colmap_extra_args.extend(["--data_type", "video"])

                num_images = sum(
                    1
                    for path in image_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                )

                scene_info = SceneInfo(
                    dataset=self.dataset_name,
                    category=category,
                    scene=scene,
                    num_images=num_images,
                    workspace_path=workspace_path,
                    image_path=image_path,
                    sparse_gt_path=sparse_gt_path,
                    has_camera_priors=True,
                    colmap_extra_args=colmap_extra_args,
                )

                scene_infos.append(scene_info)

        if not scene_infos and not dataset_path.is_dir():
            raise RuntimeError(
                f"No {self.dataset_name} scenes found. Download them with "
                f"`python download.py --datasets {self.dataset_name}`."
            )
        return scene_infos

    def prepare_scene(self, scene_info: SceneInfo) -> None:
        # Nothing to prepare for ETH3D.
        pass


class DatasetETH3DUndistorted(_DatasetETH3D):
    """ETH3D undistorted reconstruction benchmark.

    Built from the `*_undistorted.7z` downloads, which cover both the `dslr`
    and `rig` categories. The images are already undistorted and the ground
    truth uses PINHOLE cameras.
    """

    dataset_name = "eth3d"
    calibration_glob = "*_calibration_undistorted"


class DatasetETH3DDistorted(_DatasetETH3D):
    """ETH3D distorted DSLR JPEG reconstruction benchmark.

    The ground truth uses THIN_PRISM_FISHEYE cameras, and rig data is not
    published for this set. ETH3D's separate `_raw.7z` downloads contain
    undeveloped camera RAW files and are not this dataset. The JPEGs have no
    EXIF, so `--uncalibrated` starts from a focal-length guess of
    1.2 * max(width, height).
    """

    dataset_name = "eth3d-distorted"
    calibration_glob = "*_calibration_jpg"
