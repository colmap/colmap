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
    def position_accuracy_gt(self):
        return 0.001

    @property
    def supports_covisibility_filtering(self) -> bool:
        return True

    def list_scenes(self):
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

                colmap_extra_args = []
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

    def prepare_scene(self, scene_info):
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
