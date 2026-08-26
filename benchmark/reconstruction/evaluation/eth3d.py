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

from pathlib import Path

import pycolmap

from .utils import DATASET_VARIANTS, Dataset, SceneInfo

_VARIANT_PATH_GLOBS = {
    "undistorted": ("*_images_undistorted", "*_calibration_undistorted"),
    "distorted": ("*_images", "*_calibration_jpg"),
}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class DatasetETH3D(Dataset):
    """ETH3D reconstruction benchmark with selectable image variants.

    Bare `eth3d` defaults to undistorted images with PINHOLE ground truth for
    backward compatibility. `eth3d:distorted` selects distorted DSLR JPEGs with
    THIN_PRISM_FISHEYE ground truth; distorted rig data is not included. ETH3D's
    separate `_raw.7z` downloads are undeveloped camera RAW files and are not
    this variant. Combining `eth3d:distorted` with `--uncalibrated` exercises
    the harder SIMPLE_RADIAL self-calibration path: the JPEGs have no EXIF, so
    the initial focal length is guessed as 1.2 * max(width, height).
    """

    def __init__(
        self,
        data_path: Path,
        categories: list[str],
        scenes: list[Path],
        run_path: Path,
        run_name: str,
        variant: str = "undistorted",
    ):
        super().__init__(
            data_path=data_path,
            categories=categories,
            scenes=scenes,
            run_path=run_path,
            run_name=run_name,
        )
        if variant not in DATASET_VARIANTS["eth3d"]:
            valid = ", ".join(DATASET_VARIANTS["eth3d"])
            raise ValueError(
                f"Unsupported ETH3D variant {variant!r}. "
                f"Valid variants: {valid}"
            )
        # Keep this default aligned with DATASET_VARIANTS: bare `eth3d` must
        # always retain its historical undistorted behavior.
        self.variant = variant

    @property
    def position_accuracy_gt(self):
        return 0.001

    @property
    def supports_covisibility_filtering(self) -> bool:
        return True

    def list_scenes(self):
        image_glob, calibration_glob = _VARIANT_PATH_GLOBS[self.variant]
        scene_infos = []
        dataset_path = self.data_path / "eth3d"
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
                image_dirs = sorted(image_path.glob(image_glob))
                sparse_gt_paths = sorted(scene_path.glob(calibration_glob))
                if not image_dirs or not sparse_gt_paths:
                    pycolmap.logging.warning(
                        f"Skipping ETH3D scene {category}/{scene}: variant "
                        f"{self.variant!r} requires images/{image_glob} and "
                        f"{calibration_glob}"
                    )
                    continue

                workspace_path = (
                    self.run_path / self.run_name / "eth3d" / category / scene
                )
                # Keep the shared images root: ETH3D GT image names include the
                # variant subdirectory prefix, such as dslr_images/DSC_0286.JPG.
                sparse_gt_path = sparse_gt_paths[0]

                colmap_extra_args = []
                if category == "dslr":
                    colmap_extra_args.extend(["--data_type", "individual"])
                elif category == "rig":
                    colmap_extra_args.extend(["--data_type", "video"])
                # This full-set list is inert for undistorted-only trees and
                # prevents sibling distorted/undistorted image sets from mixing.
                colmap_extra_args.extend(
                    [
                        "--image_list_path",
                        str(workspace_path / "image_list.txt"),
                    ]
                )

                num_images = sum(
                    1
                    for image_dir in image_dirs
                    for path in image_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                )

                scene_info = SceneInfo(
                    dataset="eth3d",
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

        if not scene_infos:
            download_spec = (
                "eth3d:distorted"
                if self.variant == "distorted"
                else "eth3d:undistorted"
            )
            raise RuntimeError(
                f"No ETH3D scenes found for variant {self.variant!r}. Download "
                f"it with `python download.py --datasets {download_spec}`. "
                "The distorted variant is DSLR-only, so use "
                "`--categories dslr`."
            )
        return scene_infos

    def prepare_scene(self, scene_info):
        scene_info.workspace_path.mkdir(parents=True, exist_ok=True)

        marker_path = scene_info.workspace_path / "variant.txt"
        if marker_path.exists():
            workspace_variant = marker_path.read_text().strip()
            if workspace_variant != self.variant:
                raise RuntimeError(
                    f"ETH3D workspace {scene_info.workspace_path} contains "
                    f"variant {workspace_variant!r}, but {self.variant!r} was "
                    "requested. Use a fresh --run_name."
                )
        else:
            has_reusable_state = (
                scene_info.workspace_path / "database.db"
            ).exists() or (scene_info.workspace_path / "sparse").exists()
            if self.variant == "distorted" and has_reusable_state:
                raise RuntimeError(
                    f"ETH3D workspace {scene_info.workspace_path} predates "
                    "variant markers and contains reusable state, which is "
                    "assumed to be undistorted. Use a fresh --run_name."
                )
            marker_path.write_text(self.variant + "\n")

        sparse_gt = pycolmap.Reconstruction(scene_info.sparse_gt_path)
        image_names = sorted(image.name for image in sparse_gt.images.values())
        if len(image_names) != scene_info.num_images:
            pycolmap.logging.warning(
                f"ETH3D scene {scene_info.category}/{scene_info.scene} has "
                f"{scene_info.num_images} on-disk {self.variant} images but "
                f"{len(image_names)} ground-truth images"
            )
        (scene_info.workspace_path / "image_list.txt").write_text(
            "\n".join(image_names) + "\n"
        )
