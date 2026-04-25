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

import tempfile
from pathlib import Path

import numpy as np

import pycolmap

from .utils import Dataset, SceneInfo

_POINTS3D_FILENAME = "points3D.txt"


def _read_points3D_lenient(
    path: Path, valid_image_ids: set[int]
) -> dict[int, pycolmap.Point3D]:
    """Parse points3D.txt, dropping track elements whose image_id is not in
    valid_image_ids and skipping any point3D whose track becomes empty."""
    points3D: dict[int, pycolmap.Point3D] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            point3D_id = int(tokens[0])
            xyz = [float(v) for v in tokens[1:4]]
            rgb = [int(v) for v in tokens[4:7]]
            error = float(tokens[7])
            elements = []
            track_tokens = tokens[8:]
            for i in range(0, len(track_tokens), 2):
                image_id = int(track_tokens[i])
                point2D_idx = int(track_tokens[i + 1])
                if image_id in valid_image_ids:
                    elements.append(
                        pycolmap.TrackElement(image_id, point2D_idx)
                    )
            if not elements:
                continue
            point3D = pycolmap.Point3D()
            point3D.xyz = np.array(xyz)
            point3D.color = np.array(rgb, dtype=np.uint8)
            point3D.error = error
            point3D.track = pycolmap.Track(elements)
            points3D[point3D_id] = point3D
    return points3D


def _read_sparse_sfm_without_points3D(
    sfm_path: Path,
) -> pycolmap.Reconstruction:
    """Read an SfM reconstruction's cameras/rigs/frames/images, skipping
    points3D entirely.

    Some IMC scenes ship with points3D.txt referencing image_ids absent from
    images.txt, which makes the strict pycolmap reader abort. Callers can
    parse points3D separately via _read_points3D_lenient if needed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for src_file in sfm_path.iterdir():
            if src_file.name == _POINTS3D_FILENAME:
                continue
            (tmp_path / src_file.name).symlink_to(src_file.resolve())
        (tmp_path / _POINTS3D_FILENAME).touch()
        return pycolmap.Reconstruction(tmp_path)


class _DatasetIMC(Dataset):
    @property
    def position_accuracy_gt(self):
        return 0.02

    def list_scenes(self):
        folder_name = f"imc{self.year}"

        scene_infos = []
        for category_path in Path(
            self.data_path / f"{folder_name}/train"
        ).iterdir():
            if not category_path.is_dir() or (
                self.categories and category_path.name not in self.categories
            ):
                continue

            category = category_path.name

            for scene_path in category_path.iterdir():
                if not scene_path.is_dir():
                    continue

                scene = scene_path.name
                if self.scenes and scene not in self.scenes:
                    continue

                sfm_path = scene_path / "sfm"
                if not sfm_path.exists():
                    pycolmap.logging.warning(
                        f"Skipping dataset=IMC{self.year}, "
                        f"category={category}, scene={scene}, "
                        "because the GT reconstruction is missing"
                    )
                    continue

                workspace_path = (
                    self.run_path
                    / self.run_name
                    / folder_name
                    / category
                    / scene
                )
                image_path = scene_path / "images"
                sparse_gt_path = scene_path / "sparse_gt"

                num_images = sum(
                    1
                    for p in image_path.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )

                scene_info = SceneInfo(
                    dataset=f"IMC{self.year}",
                    category=category,
                    scene=scene,
                    num_images=num_images,
                    workspace_path=workspace_path,
                    image_path=image_path,
                    sparse_gt_path=sparse_gt_path,
                    has_camera_priors=False,
                    colmap_extra_args=None,
                )

                scene_infos.append(scene_info)

        return scene_infos

    def prepare_scene(self, scene_info):
        if scene_info.sparse_gt_path.exists():
            return

        scene_path = scene_info.image_path.parent
        sfm_path = scene_path / "sfm"

        sparse_sfm = _read_sparse_sfm_without_points3D(sfm_path)
        sparse_gt = pycolmap.Reconstruction()

        train_image_ids = set()
        train_image_names = set(
            image.name for image in scene_info.image_path.iterdir()
        )
        for image in sparse_sfm.images.values():
            if image.name in train_image_names:
                train_image_ids.add(image.image_id)
        for camera in sparse_sfm.cameras.values():
            sparse_gt.add_camera(camera)
        for rig in sparse_sfm.rigs.values():
            sparse_gt.add_rig(rig)
        for frame in sparse_sfm.frames.values():
            has_train_image = False
            for data_id in frame.image_ids:
                if data_id.id in train_image_ids:
                    has_train_image = True
                    break
            if has_train_image:
                frame.reset_rig_ptr()
                sparse_gt.add_frame(frame)
        for image in sparse_sfm.images.values():
            if image.image_id not in train_image_ids:
                continue
            if image.camera_id not in sparse_gt.cameras:
                sparse_gt.add_camera(image.camera)
            image.reset_camera_ptr()
            image.reset_frame_ptr()
            sparse_gt.add_image(image)

        points3D = _read_points3D_lenient(
            sfm_path / _POINTS3D_FILENAME, train_image_ids
        )
        for point3D_id, point3D in points3D.items():
            sparse_gt.add_point3D_with_id(point3D_id, point3D)

        scene_info.sparse_gt_path.mkdir(exist_ok=True)
        sparse_gt.write(scene_info.sparse_gt_path)


class DatasetIMC2023(_DatasetIMC):
    def __init__(
        self,
        data_path: Path,
        categories: list[str],
        scenes: list[Path],
        run_path: Path,
        run_name: str,
    ):
        super().__init__(
            data_path=data_path,
            categories=categories,
            scenes=scenes,
            run_path=run_path,
            run_name=run_name,
        )
        self.year = 2023


class DatasetIMC2024(_DatasetIMC):
    def __init__(
        self,
        data_path: Path,
        categories: list[str],
        scenes: list[Path],
        run_path: Path,
        run_name: str,
    ):
        super().__init__(
            data_path=data_path,
            categories=categories,
            scenes=scenes,
            run_path=run_path,
            run_name=run_name,
        )
        self.year = 2024
