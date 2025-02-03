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

import numpy as np
from PIL import Image

import pycolmap

from .utils import Dataset, SceneInfo


class DatasetBlendedMVS(Dataset):
    def __init__(
        self,
        data_path: Path,
        categories: list[str],
        scenes: list[Path],
        run_path: Path,
        run_name: str,
    ):
        super().__init__()
        self.data_path = data_path
        self.categories = categories
        self.scenes = scenes
        self.run_path = run_path
        self.run_name = run_name

    @property
    def position_accuracy_gt(self):
        return 0.001

    def list_scenes(self):
        scene_infos = []
        for category_path in (self.data_path / "blended-mvs").iterdir():
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

                workspace_path = (
                    self.run_path
                    / self.run_name
                    / "blended-mvs"
                    / category
                    / scene
                )
                image_path = scene_path / "blended_images"
                image_list_path = scene_path / "images.txt"
                with open(image_list_path, "w") as fid:
                    for filepath in sorted(image_path.iterdir()):
                        image_name = str(filepath.name)
                        if (
                            image_name.endswith(".jpg")
                            and "masked" not in image_name
                        ):
                            fid.write(image_name + "\n")

                sparse_gt_path = scene_path / "sparse_gt"
                colmap_extra_args = ["--image_list_path", image_list_path]

                scene_info = SceneInfo(
                    dataset="blended-mvs",
                    category=category,
                    scene=scene,
                    workspace_path=workspace_path,
                    image_path=image_path,
                    sparse_gt_path=sparse_gt_path,
                    camera_priors_from_sparse_gt=True,
                    colmap_extra_args=colmap_extra_args,
                )

                scene_infos.append(scene_info)

        return scene_infos

    def prepare_scene(self, scene_info):
        if scene_info.sparse_gt_path.exists():
            return

        scene_path = scene_info.image_path.parent

        sparse_gt = pycolmap.Reconstruction()
        for i, filepath in enumerate(sorted((scene_path / "cams").iterdir())):
            filename = str(filepath.name)
            if not filename.endswith("_cam.txt"):
                continue
            image_name = filename[:-8] + ".jpg"
            width, height = Image.open(
                scene_path / "blended_images" / image_name
            ).size[:2]
            with open(filepath, encoding="ascii") as fid:
                lines = list(map(lambda b: b.strip(), fid.readlines()))
                extrinsic = np.fromstring(
                    " ".join(lines[1:4]),
                    count=12,
                    sep=" ",
                ).reshape(3, 4)
                intrinsic = np.fromstring(
                    " ".join(lines[7:10]),
                    count=9,
                    sep=" ",
                ).reshape(3, 3)
            camera = pycolmap.Camera(
                camera_id=i,
                model=pycolmap.CameraModelId.PINHOLE,
                width=width,
                height=height,
                params=intrinsic[(0, 1, 0, 1), (0, 1, 2, 2)],
            )
            image = pycolmap.Image(
                image_id=i,
                camera_id=i,
                name=image_name,
                cam_from_world=pycolmap.Rigid3d(extrinsic),
            )
            sparse_gt.add_camera(camera)
            sparse_gt.add_image(image)

        scene_info.sparse_gt_path.mkdir(exist_ok=True)
        sparse_gt.write(scene_info.sparse_gt_path)
