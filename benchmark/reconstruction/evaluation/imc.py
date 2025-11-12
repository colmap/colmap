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

from .utils import Dataset, SceneInfo


class _DatasetIMC(Dataset):
    def __init__(
        self,
        data_path: Path,
        categories: list[str],
        scenes: list[Path],
        run_path: Path,
        run_name: str,
        year: int,
    ):
        super().__init__()
        self.data_path = data_path
        self.categories = categories
        self.scenes = scenes
        self.run_path = run_path
        self.run_name = run_name
        self.year = year

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

                scene_info = SceneInfo(
                    dataset=f"IMC{self.year}",
                    category=category,
                    scene=scene,
                    workspace_path=workspace_path,
                    image_path=image_path,
                    sparse_gt_path=sparse_gt_path,
                    camera_priors_from_sparse_gt=False,
                    colmap_extra_args=None,
                )

                scene_infos.append(scene_info)

        return scene_infos

    def prepare_scene(self, scene_info):
        if scene_info.sparse_gt_path.exists():
            return

        scene_path = scene_info.image_path.parent
        sfm_path = scene_path / "sfm"
        train_image_names = set(
            image.name for image in scene_info.image_path.iterdir()
        )
        sparse_sfm = pycolmap.Reconstruction(sfm_path)
        sparse_gt = pycolmap.Reconstruction()
        for image in sparse_sfm.images.values():
            if image.name not in train_image_names:
                continue
            if image.camera_id not in sparse_gt.cameras:
                sparse_gt.add_camera(image.camera)
            image.reset_camera_ptr()
            sparse_gt.add_image(image)
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
            year=2023,
        )


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
            year=2024,
        )
