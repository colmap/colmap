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

import csv
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image as PilImage

import pycolmap

from .utils import Dataset, SceneInfo

_POINTS3D_FILENAME = "points3D.txt"
_TRAIN_LABELS_FILENAME = "train_labels.csv"
# Scene label used by IMC2025 to mark images that do not belong to any scene.
_OUTLIER_SCENE = "outliers"
# Fallback image dimensions if an image file cannot be opened. Only used to
# build placeholder GT cameras, which do not affect relative pose evaluation.
_DEFAULT_IMAGE_WIDTH = 1024
_DEFAULT_IMAGE_HEIGHT = 768


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

    def _has_ground_truth(self, scene_path: Path) -> bool:
        """Whether ground truth is available for a scene.

        The default IMC layout ships a per-scene COLMAP reconstruction under
        sfm/; scenes without it are skipped.
        """
        return (scene_path / "sfm").exists()

    def _image_path(self, scene_path: Path) -> Path:
        """Folder holding a scene's images (a dedicated images/ subfolder)."""
        return scene_path / "images"

    def _count_images(self, scene: str, image_path: Path) -> int:
        """Number of images for a scene (image files present on disk)."""
        return sum(
            1
            for p in image_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

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

                if not self._has_ground_truth(scene_path):
                    pycolmap.logging.warning(
                        f"Skipping dataset=IMC{self.year}, "
                        f"category={category}, scene={scene}, "
                        "because the GT reconstruction is missing"
                    )
                    continue

                image_path = self._image_path(scene_path)
                sparse_gt_path = scene_path / "sparse_gt"
                workspace_path = (
                    self.run_path
                    / self.run_name
                    / folder_name
                    / category
                    / scene
                )

                scene_info = SceneInfo(
                    dataset=f"IMC{self.year}",
                    category=category,
                    scene=scene,
                    num_images=self._count_images(scene, image_path),
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


def _parse_floats(text: str) -> list[float] | None:
    """Parse a list of floats from an IMC label field.

    IMC stores matrices/vectors as ';'-separated values inside a single CSV
    field (falls back to whitespace separation). Returns None if the field is
    empty or cannot be fully parsed as floats.
    """
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    tokens = text.split(";") if ";" in text else text.split()
    try:
        return [float(t) for t in tokens if t.strip() != ""]
    except ValueError:
        return None


def _read_imc2025_labels(
    path: Path,
) -> tuple[dict[tuple[str, str], list[dict]], dict[str, list[str]]]:
    """Parse IMC2025 train_labels.csv.

    Expected columns: dataset, scene, image, rotation_matrix (row-major 3x3,
    cam_from_world), translation_vector (cam_from_world).

    Returns a tuple of:
      - gt_rows_by_scene: {(dataset, scene): [{image, R, t}, ...]} for all
        non-outlier images with a parseable pose (the ground truth).
      - images_by_dataset: {dataset: [image, ...]} for every image, including
        outliers (the full input to the reconstruction problem).
    """
    gt_rows_by_scene: dict[tuple[str, str], list[dict]] = defaultdict(list)
    images_by_dataset: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            scene = (row.get("scene") or "").strip()
            image = (row.get("image") or "").strip()
            if not dataset or not image:
                continue
            images_by_dataset[dataset].append(image)
            if not scene or scene == _OUTLIER_SCENE:
                continue
            rotation = _parse_floats(row.get("rotation_matrix", ""))
            translation = _parse_floats(row.get("translation_vector", ""))
            if (
                rotation is None
                or translation is None
                or len(rotation) != 9
                or len(translation) != 3
            ):
                continue
            gt_rows_by_scene[(dataset, scene)].append(
                {
                    "image": image,
                    "R": np.array(rotation, dtype=np.float64).reshape(3, 3),
                    "t": np.array(translation, dtype=np.float64),
                }
            )
    return gt_rows_by_scene, images_by_dataset


def _build_imc2025_gt_reconstruction(
    rows: list[dict], image_dir: Path
) -> pycolmap.Reconstruction:
    """Build a placeholder GT reconstruction from IMC2025 label rows.

    Each row provides {image, R, t} with cam_from_world extrinsics. IMC2025
    ships neither intrinsics nor 3D points, so we attach a rough pinhole camera
    sized from each image (falling back to default dimensions if the file
    cannot be opened) and add no points3D. The result has one
    rig/frame/camera/image per row and is only used for pose-based evaluation
    and covisibility heuristics, not for anything that depends on intrinsics.
    """
    reconstruction = pycolmap.Reconstruction()
    for idx, row in enumerate(rows, start=1):
        image_name = row["image"]
        image_file = image_dir / image_name
        try:
            width, height = PilImage.open(image_file).size
        except (OSError, ValueError):
            pycolmap.logging.warning(
                f"Could not read dimensions for {image_file}, "
                "using placeholder camera size"
            )
            width = _DEFAULT_IMAGE_WIDTH
            height = _DEFAULT_IMAGE_HEIGHT

        # Intrinsics are not provided by IMC; use a rough pinhole guess. This
        # only affects covisibility/alignment heuristics, not the relative pose
        # error, which depends solely on poses.
        focal = 1.2 * max(width, height)
        camera = pycolmap.Camera(
            camera_id=idx,
            model=pycolmap.CameraModelId.PINHOLE,
            width=width,
            height=height,
            params=[focal, focal, width / 2.0, height / 2.0],
        )
        rig = pycolmap.Rig(rig_id=idx)
        rig.add_ref_sensor(camera.sensor_id)
        image = pycolmap.Image(
            image_id=idx,
            camera_id=idx,
            name=image_name,
        )
        image.frame_id = idx
        frame = pycolmap.Frame(frame_id=idx)
        frame.rig_id = idx
        frame.add_data_id(image.data_id)
        # IMC stores cam_from_world (x_cam = R * x_world + t).
        extrinsic = np.hstack([row["R"], row["t"].reshape(3, 1)])
        frame.rig_from_world = pycolmap.Rigid3d(extrinsic)
        reconstruction.add_camera(camera)
        reconstruction.add_rig(rig)
        reconstruction.add_frame(frame)
        reconstruction.add_image(image)
    return reconstruction


class DatasetIMC2025(_DatasetIMC):
    """IMC2025 benchmark dataset.

    Unlike IMC2023/IMC2024, the ground truth is provided as poses in a single
    train_labels.csv rather than per-scene COLMAP reconstructions, and each
    IMC "dataset" folder mixes images from multiple scenes plus outliers that
    belong to no scene.

    Here each IMC dataset is a single benchmark scene: all of its images (every
    scene plus outliers) are fed into one reconstruction problem, which
    typically yields several sub-models. Evaluation uses the base-class
    set-based relative pose metric, driven by a per-GT-scene grouping supplied
    via scene_info.image_name_to_gt_recon_ids, which jointly penalizes wrong
    merges, registered outliers, and failed/fragmented registrations.

    Ground-truth 3D points and intrinsics are unavailable, so the GT
    reconstruction (used only for stats and covisibility heuristics) uses
    placeholder pinhole cameras built from the actual image dimensions. All
    scenes are stored in one GT reconstruction, each with its own gauge; the
    metric only ever compares poses within the same scene, so the (arbitrary)
    relative placement of different scenes does not matter.

    IMC2025 has no coarse category grouping: the downloaded layout places every
    dataset under train/all/<dataset>, so the base iteration reports "all" as
    the (single) category and each dataset as its own scene.
    """

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
        self.year = 2025
        # Lazily-parsed {dataset: [image, ...]} from train_labels.csv.
        self._images_by_dataset_cache: dict[str, list[str]] | None = None

    def _labels_path(self) -> Path:
        """Path to train_labels.csv (sibling of the train/ folder)."""
        return self.data_path / "imc2025" / _TRAIN_LABELS_FILENAME

    def _images_by_dataset(self) -> dict[str, list[str]]:
        """Parse (and cache) the {dataset: [image, ...]} map from the labels."""
        if self._images_by_dataset_cache is None:
            _, self._images_by_dataset_cache = _read_imc2025_labels(
                self._labels_path()
            )
        return self._images_by_dataset_cache

    def _has_ground_truth(self, scene_path: Path) -> bool:
        """IMC2025 ships GT as poses in train_labels.csv rather than a
        per-scene sfm/ reconstruction, so availability is determined by the
        existence of the labels file."""
        return self._labels_path().exists()

    def _image_path(self, scene_path: Path) -> Path:
        # All of a dataset's images live directly in the dataset folder.
        return scene_path

    def _count_images(self, scene: str, image_path: Path) -> int:
        # The folder contains exactly the labeled images; count them from the
        # labels and warn about any listed image missing on disk.
        num_images = 0
        num_missing = 0
        for name in self._images_by_dataset().get(scene, []):
            if (image_path / name).exists():
                num_images += 1
            else:
                num_missing += 1
        if num_missing:
            pycolmap.logging.warning(
                f"IMC2025 dataset={scene}: {num_missing} listed "
                "image(s) not found on disk"
            )
        return num_images

    def prepare_scene(self, scene_info):
        gt_rows_by_scene, _ = _read_imc2025_labels(self._labels_path())
        dataset = scene_info.scene

        # Each GT scene within this IMC dataset is its own reconstruction; map
        # every GT image name to a distinct integer id so the base-class
        # set-based metric only ever compares poses within the same scene.
        # Outliers are already excluded from gt_rows_by_scene, so they are
        # absent from the mapping and penalized when registered.
        scene_to_recon_id: dict[str, int] = {}
        image_name_to_gt_recon_ids: dict[str, int] = {}
        rows = []
        for (ds, scene), scene_rows in sorted(gt_rows_by_scene.items()):
            if ds != dataset:
                continue
            recon_id = scene_to_recon_id.setdefault(
                scene, len(scene_to_recon_id)
            )
            for row in scene_rows:
                image_name_to_gt_recon_ids[row["image"]] = recon_id
            rows.extend(scene_rows)
        scene_info.image_name_to_gt_recon_ids = image_name_to_gt_recon_ids

        if scene_info.sparse_gt_path.exists():
            return

        sparse_gt = _build_imc2025_gt_reconstruction(
            rows, scene_info.image_path
        )

        scene_info.sparse_gt_path.mkdir(parents=True, exist_ok=True)
        sparse_gt.write(scene_info.sparse_gt_path)

    def compute_scene_errors(
        self, args, scene_info, sub_models, sparse_gt, position_accuracy_gt
    ):
        if not args.error_type.startswith("relative"):
            raise ValueError(
                "IMC2025 evaluation only supports relative error types, "
                f"got: {args.error_type}"
            )
        return super().compute_scene_errors(
            args=args,
            scene_info=scene_info,
            sub_models=sub_models,
            sparse_gt=sparse_gt,
            position_accuracy_gt=position_accuracy_gt,
        )
