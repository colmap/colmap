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

from .geometry import vec_angular_dist_deg
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


def _imc2025_image_dir(dataset_dir: Path) -> Path:
    """Return the directory holding the images of an IMC2025 dataset."""
    images_subdir = dataset_dir / "images"
    return images_subdir if images_subdir.is_dir() else dataset_dir


def _imc2025_image_name(image_dir: Path, raw_image: str) -> str:
    """Return the image name relative to image_dir, as COLMAP will see it."""
    if (image_dir / raw_image).exists():
        return raw_image
    return Path(raw_image).name


def _within_group_relative_poses(
    name_to_cam_from_world: dict[str, pycolmap.Rigid3d],
) -> dict[tuple[str, str], pycolmap.Rigid3d]:
    """Relative poses cam_j_from_cam_i for all image pairs (i < j) in a group.

    Keys are (name_i, name_j) with name_i < name_j so that estimated and GT
    edges can be compared directly.
    """
    rel_poses: dict[tuple[str, str], pycolmap.Rigid3d] = {}
    names = sorted(name_to_cam_from_world)
    for a in range(len(names)):
        cam_i_from_world = name_to_cam_from_world[names[a]]
        world_from_cam_i = cam_i_from_world.inverse()
        for b in range(a + 1, len(names)):
            cam_j_from_world = name_to_cam_from_world[names[b]]
            rel_poses[(names[a], names[b])] = (
                cam_j_from_world * world_from_cam_i
            )
    return rel_poses


def _relative_pose_error_deg(
    rel_est: pycolmap.Rigid3d,
    rel_gt: pycolmap.Rigid3d,
    min_proj_center_dist: float,
) -> float:
    """Combined rotation/translation error (in degrees) of a relative pose.

    The rotation error is the geodesic angle; the translation error is the
    angular distance between the (gauge-independent) relative translation
    directions. Returns the maximum of the two. If the GT baseline is shorter
    than min_proj_center_dist, the translation direction is unstable and only
    the rotation error is used.
    """
    estimated_from_gt = rel_est.inverse() * rel_gt
    dR = np.rad2deg(estimated_from_gt.rotation.angle())
    if np.linalg.norm(rel_gt.translation) < min_proj_center_dist:
        dt = 0.0
    else:
        dt = vec_angular_dist_deg(rel_est.translation, rel_gt.translation)
    return max(dt, dR)


def compute_grouped_rel_errors(
    gt_scene_poses: dict[str, tuple[str, pycolmap.Rigid3d]],
    sub_models: list[pycolmap.Reconstruction],
    min_proj_center_dist: float,
) -> np.ndarray:
    """Graph-edge relative pose errors for the IMC2025 mixed-scene setting.

    Let A be the set of image pairs that share an estimated sub-model and B the
    set of pairs that share a ground-truth scene. For pairs in A n B we measure
    the relative pose error; for pairs in the symmetric difference (grouped in
    only one of the two, i.e. wrong merges/outliers or failed registrations) we
    assign the maximum error of 180 degrees. The AUC is then computed over all
    pairs in A u B.
    """
    # Set B: within-GT-scene relative poses.
    scene_to_poses: dict[str, dict[str, pycolmap.Rigid3d]] = defaultdict(dict)
    for name, (scene, cam_from_world) in gt_scene_poses.items():
        scene_to_poses[scene][name] = cam_from_world
    gt_rel: dict[tuple[str, str], pycolmap.Rigid3d] = {}
    for poses in scene_to_poses.values():
        gt_rel.update(_within_group_relative_poses(poses))

    # Set A: within-estimated-sub-model relative poses. An image may appear in
    # several sub-models, so collect all candidate estimates per pair.
    est_rel: dict[tuple[str, str], list[pycolmap.Rigid3d]] = defaultdict(list)
    for model in sub_models:
        name_to_cam_from_world = {
            image.name: image.cam_from_world()
            for image in model.images.values()
        }
        for pair, rel in _within_group_relative_poses(
            name_to_cam_from_world
        ).items():
            est_rel[pair].append(rel)

    all_pairs = set(gt_rel) | set(est_rel)
    if not all_pairs:
        # No evaluable pairs (e.g. only singleton scenes). Report worst case
        # rather than raising downstream on an empty error array.
        return np.array([180.0])

    errors = []
    for pair in all_pairs:
        if pair in gt_rel and pair in est_rel:
            errors.append(
                min(
                    _relative_pose_error_deg(
                        rel_est, gt_rel[pair], min_proj_center_dist
                    )
                    for rel_est in est_rel[pair]
                )
            )
        else:
            errors.append(180.0)
    return np.array(errors)


def _imc2025_gt_scene_poses(
    labels_path: Path, dataset: str, image_dir: Path
) -> dict[str, tuple[str, pycolmap.Rigid3d]]:
    """Map each GT image name of a dataset to its (scene, cam_from_world)."""
    gt_rows_by_scene, _ = _read_imc2025_labels(labels_path)
    scene_poses: dict[str, tuple[str, pycolmap.Rigid3d]] = {}
    for (ds, scene), rows in gt_rows_by_scene.items():
        if ds != dataset:
            continue
        for row in rows:
            name = _imc2025_image_name(image_dir, row["image"])
            extrinsic = np.hstack([row["R"], row["t"].reshape(3, 1)])
            scene_poses[name] = (scene, pycolmap.Rigid3d(extrinsic))
    return scene_poses


class DatasetIMC2025(Dataset):
    """IMC2025 benchmark dataset.

    Unlike IMC2023/IMC2024, the ground truth is provided as poses in a single
    train_labels.csv rather than per-scene COLMAP reconstructions, and each
    IMC "dataset" folder mixes images from multiple scenes plus outliers that
    belong to no scene.

    Here each IMC dataset is a single benchmark scene: all of its images (every
    scene plus outliers) are fed into one reconstruction problem, which
    typically yields several sub-models. Evaluation uses a graph-edge relative
    pose metric (see compute_grouped_rel_errors) that jointly penalizes wrong
    merges, registered outliers, and failed/fragmented registrations.

    Ground-truth 3D points and intrinsics are unavailable, so the GT
    reconstruction (used only for stats and covisibility heuristics) uses
    placeholder pinhole cameras built from the actual image dimensions. All
    scenes are stored in one GT reconstruction, each with its own gauge; the
    metric only ever compares poses within the same scene, so the (arbitrary)
    relative placement of different scenes does not matter.
    """

    # IMC2025 has no coarse category grouping, so all datasets share one
    # category and each dataset is reported as its own scene.
    _CATEGORY = "all"

    @property
    def position_accuracy_gt(self):
        return 0.02

    def _train_dir(self) -> Path:
        """Directory containing the per-dataset image folders."""
        return self.data_path / "imc2025" / "train"

    def _labels_path(self) -> Path:
        """Path to train_labels.csv (sibling of the train/ folder)."""
        return self.data_path / "imc2025" / _TRAIN_LABELS_FILENAME

    def list_scenes(self):
        train_dir = self._train_dir()
        labels_path = self._labels_path()
        if not labels_path.exists():
            pycolmap.logging.warning(
                f"Skipping IMC2025, because {labels_path} is missing"
            )
            return []

        if self.categories and self._CATEGORY not in self.categories:
            return []

        _, images_by_dataset = _read_imc2025_labels(labels_path)

        scene_infos = []
        for dataset in sorted(images_by_dataset):
            if self.scenes and dataset not in self.scenes:
                continue

            dataset_dir = train_dir / dataset
            image_dir = _imc2025_image_dir(dataset_dir)

            # Feed every image (all scenes + outliers) into one reconstruction,
            # restricted via an image list so unrelated files in the folder are
            # ignored. Missing files are dropped with a warning.
            names = []
            num_missing = 0
            for raw_image in images_by_dataset[dataset]:
                name = _imc2025_image_name(image_dir, raw_image)
                if (image_dir / name).exists():
                    names.append(name)
                else:
                    num_missing += 1
            if num_missing:
                pycolmap.logging.warning(
                    f"IMC2025 dataset={dataset}: {num_missing} listed "
                    "image(s) not found on disk, skipping them"
                )

            image_list_path = dataset_dir / "image_list.txt"
            with open(image_list_path, "w") as f:
                f.write("\n".join(names) + "\n")

            workspace_path = (
                self.run_path
                / self.run_name
                / "imc2025"
                / self._CATEGORY
                / dataset
            )
            sparse_gt_path = dataset_dir / "sparse_gt"

            scene_infos.append(
                SceneInfo(
                    dataset="IMC2025",
                    category=self._CATEGORY,
                    scene=dataset,
                    num_images=len(names),
                    workspace_path=workspace_path,
                    image_path=image_dir,
                    sparse_gt_path=sparse_gt_path,
                    has_camera_priors=False,
                    colmap_extra_args=[
                        "--image_list_path",
                        str(image_list_path),
                    ],
                )
            )

        return scene_infos

    def prepare_scene(self, scene_info):
        if scene_info.sparse_gt_path.exists():
            return

        gt_rows_by_scene, _ = _read_imc2025_labels(self._labels_path())
        image_dir = scene_info.image_path
        dataset = scene_info.scene

        sparse_gt = pycolmap.Reconstruction()
        idx = 0
        for (ds, _scene), rows in sorted(gt_rows_by_scene.items()):
            if ds != dataset:
                continue
            for row in rows:
                idx += 1
                image_name = _imc2025_image_name(image_dir, row["image"])
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

                # Intrinsics are not provided by IMC; use a rough pinhole
                # guess. This only affects covisibility/alignment heuristics,
                # not the relative pose error, which depends solely on poses.
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
                sparse_gt.add_camera(camera)
                sparse_gt.add_rig(rig)
                sparse_gt.add_frame(frame)
                sparse_gt.add_image(image)

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
        gt_scene_poses = _imc2025_gt_scene_poses(
            self._labels_path(),
            scene_info.scene,
            scene_info.image_path,
        )
        return compute_grouped_rel_errors(
            gt_scene_poses=gt_scene_poses,
            sub_models=sub_models,
            min_proj_center_dist=position_accuracy_gt,
        )
