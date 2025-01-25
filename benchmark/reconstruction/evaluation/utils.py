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

import argparse
import collections
import copy
import dataclasses
import datetime
import functools
import multiprocessing
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

import pycolmap


@dataclasses.dataclass(kw_only=True)
class SceneInfo:
    # Dataset name.
    dataset: str
    # Category name.
    category: str
    # Scene name.
    scene: str
    # Path to the workspace directory in the run directory.
    workspace_path: Path
    # Path to the input images.
    image_path: Path
    # Path to the ground-truth sparse reconstruction.
    sparse_gt_path: Path
    # Whether to update camera priors from the ground-truth reconstruction.
    camera_priors_from_sparse_gt: bool
    # Additional arguments for the COLMAP reconstruction command.
    colmap_extra_args: list[str]


@dataclasses.dataclass(kw_only=True)
class SceneResult:
    # Scene information for which the result was computed.
    scene_info: SceneInfo
    # Flat list of errors.
    errors: np.ndarray
    # Number of images in the scene.
    num_images: int
    # Number of registered images in the scene (over all components).
    num_reg_images: int
    # Number of components in the scene.
    num_components: int
    # Number of images in the largest component.
    largest_component: int


@dataclasses.dataclass(kw_only=True)
class SceneMetrics:
    # Area under the curve (AUC) scores at specified error thresholds.
    aucs: np.ndarray
    error_thresholds: np.ndarray
    error_type: str
    # Number of images in the scene.
    num_images: int
    # Number of registered images in the scene (over all components).
    num_reg_images: int
    # Number of components in the scene.
    num_components: int
    # Number of images in the largest component.
    largest_component: int


class Dataset(ABC):
    @property
    @abstractmethod
    def position_accuracy_gt(self) -> float:
        """Ground-truth position accuracy in meters."""
        pass

    @abstractmethod
    def list_scenes(self) -> list[SceneInfo]:
        """List all scenes to evaluate."""
        pass

    @abstractmethod
    def prepare_scene(self, scene_info: SceneInfo) -> None:
        """Prepare the scene for reconstruction."""
        pass


def parse_args() -> argparse.Namespace:
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default=Path(__file__).parent / "data", type=Path
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eth3d", "blended-mvs", "imc2023", "imc2024"],
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="Categories to evaluate, if empty all categories are evaluated.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[],
        help="Scenes to evaluate, if empty all scenes are evaluated.",
    )
    parser.add_argument(
        "--run_path", default=Path(__file__).parent / "runs", type=Path
    )
    parser.add_argument("--run_name", default=datetime_str)
    parser.add_argument("--report_name", default=f"report-{datetime_str}")
    parser.add_argument(
        "--overwrite_database", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_reconstruction", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_alignment", default=False, action="store_true"
    )
    parser.add_argument("--colmap_path", required=True)
    parser.add_argument("--use_gpu", default=True, action="store_true")
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of processes for parallel reconstruction.",
    )
    parser.add_argument("--quality", default="high")
    parser.add_argument(
        "--error_type",
        default="relative",
        choices=["relative", "absolute"],
        help="Whether to evaluate relative pairwise pose errors in angular "
        "distance or absolute pose errors through GT alignment.",
    )
    parser.add_argument(
        "--rel_error_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 1, 5, 10],
        help="Evaluation thresholds in degrees.",
    )
    parser.add_argument(
        "--abs_error_thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.2, 0.5],
        help="Evaluation thresholds in meters.",
    )
    args = parser.parse_args()
    args.colmap_path = Path(args.colmap_path).resolve()
    if args.overwrite_database:
        pycolmap.logging.info(
            "Overwriting database also overwrites reconstruction"
        )
        args.overwrite_reconstruction = True
    if args.overwrite_reconstruction:
        pycolmap.logging.info(
            "Overwriting reconstruction also overwrites alignment"
        )
        args.overwrite_alignment = True
    return args


def update_camera_priors_from_sparse_gt(
    database_path: Path, camera_priors_sparse_gt: pycolmap.Reconstruction
) -> None:
    pycolmap.logging.info("Setting prior cameras from GT")

    database = pycolmap.Database()
    database.open(database_path)

    camera_id_gt_to_camera_id = {}
    for camera_id_gt, camera_gt in camera_priors_sparse_gt.cameras.items():
        camera_gt.has_prior_focal_length = True
        camera_id = database.write_camera(camera_gt)
        camera_id_gt_to_camera_id[camera_id_gt] = camera_id

    images_gt_by_name = {}
    for image_gt in camera_priors_sparse_gt.images.values():
        images_gt_by_name[image_gt.name] = image_gt

    for image in database.read_all_images():
        if image.name not in images_gt_by_name:
            pycolmap.logging.warning(
                f"Not setting prior camera for image {image.name}, "
                "because it does not exist in GT"
            )
            continue
        image_gt = images_gt_by_name[image.name]
        camera_id = camera_id_gt_to_camera_id[image_gt.camera_id]
        image.camera_id = camera_id
        database.update_image(image)

    database.close()


def colmap_reconstruction(
    args: argparse.Namespace,
    workspace_path: Path,
    image_path: Path,
    camera_priors_sparse_gt: pycolmap.Reconstruction = None,
    colmap_extra_args: list = None,
    num_threads: int = 1,
) -> None:
    workspace_path.mkdir(parents=True, exist_ok=True)

    database_path = workspace_path / "database.db"
    if args.overwrite_database and database_path.exists():
        database_path.unlink()

    sparse_path = workspace_path / "sparse"
    if args.overwrite_reconstruction and sparse_path.exists():
        shutil.rmtree(sparse_path)

    if sparse_path.exists():
        pycolmap.logging.info("Skipping reconstruction, as it already exists")
        return

    # TODO: Expose automatic reconstruction through pycolmap bindings instead
    # of using the command line interface. One blocker for this is that we
    # currently do not produce CUDA enabled pycolmap packages.
    colmap_args = [
        args.colmap_path,
        "automatic_reconstructor",
        "--image_path",
        image_path,
        "--workspace_path",
        workspace_path,
        "--vocab_tree_path",
        args.data_path / "vocab_tree_flickr100K_words256K.bin",
        "--use_gpu",
        "1" if args.use_gpu else "0",
        "--num_threads",
        str(num_threads),
        "--quality",
        args.quality,
    ]

    subprocess.check_call(
        colmap_args
        + (colmap_extra_args or [])
        + [
            "--extraction",
            "1",
            "--matching",
            "0",
            "--sparse",
            "0",
            "--dense",
            "0",
        ],
        cwd=workspace_path,
    )

    if camera_priors_sparse_gt is not None:
        update_camera_priors_from_sparse_gt(
            database_path, camera_priors_sparse_gt
        )

    subprocess.check_call(
        colmap_args
        + (colmap_extra_args or [])
        + [
            "--extraction",
            "0",
            "--matching",
            "1",
            "--sparse",
            "0",
            "--dense",
            "0",
        ],
        cwd=workspace_path,
    )

    # Decouple matching from sparse reconstruction, because matching will
    # initialize an OpenGL context and Mac on Apple silicon tends to assign GUI
    # applications to the low efficiency cores but we want to use the
    # performance cores.
    subprocess.check_call(
        colmap_args
        + (colmap_extra_args or [])
        + [
            "--extraction",
            "0",
            "--matching",
            "0",
            "--sparse",
            "1",
            "--dense",
            "0",
        ],
        cwd=workspace_path,
    )


def colmap_alignment(
    args: argparse.Namespace,
    sparse_path: Path,
    sparse_gt_path: Path,
    sparse_aligned_path: Path,
    max_ref_model_error: float,
) -> None:
    if args.overwrite_alignment and sparse_aligned_path.exists():
        shutil.rmtree(sparse_aligned_path)
    if sparse_aligned_path.exists():
        pycolmap.logging.info("Skipping alignment, as it already exists")
        return

    if sparse_path.exists():
        sparse_aligned_path.mkdir(parents=True, exist_ok=True)
        subprocess.call(
            [
                args.colmap_path,
                "model_aligner",
                "--input_path",
                sparse_path,
                "--ref_model_path",
                sparse_gt_path,
                "--output_path",
                sparse_aligned_path,
                "--alignment_max_error",
                str(max_ref_model_error),
            ]
        )


def process_scene(
    args: argparse.Namespace,
    scene_info: SceneInfo,
    prepare_scene: callable,
    position_accuracy_gt: float,
    num_threads: int,
) -> SceneResult:
    pycolmap.logging.info(
        f"Processing dataset={scene_info.dataset}, "
        f"category={scene_info.category}, "
        f"scene={scene_info.scene}"
    )

    prepare_scene(scene_info)

    sparse_gt = pycolmap.Reconstruction(scene_info.sparse_gt_path)

    colmap_reconstruction(
        args=args,
        workspace_path=scene_info.workspace_path,
        image_path=scene_info.image_path,
        camera_priors_sparse_gt=(
            sparse_gt if scene_info.camera_priors_from_sparse_gt else None
        ),
        num_threads=num_threads,
        colmap_extra_args=scene_info.colmap_extra_args,
    )

    # Merge all sub-models into a single reconstruction. Each sub-model will be
    # "randomly" aligned to the other sub-models. We then compute the overall
    # error over the merged reconstruction. With this simple appraoch, there is
    # a small chance that the randomly aligned images in one sub-model are
    # correctly aligned with other sub-models and the error is therefore
    # underestimated. However, this is very unlikely to happen.
    sparse_merged = pycolmap.Reconstruction()
    num_components = 0
    largest_component = 0
    for sparse_path in (scene_info.workspace_path / "sparse").iterdir():
        if not sparse_path.is_dir():
            continue
        num_components += 1
        sparse = None
        if args.error_type == "relative":
            sparse = pycolmap.Reconstruction(sparse_path)
        elif args.error_type == "absolute":
            sparse_aligned_path = scene_info.workspace_path / "sparse_aligned"
            colmap_alignment(
                args=args,
                sparse_path=sparse_path,
                sparse_gt_path=scene_info.sparse_gt_path,
                sparse_aligned_path=sparse_aligned_path,
                max_ref_model_error=position_accuracy_gt,
            )
            if (sparse_aligned_path / "images.bin").exists():
                sparse = pycolmap.Reconstruction(sparse_aligned_path)
        else:
            raise ValueError(f"Invalid error type: {args.error_type}")

        if sparse is not None:
            largest_component = max(largest_component, sparse.num_images())
            for image in sparse.images.values():
                if image.image_id in sparse_merged.images:
                    continue
                if image.camera_id not in sparse_merged.cameras:
                    sparse_merged.add_camera(image.camera)
                image.reset_camera_ptr()
                sparse_merged.add_image(image)

    if args.error_type == "relative":
        dts, dRs = compute_rel_errors(
            sparse_gt=sparse_gt,
            sparse=sparse_merged,
            min_proj_center_dist=position_accuracy_gt,
        )
        errors = np.maximum(dts, dRs)
    elif args.error_type == "absolute":
        dts, dRs = compute_abs_errors(
            sparse_gt=sparse_gt,
            sparse=sparse_merged,
        )
        errors = dts
    else:
        raise ValueError(f"Invalid error type: {args.error_type}")

    return SceneResult(
        scene_info=scene_info,
        errors=errors,
        num_images=sparse_gt.num_images(),
        num_reg_images=sparse_merged.num_images(),
        num_components=num_components,
        largest_component=largest_component,
    )


def process_scenes(
    args: argparse.Namespace,
    scene_infos: list[SceneInfo],
    prepare_scene: callable,
    position_accuracy_gt: float,
) -> dict[str, dict[str, SceneMetrics]]:
    error_thresholds = get_error_thresholds(args)

    num_threads = min(
        args.parallelism, 2 * max(1, int(args.parallelism / len(scene_infos)))
    )
    with multiprocessing.Pool(processes=args.parallelism) as p:
        results = p.map(
            functools.partial(
                process_scene,
                args,
                prepare_scene=prepare_scene,
                position_accuracy_gt=position_accuracy_gt,
                num_threads=num_threads,
            ),
            scene_infos,
        )

    metrics = collections.defaultdict(dict)
    errors_by_category = collections.defaultdict(list)
    total_num_images = 0
    total_num_reg_images = 0
    total_num_components = 0
    total_largest_components = 0
    num_scenes = len(results)
    for result in results:
        errors_by_category[result.scene_info.category].extend(result.errors)
        total_num_images += result.num_images
        total_num_reg_images += result.num_reg_images
        total_num_components += result.num_components
        total_largest_components += result.largest_component
        metrics[result.scene_info.category][result.scene_info.scene] = (
            SceneMetrics(
                aucs=compute_auc(
                    result.errors,
                    error_thresholds,
                    min_error=position_accuracy_gt,
                ),
                error_thresholds=error_thresholds,
                error_type=args.error_type,
                num_images=result.num_images,
                num_reg_images=result.num_reg_images,
                num_components=result.num_components,
                largest_component=result.largest_component,
            )
        )

    for category, errors in errors_by_category.items():
        metrics[category]["__all__"] = SceneMetrics(
            aucs=compute_auc(
                errors,
                error_thresholds,
                min_error=position_accuracy_gt,
            ),
            error_thresholds=error_thresholds,
            error_type=args.error_type,
            num_images=total_num_images,
            num_reg_images=total_num_reg_images,
            num_components=total_num_components,
            largest_component=total_largest_components,
        )
        metrics[category]["__avg__"] = SceneMetrics(
            aucs=compute_avg_auc(metrics[category]),
            error_thresholds=error_thresholds,
            error_type=args.error_type,
            num_images=int(round(total_num_images / num_scenes)),
            num_reg_images=int(round(total_num_reg_images / num_scenes)),
            num_components=int(round(total_num_components / num_scenes)),
            largest_component=int(round(total_largest_components / num_scenes)),
        )

    return metrics


def normalize_vec(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return vec / max(eps, np.linalg.norm(vec))


def vec_angular_dist_deg(vec1: np.ndarray, vec2: np.ndarray) -> float:
    cos_dist = np.clip(np.dot(normalize_vec(vec1), normalize_vec(vec2)), -1, 1)
    return np.rad2deg(np.acos(cos_dist))


def get_error_thresholds(args: argparse.Namespace) -> list[float]:
    if args.error_type == "relative":
        return np.array(args.rel_error_thresholds)
    elif args.error_type == "absolute":
        return np.array(args.abs_error_thresholds)
    else:
        raise ValueError(f"Invalid error type: {args.error_type}")


def compute_rel_errors(
    sparse_gt: pycolmap.Reconstruction,
    sparse: pycolmap.Reconstruction,
    min_proj_center_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes angular relative pose errors across all image pairs.

    Notice that this approach leads to a super-linear decrease in the AUC scores
    when multiple images fail to register. Consider that we have N images in
    total in a dataset and M images are registered in the evaluated
    reconstruction. In this case, we can compute "finite" errors for (N-M)^2
    pairs while the dataset has a total of N^2 pairs. In case of many
    unregistered images, the AUC score will drop much more than the
    (intuitively) expected (N-M) / N ratio. One could appropriately normalize by
    computing a single score per image through a suitable normalization of all
    pairwise errors per image. However, this becomes difficult when multiple
    sub-components are incorrectly stitched together in the same reconstruction
    (e.g., in the case of symmetry issues).
    """

    if sparse is None:
        pycolmap.logging.error("Reconstruction failed")
        return len(sparse_gt.images) * [np.inf], len(sparse_gt.images) * [180]

    images = {}
    for image in sparse.images.values():
        images[image.name] = image

    dts = []
    dRs = []
    for this_image_gt in sparse_gt.images.values():
        if this_image_gt.name not in images:
            for _ in range(sparse_gt.num_images() - 1):
                dts.append(np.inf)
                dRs.append(180)
            continue

        this_image = images[this_image_gt.name]

        for other_image_gt in sparse_gt.images.values():
            if this_image_gt.image_id == other_image_gt.image_id:
                continue

            if other_image_gt.name not in images:
                dts.append(np.inf)
                dRs.append(180)
                continue

            other_image = images[other_image_gt.name]

            other_from_this = (
                other_image.cam_from_world * this_image.cam_from_world.inverse()
            )
            other_from_this_gt = (
                other_image_gt.cam_from_world
                * this_image_gt.cam_from_world.inverse()
            )

            estimated_from_gt = other_from_this.inverse() * other_from_this_gt

            if (
                np.linalg.norm(other_from_this_gt.translation)
                < min_proj_center_dist
            ):
                # If the cameras almost coincide, then the angular direction
                # distance is unstable, because a small position change can
                # cause a large rotational error. In this case, we only measure
                # rotational relative pose error.
                dt = 0
            else:
                dt = vec_angular_dist_deg(
                    other_from_this.translation, other_from_this_gt.translation
                )

            dR = np.rad2deg(estimated_from_gt.rotation.angle())

            dts.append(dt)
            dRs.append(dR)

    return np.array(dts), np.array(dRs)


def compute_abs_errors(
    sparse_gt: pycolmap.Reconstruction, sparse: pycolmap.Reconstruction
) -> tuple[np.ndarray, np.ndarray]:
    """Computes rotational and translational absolute pose errors.

    Assumes that the input reconstructions are aligned in the same coordinate
    system. Computes one error per ground-truth image.
    """

    dts = np.full(len(sparse_gt.images), fill_value=np.inf, dtype=np.float64)
    dRs = np.full(len(sparse_gt.images), fill_value=180, dtype=np.float64)

    if sparse is None:
        pycolmap.logging.error("Reconstruction or alignment failed")
        return dts, dRs

    images = {}
    for image in sparse.images.values():
        images[image.name] = image

    dts = np.full(len(sparse_gt.images), fill_value=np.inf, dtype=np.float64)
    dRs = np.full(len(sparse_gt.images), fill_value=180, dtype=np.float64)
    for i, image_gt in enumerate(sparse_gt.images.values()):
        if image_gt.name not in images:
            continue

        image = images[image_gt.name]

        estimated_from_gt = (
            image.cam_from_world * image_gt.cam_from_world.inverse()
        )

        dts[i] = np.linalg.norm(estimated_from_gt.translation)
        dRs[i] = np.rad2deg(estimated_from_gt.rotation.angle())

    return dts, dRs


def compute_recall(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_elements = len(errors)
    errors = np.sort(errors)
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall


def compute_auc(
    errors: np.ndarray, thresholds: list[float], min_error: float = 0
) -> list[float]:
    if len(errors) == 0:
        raise ValueError("No errors to evaluate")

    errors, recall = compute_recall(errors)

    if min_error > 0:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = np.zeros(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapezoid(r, x=e) / t
        aucs[i] = auc * 100
    return aucs / 1.1


def compute_avg_auc(scene_metrics: dict[str, SceneMetrics]) -> list[float]:
    auc_sum = None
    num_scenes = 0
    for scene, metrics in scene_metrics.items():
        if scene.startswith("__") and scene.endswith("__"):
            continue
        num_scenes += 1
        if auc_sum is None:
            auc_sum = copy.copy(metrics.aucs)
        else:
            for i in range(len(auc_sum)):
                auc_sum[i] += metrics.aucs[i]
    return np.array(auc_sum) / num_scenes


def diff_metrics(
    metrics_a: dict[str, dict[str, SceneMetrics]],
    metrics_b: dict[str, dict[str, SceneMetrics]],
):
    """Computes difference between two sets of metrics.

    Raises exception if the metrics are inconsistent.
    """
    metrics_diff = copy.deepcopy(metrics_a)
    for dataset, category_metrics_a in metrics_a.items():
        if dataset not in metrics_b:
            raise ValueError(f"Dataset {dataset} not found in metrics_b")
        category_metrics_b = metrics_b[dataset]
        for category, scene_metrics_a in category_metrics_a.items():
            if category not in category_metrics_b:
                raise ValueError(f"Category {category} not found in metrics_b")
            scene_metrics_b = category_metrics_b[category]
            for scene, metrics_a in scene_metrics_a.items():
                if scene not in scene_metrics_b:
                    raise ValueError(f"Scene {scene} not found in metrics_b")
                metrics_b = scene_metrics_b[scene]
                if metrics_a.error_type != metrics_b.error_type or not np.all(
                    metrics_a.error_thresholds == metrics_b.error_thresholds
                ):
                    raise ValueError("Inconsistent error thresholds or types")
                metrics_diff[dataset][category][scene] = SceneMetrics(
                    aucs=metrics_a.aucs - metrics_b.aucs,
                    error_thresholds=metrics_a.error_thresholds,
                    error_type=metrics_a.error_type,
                    num_images=metrics_a.num_images - metrics_b.num_images,
                    num_reg_images=metrics_a.num_reg_images
                    - metrics_b.num_reg_images,
                    num_components=metrics_a.num_components
                    - metrics_b.num_components,
                    largest_component=metrics_a.largest_component
                    - metrics_b.largest_component,
                )
    return metrics_diff


def create_result_table(
    dataset_metrics: dict[str, dict[str, SceneMetrics]],
) -> str:
    first_metrics = next(
        iter(next(iter(next(iter(dataset_metrics.values())).values())).values())
    )
    if first_metrics.error_type == "relative":
        label = "AUC @ X deg (%)"
        thresholds = first_metrics.error_thresholds
    elif first_metrics.error_type == "absolute":
        label = "AUC @ X cm (%)"
        thresholds = 100 * first_metrics.error_thresholds
    else:
        raise ValueError(f"Invalid error type: {first_metrics.error_type}")

    column = "scenes"
    size_scenes = max(
        len(column) + 2,
        max(
            len(s)
            for d in dataset_metrics.values()
            for c in d.values()
            for s in c
        ),
    )
    size_aucs = max(len(label) + 2, len(thresholds) * 7 - 1)
    size_imgs = 12
    size_comps = 12
    size_sep = size_scenes + size_aucs + size_imgs + size_comps + 3
    header = (
        f"{column:=^{size_scenes}} {label:=^{size_aucs}} "
        f"{"images":=^{size_imgs}} {"components":=^{size_comps}}"
    )
    header += "\n" + " " * (size_scenes + 1)
    header += " ".join(f'{str(t).rstrip("."):^6}' for t in thresholds)
    header += "    reg   all  num largest"
    text = [header]
    for dataset, category_metrics in dataset_metrics.items():
        for category, scene_metrics in category_metrics.items():
            text.append(f"\n{dataset + '=' + category:=^{size_sep}}")
            for scene, metrics in scene_metrics.items():
                assert len(metrics.aucs) == len(thresholds)
                row = ""
                if scene == "__avg__":
                    scene = "average"
                    row += "-" * size_sep + "\n"
                if scene == "__all__":
                    scene = "overall"
                    row += "-" * size_sep + "\n"
                row += f"{scene:<{size_scenes}} "
                row += " ".join(f"{auc:>6.2f}" for auc in metrics.aucs)
                row += f" {metrics.num_reg_images:6d}"
                row += f"{metrics.num_images:6d}"
                row += f" {metrics.num_components:4d}"
                row += f"{metrics.largest_component:8d}"
                text.append(row)
    return "\n".join(text)
