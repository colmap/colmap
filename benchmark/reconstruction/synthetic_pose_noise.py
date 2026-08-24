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
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Measure relative-pose AUCs after perturbing dataset ground-truth poses.

The experiment assumes perfect registration within every GT component and
therefore isolates the ceiling imposed by pose noise. Rotation noise is an
isotropic Gaussian in the camera orientation's tangent space: every component
of the axis-angle perturbation has standard deviation ``sigma_R`` degrees.
Translation noise is applied to camera centers in world coordinates, with
standard deviation ``sigma_C`` meters independently in x, y, and z.

Example:
  python benchmark/reconstruction/synthetic_pose_noise.py \
    --dataset eth3d --categories dslr \
    --data_path benchmark/reconstruction/data \
    --output_dir benchmark/reconstruction/runs/pose-noise-upper-bound
"""

import argparse
import csv
import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

import pycolmap

if __package__:
    from .evaluation.datasets import DATASET_TYPES
    from .evaluation.statistics import NoiseCeiling, save_noise_ceiling
    from .evaluation.utils import OUTLIER_COMPONENT_ID
else:
    from evaluation.datasets import DATASET_TYPES
    from evaluation.statistics import NoiseCeiling, save_noise_ceiling
    from evaluation.utils import OUTLIER_COMPONENT_ID

DEFAULT_ROTATION_SIGMAS_DEG = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
DEFAULT_TRANSLATION_SIGMAS_M = [
    0,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
]
DEFAULT_THRESHOLDS_DEG = [0.5, 1, 5, 10]


@dataclass(frozen=True)
class ScenePoses:
    dataset: str
    category: str
    name: str
    position_accuracy_gt: float
    rotations: npt.NDArray[np.float64]
    centers: npt.NDArray[np.float64]
    src_indices: npt.NDArray[np.int64]
    tgt_indices: npt.NDArray[np.int64]
    gt_rel_rotations: npt.NDArray[np.float64]
    gt_rel_translations: npt.NDArray[np.float64]
    gt_baselines: npt.NDArray[np.float64]

    @property
    def num_images(self) -> int:
        return len(self.rotations)

    @property
    def num_pairs(self) -> int:
        return len(self.src_indices)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.category}/{self.name}"


def axis_angle_to_rotation_matrices(
    axis_angles: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert an N x 3 axis-angle array to N rotation matrices."""
    axis_angles = np.asarray(axis_angles, dtype=np.float64)
    theta = np.linalg.norm(axis_angles, axis=1)
    theta2 = theta * theta
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    small = theta < 1e-8
    a[small] = 1 - theta2[small] / 6 + theta2[small] ** 2 / 120
    b[small] = 0.5 - theta2[small] / 24 + theta2[small] ** 2 / 720
    a[~small] = np.sin(theta[~small]) / theta[~small]
    b[~small] = (1 - np.cos(theta[~small])) / theta2[~small]

    skew = np.zeros((len(axis_angles), 3, 3), dtype=np.float64)
    x, y, z = axis_angles.T
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x
    identity = np.broadcast_to(np.eye(3), skew.shape)
    return identity + a[:, None, None] * skew + b[:, None, None] * (skew @ skew)


def _ordered_pair_indices(
    component_ids: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    num_images = len(component_ids)
    indices = np.arange(num_images)
    src = np.repeat(indices, num_images)
    tgt = np.tile(indices, num_images)
    keep = (
        (src != tgt)
        & (component_ids[src] != OUTLIER_COMPONENT_ID)
        & (component_ids[src] == component_ids[tgt])
    )
    return src[keep], tgt[keep]


def _relative_poses(
    rotations: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    src: npt.NDArray[np.int64],
    tgt: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rel_rotations = rotations[tgt] @ rotations[src].transpose(0, 2, 1)
    baselines_in_world = centers[src] - centers[tgt]
    rel_translations = np.einsum(
        "nij,nj->ni", rotations[tgt], baselines_in_world
    )
    return rel_rotations, rel_translations


def load_dataset_scenes(
    data_path: Path,
    dataset_name: str,
    categories: list[str],
    scene_names: list[str],
    run_path: Path,
) -> list[ScenePoses]:
    if dataset_name not in DATASET_TYPES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    dataset = DATASET_TYPES[dataset_name](
        data_path=data_path,
        categories=categories,
        scenes=scene_names,
        run_path=run_path,
        run_name="synthetic-pose-noise",
    )
    scenes = []
    scene_infos = dataset.list_scenes()
    for scene_info in scene_infos:
        dataset.prepare_scene(scene_info)
        reconstruction = pycolmap.Reconstruction(scene_info.sparse_gt_path)
        images = sorted(
            reconstruction.images.values(), key=lambda image: image.name
        )
        normalized_rotations = []
        for image in images:
            rotation = pycolmap.Rotation3d(image.cam_from_world().rotation.quat)
            rotation.normalize()
            normalized_rotations.append(rotation.matrix())
        rotations = np.stack(normalized_rotations)
        centers = np.stack([image.projection_center() for image in images])
        component_by_name = scene_info.image_name_to_component
        component_ids = np.array(
            [
                component_by_name.get(image.name, OUTLIER_COMPONENT_ID)
                if component_by_name
                else 0
                for image in images
            ],
            dtype=np.int64,
        )
        src, tgt = _ordered_pair_indices(component_ids)
        if len(src) == 0:
            raise ValueError(
                f"Scene {scene_info.dataset}/{scene_info.category}/"
                f"{scene_info.scene} has no evaluable relative-pose pairs"
            )
        gt_rel_rotations, gt_rel_translations = _relative_poses(
            rotations, centers, src, tgt
        )
        scenes.append(
            ScenePoses(
                # Match the outer dataset key used in evaluate.py reports.
                dataset=dataset_name,
                category=scene_info.category,
                name=scene_info.scene,
                position_accuracy_gt=dataset.position_accuracy_gt,
                rotations=rotations,
                centers=centers,
                src_indices=src,
                tgt_indices=tgt,
                gt_rel_rotations=gt_rel_rotations,
                gt_rel_translations=gt_rel_translations,
                gt_baselines=np.linalg.norm(gt_rel_translations, axis=1),
            )
        )
    if not scenes:
        raise ValueError(f"No ground-truth scenes found for {dataset_name}")
    return scenes


def compute_relative_pose_errors(
    scene: ScenePoses,
    rotations: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    min_baseline_m: float | None = None,
) -> npt.NDArray[np.float64]:
    """Reproduce the reconstruction benchmark's max(dt, dR) pair error."""
    return compute_relative_pose_error_grid(
        scene,
        rotations[None, ...],
        centers[None, ...],
        scene.position_accuracy_gt
        if min_baseline_m is None
        else min_baseline_m,
    )[0, 0]


def compute_relative_pose_error_grid(
    scene: ScenePoses,
    rotations: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    min_baseline_m: float | None = None,
) -> npt.NDArray[np.float64]:
    """Compute errors for R rotation levels by T camera-center levels."""
    est_rel_rotations = rotations[:, scene.tgt_indices] @ rotations[
        :, scene.src_indices
    ].transpose(0, 1, 3, 2)
    estimated_from_gt = (
        est_rel_rotations.transpose(0, 1, 3, 2)
        @ scene.gt_rel_rotations[None, ...]
    )
    rotation_cosines = np.clip(
        (np.trace(estimated_from_gt, axis1=2, axis2=3) - 1) / 2, -1, 1
    )
    rotation_errors = np.rad2deg(np.arccos(rotation_cosines))

    baselines_in_world = (
        centers[:, scene.src_indices] - centers[:, scene.tgt_indices]
    )
    est_rel_translations = np.einsum(
        "rpij,tpj->rtpi",
        rotations[:, scene.tgt_indices],
        baselines_in_world,
    )
    est_norms = np.linalg.norm(est_rel_translations, axis=3)
    gt_norms = np.linalg.norm(scene.gt_rel_translations, axis=1)
    denom = est_norms * gt_norms[None, None, :]
    translation_cosines = np.ones_like(denom)
    valid = denom > np.finfo(np.float64).tiny
    dot_products = np.einsum(
        "rtpi,pi->rtp", est_rel_translations, scene.gt_rel_translations
    )
    translation_cosines[valid] = dot_products[valid] / denom[valid]
    translation_errors = np.rad2deg(
        np.arccos(np.clip(translation_cosines, -1, 1))
    )
    if min_baseline_m is None:
        min_baseline_m = scene.position_accuracy_gt
    translation_errors[:, :, scene.gt_baselines < min_baseline_m] = 0
    return np.maximum(translation_errors, rotation_errors[:, None, :])


def aucs_from_errors(
    errors: npt.NDArray[np.float64],
    thresholds_deg: npt.NDArray[np.float64],
    min_error: float = 0,
) -> npt.NDArray[np.float64]:
    """Compute the benchmark's trapezoidal empirical-CDF AUCs."""
    errors = np.sort(errors)
    recalls = (np.arange(len(errors)) + 1) / len(errors)
    if min_error > 0:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_recall = min_index / len(errors)
        recalls = np.r_[min_recall, min_recall, recalls[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recalls = np.r_[0, recalls]
        errors = np.r_[0, errors]

    aucs = np.zeros(len(thresholds_deg), dtype=np.float64)
    for index, threshold in enumerate(thresholds_deg):
        last_index = np.searchsorted(errors, threshold, side="right")
        threshold_recalls = np.r_[recalls[:last_index], recalls[last_index - 1]]
        threshold_errors = np.r_[errors[:last_index], threshold]
        aucs[index] = (
            100
            * np.trapezoid(threshold_recalls, x=threshold_errors)
            / threshold
        )
    return aucs


def run_experiment(
    scenes: list[ScenePoses],
    rotation_sigmas_deg: npt.NDArray[np.float64],
    translation_sigmas_m: npt.NDArray[np.float64],
    thresholds_deg: npt.NDArray[np.float64],
    num_trials: int,
    seed: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """Run common-random-number Monte Carlo trials over the noise grid."""
    shape = (
        num_trials,
        len(rotation_sigmas_deg),
        len(translation_sigmas_m),
        len(thresholds_deg),
    )
    scene_trials = np.zeros(
        (num_trials, len(scenes), *shape[1:]), dtype=np.float64
    )
    pooled_trials = np.zeros(shape, dtype=np.float64)
    total_pairs = sum(scene.num_pairs for scene in scenes)
    rng = np.random.default_rng(seed)

    for trial in range(num_trials):
        pooled_errors = np.empty(
            (
                len(rotation_sigmas_deg),
                len(translation_sigmas_m),
                total_pairs,
            ),
            dtype=np.float64,
        )
        pair_offset = 0
        for scene_idx, scene in enumerate(scenes):
            rotation_standard_normal = rng.standard_normal(
                (scene.num_images, 3)
            )
            translation_standard_normal = rng.standard_normal(
                (scene.num_images, 3)
            )
            noisy_rotations = np.stack(
                [
                    axis_angle_to_rotation_matrices(
                        np.deg2rad(sigma_deg) * rotation_standard_normal
                    )
                    @ scene.rotations
                    for sigma_deg in rotation_sigmas_deg
                ]
            )
            noisy_centers = np.stack(
                [
                    scene.centers + sigma_m * translation_standard_normal
                    for sigma_m in translation_sigmas_m
                ]
            )
            error_grid = compute_relative_pose_error_grid(
                scene, noisy_rotations, noisy_centers
            )

            for rotation_idx in range(len(rotation_sigmas_deg)):
                for translation_idx in range(len(translation_sigmas_m)):
                    errors = error_grid[rotation_idx, translation_idx]
                    aucs = aucs_from_errors(
                        errors,
                        thresholds_deg,
                        min_error=scene.position_accuracy_gt,
                    )
                    scene_trials[
                        trial, scene_idx, rotation_idx, translation_idx
                    ] = aucs
                    pooled_errors[
                        rotation_idx,
                        translation_idx,
                        pair_offset : pair_offset + scene.num_pairs,
                    ] = errors
            pair_offset += scene.num_pairs
        for rotation_idx in range(len(rotation_sigmas_deg)):
            for translation_idx in range(len(translation_sigmas_m)):
                pooled_trials[trial, rotation_idx, translation_idx] = (
                    aucs_from_errors(
                        pooled_errors[rotation_idx, translation_idx],
                        thresholds_deg,
                        min_error=min(
                            scene.position_accuracy_gt for scene in scenes
                        ),
                    )
                )
        print(f"trial {trial + 1}/{num_trials}", flush=True)

    macro_trials = np.mean(scene_trials, axis=1)
    return {
        "macro_mean": np.mean(macro_trials, axis=0),
        "macro_std": np.std(macro_trials, axis=0, ddof=1)
        if num_trials > 1
        else np.zeros(shape[1:]),
        "pooled_mean": np.mean(pooled_trials, axis=0),
        "pooled_std": np.std(pooled_trials, axis=0, ddof=1)
        if num_trials > 1
        else np.zeros(shape[1:]),
        "macro_trials": macro_trials,
        "pooled_trials": pooled_trials,
        "scene_trials": scene_trials,
    }


def _format_level(value: float) -> str:
    return f"{value:g}"


def _interpolate_color(value: float) -> str:
    stops = [
        (0.0, (68, 1, 84)),
        (0.35, (49, 104, 142)),
        (0.7, (53, 183, 121)),
        (1.0, (253, 231, 37)),
    ]
    unit = np.clip(value / 100, 0, 1)
    for (left_x, left), (right_x, right) in zip(stops, stops[1:], strict=False):
        if unit <= right_x:
            alpha = (unit - left_x) / (right_x - left_x)
            rgb = tuple(
                round((1 - alpha) * a + alpha * b)
                for a, b in zip(left, right, strict=True)
            )
            return "#" + "".join(f"{channel:02x}" for channel in rgb)
    return "#fde725"


def write_heatmap_svg(
    path: Path,
    values: npt.NDArray[np.float64],
    rotation_sigmas_deg: npt.NDArray[np.float64],
    translation_sigmas_m: npt.NDArray[np.float64],
    thresholds_deg: npt.NDArray[np.float64],
    title: str,
) -> None:
    panel_width, panel_height = 760, 540
    width, height = 2 * panel_width, 2 * panel_height + 50
    cell_width = 58
    cell_height = 39
    grid_width = cell_width * len(translation_sigmas_m)
    grid_height = cell_height * len(rotation_sigmas_deg)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:13px}"
        ".cell{font-size:11px;font-weight:600}.panel{font-size:19px;font-weight:600}"
        ".title{font-size:24px;font-weight:700}</style>",
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">'
        f"{html.escape(title)}</text>",
    ]
    for threshold_idx, threshold in enumerate(thresholds_deg):
        panel_col = threshold_idx % 2
        panel_row = threshold_idx // 2
        panel_x = panel_col * panel_width
        panel_y = 45 + panel_row * panel_height
        grid_x = panel_x + 175
        grid_y = panel_y + 70
        parts.append(
            f'<text class="panel" x="{panel_x + panel_width / 2}" '
            f'y="{panel_y + 25}" text-anchor="middle">'
            f"AUC @ {threshold:g}°</text>"
        )
        parts.append(
            f'<text x="{grid_x + grid_width / 2}" '
            f'y="{grid_y + grid_height + 67}" '
            'text-anchor="middle">Camera-center noise σ (m)</text>'
        )
        parts.append(
            f'<text transform="translate({grid_x - 125} '
            f'{grid_y + grid_height / 2}) rotate(-90)" text-anchor="middle">'
            "Rotation tangent noise σ (deg/axis)</text>"
        )
        for col, sigma in enumerate(translation_sigmas_m):
            x = grid_x + (col + 0.5) * cell_width
            parts.append(
                f'<text class="tick" x="{x}" y="{grid_y + grid_height + 23}" '
                f'text-anchor="middle">{_format_level(sigma)}</text>'
            )
        for row, sigma in enumerate(rotation_sigmas_deg):
            y = grid_y + (row + 0.5) * cell_height
            parts.append(
                f'<text class="tick" x="{grid_x - 12}" y="{y + 5}" '
                f'text-anchor="end">{_format_level(sigma)}</text>'
            )
            for col, _ in enumerate(translation_sigmas_m):
                value = values[row, col, threshold_idx]
                x = grid_x + col * cell_width
                y0 = grid_y + row * cell_height
                color = _interpolate_color(value)
                text_color = "#111" if value >= 62 else "#fff"
                parts.extend(
                    [
                        f'<rect x="{x}" y="{y0}" width="{cell_width}" '
                        f'height="{cell_height}" fill="{color}" '
                        'stroke="white"/>',
                        f'<text class="cell" x="{x + cell_width / 2}" '
                        f'y="{y0 + cell_height / 2 + 4}" text-anchor="middle" '
                        f'style="fill:{text_color}">{value:.1f}</text>',
                    ]
                )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_csv(
    path: Path,
    results: dict[str, npt.NDArray[np.float64]],
    rotation_sigmas_deg: npt.NDArray[np.float64],
    translation_sigmas_m: npt.NDArray[np.float64],
    thresholds_deg: npt.NDArray[np.float64],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "aggregation",
                "rotation_sigma_deg_per_axis",
                "translation_sigma_m_per_axis",
                "auc_threshold_deg",
                "auc_mean_percent",
                "auc_std_percent",
            ]
        )
        for aggregation in ["macro", "pooled"]:
            means = results[f"{aggregation}_mean"]
            stds = results[f"{aggregation}_std"]
            for rotation_idx, rotation_sigma in enumerate(rotation_sigmas_deg):
                for translation_idx, translation_sigma in enumerate(
                    translation_sigmas_m
                ):
                    for threshold_idx, threshold in enumerate(thresholds_deg):
                        writer.writerow(
                            [
                                aggregation,
                                rotation_sigma,
                                translation_sigma,
                                threshold,
                                means[
                                    rotation_idx, translation_idx, threshold_idx
                                ],
                                stds[
                                    rotation_idx, translation_idx, threshold_idx
                                ],
                            ]
                        )


def _markdown_grid(
    values: npt.NDArray[np.float64],
    rotation_sigmas_deg: npt.NDArray[np.float64],
    translation_sigmas_m: npt.NDArray[np.float64],
) -> list[str]:
    lines = [
        "| σR deg/axis \\ σC m/axis | "
        + " | ".join(_format_level(x) for x in translation_sigmas_m)
        + " |",
        "|---|" + "---:|" * len(translation_sigmas_m),
    ]
    for sigma, row in zip(rotation_sigmas_deg, values, strict=True):
        lines.append(
            f"| {_format_level(sigma)} | "
            + " | ".join(f"{value:.2f}" for value in row)
            + " |"
        )
    return lines


def write_markdown_report(
    path: Path,
    scenes: list[ScenePoses],
    results: dict[str, npt.NDArray[np.float64]],
    rotation_sigmas_deg: npt.NDArray[np.float64],
    translation_sigmas_m: npt.NDArray[np.float64],
    thresholds_deg: npt.NDArray[np.float64],
    num_trials: int,
    seed: int,
) -> None:
    dataset_labels = sorted({f"{s.dataset}/{s.category}" for s in scenes})
    lines = [
        "# Synthetic pose-noise upper bound",
        "",
        f"- Dataset categories: {', '.join(dataset_labels)}",
        f"- Scenes: {len(scenes)} "
        f"({sum(s.num_images for s in scenes)} cameras, "
        f"{sum(s.num_pairs for s in scenes):,} ordered pairs)",
        f"- Monte Carlo trials: {num_trials}; random seed: {seed}",
        "- Noise: independent per-camera Gaussian SO(3) tangent components "
        "(σR degrees/component) and world-frame camera-center components "
        "(σC meters/component)",
        "- Metric: the reconstruction benchmark's max(rotation error, "
        "translation-direction error), followed by its AUC integration",
        "- Assumption: every GT component is reconstructed perfectly; only "
        "synthetic pose noise limits the score",
        "- Primary aggregate: macro mean of per-scene AUCs. The pooled-pair "
        "aggregate weights scenes by their number of ordered camera pairs.",
        "",
        "## Macro scene-average AUC mean (%)",
        "",
    ]
    for threshold_idx, threshold in enumerate(thresholds_deg):
        lines.extend(
            [
                f"### AUC @ {threshold:g}°",
                "",
                *_markdown_grid(
                    results["macro_mean"][:, :, threshold_idx],
                    rotation_sigmas_deg,
                    translation_sigmas_m,
                ),
                "",
            ]
        )
    lines.extend(["## Pooled ordered-pair AUC mean (%)", ""])
    for threshold_idx, threshold in enumerate(thresholds_deg):
        lines.extend(
            [
                f"### AUC @ {threshold:g}°",
                "",
                *_markdown_grid(
                    results["pooled_mean"][:, :, threshold_idx],
                    rotation_sigmas_deg,
                    translation_sigmas_m,
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_TYPES),
        default="eth3d",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="Dataset categories to include; empty includes every category.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(__file__).parent / "data",
    )
    parser.add_argument("--scenes", nargs="+", default=[])
    parser.add_argument(
        "--rotation_sigmas_deg",
        nargs="+",
        type=float,
        default=DEFAULT_ROTATION_SIGMAS_DEG,
    )
    parser.add_argument(
        "--translation_sigmas_m",
        nargs="+",
        type=float,
        default=DEFAULT_TRANSLATION_SIGMAS_M,
    )
    parser.add_argument(
        "--thresholds_deg",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS_DEG,
    )
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ceiling_rotation_sigma_deg",
        type=float,
        default=None,
        help="Rotation sigma to select for the optional ceiling artifact.",
    )
    parser.add_argument(
        "--ceiling_translation_sigma_m",
        type=float,
        default=None,
        help="Center sigma to select for the optional ceiling artifact.",
    )
    parser.add_argument(
        "--ceiling_output",
        type=Path,
        default=None,
        help="Optional ceiling artifact path; defaults below output_dir.",
    )
    parser.add_argument(
        "--ceiling_calibrated",
        action="store_true",
        help="Mark the selected uncertainty model as externally calibrated. "
        "Do not set this for a hypothetical sensitivity profile.",
    )
    parser.add_argument(
        "--ceiling_description",
        default="",
        help="Provenance or description of the selected uncertainty model.",
    )
    args = parser.parse_args()
    if args.num_trials <= 0:
        parser.error("--num_trials must be positive")
    for name in ["rotation_sigmas_deg", "translation_sigmas_m"]:
        if any(value < 0 for value in getattr(args, name)):
            parser.error(f"--{name} values must be non-negative")
    if any(value <= 0 for value in args.thresholds_deg):
        parser.error("--thresholds_deg values must be positive")
    ceiling_values = (
        args.ceiling_rotation_sigma_deg,
        args.ceiling_translation_sigma_m,
    )
    if (ceiling_values[0] is None) != (ceiling_values[1] is None):
        parser.error(
            "provide both --ceiling_rotation_sigma_deg and "
            "--ceiling_translation_sigma_m"
        )
    if args.ceiling_output is not None and ceiling_values[0] is None:
        parser.error("--ceiling_output requires a selected ceiling noise level")
    if args.ceiling_calibrated and ceiling_values[0] is None:
        parser.error("--ceiling_calibrated requires a selected ceiling model")
    if args.ceiling_calibrated and not args.ceiling_description:
        parser.error(
            "--ceiling_calibrated requires --ceiling_description provenance"
        )
    if ceiling_values[0] is not None:
        if any(value < 0 for value in ceiling_values):
            parser.error("selected ceiling noise levels must be non-negative")
        for selected, levels, name in [
            (
                ceiling_values[0],
                args.rotation_sigmas_deg,
                "rotation sigma",
            ),
            (
                ceiling_values[1],
                args.translation_sigmas_m,
                "translation sigma",
            ),
        ]:
            if not any(
                np.isclose(selected, level, rtol=0, atol=1e-12)
                for level in levels
            ):
                parser.error(f"selected {name} must be in its sampled levels")
    if args.output_dir is None:
        category_suffix = (
            "_" + "-".join(args.categories) if args.categories else ""
        )
        args.output_dir = (
            Path(__file__).parent
            / "runs"
            / f"synthetic_pose_noise_{args.dataset}{category_suffix}"
        )
    return args


def _find_level_index(values: np.ndarray, selected: float, name: str) -> int:
    matches = np.flatnonzero(np.isclose(values, selected, rtol=0, atol=1e-12))
    if len(matches) != 1:
        raise ValueError(
            f"Selected {name}={selected:g} is not a unique sampled level"
        )
    return int(matches[0])


def _ground_truth_digest(scenes: list[ScenePoses]) -> str:
    """Digest pose/evaluation inputs to identify stale ceiling artifacts."""
    digest = hashlib.sha256()
    for scene in scenes:
        digest.update(scene.key.encode())
        digest.update(np.float64(scene.position_accuracy_gt).tobytes())
        for array in [
            scene.rotations,
            scene.centers,
            scene.src_indices,
            scene.tgt_indices,
        ]:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    rotation_sigmas_deg = np.asarray(args.rotation_sigmas_deg, dtype=np.float64)
    translation_sigmas_m = np.asarray(
        args.translation_sigmas_m, dtype=np.float64
    )
    thresholds_deg = np.asarray(args.thresholds_deg, dtype=np.float64)
    scenes = load_dataset_scenes(
        data_path=args.data_path,
        dataset_name=args.dataset,
        categories=args.categories,
        scene_names=args.scenes,
        run_path=args.output_dir,
    )
    print(
        f"Loaded {len(scenes)} scenes, {sum(s.num_images for s in scenes)} "
        f"cameras, {sum(s.num_pairs for s in scenes)} ordered pairs"
    )
    results = run_experiment(
        scenes=scenes,
        rotation_sigmas_deg=rotation_sigmas_deg,
        translation_sigmas_m=translation_sigmas_m,
        thresholds_deg=thresholds_deg,
        num_trials=args.num_trials,
        seed=args.seed,
    )
    ground_truth_digest = _ground_truth_digest(scenes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "summary.csv",
        results,
        rotation_sigmas_deg,
        translation_sigmas_m,
        thresholds_deg,
    )
    metadata = {
        "dataset": args.dataset,
        "categories": sorted({scene.category for scene in scenes}),
        "scene_keys": [scene.key for scene in scenes],
        "num_images": sum(scene.num_images for scene in scenes),
        "num_ordered_pairs": sum(scene.num_pairs for scene in scenes),
        "num_trials": args.num_trials,
        "seed": args.seed,
        "ground_truth_digest_sha256": ground_truth_digest,
        "rotation_sigmas_deg_per_axis": rotation_sigmas_deg.tolist(),
        "translation_sigmas_m_per_axis": translation_sigmas_m.tolist(),
        "thresholds_deg": thresholds_deg.tolist(),
        "macro_mean": results["macro_mean"].tolist(),
        "macro_std": results["macro_std"].tolist(),
        "pooled_mean": results["pooled_mean"].tolist(),
        "pooled_std": results["pooled_std"].tolist(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "trials.npz",
        macro=results["macro_trials"],
        pooled=results["pooled_trials"],
        per_scene=results["scene_trials"],
        scene_keys=np.asarray([scene.key for scene in scenes]),
        rotation_sigmas_deg=rotation_sigmas_deg,
        translation_sigmas_m=translation_sigmas_m,
        thresholds_deg=thresholds_deg,
    )
    write_markdown_report(
        args.output_dir / "report.md",
        scenes,
        results,
        rotation_sigmas_deg,
        translation_sigmas_m,
        thresholds_deg,
        args.num_trials,
        args.seed,
    )
    write_heatmap_svg(
        args.output_dir / "auc_heatmaps_macro.svg",
        results["macro_mean"],
        rotation_sigmas_deg,
        translation_sigmas_m,
        thresholds_deg,
        f"{args.dataset} pose-noise ceiling — macro scene-average AUC (%)",
    )
    write_heatmap_svg(
        args.output_dir / "auc_heatmaps_pooled.svg",
        results["pooled_mean"],
        rotation_sigmas_deg,
        translation_sigmas_m,
        thresholds_deg,
        f"{args.dataset} pose-noise ceiling — pooled-pair AUC (%)",
    )
    if args.ceiling_rotation_sigma_deg is not None:
        rotation_idx = _find_level_index(
            rotation_sigmas_deg,
            args.ceiling_rotation_sigma_deg,
            "rotation sigma",
        )
        translation_idx = _find_level_index(
            translation_sigmas_m,
            args.ceiling_translation_sigma_m,
            "translation sigma",
        )
        ceiling_path = args.ceiling_output or (
            args.output_dir / "noise_ceiling.npz"
        )
        ceiling_scores = results["scene_trials"][
            :, :, rotation_idx, translation_idx, :
        ].transpose(1, 0, 2)
        save_noise_ceiling(
            ceiling_path,
            NoiseCeiling(
                scene_keys=tuple(scene.key for scene in scenes),
                scores=ceiling_scores,
                thresholds=thresholds_deg,
                error_type="relative_auc",
                metadata={
                    "schema_version": 1,
                    "dataset": args.dataset,
                    "categories": sorted({scene.category for scene in scenes}),
                    "calibrated": args.ceiling_calibrated,
                    "description": args.ceiling_description,
                    "uncertainty_model": "iid_gaussian_pose",
                    "rotation_sigma_deg_per_axis": (
                        args.ceiling_rotation_sigma_deg
                    ),
                    "translation_sigma_m_per_axis": (
                        args.ceiling_translation_sigma_m
                    ),
                    "num_monte_carlo_draws": args.num_trials,
                    "random_seed": args.seed,
                    "ground_truth_digest_sha256": ground_truth_digest,
                },
            ),
        )
        print(f"Wrote noise ceiling artifact to {ceiling_path}")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
