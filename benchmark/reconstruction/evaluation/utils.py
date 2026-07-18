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
import ctypes
import dataclasses
import datetime
import functools
import multiprocessing
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

import pycolmap

from .covisibility import filter_covisibility  # noqa: F401
from .geometry import normalize_vec, vec_angular_dist_deg  # noqa: F401

_PR_SET_PDEATHSIG = 1
_LIBC = (
    ctypes.CDLL("libc.so.6", use_errno=True)
    if platform.system() == "Linux"
    else None
)


def _set_pdeathsig() -> None:
    """preexec_fn: ensure child process is killed if parent dies."""
    if _LIBC is not None:
        _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)


def _init_pool_worker() -> None:
    """Pool initializer: ignore SIGINT in workers so the main process
    handles KeyboardInterrupt and terminates the pool cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_with_log(
    cmd: list, log_path: Path, check: bool = True, **kwargs
) -> int:
    """Run a subprocess, redirecting stdout+stderr to log_path (overwrite).

    Always uses preexec_fn=_set_pdeathsig so children die with their parent.
    Raises CalledProcessError on non-zero exit when check=True.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as fh:
        runner = subprocess.check_call if check else subprocess.call
        return runner(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            preexec_fn=_set_pdeathsig,
            **kwargs,
        )


@dataclasses.dataclass(kw_only=True)
class SceneInfo:
    # Dataset name.
    dataset: str
    # Category name.
    category: str
    # Scene name.
    scene: str
    # Number of input images in the scene.
    num_images: int
    # Path to the workspace directory in the run directory.
    workspace_path: Path
    # Path to the input images.
    image_path: Path
    # Path to the ground-truth sparse reconstruction.
    sparse_gt_path: Path
    # Whether the dataset has camera priors.
    has_camera_priors: bool
    # Additional arguments for the COLMAP reconstruction command.
    colmap_extra_args: list[str]


@dataclasses.dataclass(kw_only=True)
class SceneResult:
    # Scene information for which the result was computed.
    scene_info: SceneInfo
    # Flat list of errors.
    errors: npt.NDArray[np.floating]
    # Number of images in the scene.
    num_images: int
    # Number of registered images in the scene (over all components).
    num_reg_images: int
    # Number of components in the scene.
    num_components: int
    # Number of images in the largest component.
    largest_component: int


@dataclasses.dataclass(kw_only=True)
class Metrics:
    # Recall at specified error thresholds.
    recalls: npt.NDArray[np.floating]
    # Area under the curve (AUC) scores at specified error thresholds.
    aucs: npt.NDArray[np.floating]
    error_thresholds: npt.NDArray[np.floating]
    error_type: str
    # Number of images in the scene.
    num_images: int
    # Number of registered images in the scene (over all components).
    num_reg_images: int
    # Number of components in the scene.
    num_components: int
    # Number of images in the largest component.
    largest_component: int
    # Raw errors that produced aucs/recalls. Empty for entries (like __avg__)
    # where no underlying error pool exists. Retained so higher-level summaries
    # (per-dataset, overall) can recompute pooled __all__ statistics.
    errors: npt.NDArray[np.floating] = dataclasses.field(
        default_factory=lambda: np.array([])
    )
    # Ground-truth position accuracy (used as min_error when computing AUC).
    # Carried so cross-dataset aggregations can pick a sensible value.
    position_accuracy_gt: float = 0.0


MetricsByScene = dict[str, Metrics]
MetricsByCatByScene = dict[str, MetricsByScene]
MetricsByDatasetByCatByScene = dict[str, MetricsByCatByScene]

# Identifies one scene across reports: (dataset, category, scene).
SceneKey = tuple[str, str, str]


class Dataset(ABC):
    def __init__(
        self,
        data_path: Path,
        categories: list[str],
        scenes: list[Path],
        run_path: Path,
        run_name: str,
    ):
        self.data_path = data_path
        self.categories = categories
        self.scenes = scenes
        self.run_path = run_path
        self.run_name = run_name

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


class _PhaseTracker:
    """Worker-side helper that publishes the current phase for a scene to a
    shared dict. No-op when status_dict is None."""

    def __init__(self, status_dict=None, scene_key: str = "") -> None:
        self._dict = status_dict
        self._key = scene_key

    def set(self, phase: str) -> None:
        if self._dict is not None:
            self._dict[self._key] = phase


def _scene_key(scene_info: SceneInfo) -> str:
    return f"{scene_info.dataset}/{scene_info.category}/{scene_info.scene}"


def _run_progress_monitor(
    status_dict, total: int, stop_event: threading.Event
) -> None:
    """Render a live progress display of in-flight scenes until stop_event is
    set. Counts entries marked "done" toward overall completion; everything
    else is shown as an in-progress task with a spinner and elapsed time."""
    from rich.console import Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    overall = Progress(
        TextColumn("[bold]Scenes[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    scenes = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TextColumn("[cyan]{task.fields[phase]:>14}[/cyan]"),
        TimeElapsedColumn(),
    )
    overall_task = overall.add_task("scenes", total=total)
    scene_tasks: dict[str, int] = {}

    def refresh() -> None:
        snapshot = dict(status_dict)
        done = sum(1 for v in snapshot.values() if v == "finished")
        overall.update(overall_task, completed=done)
        in_progress = {k: v for k, v in snapshot.items() if v != "finished"}
        for key in list(scene_tasks):
            if key not in in_progress:
                scenes.remove_task(scene_tasks.pop(key))
        for key, phase in in_progress.items():
            if key in scene_tasks:
                scenes.update(scene_tasks[key], phase=phase)
            else:
                scene_tasks[key] = scenes.add_task(key, total=None, phase=phase)

    with Live(Group(overall, scenes), refresh_per_second=4):
        while not stop_event.is_set():
            refresh()
            stop_event.wait(0.5)
        refresh()


def filter_smallest_scenes_per_category(
    scene_infos: list[SceneInfo], num_scenes: int
) -> list[SceneInfo]:
    """Keep only the `num_scenes` smallest scenes (by num_images) per category,
    preserving the original order."""
    indices_by_category: dict[str, list[int]] = collections.defaultdict(list)
    for i, scene_info in enumerate(scene_infos):
        indices_by_category[scene_info.category].append(i)

    keep: set[int] = set()
    for indices in indices_by_category.values():
        smallest = sorted(indices, key=lambda i: scene_infos[i].num_images)[
            :num_scenes
        ]
        keep.update(smallest)

    return [scene_infos[i] for i in sorted(keep)]


def parse_args(description: str | None = None) -> argparse.Namespace:
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_path", default=Path(__file__).parent.parent / "data", type=Path
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
        "--progress",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Show a live progress display of in-flight scenes "
        "(default: enabled when stdout is a TTY).",
    )
    parser.add_argument(
        "--fast",
        default=False,
        action="store_true",
        help="Fast mode: only evaluate the N smallest scenes per category, "
        "where N is set by --fast_num_scenes.",
    )
    parser.add_argument(
        "--fast_num_scenes",
        type=int,
        default=1,
        help="Number of smallest scenes per category to evaluate in --fast "
        "mode.",
    )
    parser.add_argument(
        "--run_path", default=Path(__file__).parent.parent / "runs", type=Path
    )
    parser.add_argument("--run_name", default=datetime_str)
    parser.add_argument(
        "--report_name",
        default=f"report-{datetime_str}",
        help="Report file stem, without the .pkl suffix. With "
        "--seeds/--num_seeds it is the stem of every run, which are written "
        "as <report_name>_s<seed>.pkl; compare.py reads that naming back via "
        "--report_a_path_prefix/--report_b_path_prefix.",
    )
    parser.add_argument(
        "--overwrite_database", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_matches", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_two_view_geometries",
        default=False,
        action="store_true",
        help="Clear two-view geometries (inlier matches) but keep raw matches, "
        "so geometric verification is recomputed from the cached matches. "
        "Useful for re-tuning geometric verification without re-matching.",
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
        "--num_threads",
        type=int,
        default=-1,
        help=(
            "Total number of threads to use across all parallel scenes. "
            "Defaults to 2x the number of logical CPU cores (-1)."
        ),
    )
    parser.add_argument(
        "--num_parallel_scenes",
        type=int,
        default=-1,
        help=(
            "Number of scenes to reconstruct in parallel. "
            "Defaults to max(1, num_threads // 4) (-1)."
        ),
    )
    parser.add_argument(
        "--threads_per_scene",
        type=int,
        default=-1,
        help=(
            "Override the number of threads used within each scene. "
            "Defaults to num_threads // num_parallel_scenes (-1). Set to 1 "
            "for reproducible runs: RANSAC seeds per thread as "
            "random_seed + omp_get_thread_num(), so with a fixed --random_seed "
            "only single-threaded scenes are deterministic (required for "
            "paired common-random-number A/B comparison)."
        ),
    )
    parser.add_argument(
        "--gpu_index",
        type=str,
        default="-1",
        help="GPU indices to use for reconstruction. "
        "Use '-1' to auto-detect and use all available GPUs. "
        "Use comma-separated indices like '0,1,2' to specify exact GPUs.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=-1,
        help="Random seed forwarded to colmap's automatic_reconstructor "
        "(two-view RANSAC + mapper). -1 (default) uses a nondeterministic "
        "random device, matching prior harness behavior. Fix it to a "
        "non-negative value for reproducible / paired A-B runs.",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run the evaluation once per seed, writing "
        "<report_name>_s<seed>.pkl each time, to measure run-to-run spread. "
        "Mutually exclusive with --random_seed.",
    )
    seed_group.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Shorthand for --seeds 0 1 ... N-1.",
    )
    parser.add_argument(
        "--feature",
        default="sift",
        choices=["sift", "aliked"],
    )
    parser.add_argument(
        "--mapper",
        default="incremental",
        choices=["incremental", "hierarchical", "global"],
    )
    parser.add_argument(
        "--quality", default="high", choices=["low", "medium", "high"]
    )
    parser.add_argument(
        "--uncalibrated",
        default=False,
        action="store_true",
        help="Whether to evaluate the setting of uncalibrated input cameras, "
        "even if normal setting for the dataset contains calibrated inputs. "
        "This is useful for evaluating the performance of self-calibration.",
    )
    parser.add_argument(
        "--filter_covisibility",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Filter out non-covisible image pairs based on GT camera poses. "
        "Use --no-filter_covisibility to disable.",
    )
    parser.add_argument(
        "--covisibility_frustum_near",
        type=float,
        default=None,
        help="Near plane for frustum co-visibility check. "
        "Auto-detected from GT points if not specified.",
    )
    parser.add_argument(
        "--covisibility_frustum_far",
        type=float,
        default=None,
        help="Far plane for frustum co-visibility check. "
        "Auto-detected from GT points if not specified.",
    )
    parser.add_argument(
        "--covisibility_max_viewing_angle",
        type=float,
        default=120.0,
        help="Maximum viewing angle in degrees for co-visibility check.",
    )
    parser.add_argument(
        "--covisibility_min_shared_points",
        type=int,
        default=5,
        help="Minimum number of shared GT 3D points for two images to be "
        "considered covisible. If GT tracks are available and this is > 0, "
        "track-based covisibility is preferred over frustum-based checking.",
    )
    parser.add_argument(
        "--error_type",
        default="relative_auc",
        choices=[
            "relative_auc",
            "absolute_auc",
            "relative_recall",
            "absolute_recall",
        ],
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
    if args.num_seeds is not None:
        if args.num_seeds <= 0:
            parser.error("--num_seeds must be > 0")
        args.seeds = list(range(args.num_seeds))
    if args.seeds is not None:
        if args.random_seed >= 0:
            parser.error(
                "--random_seed sets the seed of a single run; use "
                "--seeds/--num_seeds to run once per seed"
            )
        if any(seed < 0 for seed in args.seeds):
            parser.error("--seeds must be non-negative")
    if args.fast and args.fast_num_scenes <= 0:
        parser.error("--fast_num_scenes must be > 0 when --fast is set")
    if args.progress is None:
        args.progress = sys.stdout.isatty()
    if args.num_threads <= 0:
        args.num_threads = 2 * multiprocessing.cpu_count()
    if args.num_parallel_scenes <= 0:
        args.num_parallel_scenes = max(1, args.num_threads // 4)
    if args.overwrite_database:
        pycolmap.logging.info(
            "Overwriting database also overwrites reconstruction"
        )
        args.overwrite_reconstruction = True
    if args.overwrite_matches:
        pycolmap.logging.info(
            "Overwriting matches also overwrites reconstruction"
        )
        args.overwrite_reconstruction = True
    if args.overwrite_two_view_geometries:
        pycolmap.logging.info(
            "Overwriting two-view geometries also overwrites reconstruction"
        )
        args.overwrite_reconstruction = True
    if args.overwrite_reconstruction:
        pycolmap.logging.info(
            "Overwriting reconstruction also overwrites alignment"
        )
        args.overwrite_alignment = True
    return args


def set_camera_priors(
    database_path: Path, camera_priors_sparse_gt: pycolmap.Reconstruction
) -> None:
    pycolmap.logging.info("Setting prior cameras from GT")

    with pycolmap.Database.open(str(database_path)) as database:
        images_gt_by_name = {}
        for image_gt in camera_priors_sparse_gt.images.values():
            images_gt_by_name[image_gt.name] = image_gt

        updated_camera_ids = set()
        for image in database.read_all_images():
            if image.name not in images_gt_by_name:
                pycolmap.logging.warning(
                    f"Not setting prior camera for image {image.name}, "
                    "because it does not exist in GT"
                )
                continue
            image_gt = images_gt_by_name[image.name]
            if image.camera_id in updated_camera_ids:
                continue
            camera_gt = camera_priors_sparse_gt.cameras[image_gt.camera_id]
            camera_gt.camera_id = image.camera_id
            camera_gt.has_prior_focal_length = True
            database.update_camera(camera_gt)
            updated_camera_ids.add(image.camera_id)


def colmap_reconstruction(
    args: argparse.Namespace,
    workspace_path: Path,
    image_path: Path,
    camera_priors_sparse_gt: pycolmap.Reconstruction | None = None,
    covisibility_sparse_gt: pycolmap.Reconstruction | None = None,
    colmap_extra_args: list | None = None,
    num_threads: int = 1,
    gpu_index: str = "-1",
    phase_tracker: _PhaseTracker | None = None,
) -> None:
    phase_tracker = phase_tracker or _PhaseTracker()
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

    # Clearing matches also clears two-view geometries, so there is no need to
    # clear the latter separately when both flags are set.
    if args.overwrite_matches:
        cleaner_type = "matches"
    elif args.overwrite_two_view_geometries:
        cleaner_type = "two_view_geometries"
    else:
        cleaner_type = None
    if cleaner_type is not None:
        subprocess.check_call(
            [
                args.colmap_path,
                "database_cleaner",
                "--database_path",
                database_path,
                "--type",
                cleaner_type,
            ],
            cwd=workspace_path,
            preexec_fn=_set_pdeathsig,
        )

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
        "--use_gpu",
        "1" if args.use_gpu else "0",
        "--gpu_index",
        gpu_index,
        "--num_threads",
        str(num_threads),
        "--random_seed",
        str(args.random_seed),
        "--feature",
        args.feature,
        "--mapper",
        args.mapper,
        "--quality",
        args.quality,
    ]

    phase_tracker.set("extraction")
    _run_with_log(
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
        workspace_path / "extraction.log",
        cwd=workspace_path,
    )

    if camera_priors_sparse_gt is not None:
        set_camera_priors(database_path, camera_priors_sparse_gt)

    phase_tracker.set("matching")
    _run_with_log(
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
        workspace_path / "matching.log",
        cwd=workspace_path,
    )

    if covisibility_sparse_gt is not None:
        filter_covisibility(
            database_path,
            covisibility_sparse_gt,
            args.covisibility_frustum_near,
            args.covisibility_frustum_far,
            args.covisibility_max_viewing_angle,
            args.covisibility_min_shared_points,
        )

    # Decouple matching from sparse reconstruction, because matching will
    # initialize an OpenGL context and Mac on Apple silicon tends to assign GUI
    # applications to the low efficiency cores but we want to use the
    # performance cores.
    phase_tracker.set("reconstruction")
    _run_with_log(
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
        workspace_path / "reconstruction.log",
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
        _run_with_log(
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
            ],
            sparse_aligned_path.parent / "alignment.log",
            check=False,
        )


def process_scene(
    args: argparse.Namespace,
    scene_info: SceneInfo,
    prepare_scene: Callable[[SceneInfo], None],
    position_accuracy_gt: float,
    num_threads: int,
    gpu_index: str = "-1",
    progress_status=None,
) -> SceneResult:
    pycolmap.logging.info(
        f"Processing dataset={scene_info.dataset}, "
        f"category={scene_info.category}, "
        f"scene={scene_info.scene}"
    )

    tracker = _PhaseTracker(progress_status, _scene_key(scene_info))

    tracker.set("setup")
    prepare_scene(scene_info)

    sparse_gt = pycolmap.Reconstruction(str(scene_info.sparse_gt_path))

    colmap_reconstruction(
        args=args,
        workspace_path=scene_info.workspace_path,
        image_path=scene_info.image_path,
        camera_priors_sparse_gt=(
            sparse_gt
            if not args.uncalibrated and scene_info.has_camera_priors
            else None
        ),
        covisibility_sparse_gt=(
            sparse_gt if args.filter_covisibility else None
        ),
        num_threads=num_threads,
        colmap_extra_args=scene_info.colmap_extra_args,
        gpu_index=gpu_index,
        phase_tracker=tracker,
    )

    tracker.set("evaluation")

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
        if args.error_type.startswith("relative"):
            sparse = pycolmap.Reconstruction(str(sparse_path))
        elif args.error_type.startswith("absolute"):
            sparse_aligned_path = scene_info.workspace_path / "sparse_aligned"
            colmap_alignment(
                args=args,
                sparse_path=sparse_path,
                sparse_gt_path=scene_info.sparse_gt_path,
                sparse_aligned_path=sparse_aligned_path,
                max_ref_model_error=position_accuracy_gt,
            )
            if (sparse_aligned_path / "images.bin").exists():
                sparse = pycolmap.Reconstruction(str(sparse_aligned_path))
        else:
            raise ValueError(f"Invalid error type: {args.error_type}")

        if sparse is not None:
            largest_component = max(largest_component, sparse.num_images())
            for image in sparse.images.values():
                if image.image_id in sparse_merged.images:
                    continue
                if image.camera_id not in sparse_merged.cameras:
                    sparse_merged.add_camera(image.camera)
                if image.frame_id not in sparse_merged.frames:
                    if image.frame.rig_id not in sparse_merged.rigs:
                        sparse_merged.add_rig(image.frame.rig)
                    image.frame.reset_rig_ptr()
                    sparse_merged.add_frame(image.frame)
                image.reset_camera_ptr()
                image.reset_frame_ptr()
                sparse_merged.add_image(image)

    if args.error_type.startswith("relative"):
        dts, dRs = compute_rel_errors(
            sparse_gt=sparse_gt,
            sparse=sparse_merged,
            min_proj_center_dist=position_accuracy_gt,
        )
        errors = np.maximum(dts, dRs)
    elif args.error_type.startswith("absolute"):
        dts, dRs = compute_abs_errors(
            sparse_gt=sparse_gt,
            sparse=sparse_merged,
        )
        errors = dts
    else:
        raise ValueError(f"Invalid error type: {args.error_type}")

    tracker.set("finished")
    return SceneResult(
        scene_info=scene_info,
        errors=errors,
        num_images=sparse_gt.num_images(),
        num_reg_images=sparse_merged.num_images(),
        num_components=num_components,
        largest_component=largest_component,
    )


def _parse_gpu_index(args: argparse.Namespace) -> list[int]:
    if args.gpu_index == "-1":
        if not pycolmap.has_cuda:
            return [-1]
        num_devices = pycolmap.get_num_cuda_devices()  # type: ignore[attr-defined]
        if num_devices <= 0:
            return [-1]
        return list(range(num_devices))
    indices = [int(idx) for idx in args.gpu_index.split(",") if idx.strip()]
    return indices if indices else [-1]


def _process_scene_with_gpu(
    scene_info_and_gpu: tuple[SceneInfo, str],
    args: argparse.Namespace,
    prepare_scene: Callable[[SceneInfo], None],
    position_accuracy_gt: float,
    num_threads: int,
    progress_status=None,
) -> SceneResult:
    scene_info, gpu_index = scene_info_and_gpu
    return process_scene(
        args=args,
        scene_info=scene_info,
        prepare_scene=prepare_scene,
        position_accuracy_gt=position_accuracy_gt,
        num_threads=num_threads,
        gpu_index=gpu_index,
        progress_status=progress_status,
    )


def process_scenes(
    args: argparse.Namespace,
    scene_infos: list[SceneInfo],
    prepare_scene: Callable[[SceneInfo], None],
    position_accuracy_gt: float,
) -> MetricsByCatByScene:
    error_thresholds = get_error_thresholds(args)

    gpu_index = _parse_gpu_index(args)
    scene_gpu_pairs = [
        (scene_info, str(gpu_index[i % len(gpu_index)]))
        for i, scene_info in enumerate(scene_infos)
    ]

    num_parallel_scenes = min(args.num_parallel_scenes, len(scene_infos))
    if args.threads_per_scene > 0:
        num_threads_per_scene = args.threads_per_scene
    else:
        num_threads_per_scene = max(1, args.num_threads // num_parallel_scenes)
    if args.random_seed >= 0 and num_threads_per_scene != 1:
        pycolmap.logging.warning(
            f"--random_seed={args.random_seed} is set but "
            f"num_threads_per_scene={num_threads_per_scene} > 1: RANSAC "
            "seeds per thread, so results are NOT deterministic. Pass "
            "--threads_per_scene 1 for reproducible / paired A-B runs."
        )

    manager = None
    progress_status = None
    monitor_thread = None
    stop_event = threading.Event()
    if args.progress:
        manager = multiprocessing.Manager()
        progress_status = manager.dict()
        monitor_thread = threading.Thread(
            target=_run_progress_monitor,
            args=(progress_status, len(scene_infos), stop_event),
            daemon=True,
        )
        monitor_thread.start()

    try:
        p = multiprocessing.Pool(
            processes=num_parallel_scenes, initializer=_init_pool_worker
        )
        try:
            results = list(
                p.imap_unordered(
                    functools.partial(
                        _process_scene_with_gpu,
                        args=args,
                        prepare_scene=prepare_scene,
                        position_accuracy_gt=position_accuracy_gt,
                        num_threads=num_threads_per_scene,
                        progress_status=progress_status,
                    ),
                    scene_gpu_pairs,
                    chunksize=1,
                )
            )
        except KeyboardInterrupt:
            pycolmap.logging.warning(
                "Interrupted, terminating workers and child processes..."
            )
            p.terminate()
            raise
        except BaseException:
            p.terminate()
            raise
        else:
            p.close()
        finally:
            p.join()
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join()
        if manager is not None:
            manager.shutdown()

    metrics: MetricsByCatByScene = collections.defaultdict(dict)
    for result in results:
        metrics[result.scene_info.category][result.scene_info.scene] = Metrics(
            aucs=compute_auc(
                result.errors,
                error_thresholds,
                min_error=position_accuracy_gt,
            ),
            recalls=compute_recall(result.errors, error_thresholds),
            error_thresholds=error_thresholds,
            error_type=args.error_type,
            num_images=result.num_images,
            num_reg_images=result.num_reg_images,
            num_components=result.num_components,
            largest_component=result.largest_component,
            errors=np.asarray(result.errors),
            position_accuracy_gt=position_accuracy_gt,
        )

    for category in metrics:
        metrics[category].update(
            aggregate_scene_metrics(
                metrics[category].items(),
                error_thresholds=error_thresholds,
                error_type=args.error_type,
            )
        )

    return metrics


def aggregate_scene_metrics(
    scene_metrics: Iterable[tuple[str, Metrics]],
    error_thresholds: npt.NDArray[np.floating],
    error_type: str,
) -> dict[str, Metrics]:
    """Compute __avg__ (mean of per-scene metrics) and __all__ (recomputed
    from the pool of raw errors) summary entries from per-scene Metrics.

    Skips entries whose key starts and ends with "__" so this can be applied
    iteratively at higher levels (category -> dataset -> overall) without
    double-counting previously-emitted summaries.
    """
    real = [
        m
        for k, m in scene_metrics
        if not (k.startswith("__") and k.endswith("__"))
    ]
    if not real:
        return {}

    n = len(real)
    sum_num_images = sum(m.num_images for m in real)
    sum_num_reg_images = sum(m.num_reg_images for m in real)
    sum_num_components = sum(m.num_components for m in real)
    sum_largest_component = sum(m.largest_component for m in real)
    min_pos_acc = min(m.position_accuracy_gt for m in real)
    pooled_errors = np.concatenate([m.errors for m in real])

    summary = {
        "__avg__": Metrics(
            aucs=np.mean([m.aucs for m in real], axis=0),
            recalls=np.mean([m.recalls for m in real], axis=0),
            error_thresholds=error_thresholds,
            error_type=error_type,
            num_images=int(round(sum_num_images / n)),
            num_reg_images=int(round(sum_num_reg_images / n)),
            num_components=int(round(sum_num_components / n)),
            largest_component=int(round(sum_largest_component / n)),
            position_accuracy_gt=min_pos_acc,
        ),
    }
    if pooled_errors.size:
        summary["__all__"] = Metrics(
            aucs=compute_auc(
                pooled_errors, error_thresholds, min_error=min_pos_acc
            ),
            recalls=compute_recall(pooled_errors, error_thresholds),
            error_thresholds=error_thresholds,
            error_type=error_type,
            num_images=sum_num_images,
            num_reg_images=sum_num_reg_images,
            num_components=sum_num_components,
            largest_component=sum_largest_component,
            errors=pooled_errors,
            position_accuracy_gt=min_pos_acc,
        )
    return summary


def get_error_thresholds(args: argparse.Namespace) -> npt.NDArray[np.floating]:
    if args.error_type.startswith("relative"):
        return np.array(args.rel_error_thresholds)
    elif args.error_type.startswith("absolute"):
        return np.array(args.abs_error_thresholds)
    else:
        raise ValueError(f"Invalid error type: {args.error_type}")


def get_scores(error_type: str, metrics: Metrics) -> npt.NDArray[np.floating]:
    if error_type.endswith("auc"):
        return metrics.aucs
    elif error_type.endswith("recall"):
        return metrics.recalls
    else:
        raise ValueError(f"Invalid error type: {error_type}")


def compute_rel_errors(
    sparse_gt: pycolmap.Reconstruction,
    sparse: pycolmap.Reconstruction,
    min_proj_center_dist: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
                other_image.cam_from_world()
                * this_image.cam_from_world().inverse()
            )
            other_from_this_gt = (
                other_image_gt.cam_from_world()
                * this_image_gt.cam_from_world().inverse()
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
                dt = 0.0
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
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
            image.cam_from_world() * image_gt.cam_from_world().inverse()
        )

        dts[i] = np.linalg.norm(estimated_from_gt.translation)
        dRs[i] = np.rad2deg(estimated_from_gt.rotation.angle())

    return dts, dRs


def compute_auc(
    errors: npt.NDArray[np.floating],
    thresholds: npt.NDArray[np.floating],
    min_error: float = 0,
) -> npt.NDArray[np.floating]:
    num_elems = len(errors)
    if len(errors) == 0:
        raise ValueError("No errors to evaluate")

    errors = np.sort(errors)
    recalls = (np.arange(num_elems) + 1) / num_elems

    if min_error > 0:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_recall = min_index / num_elems
        recalls = np.r_[min_recall, min_recall, recalls[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recalls = np.r_[0, recalls]
        errors = np.r_[0, errors]

    aucs = np.zeros(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recalls[:last_index], recalls[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapezoid(r, x=e) / t
        aucs[i] = auc * 100

    return aucs


def compute_recall(
    errors: npt.NDArray[np.floating],
    thresholds: npt.NDArray[np.floating],
    min_error: float = 0,
) -> npt.NDArray[np.floating]:
    num_elems = len(errors)
    if num_elems == 0:
        raise ValueError("No errors to evaluate")

    recalls = np.zeros(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        recalls[i] = 100 * np.sum(errors <= t) / num_elems

    return recalls


def compute_avg_metrics(
    scene_metrics: MetricsByScene,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    auc_sum = None
    recall_sum = None
    num_scenes = 0
    for scene, metrics in scene_metrics.items():
        if scene.startswith("__") and scene.endswith("__"):
            continue
        num_scenes += 1
        if auc_sum is None:
            auc_sum = copy.copy(metrics.aucs)
        if recall_sum is None:
            recall_sum = copy.copy(metrics.recalls)
        else:
            for i in range(len(auc_sum)):
                auc_sum[i] += metrics.aucs[i]
            for i in range(len(recall_sum)):
                recall_sum[i] += metrics.recalls[i]
    return np.array(auc_sum) / num_scenes, np.array(recall_sum) / num_scenes


def diff_metrics(
    metrics_a: MetricsByDatasetByCatByScene,
    metrics_b: MetricsByDatasetByCatByScene,
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
            for scene, metrics_a_item in scene_metrics_a.items():
                if scene not in scene_metrics_b:
                    raise ValueError(f"Scene {scene} not found in metrics_b")
                metrics_b_item = scene_metrics_b[scene]
                if (
                    metrics_a_item.error_type != metrics_b_item.error_type
                    or not np.all(
                        metrics_a_item.error_thresholds
                        == metrics_b_item.error_thresholds
                    )
                ):
                    raise ValueError("Inconsistent error thresholds or types")
                metrics_diff[dataset][category][scene] = Metrics(
                    aucs=metrics_a_item.aucs - metrics_b_item.aucs,
                    recalls=metrics_a_item.recalls - metrics_b_item.recalls,
                    error_thresholds=metrics_a_item.error_thresholds,
                    error_type=metrics_a_item.error_type,
                    num_images=metrics_a_item.num_images
                    - metrics_b_item.num_images,
                    num_reg_images=metrics_a_item.num_reg_images
                    - metrics_b_item.num_reg_images,
                    num_components=metrics_a_item.num_components
                    - metrics_b_item.num_components,
                    largest_component=metrics_a_item.largest_component
                    - metrics_b_item.largest_component,
                )
    return metrics_diff


def _is_summary_scene(scene: str) -> bool:
    """Whether a scene name is an aggregate row (e.g. __avg__, __all__)."""
    return scene.startswith("__") and scene.endswith("__")


def create_result_table(
    dataset_metrics: MetricsByDatasetByCatByScene,
) -> str:
    first_metrics = next(
        iter(next(iter(next(iter(dataset_metrics.values())).values())).values())
    )

    is_auc = first_metrics.error_type.endswith("auc")
    is_relative = first_metrics.error_type.startswith("relative")
    score_type = "AUC" if is_auc else "Recall"
    score_unit = "deg" if is_relative else "cm"
    label = f"{score_type} @ X {score_unit} (%)"
    if is_relative:
        thresholds = first_metrics.error_thresholds
    else:
        thresholds = 100 * first_metrics.error_thresholds  # cm

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
        f"{'images':=^{size_imgs}} {'components':=^{size_comps}}"
    )
    header += "\n" + " " * (size_scenes + 1)
    header += " ".join(f"{str(t).rstrip('.'):^6}" for t in thresholds)
    header += "    reg   all  num largest"
    text = [header]

    def render_block(
        header_text: str,
        scene_metrics: MetricsByScene,
        header_fill: str = "=",
    ) -> None:
        text.append(f"\n{header_text:{header_fill}^{size_sep}}")
        any_scene_row = False
        summary_separator_drawn = False
        for scene, metrics in sorted(
            scene_metrics.items(),
            key=lambda x: (
                x[0].startswith("__"),
                x[0],
            ),
        ):
            scores = get_scores(first_metrics.error_type, metrics)
            assert len(scores) == len(thresholds)
            row = ""
            is_summary = _is_summary_scene(scene)
            if is_summary and any_scene_row and not summary_separator_drawn:
                row += "-" * size_sep + "\n"
                summary_separator_drawn = True
            if not is_summary:
                any_scene_row = True
            if scene == "__avg__":
                scene = "average"
            if scene == "__all__":
                scene = "overall"
            row += f"{scene:<{size_scenes}} "
            row += " ".join(f"{score:>6.2f}" for score in scores)
            row += f" {metrics.num_reg_images:6d}"
            row += f"{metrics.num_images:6d}"
            row += f" {metrics.num_components:4d}"
            row += f"{metrics.largest_component:8d}"
            text.append(row)

    overall_scene_metrics: list[tuple[str, Metrics]] = []
    for dataset, category_metrics in dataset_metrics.items():
        dataset_scene_metrics: list[tuple[str, Metrics]] = []
        for category, scene_metrics in category_metrics.items():
            render_block(f"{dataset}={category}", scene_metrics)
            dataset_scene_metrics.extend(scene_metrics.items())
        if len(category_metrics) > 1:
            render_block(
                dataset,
                aggregate_scene_metrics(
                    dataset_scene_metrics,
                    error_thresholds=first_metrics.error_thresholds,
                    error_type=first_metrics.error_type,
                ),
                header_fill="#",
            )
        overall_scene_metrics.extend(dataset_scene_metrics)

    if len(dataset_metrics) > 1:
        render_block(
            "overall",
            aggregate_scene_metrics(
                overall_scene_metrics,
                error_thresholds=first_metrics.error_thresholds,
                error_type=first_metrics.error_type,
            ),
            header_fill="#",
        )

    return "\n".join(text)


def load_report(path: Path) -> MetricsByDatasetByCatByScene:
    """Loads a report pickled by evaluate.py."""
    with open(path, "rb") as report_file:
        return pickle.load(report_file)


def _collect_reports(prefix: Path) -> dict[int, Path] | Path:
    """Collects the reports of one variant given as a report path prefix.

    Returns seed -> path for the <prefix>_s<seed>.pkl reports of a seeded run,
    or the single <prefix>.pkl path of an unseeded one.
    """
    reports: dict[int, Path] = {}
    for path in prefix.parent.glob(f"{prefix.name}_s*.pkl"):
        seed = path.stem[len(prefix.name) + 2 :]
        if seed.isdigit():
            reports[int(seed)] = path
    if reports:
        return reports
    path = prefix.with_name(f"{prefix.name}.pkl")
    if not path.exists():
        raise SystemExit(
            f"found neither {prefix.name}_s<seed>.pkl reports nor {path}"
        )
    return path


def collect_reports(
    path: Path | None, prefix: Path | None
) -> dict[int, Path] | Path:
    """Collects one variant's reports from a report path or a path prefix.

    Returns the single report path, or seed -> path for the
    <prefix>_s<seed>.pkl reports of a seeded run.
    """
    if path is not None:
        if not path.exists():
            raise SystemExit(f"no such report: {path}")
        return path
    assert prefix is not None
    return _collect_reports(prefix)


def pair_reports(
    reports_a: dict[int, Path] | Path,
    reports_b: dict[int, Path] | Path,
    labels: Sequence[str] = ("A", "B"),
    seeds: list[int] | None = None,
) -> tuple[list[Path], list[Path], list[int] | None]:
    """Pairs up the collected reports of two variants.

    Seeded variants are matched on the seeds they have in common; pass seeds
    to restrict the comparison to a subset. A variant that is a single report
    is compared against every seed of the other.
    """
    label_a, label_b = labels

    if isinstance(reports_a, dict) and isinstance(reports_b, dict):
        # Both seeded: pair them up on the seeds they have in common.
        shared = sorted(set(reports_a) & set(reports_b))
        if not shared:
            raise SystemExit(
                f"{label_a} and {label_b} share no seed, but the "
                "comparison is paired per seed "
                f"({label_a}: {sorted(reports_a)}, "
                f"{label_b}: {sorted(reports_b)})"
            )
        for label, reports in ((label_a, reports_a), (label_b, reports_b)):
            dropped = sorted(set(reports) - set(shared))
            if dropped:
                pycolmap.logging.warning(
                    f"ignoring {len(dropped)} of {label}'s "
                    f"{len(reports)} reports, seed(s) {dropped}: the other "
                    "variant has no report for them"
                )
    elif isinstance(reports_a, dict):
        # Only A seeded: B is a single run, compared against each of A's seeds.
        shared = sorted(reports_a)
    elif isinstance(reports_b, dict):
        shared = sorted(reports_b)
    else:
        # Neither seeded: a plain one-to-one comparison.
        return [reports_a], [reports_b], None

    if seeds is not None:
        unknown = sorted(set(seeds) - set(shared))
        if unknown:
            pycolmap.logging.warning(
                f"skipping seed(s) {unknown}: not present for both variants"
            )
        shared = [seed for seed in shared if seed in set(seeds)]
        if not shared:
            raise SystemExit(f"none of the seeds {seeds} is available")

    def paths(reports: dict[int, Path] | Path) -> list[Path]:
        if isinstance(reports, Path):
            return [reports] * len(shared)
        return [reports[seed] for seed in shared]

    return paths(reports_a), paths(reports_b), shared


def _first_metrics(report: MetricsByDatasetByCatByScene) -> Metrics:
    return next(iter(next(iter(next(iter(report.values())).values())).values()))


def _common_scene_keys(
    reports: list[MetricsByDatasetByCatByScene],
) -> list[SceneKey]:
    """Scene keys present in every report, ordered as in the first report."""
    key_sets: list[set[SceneKey]] = []
    for report in reports:
        keys: set[SceneKey] = set()
        for dataset, cat_metrics in report.items():
            for category, scene_metrics in cat_metrics.items():
                for scene in scene_metrics:
                    keys.add((dataset, category, scene))
        key_sets.append(keys)
    shared: set[SceneKey] = set()
    if key_sets:
        shared = key_sets[0].intersection(*key_sets[1:])
    ordered: list[SceneKey] = []
    for dataset, cat_metrics in reports[0].items():
        for category, scene_metrics in cat_metrics.items():
            for scene in scene_metrics:
                if (dataset, category, scene) in shared:
                    ordered.append((dataset, category, scene))
    return ordered


def _stack_scores(
    reports: list[MetricsByDatasetByCatByScene],
    key: SceneKey,
    error_type: str,
) -> npt.NDArray[np.floating]:
    """Scores for one scene across reports, shaped (num_reports, num_thr)."""
    dataset, category, scene = key
    return np.array(
        [
            get_scores(error_type, report[dataset][category][scene])
            for report in reports
        ]
    )


def _render_meanstd(
    title: str,
    per_key_stack: dict[SceneKey, npt.NDArray[np.floating]],
    keys: list[SceneKey],
    error_type: str,
    thresholds: npt.NDArray[np.floating],
    num_runs: int,
    signed: bool = False,
    num_reported: int | None = None,
) -> str:
    """Renders one mean +/- std table over runs, per scene x threshold.

    per_key_stack maps each key to a (num_runs, num_thresholds) score array.
    num_reported is shown in the trailing N column as the number of distinct
    runs behind the statistics, which is 1 for a single run compared against
    every seed of the other variant, and whose std is then zero.
    """
    is_relative = error_type.startswith("relative")
    thresholds_disp = thresholds if is_relative else 100 * thresholds
    size_scene = max(8, max(len(s) for _, _, s in keys))
    size_cell = 12  # "+dd.dd±dd.dd"
    fmt = "{:+6.2f}±{:5.2f}" if signed else "{:6.2f}±{:5.2f}"
    header = f"{'scene':<{size_scene}} {'N':>3} " + " ".join(
        f"{'@' + str(t).rstrip('.'):^{size_cell}}" for t in thresholds_disp
    )
    num_reported = num_runs if num_reported is None else num_reported
    lines: list[str] = [title, header, "-" * len(header)]
    prev_summary = False
    for key in sorted(keys, key=lambda k: (_is_summary_scene(k[2]), k)):
        scene = key[2]
        if _is_summary_scene(scene) and not prev_summary:
            lines.append("-" * len(header))
            prev_summary = True
        scores = per_key_stack[key]  # (num_runs, num_thresholds)
        mean = scores.mean(axis=0)
        std = (
            scores.std(axis=0, ddof=1) if num_runs > 1 else np.zeros_like(mean)
        )
        label = {"__avg__": "average", "__all__": "overall"}.get(scene, scene)
        cells = " ".join(
            fmt.format(m, s) for m, s in zip(mean, std, strict=True)
        )
        lines.append(f"{label:<{size_scene}} {num_reported:>3} {cells}")
    return "\n".join(lines)


def compare_reports(
    report_a_paths: list[Path],
    report_b_paths: list[Path],
    labels: Sequence[str] = ("A", "B"),
    seeds: list[int] | None = None,
) -> None:
    """Logs an A vs B comparison of two sets of paired reports.

    With one report per variant this prints the usual result tables for A, B
    and A - B. With several reports per variant (one per seed, paired by
    position) it instead prints mean +/- std over the runs, with A - B computed
    per seed before averaging.
    """
    if len(report_a_paths) != len(report_b_paths):
        raise SystemExit(
            "A and B must have the same number of reports (paired by seed): "
            f"{len(report_a_paths)} vs {len(report_b_paths)}"
        )
    num_runs = len(report_a_paths)
    if num_runs == 0:
        raise SystemExit("no reports to compare")
    if seeds is not None and len(seeds) != num_runs:
        raise SystemExit(
            f"seed count ({len(seeds)}) != report count ({num_runs})"
        )
    label_a, label_b = labels

    reports_a = [load_report(path) for path in report_a_paths]
    reports_b = [load_report(path) for path in report_b_paths]

    if num_runs == 1:
        metrics_a, metrics_b = reports_a[0], reports_b[0]
        metrics_diff = diff_metrics(metrics_a, metrics_b)
        pycolmap.logging.info(
            f"Results {label_a}:\n" + create_result_table(metrics_a)
        )
        pycolmap.logging.info(
            f"Results {label_b}:\n" + create_result_table(metrics_b)
        )
        pycolmap.logging.info(
            f"Results {label_a} - {label_b}:\n"
            + create_result_table(metrics_diff)
        )
        return

    keys = _common_scene_keys(reports_a + reports_b)
    if not keys:
        raise SystemExit("No scenes shared across all reports.")

    first_metrics = _first_metrics(reports_a[0])
    error_type = first_metrics.error_type
    thresholds = np.asarray(first_metrics.error_thresholds)
    score = "AUC" if error_type.endswith("auc") else "Recall"

    # A variant may be a single run compared against every seed of the other,
    # in which case it has no spread of its own and is labelled as such.
    single_a = len(set(report_a_paths)) == 1 and num_runs > 1
    single_b = len(set(report_b_paths)) == 1 and num_runs > 1
    # Both variants ran on the same seeds, so the difference is paired; a
    # single run broadcast against the other's seeds is not.
    shared = not (single_a or single_b)
    over_seeds = f"{score} mean ± std over {num_runs} seeds"

    seeds_label = " ".join(map(str, seeds)) if seeds is not None else "-"
    num_scenes = len([k for k in keys if not _is_summary_scene(k[2])])
    num_reconstructions = num_runs * (2 - single_a - single_b)
    pycolmap.logging.info(
        f"{label_a} vs {label_b}: {num_runs} seeds; "
        f"{num_reconstructions} reconstruction runs over {num_scenes} scenes; "
        f"seeds: [{seeds_label}]"
    )
    if single_a or single_b:
        single, seeded = (label_a, label_b) if single_a else (label_b, label_a)
        pycolmap.logging.warning(
            f"{single} is a single run compared against every seed of "
            f"{seeded}: the difference therefore carries only {seeded}'s "
            f"spread, and is not a paired (common random number) comparison. "
            f"Run {single} on the same seeds for that."
        )

    stacks_a = {k: _stack_scores(reports_a, k, error_type) for k in keys}
    stacks_b = {k: _stack_scores(reports_b, k, error_type) for k in keys}
    stacks_diff = {k: stacks_a[k] - stacks_b[k] for k in keys}

    common = (keys, error_type, thresholds, num_runs)
    pycolmap.logging.info(
        "\n"
        + _render_meanstd(
            f"A = {label_a}  "
            + (f"({score}, single run)" if single_a else f"({over_seeds})"),
            stacks_a,
            *common,
            num_reported=1 if single_a else num_runs,
        )
    )
    pycolmap.logging.info(
        "\n"
        + _render_meanstd(
            f"B = {label_b}  "
            + (f"({score}, single run)" if single_b else f"({over_seeds})"),
            stacks_b,
            *common,
            num_reported=1 if single_b else num_runs,
        )
    )
    pycolmap.logging.info(
        "\n"
        + _render_meanstd(
            f"A - B = {label_a} - {label_b}  ({over_seeds}, "
            + (
                "shared seeds -> paired)"
                if shared
                else f"seeds NOT shared -> {label_b if single_a else label_a}"
                "'s spread only)"
            ),
            stacks_diff,
            *common,
            signed=True,
        )
    )
