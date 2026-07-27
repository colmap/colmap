# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

MANIFEST_PATH = Path(__file__).with_name("tartanair_v2_manifest.json")


@dataclasses.dataclass(frozen=True)
class SceneSelection:
    category: str
    environment: str
    difficulty: str
    trajectory: str

    @property
    def name(self) -> str:
        return f"{self.environment}-{self.difficulty}-{self.trajectory}"

    @property
    def source_archive(self) -> str:
        return (
            f"{self.environment}/Data_{self.difficulty}/image_lcam_equirect.zip"
        )

    @property
    def depth_source_archive(self) -> str:
        return (
            f"{self.environment}/Data_{self.difficulty}/depth_lcam_equirect.zip"
        )


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def list_scenes(manifest: dict | None = None) -> list[SceneSelection]:
    manifest = manifest or load_manifest()
    scenes = []
    for category, selections in manifest["selections"].items():
        for selection in selections:
            environment, difficulty, trajectory = selection.split(":")
            scenes.append(
                SceneSelection(
                    category=category,
                    environment=environment,
                    difficulty=difficulty,
                    trajectory=trajectory,
                )
            )
    return scenes


def shard_name(manifest: dict, shard_index: int) -> str:
    version = manifest["version"]
    return f"tartanair-v2-v{version}-shard-{shard_index:03d}.tar"


def scene_shards(manifest: dict | None = None) -> list[list[SceneSelection]]:
    manifest = manifest or load_manifest()
    scenes = list_scenes(manifest)
    size = manifest["release"]["scenes_per_shard"]
    return [scenes[i : i + size] for i in range(0, len(scenes), size)]


def quaternion_angular_distance_deg(
    quaternion1: np.ndarray, quaternion2: np.ndarray
) -> float:
    dot = abs(float(np.dot(quaternion1, quaternion2)))
    return math.degrees(2.0 * math.acos(np.clip(dot, -1.0, 1.0)))


def select_frame_window(
    poses: np.ndarray,
    num_frames: int,
    max_adjacent_translation_m: float,
    max_adjacent_rotation_deg: float,
) -> range:
    """Select a contiguous, overlapping window with maximum spatial extent."""
    if len(poses) < num_frames:
        raise ValueError(
            f"Trajectory has {len(poses)} poses, fewer than {num_frames}"
        )

    best_score = None
    best_start = None
    for start in range(len(poses) - num_frames + 1):
        window = poses[start : start + num_frames]
        translations = np.linalg.norm(np.diff(window[:, :3], axis=0), axis=1)
        rotations = np.array(
            [
                quaternion_angular_distance_deg(q1, q2)
                for q1, q2 in zip(
                    window[:-1, 3:7], window[1:, 3:7], strict=True
                )
            ]
        )
        if (
            translations.max() > max_adjacent_translation_m
            or rotations.max() > max_adjacent_rotation_deg
        ):
            continue

        bbox_diagonal = np.linalg.norm(
            window[:, :3].max(axis=0) - window[:, :3].min(axis=0)
        )
        path_length = translations.sum()
        score = (bbox_diagonal, path_length, -start)
        if best_score is None or score > best_score:
            best_score = score
            best_start = start

    if best_start is None:
        raise ValueError(
            "Trajectory has no frame window satisfying motion limits"
        )
    return range(best_start, best_start + num_frames)
