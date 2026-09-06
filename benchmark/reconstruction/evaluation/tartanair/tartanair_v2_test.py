import json

import numpy as np
import pytest

from .tartanair_v2 import (
    MANIFEST_PATH,
    list_scenes,
    load_manifest,
    scene_shards,
    select_frame_window,
    shard_name,
)


def test_manifest_invariants() -> None:
    manifest = load_manifest()
    scenes = list_scenes(manifest)

    assert len(scenes) == 100
    assert len({scene.name for scene in scenes}) == 100
    assert len({scene.environment for scene in scenes}) == 74
    assert sum(scene.difficulty == "easy" for scene in scenes) == 50
    assert sum(scene.difficulty == "hard" for scene in scenes) == 50
    assert set(manifest["selections"]) == {
        "domestic",
        "infrastructure",
        "nature",
        "rural",
        "thematic",
        "urban",
    }
    assert set(manifest["frame_starts"]) == {
        f"{scene.environment}:{scene.difficulty}:{scene.trajectory}"
        for scene in scenes
    }
    assert len(scene_shards(manifest)) == 13
    checksum_path = MANIFEST_PATH.with_name("tartanair_v2_checksums.json")
    checksums = json.loads(checksum_path.read_text())
    assert set(checksums) == {
        shard_name(manifest, index)
        for index in range(len(scene_shards(manifest)))
    }


def test_select_frame_window_maximizes_extent() -> None:
    poses = np.zeros((8, 7))
    poses[:, 6] = 1.0
    poses[:, 0] = [0.0, 0.1, 0.2, 0.3, 0.4, 1.3, 2.2, 3.1]

    selected = select_frame_window(
        poses,
        num_frames=4,
        max_adjacent_translation_m=1.0,
        max_adjacent_rotation_deg=30.0,
    )

    assert selected == range(4, 8)


def test_select_frame_window_rejects_motion_gaps() -> None:
    poses = np.zeros((4, 7))
    poses[:, 6] = 1.0
    poses[:, 0] = [0.0, 0.1, 2.0, 2.1]

    with pytest.raises(ValueError, match="no frame window"):
        select_frame_window(
            poses,
            num_frames=4,
            max_adjacent_translation_m=1.0,
            max_adjacent_rotation_deg=30.0,
        )
