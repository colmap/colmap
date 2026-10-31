# SPDX-License-Identifier: BSD-3-Clause

import json

import numpy as np
import pytest

from . import tartanair_v2
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


@pytest.mark.parametrize("num_frames", [2, 20, 200])
def test_frame_window_evaluates_each_relative_rotation_once(
    monkeypatch, num_frames
):
    poses = np.zeros((200, 7))
    poses[:, 0] = np.arange(len(poses))
    poses[:, 6] = 1.0
    calls = 0
    angular_distance = tartanair_v2.quaternion_angular_distance_deg

    def counted_distance(q1, q2):
        nonlocal calls
        calls += 1
        return angular_distance(q1, q2)

    monkeypatch.setattr(
        tartanair_v2, "quaternion_angular_distance_deg", counted_distance
    )
    selected = select_frame_window(poses, num_frames, 1.0, 30.0)
    # Every equally spaced window has the same score: retain the earliest one.
    assert selected == range(num_frames)
    assert calls == len(poses) - 1


@pytest.mark.parametrize("rotation_jump", [False, True])
@pytest.mark.parametrize("jump_at,expected_start", [(1, 1), (3, 0), (5, 0)])
def test_frame_window_motion_limits_include_only_internal_edges(
    rotation_jump, jump_at, expected_start
):
    poses = np.zeros((6, 7))
    poses[:, 0] = np.arange(len(poses))
    poses[:, 6] = 1.0
    if rotation_jump:
        poses[jump_at:, 3:7] = [0.0, 0.0, 1.0, 0.0]
    else:
        poses[jump_at:, 0] += 10.0
    selected = select_frame_window(poses, 3, 1.0, 30.0)
    assert selected == range(expected_start, expected_start + 3)
