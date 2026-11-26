# SPDX-License-Identifier: BSD-3-Clause

import sys
from pathlib import Path

import convert_legacy_rotation_averaging_format as converter
import pytest


@pytest.mark.parametrize("bad_gravity", [False, True])
def test_truncated_input_preserves_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_gravity: bool
) -> None:
    database = tmp_path / "out.db"
    database.write_bytes(b"original")
    poses = tmp_path / "poses.txt"
    gravity = tmp_path / "gravity.txt"
    poses.write_text("# qw qx qy qz tx ty tz\na b 1 0 0 0 0 0 0\n")
    gravity.write_text("# gx gy gz\na 0 1 0\n")
    bad_file = gravity if bad_gravity else poses
    with bad_file.open("a") as f:
        f.write("a 0\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert",
            "--relpose_path",
            str(poses),
            "--gravity_path",
            str(gravity),
            "--database_path",
            str(database),
        ],
    )
    with pytest.raises(ValueError, match=rf"{bad_file.name}:3"):
        converter.main()
    assert database.read_bytes() == b"original"
