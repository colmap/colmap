# SPDX-License-Identifier: BSD-3-Clause

import sys
from pathlib import Path

import convert_legacy_rotation_averaging_format as converter
import pytest


def test_missing_gravity_path_preserves_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poses = tmp_path / "poses.txt"
    poses.write_text("a b 1 0 0 0 0 0 0\n")
    database = tmp_path / "out.db"
    database.write_bytes(b"original")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert",
            "--relpose_path",
            str(poses),
            "--gravity_path",
            str(tmp_path / "missing.txt"),
            "--database_path",
            str(database),
        ],
    )

    assert converter.main() != 0
    assert database.read_bytes() == b"original"
