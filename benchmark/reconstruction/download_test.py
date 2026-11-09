# SPDX-License-Identifier: BSD-3-Clause

import json
import shutil
import zipfile
from pathlib import Path

import pytest


def test_imc2024_relative_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("py7zr")
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import download

    monkeypatch.chdir(tmp_path)
    data_path = Path("data/imc2024")
    data_path.mkdir(parents=True)
    with zipfile.ZipFile(
        data_path / "image-matching-challenge-2024.zip", "w"
    ) as archive:
        archive.writestr("train/scene/images/one.jpg", b"image")
        archive.writestr("train/scene2/images/two.jpg", b"image2")
    monkeypatch.setattr(download.subprocess, "check_call", lambda cmd: 0)
    download.download_imc2024(data_path)
    image = data_path / "train/all/scene/images/one.jpg"
    assert image.read_bytes() == b"image"
    image2 = data_path / "train/all/scene2/images/two.jpg"
    assert image2.read_bytes() == b"image2"
    assert [p.name for p in (data_path / "train").iterdir()] == ["all"]


def test_eth3d_undistorted_deletes_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    py7zr = pytest.importorskip("py7zr")
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import download

    release_path = tmp_path / "release"
    release_path.mkdir()
    filename = "eth3d-undistorted-v1-scene.7z"
    with py7zr.SevenZipFile(release_path / filename, "w") as archive:
        archive.writestr("image", "scene/images/one.jpg")
    checksums_path = tmp_path / "checksums.json"
    checksums_path.write_text(
        json.dumps({filename: download._sha256(release_path / filename)})
    )
    monkeypatch.setattr(
        download.package_eth3d_undistorted, "CHECKSUMS_PATH", checksums_path
    )
    monkeypatch.setattr(
        download,
        "download_file",
        lambda url, folder: shutil.copy(
            release_path / url.split("/")[-1], folder
        ),
    )
    data_path = tmp_path / "eth3d-undistorted"
    download.download_eth3d_undistorted(data_path)
    assert (data_path / "dslr/scene/images/one.jpg").read_text() == "image"
    assert not (data_path / "dslr" / filename).exists()
