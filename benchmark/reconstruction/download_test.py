# SPDX-License-Identifier: BSD-3-Clause

import json
import shutil
import zipfile
from pathlib import Path
from types import ModuleType

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


@pytest.fixture
def eth3d_undistorted_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModuleType, Path, str]:
    """Patches download.py to fetch one scene archive from a local release."""
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

    def fake_download_file(url: str, folder: Path, name: str) -> str:
        shutil.copy(release_path / url.split("/")[-1], folder / name)
        return name

    monkeypatch.setattr(download, "download_file", fake_download_file)
    return download, release_path, filename


def test_eth3d_undistorted_deletes_archives(
    tmp_path: Path, eth3d_undistorted_release: tuple[ModuleType, Path, str]
) -> None:
    download, _, _ = eth3d_undistorted_release
    data_path = tmp_path / "eth3d-undistorted"
    download.download_eth3d_undistorted(data_path)
    assert (data_path / "dslr/scene/images/one.jpg").read_text() == "image"
    assert sorted(p.name for p in (data_path / "dslr").iterdir()) == ["scene"]


def test_eth3d_undistorted_reuses_valid_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eth3d_undistorted_release: tuple[ModuleType, Path, str],
) -> None:
    download, release_path, filename = eth3d_undistorted_release
    data_path = tmp_path / "eth3d-undistorted"
    (data_path / "dslr").mkdir(parents=True)
    shutil.copy(release_path / filename, data_path / "dslr" / filename)

    def fail(*args: object) -> None:
        raise AssertionError("Unexpected download")

    monkeypatch.setattr(download, "download_file", fail)
    download.download_eth3d_undistorted(data_path)
    assert (data_path / "dslr/scene/images/one.jpg").read_text() == "image"


def test_eth3d_undistorted_rejects_corrupt_archives(
    tmp_path: Path, eth3d_undistorted_release: tuple[ModuleType, Path, str]
) -> None:
    download, release_path, filename = eth3d_undistorted_release
    data_path = tmp_path / "eth3d-undistorted"
    # A truncated archive from an interrupted run is downloaded again, and
    # here also corrupt in the release.
    (data_path / "dslr").mkdir(parents=True)
    (data_path / "dslr" / filename).write_bytes(b"truncated")
    (release_path / filename).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download.download_eth3d_undistorted(data_path)
    assert not (data_path / "dslr" / f"{filename}.part").exists()
    assert not (data_path / "dslr" / "scene").exists()
