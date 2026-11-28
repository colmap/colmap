# SPDX-License-Identifier: BSD-3-Clause

import importlib
import json
import shutil
import zipfile
from pathlib import Path
from types import ModuleType

import py7zr
import pytest


@pytest.fixture
def download(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # download.py is a script that imports the sibling evaluation package.
    monkeypatch.syspath_prepend(Path(__file__).parent)
    return importlib.import_module("download")


def write_zip(path: Path, member: str) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(member, "data")


def patch_kaggle(
    monkeypatch: pytest.MonkeyPatch, download: ModuleType, valid: bool
) -> None:
    def fake_check_call(args: list[str]) -> None:
        archive_path = Path(args[-1]) / f"{args[-3]}.zip"
        if valid:
            write_zip(archive_path, "train/scene/image.jpg")
        else:
            archive_path.write_bytes(b"corrupt")

    monkeypatch.setattr(download.subprocess, "check_call", fake_check_call)


@pytest.mark.parametrize(
    ("year", "scene_path"),
    [(2023, "train/scene"), (2024, "train/all/scene")],
)
def test_download_imc_deletes_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download: ModuleType,
    year: int,
    scene_path: str,
) -> None:
    patch_kaggle(monkeypatch, download, valid=True)
    getattr(download, f"download_imc{year}")(tmp_path)
    assert (tmp_path / scene_path / "image.jpg").exists()
    assert not (tmp_path / f"image-matching-challenge-{year}.zip").exists()


def test_imc2024_relative_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, download: ModuleType
) -> None:
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


@pytest.mark.parametrize("year", [2024, 2025])
def test_imc_download_can_resume_grouped_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download: ModuleType,
    year: int,
) -> None:
    data_path = tmp_path / f"imc{year}"
    data_path.mkdir()

    def fake_check_call(args: list[str]) -> None:
        archive_path = Path(args[-1]) / f"{args[-3]}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("train/scene/images/one.jpg", b"image1")
            archive.writestr("train/scene/images/two.jpg", b"image2")
            archive.writestr("train/scene2/images/three.jpg", b"image3")

    monkeypatch.setattr(download.subprocess, "check_call", fake_check_call)
    downloader = getattr(download, f"download_imc{year}")
    downloader(data_path)

    scene = data_path / "train/all/scene"
    (scene / "images/two.jpg").unlink()
    cached_gt = scene / "sparse_gt/cameras.txt"
    cached_gt.parent.mkdir()
    cached_gt.write_text("prepared ground truth")
    downloader(data_path)
    downloader(data_path)

    assert (scene / "images/one.jpg").read_bytes() == b"image1"
    assert (scene / "images/two.jpg").read_bytes() == b"image2"
    assert (
        data_path / "train/all/scene2/images/three.jpg"
    ).read_bytes() == b"image3"
    assert cached_gt.read_text() == "prepared ground truth"
    assert [p.name for p in (data_path / "train").iterdir()] == ["all"]

    assert not (data_path / f"image-matching-challenge-{year}.zip").exists()


def test_download_imc_keeps_archive_on_extraction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, download: ModuleType
) -> None:
    patch_kaggle(monkeypatch, download, valid=False)
    with pytest.raises(zipfile.BadZipFile):
        download.download_imc2024(tmp_path)
    assert (tmp_path / "image-matching-challenge-2024.zip").exists()


def test_download_imc_keeps_archive_on_move_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, download: ModuleType
) -> None:
    patch_kaggle(monkeypatch, download, valid=True)

    def fail_move(*args: object, **kwargs: object) -> None:
        raise shutil.Error("Cannot move scene")

    monkeypatch.setattr(download.shutil, "move", fail_move)
    with pytest.raises(shutil.Error):
        download.download_imc2024(tmp_path)
    assert (tmp_path / "image-matching-challenge-2024.zip").exists()


@pytest.mark.parametrize("valid", [True, False])
def test_download_eth3d_deletes_archive_on_success_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download: ModuleType,
    valid: bool,
) -> None:
    def fake_download_file(url: str, target_folder: Path) -> str:
        filename = url.split("/")[-1]
        if valid:
            with py7zr.SevenZipFile(target_folder / filename, "w") as archive:
                archive.writestr("data", "scene/image.jpg")
        else:
            (target_folder / filename).write_bytes(b"corrupt")
        return filename

    monkeypatch.setattr(download, "download_file", fake_download_file)
    filename, category = download.ETH3D_DISTORTED_ARCHIVES[0]
    archive_path = tmp_path / category / filename
    if valid:
        download.download_eth3d_distorted(tmp_path)
        assert (tmp_path / category / "scene/image.jpg").exists()
        assert not archive_path.exists()
    else:
        with pytest.raises(py7zr.Bad7zFile):
            download.download_eth3d_distorted(tmp_path)
        assert archive_path.exists()


@pytest.mark.parametrize("valid", [True, False])
def test_download_blended_mvs_deletes_archives_on_success_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download: ModuleType,
    valid: bool,
) -> None:
    def fake_download_file(url: str, target_folder: Path) -> str:
        filename = url.split("/")[-1]
        (target_folder / filename).write_bytes(b"part")
        return filename

    def fake_check_call(args: list[str]) -> None:
        combined_zip = Path(args[-1])
        if valid:
            write_zip(combined_zip, "BlendedMVS/scene/image.jpg")
        else:
            combined_zip.write_bytes(b"corrupt")

    monkeypatch.setattr(download, "download_file", fake_download_file)
    monkeypatch.setattr(download.subprocess, "check_call", fake_check_call)
    target_folder = tmp_path / "BlendedMVS"
    if valid:
        download.download_blended_mvs(tmp_path)
        assert (target_folder / "scene/image.jpg").exists()
        assert sorted(p.name for p in target_folder.iterdir()) == ["scene"]
    else:
        with pytest.raises(zipfile.BadZipFile):
            download.download_blended_mvs(tmp_path)
        assert not (target_folder / "BlendedMVS_combined.zip").exists()
        assert len(list(target_folder.glob("BlendedMVS.z*"))) == 16


@pytest.fixture
def eth3d_undistorted_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, download: ModuleType
) -> tuple[ModuleType, Path, str]:
    """Patches download.py to fetch one scene archive from a local release."""
    release_path = tmp_path / "release"
    release_path.mkdir()
    filename = "eth3d-undistorted-v1-scene.7z"
    with py7zr.SevenZipFile(release_path / filename, "w") as archive:
        archive.writestr("image", "scene/images/one.jpg")
    checksums_path = tmp_path / "checksums.json"
    checksums_path.write_text(
        json.dumps({filename: download.sha256(release_path / filename)})
    )
    monkeypatch.setattr(
        download.package_eth3d_undistorted, "CHECKSUMS_PATH", checksums_path
    )

    def fake_download_file(
        url: str, folder: Path, name: str | None = None
    ) -> str:
        name = name or url.split("/")[-1]
        shutil.copy(release_path / url.split("/")[-1], folder / name)
        return name

    monkeypatch.setattr(download, "download_file", fake_download_file)
    # The rig archives are tested separately.
    monkeypatch.setattr(download, "ETH3D_UNDISTORTED_RIG_ARCHIVES", [])
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
    # Rerunning skips the extracted scene, whose archive was deleted.
    download.download_eth3d_undistorted(data_path)
    assert sorted(p.name for p in (data_path / "dslr").iterdir()) == ["scene"]


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


def test_eth3d_undistorted_downloads_rig_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eth3d_undistorted_release: tuple[ModuleType, Path, str],
) -> None:
    download, release_path, _ = eth3d_undistorted_release
    with py7zr.SevenZipFile(release_path / "rig.7z", "w") as archive:
        archive.writestr("image", "scene/images/cam/one.png")
    monkeypatch.setattr(
        download, "ETH3D_UNDISTORTED_RIG_ARCHIVES", [("rig.7z", "rig")]
    )
    data_path = tmp_path / "eth3d-undistorted"
    download.download_eth3d_undistorted(data_path)
    assert (data_path / "dslr/scene/images/one.jpg").read_text() == "image"
    assert (data_path / "rig/scene/images/cam/one.png").read_text() == "image"


def test_download_rejects_old_eth3d_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, download: ModuleType
) -> None:
    (tmp_path / "eth3d" / "rig").mkdir(parents=True)
    monkeypatch.setattr(
        "sys.argv",
        ["download.py", "--data_path", str(tmp_path)]
        + ["--datasets", "eth3d-distorted"],
    )
    with pytest.raises(RuntimeError, match="rmdir eth3d"):
        download.main()
