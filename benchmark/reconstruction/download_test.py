# SPDX-License-Identifier: BSD-3-Clause

import importlib
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
    # Leftover from an interrupted earlier run.
    (tmp_path / "train/all/scene").mkdir(parents=True)
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
