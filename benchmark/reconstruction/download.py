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
import hashlib
import inspect
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import py7zr
import requests
from evaluation.tartanair.tartanair_v2 import (
    MANIFEST_PATH,
    load_manifest,
    scene_shards,
    shard_name,
)
from evaluation.utils import resolve_dataset_name

import pycolmap


def download_file(url: str, target_folder: Path) -> str:
    filename = url.split("/")[-1]
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        with open(target_folder / filename, "wb") as f:
            for chunk in req.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename


def _publish_staged_tree(staging_path: Path, target_folder: Path) -> None:
    """Merge a staged archive into target_folder with rollback on failure."""
    with tempfile.TemporaryDirectory(
        prefix=".eth3d-publish-", dir=target_folder
    ) as transaction_dir:
        backups_path = Path(transaction_dir) / "backups"
        backups_path.mkdir()
        published: list[tuple[Path, Path | None]] = []

        def publish(source: Path, destination: Path) -> None:
            if source.is_dir() and destination.is_dir():
                for child in source.iterdir():
                    publish(child, destination / child.name)
                return

            backup = None
            if destination.exists():
                backup = backups_path / destination.relative_to(target_folder)
                backup.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(backup)
            try:
                source.rename(destination)
            except Exception:
                if backup is not None:
                    backup.rename(destination)
                raise
            published.append((destination, backup))

        try:
            for source in staging_path.iterdir():
                publish(source, target_folder / source.name)
        except Exception:
            for destination, backup in reversed(published):
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink(missing_ok=True)
                if backup is not None:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.rename(destination)
            raise


ETH3D_UNDISTORTED_ARCHIVES = [
    ("multi_view_training_dslr_undistorted.7z", "dslr"),
    ("multi_view_test_dslr_undistorted.7z", "dslr"),
    ("multi_view_training_rig_undistorted.7z", "rig"),
    ("multi_view_test_rig_undistorted.7z", "rig"),
]
ETH3D_DISTORTED_ARCHIVES = [
    ("multi_view_training_dslr_jpg.7z", "dslr"),
    ("multi_view_test_dslr_jpg.7z", "dslr"),
]


def _download_eth3d_archives(
    data_path: Path, archives: list[tuple[str, str]], dataset_name: str
) -> None:
    num_succeeded = 0
    for filename, category in archives:
        target_folder = data_path / category
        target_folder.mkdir(parents=True, exist_ok=True)
        archive_path = target_folder / filename
        url = "https://www.eth3d.net/data/" + filename
        try:
            pycolmap.logging.info(
                f"Downloading ETH3D category={category}, filename={filename}"
            )
            download_file(url, target_folder)

            pycolmap.logging.info(
                f"Extracting ETH3D category={category}, filename={filename}"
            )
            with tempfile.TemporaryDirectory(
                prefix=f".{filename}.", dir=target_folder
            ) as temporary_dir:
                staging_path = Path(temporary_dir)
                with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                    archive.extractall(path=staging_path)
                _publish_staged_tree(staging_path, target_folder)
            num_succeeded += 1
        except Exception as error:
            pycolmap.logging.error(
                f"Failed to download or extract ETH3D archive {url}: {error}"
            )
            archive_path.unlink(missing_ok=True)

    if num_succeeded == 0:
        raise RuntimeError(f"Failed to download any {dataset_name} archives")


def download_eth3d(data_path: Path) -> None:
    _download_eth3d_archives(
        data_path, ETH3D_UNDISTORTED_ARCHIVES, dataset_name="ETH3D"
    )


def download_eth3d_distorted(data_path: Path) -> None:
    _download_eth3d_archives(
        data_path,
        ETH3D_DISTORTED_ARCHIVES,
        dataset_name="ETH3D distorted",
    )


def download_imc2023(data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

    pycolmap.logging.info("Downloading IMC2023")
    subprocess.check_call(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "image-matching-challenge-2023",
            "-p",
            str(data_path),
        ],
    )

    pycolmap.logging.info("Extracting IMC2023")
    with zipfile.ZipFile(
        data_path / "image-matching-challenge-2023.zip", mode="r"
    ) as archive:
        archive.extractall(path=data_path)


def download_imc2024(data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

    pycolmap.logging.info("Downloading IMC2024")
    subprocess.check_call(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "image-matching-challenge-2024",
            "-p",
            str(data_path),
        ],
    )

    pycolmap.logging.info("Extracting IMC2024")
    with zipfile.ZipFile(
        data_path / "image-matching-challenge-2024.zip", mode="r"
    ) as archive:
        archive.extractall(path=data_path)

    # Move all scenes to the "all" category sub-folder.
    category_path = data_path / "train/all"
    category_path.mkdir(parents=True, exist_ok=True)
    for scene in (data_path / "train").iterdir():
        if str(scene).endswith("/all"):
            continue
        shutil.move(scene, data_path / category_path)


def download_imc2025(data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

    pycolmap.logging.info("Downloading IMC2025")
    subprocess.check_call(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "image-matching-challenge-2025",
            "-p",
            str(data_path),
        ],
    )

    pycolmap.logging.info("Extracting IMC2025")
    with zipfile.ZipFile(
        data_path / "image-matching-challenge-2025.zip", mode="r"
    ) as archive:
        archive.extractall(path=data_path)

    # Move all scenes to the "all" category sub-folder.
    category_path = data_path / "train/all"
    category_path.mkdir(parents=True, exist_ok=True)
    for scene in (data_path / "train").iterdir():
        if scene.name == "all":
            continue
        shutil.move(scene, category_path)


# TODO: BlendedMVS+ and BlendedMVS++.
def download_blended_mvs(data_path: Path) -> None:
    target_folder = data_path / "BlendedMVS"
    target_folder.mkdir(parents=True, exist_ok=True)

    pycolmap.logging.info("Downloading BlendedMVS")
    for filename in [
        "BlendedMVS.zip",
    ] + [f"BlendedMVS.z{i:02d}" for i in range(1, 16)]:
        download_file(
            "https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0/"
            + filename,
            target_folder,
        )

    pycolmap.logging.info("Merging BlendedMVS split archive")
    combined_zip = target_folder / "BlendedMVS_combined.zip"
    subprocess.check_call(
        [
            "zip",
            "-q",
            "-s",
            "0",
            str(target_folder / "BlendedMVS.zip"),
            "--out",
            str(combined_zip),
        ]
    )

    pycolmap.logging.info("Extracting BlendedMVS")
    try:
        with zipfile.ZipFile(combined_zip, mode="r") as archive:
            archive.extractall(path=data_path)
    finally:
        if combined_zip.exists():
            combined_zip.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        while chunk := fid.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_tar_safely(archive_path: Path, output_path: Path) -> None:
    output_root = output_path.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Refusing link in release archive: {member.name}"
                )
            target = (output_path / member.name).resolve()
            if not target.is_relative_to(output_root):
                raise RuntimeError(
                    f"Refusing unsafe release archive path: {member.name}"
                )
        if "filter" in inspect.signature(archive.extractall).parameters:
            archive.extractall(output_path, filter="fully_trusted")
        else:
            archive.extractall(output_path)


def download_tartanair_v2(
    data_path: Path,
    categories: list[str] | None = None,
    scenes: list[str] | None = None,
) -> None:
    manifest = load_manifest()
    categories = categories or []
    scenes = scenes or []
    shards = scene_shards(manifest)
    selected_shards = []
    for index, shard_scenes in enumerate(shards):
        if any(
            (not categories or scene.category in categories)
            and (not scenes or scene.name in scenes)
            for scene in shard_scenes
        ):
            selected_shards.append(index)

    if not selected_shards:
        pycolmap.logging.warning("No TartanAir V2 scenes matched the filters")
        return

    data_path.mkdir(parents=True, exist_ok=True)
    archive_path = data_path / ".archives"
    archive_path.mkdir(exist_ok=True)
    release = manifest["release"]
    base_url = (
        f"https://github.com/{release['repository']}/releases/download/"
        f"{release['tag']}"
    )
    checksum_path = MANIFEST_PATH.with_name("tartanair_v2_checksums.json")
    checksums = json.loads(checksum_path.read_text())
    for index in selected_shards:
        filename = shard_name(manifest, index)
        target = archive_path / filename
        expected = checksums.get(filename)
        if expected is None:
            raise RuntimeError(f"Missing release checksum for {filename}")
        if not target.exists() or (expected and _sha256(target) != expected):
            temporary = target.with_suffix(target.suffix + ".part")
            if temporary.exists():
                temporary.unlink()
            pycolmap.logging.info(f"Downloading TartanAir V2 {filename}")
            download_file(f"{base_url}/{filename}", archive_path)
            downloaded = archive_path / filename
            if downloaded != temporary:
                downloaded.replace(temporary)
            if expected and _sha256(temporary) != expected:
                temporary.unlink()
                raise RuntimeError(f"Checksum mismatch for {filename}")
            temporary.replace(target)
        pycolmap.logging.info(f"Extracting TartanAir V2 {filename}")
        _extract_tar_safely(target, data_path)


DOWNLOADERS = {
    "eth3d": download_eth3d,
    "eth3d-distorted": download_eth3d_distorted,
    "imc2023": download_imc2023,
    "imc2024": download_imc2024,
    "imc2025": download_imc2025,
    "blended-mvs": download_blended_mvs,
    "tartanair-v2": download_tartanair_v2,
}
DEFAULT_DATASETS = [name for name in DOWNLOADERS if name != "eth3d-distorted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to download by name. Valid names: "
        f"{', '.join(DOWNLOADERS)}. eth3d-distorted downloads the large "
        "DSLR-only _jpg.7z archives into data/eth3d-distorted; "
        "eth3d-undistorted is accepted as an alias for eth3d.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="TartanAir categories to download; empty downloads all.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[],
        help="TartanAir scenes to download; empty downloads all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        dataset_names = [
            resolve_dataset_name(name, DOWNLOADERS) for name in args.datasets
        ]
    except ValueError as error:
        raise SystemExit(str(error)) from error

    for name in dataset_names:
        if name == "tartanair-v2":
            download_tartanair_v2(
                args.data_path / name, args.categories, args.scenes
            )
        else:
            DOWNLOADERS[name](args.data_path / name)


if __name__ == "__main__":
    main()
