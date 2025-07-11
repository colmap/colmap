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
import shutil
import subprocess
import zipfile
from pathlib import Path

import py7zr
import requests

import pycolmap


def download_file(url: str, target_folder: Path) -> None:
    filename = url.split("/")[-1]
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        with open(target_folder / filename, "wb") as f:
            for chunk in req.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename


def download_eth3d(data_path: Path) -> None:
    for filename, category in [
        ("multi_view_training_dslr_undistorted.7z", "dslr"),
        ("multi_view_test_dslr_undistorted.7z", "dslr"),
        ("multi_view_training_rig_undistorted.7z", "rig"),
        ("multi_view_test_rig_undistorted.7z", "rig"),
    ]:
        target_folder = data_path / category
        target_folder.mkdir(parents=True, exist_ok=True)

        pycolmap.logging.info(
            f"Downloading ETH3D category={category}, filename={filename}"
        )
        download_file("https://www.eth3d.net/data/" + filename, target_folder)

        pycolmap.logging.info(
            f"Extracting ETH3D category={category}, filename={filename}"
        )
        with py7zr.SevenZipFile(target_folder / filename, mode="r") as archive:
            archive.extractall(path=target_folder)


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

    pycolmap.logging.info("Extracting BlendedMVS")
    with zipfile.ZipFile(target_folder / "BlendedMVS.zip", mode="r") as archive:
        archive.extractall(path=data_path)


DOWNLOADERS = {
    "eth3d": download_eth3d,
    "imc2023": download_imc2023,
    "imc2024": download_imc2024,
    "blended-mvs": download_blended_mvs,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DOWNLOADERS.keys(),
        choices=DOWNLOADERS.keys(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for dataset in args.datasets:
        DOWNLOADERS[dataset](args.data_path / dataset)


if __name__ == "__main__":
    main()
