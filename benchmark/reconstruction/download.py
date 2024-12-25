# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
from pathlib import Path


def download_eth3d(data_path: Path) -> None:
    for filename, category in [
        ("multi_view_training_dslr_undistorted.7z", "dslr"),
        ("multi_view_test_dslr_undistorted.7z", "dslr"),
        ("multi_view_training_rig_undistorted.7z", "rig"),
        ("multi_view_test_rig_undistorted.7z", "rig"),
    ]:
        target_folder = data_path / category
        target_folder.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["wget", "-c", "https://www.eth3d.net/data/" + filename],
            cwd=target_folder,
        )
        subprocess.check_call(["7zz", "x", filename], cwd=target_folder)


def download_imc2023(data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

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
    subprocess.check_call(
        ["unzip", "image-matching-challenge-2023.zip"], cwd=data_path
    )


def download_imc2024(data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

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
    subprocess.check_call(
        ["unzip", "image-matching-challenge-2024.zip"], cwd=data_path
    )
    # Move all scenes to the "all" category sub-folder.
    category_path = data_path / "train/all"
    category_path.mkdir(parents=True, exist_ok=True)
    for scene in (data_path / "train").iterdir():
        if str(scene).endswith("/all"):
            continue
        shutil.move(scene, data_path / category_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["eth3d", "imc2023", "imc2024"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "eth3d" in args.datasets:
        download_eth3d(args.data_path / "eth3d")
    if "imc2023" in args.datasets:
        download_imc2023(args.data_path / "imc2023")
    if "imc2024" in args.datasets:
        download_imc2024(args.data_path / "imc2024")


if __name__ == "__main__":
    main()
