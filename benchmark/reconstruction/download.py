import os
import argparse
from pathlib import Path
import subprocess


def download_eth3d(data_path: Path):
    data_path.mkdir(parents=True, exist_ok=True)

    undistorted_images = "multi_view_training_dslr_undistorted.7z"

    subprocess.call(
        ["wget", "-c", "https://www.eth3d.net/data/" + undistorted_images],
        cwd=data_path,
    )
    subprocess.call(["7zz", "x", undistorted_images], cwd=data_path)
    subprocess.call(["rm", undistorted_images], cwd=data_path)

    scan = "multi_view_training_dslr_scan_eval.7z"
    subprocess.call(
        ["wget", "-c", "https://www.eth3d.net/data/" + scan], cwd=data_path
    )
    subprocess.call(["7zz", "x", scan], cwd=data_path)
    subprocess.call(["rm", scan], cwd=data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default=Path(os.getcwd()) / "data"
    )
    args = parser.parse_args()

    download_eth3d(args.data_path)
