import os
import argparse
from pathlib import Path
import subprocess


def download_eth3d(data_path: Path):
    data_path.mkdir(parents=True, exist_ok=True)

    undistorted_images = "multi_view_training_dslr_undistorted.7z"

    subprocess.check_call(
        ["wget", "-c", "https://www.eth3d.net/data/" + undistorted_images],
        cwd=data_path,
    )
    subprocess.check_call(["7zz", "x", undistorted_images], cwd=data_path)

    scan = "multi_view_training_dslr_scan_eval.7z"
    subprocess.check_call(
        ["wget", "-c", "https://www.eth3d.net/data/" + scan], cwd=data_path
    )
    subprocess.check_call(["7zz", "x", scan], cwd=data_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default=Path(__file__).parent / "data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    download_eth3d(args.data_path / "eth3d")


if __name__ == "__main__":
    main()
