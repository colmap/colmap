import os
import argparse
from pathlib import Path
import subprocess


def download_vocab_tree(data_path: Path):
    data_path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["wget", "-c", "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"],
        cwd=data_path,
    )


def download_eth3d(data_path: Path):
    data_path.mkdir(parents=True, exist_ok=True)

    multi_view_dslr = "multi_view_training_dslr_undistorted.7z"
    subprocess.check_call(
        ["wget", "-c", "https://www.eth3d.net/data/" + multi_view_dslr],
        cwd=data_path,
    )
    subprocess.check_call(["7zz", "x", mult_view], cwd=data_path)

    multi_view_rig = "multi_view_training_rig_undistorted.7z"
    subprocess.check_call(
        ["wget", "-c", "https://www.eth3d.net/data/" + multi_view_rig],
        cwd=data_path,
    )
    subprocess.check_call(["7zz", "x", multi_view_rig], cwd=data_path)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default=Path(__file__).parent / "data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    download_vocab_tree(args.data_path)
    download_eth3d(args.data_path / "eth3d")


if __name__ == "__main__":
    main()
