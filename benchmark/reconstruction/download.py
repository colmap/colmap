import argparse
import shutil
import subprocess
from pathlib import Path


def download_vocab_tree(data_path: Path):
    data_path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "wget",
            "-c",
            "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin",
        ],
        cwd=data_path,
    )


def download_eth3d(data_path: Path):
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


def download_imc2023(data_path: Path):
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


def download_imc2024(data_path: Path):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["eth3d", "imc2023", "imc2024"]
    )
    return parser.parse_args()


def main():
    args = parse_args()

    download_vocab_tree(args.data_path)
    if "eth3d" in args.datasets:
        download_eth3d(args.data_path / "eth3d")
    if "imc2023" in args.datasets:
        download_imc2023(args.data_path / "imc2023")
    if "imc2024" in args.datasets:
        download_imc2024(args.data_path / "imc2024")


if __name__ == "__main__":
    main()
