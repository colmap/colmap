from pathlib import Path
import argparse
import subprocess
import datetime
import shutil


def run_colmap(args, workspace_path, image_path, extra_args=[]):
    workspace_path.mkdir(parents=True, exist_ok=True)

    database_path = workspace_path / "database.db"
    if args.overwrite_database and database_path.exists():
        database_path.remove()

    sparse_path = workspace_path / "sparse"
    if args.overwrite_sparse and sparse_path.exists():
        shutil.rmtree(sparse_path)

    if sparse_path.exists():
        print("Skipping reconstruction, as it already exists")
        return

    subprocess.check_call(
        [
            args.colmap_path,
            "automatic_reconstructor",
            "--image_path",
            image_path,
            "--workspace_path",
            workspace_path.resolve(),
            "--use_gpu",
            "1" if args.use_gpu else "0",
            "--num_threads",
            str(args.num_threads),
            "--quality",
            args.quality,
        ]
        + extra_args,
        cwd=workspace_path,
    )


def evaluate_eth3d(args):
    for scene_path in Path(args.data_path / "eth3d").iterdir():
        if not scene_path.is_dir():
            continue

        scene = scene_path.name
        workspace_path = args.run_path / args.run_name / "eth3d" / scene

        print("Processing ETH3D scene:", scene)

        with open(scene_path / "dslr_calibration_undistorted/cameras.txt", "r") as fid:
            for line in fid:
                if not line.startswith("#"):
                    first_camera_data = line.split()
                    camera_model = first_camera_data[1]
                    assert camera_model == "PINHOLE"
                    camera_params = first_camera_data[4:]
                    assert len(camera_params) == 4
                    break

        run_colmap(
            args,
            workspace_path,
            scene_path / "images",
            extra_args=[
                "--camera_model",
                "PINHOLE",
                "--camera_params",
                ",".join(camera_params),
            ],
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=Path(__file__).parent / "data")
    parser.add_argument("--datasets", nargs="+", default=["eth3d"])
    parser.add_argument("--run_path", default=Path(__file__).parent / "runs")
    parser.add_argument(
        "--run_name", default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument("--overwrite_database", default=False, action="store_true")
    parser.add_argument("--overwrite_sparse", default=False, action="store_true")
    parser.add_argument("--colmap_path", required=True)
    parser.add_argument("--use_gpu", default=True, action="store_true")
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--quality", default="medium")
    args = parser.parse_args()
    args.colmap_path = Path(args.colmap_path).resolve()
    return args


def main():
    args = parse_args()

    if "eth3d" in args.datasets:
        evaluate_eth3d(args)


if __name__ == "__main__":
    main()
