import os
import sys
import argparse
import urllib.request
import subprocess


def download_file(url, file_path, max_retries=3):
    if os.path.exists(file_path):
        return
    print(f"Downloading {url} to {file_path}")
    for retry in range(max_retries):
        try:
            urllib.request.urlretrieve(url, file_path)
            return
        except Exception as exc:
            print(
                f"Failed to download {url} (trial={retry+1}) to {file_path} due to {exc}"
            )


def check_small_errors_or_exit(
    max_rotation_error,
    max_proj_center_error,
    expected_num_images,
    errors_csv_path,
):
    error = False
    with open(errors_csv_path, "r") as fid:
        num_images = 0
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            rotation_error, proj_center_error = map(float, line.split(","))
            num_images += 1
            if rotation_error > max_rotation_error:
                print("Exceeded rotation error threshold:", rotation_error)
                error = True
            if proj_center_error > max_proj_center_error:
                print("Exceeded projection center error threshold:", proj_center_error)
                error = True

    if num_images != expected_num_images:
        print("Unexpected number of images:", num_images)
        error = True

    if error:
        sys.exit(1)


def process_dataset(args, dataset_name):
    print("Processing dataset:", dataset_name)

    workspace_path = os.path.join(os.path.realpath(args.workspace_path), dataset_name)
    os.makedirs(workspace_path, exist_ok=True)

    dataset_archive_path = os.path.join(workspace_path, f"{dataset_name}.7z")
    download_file(
        f"https://www.eth3d.net/data/{dataset_name}_dslr_undistorted.7z",
        dataset_archive_path,
    )

    subprocess.check_call(["7zz", "x", "-y", f"{dataset_name}.7z"], cwd=workspace_path)

    # Find undistorted parameters of first camera and initialize all images with it.
    with open(
        os.path.join(
            workspace_path,
            f"{dataset_name}/dslr_calibration_undistorted/cameras.txt",
        ),
        "r",
    ) as fid:
        for line in fid:
            if not line.startswith("#"):
                first_camera_data = line.split()
                camera_model = first_camera_data[1]
                assert camera_model == "PINHOLE"
                camera_params = first_camera_data[4:]
                assert len(camera_params) == 4
                break

    # Count the number of expected images in the GT.
    expected_num_images = 0
    with open(
        os.path.join(
            workspace_path,
            f"{dataset_name}/dslr_calibration_undistorted/images.txt",
        ),
        "r",
    ) as fid:
        for line in fid:
            if not line.startswith("#") and line.strip():
                expected_num_images += 1
    # Each image uses two consecutive lines.
    assert expected_num_images % 2 == 0
    expected_num_images /= 2

    # Run automatic reconstruction pipeline.
    subprocess.check_call(
        [
            os.path.realpath(args.colmap_path),
            "automatic_reconstructor",
            "--image_path",
            f"{dataset_name}/images/",
            "--workspace_path",
            workspace_path,
            "--use_gpu",
            "1" if args.use_gpu else "0",
            "--num_threads",
            str(args.num_threads),
            "--quality",
            "low",
            "--camera_model",
            "PINHOLE",
            "--camera_params",
            ",".join(camera_params),
        ],
        cwd=workspace_path,
    )

    # Compare reconstructed model to GT model.
    subprocess.check_call(
        [
            os.path.realpath(args.colmap_path),
            "model_comparer",
            "--input_path1",
            "sparse/0",
            "--input_path2",
            f"{dataset_name}/dslr_calibration_undistorted/",
            "--output_path",
            ".",
            "--alignment_error",
            "proj_center",
            "--max_proj_center_error",
            str(args.max_proj_center_error),
        ],
        cwd=workspace_path,
    )

    # Ensure discrepancy between reconstructed model and GT is small.
    check_small_errors_or_exit(
        args.max_rotation_error,
        args.max_proj_center_error,
        expected_num_images,
        os.path.join(workspace_path, "errors.csv"),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument("--workspace_path", required=True)
    parser.add_argument("--colmap_path", required=True)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--max_rotation_error", type=float, default=1.0)
    parser.add_argument("--max_proj_center_error", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset_name in args.dataset_names.split(","):
        process_dataset(args, dataset_name.strip())


if __name__ == "__main__":
    main()
