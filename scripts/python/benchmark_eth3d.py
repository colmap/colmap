import os
import sys
import argparse
import urllib.request
import subprocess


def download_file(url, file_path):
    if os.path.exists(file_path):
        return
    print(f"Downloading {url} to {file_path}")
    try:
        urllib.request.urlretrieve(url, file_path)
    except Exception as exc:
        print(f"Failed to download {url} to {file_path} due to {exc}")


def check_small_errors_or_exit(args, errors_csv_path):
    error = False
    with open(errors_csv_path, "r") as fid:
        num_images = 0
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            rotation_error, translation_error, proj_center_error = \
                map(float, line.split(","))
            num_images += 1
            if rotation_error > args.max_rotation_error:
                print("Exceeded rotation error threshold:", rotation_error)
                error = True
            if translation_error > args.max_translation_error:
                print("Exceeded translation error threshold:",
                      translation_error)
                error = True
            if proj_center_error > args.max_proj_center_error:
                print("Exceeded projection center error threshold:",
                      proj_center_error)
                error = True

    if args.expected_num_images >= 0 and num_images != args.expected_num_images:
        print("Unexpected number of images:", num_images)
        error = True

    if error:
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--workspace_path", required=True)
    parser.add_argument("--colmap_path", required=True)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--max_rotation_error", type=float, default=1.0)
    parser.add_argument("--max_translation_error", type=float, default=0.1)
    parser.add_argument("--max_proj_center_error", type=float, default=0.1)
    parser.add_argument("--expected_num_images", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    workspace_path = os.path.realpath(args.workspace_path)
    dataset_archive_path = os.path.join(workspace_path, "dataset.7z")
    download_file(f"https://www.eth3d.net/data/{args.dataset_name}_dslr_undistorted.7z", dataset_archive_path)

    subprocess.check_call(["7zz", "x", "dataset.7z"], cwd=workspace_path)

    subprocess.check_call([
        os.path.realpath(args.colmap_path),
        "automatic_reconstructor",
        "--image_path", f"{args.dataset_name}/images/",
        "--workspace_path", workspace_path,
        "--use_gpu", "1" if args.use_gpu else "0",
        "--num_threads", str(args.num_threads),
        "--quality", "low",
        "--camera_model", "PINHOLE"],
        cwd=workspace_path)
    
    subprocess.check_call([
        os.path.realpath(args.colmap_path),
        "model_comparer",
        "--input_path1", "sparse/0",
        "--input_path2", f"{args.dataset_name}/dslr_calibration_undistorted/",
        "--output_path", ".",
        "--alignment_error", "proj_center",
        "--max_proj_center_error", str(args.max_proj_center_error)],
        cwd=workspace_path)
    
    check_small_errors_or_exit(args, os.path.join(workspace_path, "errors.csv"))


if __name__ == "__main__":
    main()
