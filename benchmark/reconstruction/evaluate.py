import argparse
import subprocess
import datetime
import shutil
from pathlib import Path

import pycolmap
import numpy as np


def colmap_reconstruction(
    args,
    workspace_path,
    image_path,
    camera_prior_sparse_gt_path=None,
    extra_args=[],
):
    workspace_path.mkdir(parents=True, exist_ok=True)

    database_path = workspace_path / "database.db"
    if args.overwrite_database and database_path.exists():
        database_path.unlink()

    sparse_path = workspace_path / "sparse"
    if args.overwrite_reconstruction and sparse_path.exists():
        shutil.rmtree(sparse_path)

    if sparse_path.exists():
        print("Skipping reconstruction, as it already exists")
        return

    args = [
        args.colmap_path,
        "automatic_reconstructor",
        "--image_path",
        image_path,
        "--workspace_path",
        workspace_path,
        "--vocab_tree_path",
        args.data_path / "vocab_tree_flickr100K_words256K.bin",
        "--use_gpu",
        "1" if args.use_gpu else "0",
        "--num_threads",
        str(args.num_threads),
        "--quality",
        args.quality,
    ]

    subprocess.check_call(
        args
        + extra_args
        + [
            "--extraction",
            "1",
            "--matching",
            "0",
            "--sparse",
            "0",
            "--dense",
            "0",
        ],
        cwd=workspace_path,
    )

    if camera_prior_sparse_gt_path is not None:
        print("Setting prior cameras from GT")

        sparse_gt = pycolmap.Reconstruction()
        sparse_gt.read(camera_prior_sparse_gt_path)

        database = pycolmap.Database()
        database.open(database_path)

        camera_id_gt_to_camera_id = {}
        for camera_id_gt, camera_gt in sparse_gt.cameras.items():
            camera_gt.has_prior_focal_length = True
            camera_id = database.write_camera(camera_gt)
            camera_id_gt_to_camera_id[camera_id_gt] = camera_id

        images_gt_by_name = {}
        for image_gt in sparse_gt.images.values():
            images_gt_by_name[image_gt.name] = image_gt

        for image in database.read_all_images():
            if image.name not in images_gt_by_name:
                print(
                    f"Not setting prior camera for image {image.name}, "
                    "because it does not exist in GT"
                )
                continue
            image_gt = images_gt_by_name[image.name]
            camera_id = camera_id_gt_to_camera_id[image_gt.camera_id]
            image.camera_id = camera_id
            database.update_image(image)

        database.close()

    subprocess.check_call(
        args
        + extra_args
        + [
            "--extraction",
            "0",
            "--matching",
            "1",
            "--sparse",
            "1",
            "--dense",
            "0",
        ],
        cwd=workspace_path,
    )


def colmap_alignment(
    args, sparse_path, sparse_gt_path, sparse_aligned_path, max_ref_model_error
):
    if args.overwrite_alignment and sparse_aligned_path.exists():
        shutil.rmtree(sparse_aligned_path)
    if sparse_aligned_path.exists():
        print("Skipping alignment, as it already exists")
        return

    if sparse_path.exists():
        sparse_aligned_path.mkdir(parents=True, exist_ok=True)
        subprocess.call(
            [
                args.colmap_path,
                "model_aligner",
                "--input_path",
                sparse_path,
                "--ref_model_path",
                sparse_gt_path,
                "--output_path",
                sparse_aligned_path,
                "--alignment_max_error",
                str(max_ref_model_error),
            ]
        )


def compute_errors(sparse_gt_path, sparse_path):
    sparse_gt = pycolmap.Reconstruction()
    sparse_gt.read(sparse_gt_path)

    if not (sparse_path / "images.bin").exists():
        print("Reconstruction or alignment failed")
        return len(sparse_gt.images) * [np.inf], len(sparse_gt.images) * [180]

    sparse = pycolmap.Reconstruction()
    sparse.read(sparse_path)

    images = {}
    for image in sparse.images.values():
        images[image.name] = image

    dts = []
    dRs = []
    for image_gt in sparse_gt.images.values():
        if image_gt.name not in images:
            dts.append(np.inf)
            dRs.append(180)
            continue

        R_w2c_gt = image_gt.cam_from_world.rotation.matrix()
        t_c2w_gt = image_gt.projection_center()

        image = images[image_gt.name]
        R_w2c = image.cam_from_world.rotation.matrix()
        t_c2w = image.projection_center()

        dt = np.linalg.norm(t_c2w_gt - t_c2w)
        cos = np.clip(((np.trace(R_w2c_gt @ R_w2c.T)) - 1) / 2, -1, 1)
        dR = np.rad2deg(np.abs(np.arccos(cos)))

        dts.append(dt)
        dRs.append(dR)

    return dts, dRs


def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall


def compute_auc(errors, thresholds, min_error=None):
    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapezoid(r, x=e) / t
        aucs.append(auc * 100)
    return aucs


def evaluate_eth3d(args):
    results = {}
    for category_path in (args.data_path / "eth3d").iterdir():
        if not category_path.is_dir() or (
            args.categories and category_path.name not in args.categories
        ):
            continue

        category = category_path.name
        results[category] = {}

        all_dts = []

        for scene_path in category_path.iterdir():
            if not scene_path.is_dir():
                continue

            scene = scene_path.name
            workspace_path = (
                args.run_path / args.run_name / "eth3d" / category / scene
            )
            sparse_gt_path = list(scene_path.glob("*_calibration_undistorted"))[
                0
            ]

            print(f"Processing ETH3D: category={category}, scene={scene}")

            extra_args = []
            if category == "dslr":
                extra_args.extend(["--data_type", "individual"])
            elif category == "rig":
                extra_args.extend(["--data_type", "video"])

            colmap_reconstruction(
                args=args,
                workspace_path=workspace_path,
                image_path=scene_path / "images",
                camera_prior_sparse_gt_path=sparse_gt_path,
                extra_args=extra_args,
            )

            sparse_path = workspace_path / "sparse/0"
            sparse_aligned_path = workspace_path / "sparse_aligned"
            colmap_alignment(
                args=args,
                sparse_path=sparse_path,
                sparse_gt_path=sparse_gt_path,
                sparse_aligned_path=sparse_aligned_path,
                max_ref_model_error=0.1,
            )

            dts, dRs = compute_errors(
                sparse_gt_path=sparse_gt_path,
                sparse_path=sparse_aligned_path,
            )

            all_dts.extend(dts)

            # The ground truth poses are deemed accurate only up to 1mm.
            results[category][scene] = compute_auc(
                dts, args.thresholds, min_error=0.001
            )

        results[category]["__all__"] = compute_auc(
            all_dts, args.thresholds, min_error=0.02
        )

    return results


def evaluate_imc(args, year):
    folder_name = f"imc{year}"
    results = {}
    for category_path in Path(
        args.data_path / f"{folder_name}/train"
    ).iterdir():
        if not category_path.is_dir() or (
            args.categories and category_path.name not in args.categories
        ):
            continue

        category = category_path.name
        results[category] = {}

        all_dts = []

        for scene_path in category_path.iterdir():
            if not scene_path.is_dir():
                continue

            scene = scene_path.name
            workspace_path = (
                args.run_path / args.run_name / folder_name / category / scene
            )
            sparse_gt_path = scene_path / "sfm"

            print(f"Processing IMC {year}: category={category}, scene={scene}")

            colmap_reconstruction(
                args=args,
                workspace_path=workspace_path,
                image_path=scene_path / "images",
                camera_prior_sparse_gt_path=sparse_gt_path,
            )

            sparse_path = workspace_path / "sparse/0"
            sparse_aligned_path = workspace_path / "sparse_aligned"
            colmap_alignment(
                args=args,
                sparse_path=sparse_path,
                sparse_gt_path=sparse_gt_path,
                sparse_aligned_path=sparse_aligned_path,
                max_ref_model_error=0.2,
            )

            dts, dRs = compute_errors(
                sparse_gt_path=sparse_gt_path,
                sparse_path=sparse_aligned_path,
            )

            all_dts.extend(dts)

            # The ground truth poses are deemed accurate only up to 2cm.
            results[category][scene] = compute_auc(
                dts, args.thresholds, min_error=0.02
            )

        results[category]["__all__"] = compute_auc(
            all_dts, args.thresholds, min_error=0.02
        )

    return results


def format_results(results, thresholds):
    column = "scenes"
    size1 = max(
        len(column) + 2,
        max(
            len(s)
            for d in results.values()
            for c in d.values()
            for s in c.keys()
        ),
    )
    metric = "AUC @ X cm (%)"
    size2 = max(len(metric) + 2, len(thresholds) * 7 - 1)
    header = f"{column:=^{size1}} {metric:=^{size2}}"
    header += "\n" + " " * (size1 + 1)
    header += " ".join(
        f'{str(t*100).rstrip("0").rstrip("."):^6}' for t in thresholds
    )
    text = [header]
    for dataset, category_results in results.items():
        for category, scene_results in category_results.items():
            text.append(f"\n{dataset + '=' + category:=^{size1 + size2 + 1}}")
            for scene, aucs in scene_results.items():
                assert len(aucs) == len(thresholds)
                row = ""
                if scene == "__all__":
                    scene = ""
                    row += "-" * (size1 + size2 + 1) + "\n"
                row += f"{scene:<{size1}} "
                row += " ".join(f"{auc:>6.2f}" for auc in aucs)
                text.append(row)
    return "\n".join(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=Path(__file__).parent / "data")
    parser.add_argument(
        "--datasets", nargs="+", default=["eth3d", "imc2023", "imc2024"]
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="Only evaluate specific categories, if empty all categories are evaluated.",
    )
    parser.add_argument("--run_path", default=Path(__file__).parent / "runs")
    parser.add_argument(
        "--run_name",
        default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    parser.add_argument(
        "--overwrite_database", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_reconstruction", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_alignment", default=False, action="store_true"
    )
    parser.add_argument("--colmap_path", required=True)
    parser.add_argument("--use_gpu", default=True, action="store_true")
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--quality", default="high")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.5],
        help="Evaluation thresholds in meters.",
    )
    args = parser.parse_args()
    args.colmap_path = Path(args.colmap_path).resolve()
    if args.overwrite_database:
        print("Overwriting database also overwrites reconstruction")
        args.overwrite_reconstruction = True
    if args.overwrite_reconstruction:
        print("Overwriting reconstruction also overwrites alignment")
        args.overwrite_alignment = True
    return args


def main():
    args = parse_args()

    results = {}
    if "eth3d" in args.datasets:
        results["eth3d"] = evaluate_eth3d(args)
    if "imc2023" in args.datasets:
        results["imc2023"] = evaluate_imc(args, year=2023)
    if "imc2024" in args.datasets:
        results["imc2024"] = evaluate_imc(args, year=2024)

    print("\nResults\n")
    print(format_results(results, args.thresholds))


if __name__ == "__main__":
    main()
