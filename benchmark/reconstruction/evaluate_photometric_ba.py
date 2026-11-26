# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark photometric bundle adjustment on the ETH3D distorted DSLR dataset.

Protocol (for both SIFT and DaD+FlexGlue across the 13 ETH3D training scenes):
  1. Extract & match features (cached per scene & feature type), populating GT
     THIN_PRISM_FISHEYE intrinsics into the database prior to matching.
  2. Triangulate 3D points from GT poses and intrinsics (`point_triangulator`).
  3. Track A (Joint Pose + Intrinsics + 3D Recovery):
     - Run geometric `bundle_adjuster` refining poses + focal_length +
       extra_params + 3D points (starting from GT poses/intrinsics + step 2).
     - Run `photometric_bundle_adjuster` starting from geometric BA's output.
     - Measure pose recovery (relative pose AUC @ 1/3/5 deg, Sim3-aligned
       rotation & camera-center error vs. GT), intrinsics recovery (relative
       focal error %, mean grid ray-reprojection error px vs. GT), 3D point
       cloud accuracy & completeness @ [1, 2, 5] cm via
       ETH3DMultiViewEvaluation, and wall-clock runtime.
  4. Track B (Fixed-GT-Pose Triangulation Eval, PixSfM protocol):
     - Starting from GT-pose triangulated points (step 2), run
       `photometric_bundle_adjuster` with poses and intrinsics fixed.
     - Evaluate 3D point cloud accuracy & completeness @ [1, 2, 5] cm against
       ETH3D laser scans via ETH3DMultiViewEvaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from evaluation.utils import set_cameras_from_gt

import pycolmap

ETH3D_TRAINING_SCENES = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
]

DEFAULT_TOLERANCES = [0.01, 0.02, 0.05]


def run_cmd(cmd: list[str], log_path: Path) -> tuple[float, str]:
    """Run a subprocess, log combined output, and return (secs, output)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"Log: {log_path}\n--- tail ---\n{tail}"
        )
    return elapsed, proc.stdout


def extract_and_match_scene(
    colmap_bin: Path,
    feature: str,
    image_dir: Path,
    gt_sparse_dir: Path,
    db_path: Path,
    log_dir: Path,
    num_threads: int,
    overwrite: bool = False,
) -> None:
    """Extract & match features, seeding GT cameras before matching."""
    done_marker = db_path.with_suffix(".matched")
    if done_marker.exists() and db_path.exists() and not overwrite:
        return
    if db_path.exists():
        db_path.unlink()
    if done_marker.exists():
        done_marker.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    gt_rec = pycolmap.Reconstruction(str(gt_sparse_dir))

    # Build image list matching GT registered images.
    image_list_path = db_path.parent / "image_list.txt"
    gt_names = sorted(img.name for img in gt_rec.images.values())
    image_list_path.write_text("\n".join(gt_names) + "\n")

    ext_cmd = [
        str(colmap_bin),
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--image_list_path",
        str(image_list_path),
        "--FeatureExtraction.num_threads",
        str(num_threads),
        "--FeatureExtraction.use_gpu",
        "1",
    ]
    if feature == "sift":
        ext_cmd += ["--FeatureExtraction.type", "SIFT"]
    elif feature == "flexglue":
        # Matches AutomaticReconstructionController::Feature::FLEXGLUE (DaD).
        ext_cmd += ["--FeatureExtraction.type", "LOMA_B128"]
    else:
        raise ValueError(f"Unsupported feature type: {feature}")

    run_cmd(ext_cmd, log_dir / "feature_extractor.log")

    # Seed exact GT cameras, sharing, and rigs/frames into database.db before
    # matching so geometric verification uses accurate THIN_PRISM_FISHEYE
    # models.
    set_cameras_from_gt(db_path, gt_rec)

    match_cmd = [
        str(colmap_bin),
        "exhaustive_matcher",
        "--database_path",
        str(db_path),
        "--FeatureMatching.num_threads",
        str(num_threads),
        "--FeatureMatching.use_gpu",
        "1",
    ]
    if feature == "sift":
        match_cmd += ["--FeatureMatching.type", "SIFT_BRUTEFORCE"]
    elif feature == "flexglue":
        match_cmd += [
            "--FeatureMatching.type",
            "FLEXGLUE",
            "--FlexGlueMatching.image_path",
            str(image_dir),
            "--FeatureMatching.guided_matching",
            "0",
        ]

    run_cmd(match_cmd, log_dir / "exhaustive_matcher.log")
    done_marker.write_text("ok\n")


def build_db_aligned_gt_reconstruction(
    db_path: Path, gt_sparse_dir: Path, out_dir: Path
) -> None:
    """Create a GT reconstruction whose IDs match database.db."""
    gt_rec = pycolmap.Reconstruction(str(gt_sparse_dir))
    gt_by_name = {img.name: img for img in gt_rec.images.values()}

    rec = pycolmap.Reconstruction()
    with pycolmap.Database.open(str(db_path)) as db:
        for cam in db.read_all_cameras():
            rec.add_camera(cam)
        for rig in db.read_all_rigs():
            rec.add_rig(rig)
        db_images = {img.image_id: img for img in db.read_all_images()}
        for frame in db.read_all_frames():
            pose = None
            for data_id in frame.data_ids:
                if data_id.id in db_images:
                    name = db_images[data_id.id].name
                    if name in gt_by_name:
                        pose = gt_by_name[name].cam_from_world()
                        break
            if pose is None:
                continue
            frame.rig_from_world = pose
            rec.add_frame(frame)
        for img in db_images.values():
            if img.name in gt_by_name:
                rec.add_image(img)

    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))


def triangulate_from_gt(
    colmap_bin: Path,
    db_path: Path,
    image_dir: Path,
    gt_sparse_dir: Path,
    out_dir: Path,
    log_path: Path,
    overwrite: bool = False,
) -> float:
    """Triangulate 3D points with fixed GT poses & GT intrinsics."""
    if (out_dir / "points3D.bin").exists() and not overwrite:
        return 0.0
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_aligned_input = out_dir.parent / "gt_input_db_aligned"
    build_db_aligned_gt_reconstruction(db_path, gt_sparse_dir, gt_aligned_input)

    cmd = [
        str(colmap_bin),
        "point_triangulator",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--input_path",
        str(gt_aligned_input),
        "--output_path",
        str(out_dir),
        "--clear_points",
        "1",
        "--refine_intrinsics",
        "0",
    ]
    elapsed, _ = run_cmd(cmd, log_path)
    return elapsed


def run_geometric_ba(
    colmap_bin: Path,
    input_dir: Path,
    out_dir: Path,
    log_path: Path,
    refine_principal_point: bool = False,
    overwrite: bool = False,
) -> float:
    """Run geometric bundle_adjuster refining poses, intrinsics, and points."""
    if (out_dir / "points3D.bin").exists() and not overwrite:
        return 0.0
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(colmap_bin),
        "bundle_adjuster",
        "--input_path",
        str(input_dir),
        "--output_path",
        str(out_dir),
        "--BundleAdjustment.refine_focal_length",
        "1",
        "--BundleAdjustment.refine_principal_point",
        "1" if refine_principal_point else "0",
        "--BundleAdjustment.refine_extra_params",
        "1",
        "--BundleAdjustment.refine_rig_from_world",
        "1",
        "--BundleAdjustment.refine_points3D",
        "1",
        "--BundleAdjustmentCeres.max_num_iterations",
        "100",
    ]
    elapsed, _ = run_cmd(cmd, log_path)
    return elapsed


def run_photometric_ba(
    colmap_bin: Path,
    input_dir: Path,
    image_dir: Path,
    out_dir: Path,
    log_path: Path,
    *,
    refine_poses: bool = True,
    refine_intrinsics: bool = True,
    refine_principal_point: bool = False,
    pyramid_levels: str = "2,1,0",
    patch_size: int = 4,
    max_image_size: int = -1,
    warmup_iters: int = 20,
    max_num_iterations: int = 50,
    min_patch_std: float = 2.0,
    loss_type: str = "GEMAN_MCCLURE",
    loss_scale: float = 0.5,
    geo_weight: float = 1.0,
    num_threads: int = -1,
    overwrite: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Run photometric_bundle_adjuster and parse timing & filtering stats."""
    meta_path = out_dir / "photo_ba_stats.json"
    if (
        (out_dir / "points3D.bin").exists()
        and meta_path.exists()
        and not overwrite
    ):
        stats = json.loads(meta_path.read_text())
        return float(stats.get("wall_seconds", 0.0)), stats

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(colmap_bin),
        "photometric_bundle_adjuster",
        "--input_path",
        str(input_dir),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(out_dir),
        "--BundleAdjustment.refine_focal_length",
        "1" if refine_intrinsics else "0",
        "--BundleAdjustment.refine_principal_point",
        "1" if (refine_intrinsics and refine_principal_point) else "0",
        "--BundleAdjustment.refine_extra_params",
        "1" if refine_intrinsics else "0",
        "--BundleAdjustment.refine_rig_from_world",
        "1" if refine_poses else "0",
        "--BundleAdjustment.refine_points3D",
        "1",
        "--BundleAdjustmentCeres.max_num_iterations",
        str(max_num_iterations),
        "--PhotometricBundleAdjustment.pyramid_levels",
        pyramid_levels,
        "--PhotometricBundleAdjustment.patch_size",
        str(patch_size),
        "--PhotometricBundleAdjustment.max_image_size",
        str(max_image_size),
        "--PhotometricBundleAdjustment.structure_warmup_max_num_iterations",
        str(warmup_iters),
        "--PhotometricBundleAdjustment.min_patch_std",
        str(min_patch_std),
        "--PhotometricBundleAdjustment.loss_function_type",
        loss_type,
        "--PhotometricBundleAdjustment.loss_function_scale",
        str(loss_scale),
        "--PhotometricBundleAdjustment.geometric_reprojection_weight",
        str(geo_weight),
        "--PhotometricBundleAdjustment.num_threads",
        str(num_threads),
    ]
    elapsed, output = run_cmd(cmd, log_path)

    stats: dict[str, Any] = {"wall_seconds": elapsed}
    m_load = re.search(r"prepared \d+ image pyramids .* in ([0-9.]+)s", output)
    if m_load:
        stats["image_load_seconds"] = float(m_load.group(1))
    levels_parsed = []
    for m in re.finditer(
        r"Photometric BA level (\d+) \(scale 1/\d+\): "
        r"kept (\d+) / (\d+) tracks "
        r"\(([0-9.]+)% filtered: (\d+) short, (\d+) invalid/oob, "
        r"(\d+) low-texture.*\), "
        r"(\d+) patch residuals",
        output,
    ):
        levels_parsed.append(
            {
                "level": int(m.group(1)),
                "kept": int(m.group(2)),
                "total": int(m.group(3)),
                "filtered_pct": float(m.group(4)),
                "short": int(m.group(5)),
                "invalid": int(m.group(6)),
                "low_texture": int(m.group(7)),
                "patch_residuals": int(m.group(8)),
            }
        )
    stats["levels"] = levels_parsed
    meta_path.write_text(json.dumps(stats, indent=2) + "\n")
    return elapsed, stats


def compute_auc(errors: np.ndarray, thresholds: list[float]) -> list[float]:
    if errors.size == 0:
        return [0.0 for _ in thresholds]
    errs = np.sort(errors)
    recalls = np.arange(1, len(errs) + 1) / len(errs)
    aucs = []
    for t in thresholds:
        e = np.concatenate(([0.0], errs[errs < t], [t]))
        r = np.concatenate(
            (
                [0.0],
                recalls[errs < t],
                [recalls[errs < t][-1] if np.any(errs < t) else 0.0],
            )
        )
        aucs.append(float(np.trapezoid(r, x=e) / t * 100.0))
    return aucs


def align_and_eval_poses_intrinsics(
    est_dir: Path,
    gt_dir: Path,
    aligned_out_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute relative pose AUC, Sim3-aligned error, and intrinsics vs GT."""
    gt = pycolmap.Reconstruction(str(gt_dir))
    est = pycolmap.Reconstruction(str(est_dir))

    gt_by_name = {img.name: img for img in gt.images.values()}
    est_by_name = {img.name: img for img in est.images.values()}
    common_names = sorted(set(gt_by_name.keys()) & set(est_by_name.keys()))

    # 1. Pairwise relative pose errors (rotation & translation dir, scale-free).
    rel_rot_errs = []
    rel_trans_errs = []
    rel_max_errs = []
    gt_poses = {n: gt_by_name[n].cam_from_world() for n in common_names}
    est_poses = {n: est_by_name[n].cam_from_world() for n in common_names}

    for i in range(len(common_names)):
        n1 = common_names[i]
        T1_gt = gt_poses[n1]
        T1_est = est_poses[n1]
        for j in range(i + 1, len(common_names)):
            n2 = common_names[j]
            T21_gt = gt_poses[n2] * T1_gt.inverse()
            T21_est = est_poses[n2] * T1_est.inverse()

            # Rotation error (deg).
            R_diff = T21_est.rotation.matrix() @ T21_gt.rotation.matrix().T
            cos_a = np.clip(0.5 * (np.trace(R_diff) - 1.0), -1.0, 1.0)
            r_err = float(np.degrees(np.arccos(cos_a)))

            # Translation direction error (deg).
            t_gt = T21_gt.translation
            t_est = T21_est.translation
            nt_gt = np.linalg.norm(t_gt)
            nt_est = np.linalg.norm(t_est)
            if nt_gt > 1e-8 and nt_est > 1e-8:
                cos_t = np.clip(
                    np.dot(t_gt, t_est) / (nt_gt * nt_est), -1.0, 1.0
                )
                t_err = float(np.degrees(np.arccos(cos_t)))
            else:
                t_err = 0.0
            rel_rot_errs.append(r_err)
            rel_trans_errs.append(t_err)
            rel_max_errs.append(max(r_err, t_err))

    rel_max_arr = np.array(rel_max_errs, dtype=np.float64)
    rel_rot_arr = np.array(rel_rot_errs, dtype=np.float64)
    rel_trans_arr = np.array(rel_trans_errs, dtype=np.float64)

    # 2. Sim3 alignment of est camera centers to GT camera centers (Umeyama).
    src_centers = np.array(
        [est_by_name[n].projection_center() for n in common_names],
        dtype=np.float64,
    )
    tgt_centers = np.array(
        [gt_by_name[n].projection_center() for n in common_names],
        dtype=np.float64,
    )
    sim3 = pycolmap.estimate_sim3d(
        [src_centers[i] for i in range(len(common_names))],
        [tgt_centers[i] for i in range(len(common_names))],
    )
    est_aligned = pycolmap.Reconstruction(est)
    if sim3 is not None:
        est_aligned.transform(sim3)

    if aligned_out_dir is not None:
        aligned_out_dir.mkdir(parents=True, exist_ok=True)
        est_aligned.write(str(aligned_out_dir))

    est_aligned_by_name = {img.name: img for img in est_aligned.images.values()}
    abs_rot_errs_deg = []
    abs_pos_errs_mm = []
    for n in common_names:
        T_gt = gt_by_name[n].cam_from_world()
        T_al = est_aligned_by_name[n].cam_from_world()
        R_diff = T_al.rotation.matrix() @ T_gt.rotation.matrix().T
        cos_a = np.clip(0.5 * (np.trace(R_diff) - 1.0), -1.0, 1.0)
        abs_rot_errs_deg.append(float(np.degrees(np.arccos(cos_a))))
        abs_pos_errs_mm.append(
            float(
                np.linalg.norm(
                    est_aligned_by_name[n].projection_center()
                    - gt_by_name[n].projection_center()
                )
                * 1000.0
            )
        )

    # 3. Intrinsics recovery vs GT THIN_PRISM_FISHEYE cameras:
    # - Relative focal length error (%)
    # - Mean pixel reprojection difference over a 20x14 grid of rays lifted
    #   from GT camera and reprojected into estimated camera.
    focal_rel_errs_pct = []
    grid_reproj_errs_px = []
    seen_cam_pairs: set[tuple[int, int]] = set()
    for n in common_names:
        gt_cam = gt.cameras[gt_by_name[n].camera_id]
        est_cam = est.cameras[est_by_name[n].camera_id]
        pair = (gt_by_name[n].camera_id, est_by_name[n].camera_id)
        if pair in seen_cam_pairs:
            continue
        seen_cam_pairs.add(pair)

        f_gt = gt_cam.mean_focal_length()
        f_est = est_cam.mean_focal_length()
        focal_rel_errs_pct.append(abs(f_est - f_gt) / f_gt * 100.0)

        xs = np.linspace(0.05 * gt_cam.width, 0.95 * gt_cam.width, 20)
        ys = np.linspace(0.05 * gt_cam.height, 0.95 * gt_cam.height, 14)
        errs = []
        for y in ys:
            for x in xs:
                ray = gt_cam.cam_from_img(np.array([x, y]))
                if ray is None:
                    continue
                proj = est_cam.img_from_cam(np.array([ray[0], ray[1], 1.0]))
                if proj is None:
                    continue
                errs.append(float(np.hypot(proj[0] - x, proj[1] - y)))
        if errs:
            grid_reproj_errs_px.append(float(np.mean(errs)))

    return {
        "num_common_images": len(common_names),
        "num_points3D": est.num_points3D(),
        "rel_pose_auc_1_3_5": compute_auc(rel_max_arr, [1.0, 3.0, 5.0]),
        "rel_pose_auc_01_05_1": compute_auc(rel_max_arr, [0.1, 0.5, 1.0]),
        "rel_rot_median_deg": float(np.median(rel_rot_arr)),
        "rel_trans_median_deg": float(np.median(rel_trans_arr)),
        "abs_rot_median_deg": float(np.median(abs_rot_errs_deg)),
        "abs_rot_mean_deg": float(np.mean(abs_rot_errs_deg)),
        "abs_pos_median_mm": float(np.median(abs_pos_errs_mm)),
        "abs_pos_mean_mm": float(np.mean(abs_pos_errs_mm)),
        "focal_rel_err_pct": float(np.mean(focal_rel_errs_pct))
        if focal_rel_errs_pct
        else float("nan"),
        "intrinsics_grid_err_px": float(np.mean(grid_reproj_errs_px))
        if grid_reproj_errs_px
        else float("nan"),
    }


def eval_eth3d_point_cloud(
    eval_tool: Path,
    rec_dir: Path,
    scan_mlp_path: Path,
    tolerances: list[float],
    log_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export sparse reconstruction to PLY and run ETH3DMultiViewEvaluation."""
    cache_path = rec_dir / "eth3d_eval.json"
    if cache_path.exists() and not overwrite:
        cached = json.loads(cache_path.read_text())
        if cached.get("tolerances_m") == tolerances:
            return cached
    rec = pycolmap.Reconstruction(str(rec_dir))
    ply_path = rec_dir / "points3D.ply"
    rec.export_PLY(str(ply_path))
    cmd = [
        str(eval_tool),
        "--reconstruction_ply_path",
        str(ply_path),
        "--ground_truth_mlp_path",
        str(scan_mlp_path),
        "--tolerances",
        ",".join(str(t) for t in tolerances),
    ]
    _, out = run_cmd(cmd, log_path)
    acc = None
    comp = None
    for line in out.splitlines():
        if line.startswith("Accuracies:"):
            acc = [float(x) * 100.0 for x in line.split()[1:]]
        elif line.startswith("Completenesses:"):
            comp = [float(x) * 100.0 for x in line.split()[1:]]
    if acc is None or comp is None:
        raise RuntimeError(
            f"Failed to parse ETH3DMultiViewEvaluation output:\n{out}"
        )
    res = {
        "tolerances_m": tolerances,
        "accuracy_pct": acc,
        "completeness_pct": comp,
    }
    cache_path.write_text(json.dumps(res, indent=2) + "\n")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--colmap_path",
        type=Path,
        default=Path("./install/bin/colmap"),
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(__file__).parent / "data/eth3d-distorted/dslr",
    )
    parser.add_argument(
        "--scan_eval_path",
        type=Path,
        default=Path("./build-deps/eth3d_scan_eval"),
    )
    parser.add_argument(
        "--eval_tool_path",
        type=Path,
        default=None,
        help="Path to ETH3DMultiViewEvaluation binary (or found on PATH).",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("./benchmark/reconstruction/runs/photometric_ba_eth3d"),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["sift", "flexglue"],
        choices=["sift", "flexglue"],
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=ETH3D_TRAINING_SCENES,
    )
    parser.add_argument("--tag", default="default")
    parser.add_argument("--pyramid_levels", default="2,1,0")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--max_image_size", type=int, default=-1)
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--max_num_iterations", type=int, default=50)
    parser.add_argument("--min_patch_std", type=float, default=2.0)
    parser.add_argument("--loss_type", default="GEMAN_MCCLURE")
    parser.add_argument("--loss_scale", type=float, default=0.5)
    parser.add_argument("--geo_weight", type=float, default=1.0)
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--eval_workers", type=int, default=6)
    parser.add_argument("--skip_track_b", action="store_true")
    parser.add_argument("--overwrite_photo_ba", action="store_true")
    parser.add_argument("--overwrite_all", action="store_true")
    args = parser.parse_args()

    results: dict[str, Any] = {
        "config": {
            "tag": args.tag,
            "pyramid_levels": args.pyramid_levels,
            "patch_size": args.patch_size,
            "max_image_size": args.max_image_size,
            "warmup_iters": 0 if args.no_warmup else args.warmup_iters,
            "max_num_iterations": args.max_num_iterations,
            "min_patch_std": args.min_patch_std,
            "loss_type": args.loss_type,
            "loss_scale": args.loss_scale,
            "geo_weight": args.geo_weight,
        },
        "results": {},
    }

    eval_tool_path: Path | None = args.eval_tool_path
    if eval_tool_path is None:
        found = shutil.which("ETH3DMultiViewEvaluation")
        if found is not None:
            eval_tool_path = Path(found)
    has_eval_tool = eval_tool_path is not None and eval_tool_path.exists()
    if not has_eval_tool:
        print(
            "[WARN] ETH3DMultiViewEvaluation not provided/found; "
            "skipping 3D scan evaluation."
        )

    pending_3d_evals: list[tuple[str, str, str, Path, Path, Path, bool]] = []

    for feature in args.features:
        results["results"][feature] = {}
        for scene in args.scenes:
            print(f"\n=== [{feature}] {scene} ({args.tag}) ===", flush=True)
            scene_data = args.data_path / scene
            image_dir = scene_data / "images"
            gt_sparse_dir = scene_data / "dslr_calibration_jpg"
            scan_mlp = (
                args.scan_eval_path
                / scene
                / "dslr_scan_eval"
                / "scan_alignment.mlp"
            )

            ws = args.run_dir / feature / scene
            db_path = ws / "database.db"
            tri_dir = ws / "triangulated_gt"
            geo_ba_dir = ws / "geo_ba"
            photo_ba_joint_dir = ws / f"photo_ba_joint_{args.tag}"
            photo_ba_fixed_dir = ws / f"photo_ba_fixed_{args.tag}"

            # 1. Extract & match (shared across tags).
            extract_and_match_scene(
                args.colmap_path,
                feature,
                image_dir,
                gt_sparse_dir,
                db_path,
                ws / "logs",
                args.num_threads if args.num_threads > 0 else 12,
                overwrite=args.overwrite_all,
            )

            # 2. Triangulate from GT poses & intrinsics.
            tri_sec = triangulate_from_gt(
                args.colmap_path,
                db_path,
                image_dir,
                gt_sparse_dir,
                tri_dir,
                ws / "logs/point_triangulator.log",
                overwrite=args.overwrite_all,
            )

            # 3. Geometric BA refining poses + focal + extra + points.
            geo_sec = run_geometric_ba(
                args.colmap_path,
                tri_dir,
                geo_ba_dir,
                ws / "logs/geo_ba.log",
                overwrite=args.overwrite_all,
            )

            # 4. Photometric BA (Track A: joint poses + intrinsics + points).
            photo_joint_sec, photo_joint_stats = run_photometric_ba(
                args.colmap_path,
                geo_ba_dir,
                image_dir,
                photo_ba_joint_dir,
                ws / f"logs/photo_ba_joint_{args.tag}.log",
                refine_poses=True,
                refine_intrinsics=True,
                pyramid_levels=args.pyramid_levels,
                patch_size=args.patch_size,
                max_image_size=args.max_image_size,
                warmup_iters=0 if args.no_warmup else args.warmup_iters,
                max_num_iterations=args.max_num_iterations,
                min_patch_std=args.min_patch_std,
                loss_type=args.loss_type,
                loss_scale=args.loss_scale,
                geo_weight=args.geo_weight,
                num_threads=args.num_threads,
                overwrite=args.overwrite_all or args.overwrite_photo_ba,
            )

            geo_metrics = align_and_eval_poses_intrinsics(
                geo_ba_dir, gt_sparse_dir, geo_ba_dir / "aligned_gt"
            )
            photo_joint_metrics = align_and_eval_poses_intrinsics(
                photo_ba_joint_dir,
                gt_sparse_dir,
                photo_ba_joint_dir / "aligned_gt",
            )

            scene_res: dict[str, Any] = {
                "triangulate_sec": tri_sec,
                "geo_ba_sec": geo_sec,
                "photo_ba_joint_sec": photo_joint_sec,
                "photo_ba_joint_stats": photo_joint_stats,
                "geo_ba": geo_metrics,
                "photo_ba_joint": photo_joint_metrics,
            }

            if has_eval_tool and scan_mlp.exists():
                pending_3d_evals.append(
                    (
                        feature,
                        scene,
                        "geo_ba_3d",
                        geo_ba_dir / "aligned_gt",
                        scan_mlp,
                        ws / "logs/eval_3d_geo_ba.log",
                        args.overwrite_all,
                    )
                )
                pending_3d_evals.append(
                    (
                        feature,
                        scene,
                        "photo_ba_joint_3d",
                        photo_ba_joint_dir / "aligned_gt",
                        scan_mlp,
                        ws / f"logs/eval_3d_photo_joint_{args.tag}.log",
                        args.overwrite_all or args.overwrite_photo_ba,
                    )
                )

            # 5. Track B: Fixed-GT-pose triangulation eval (PixSfM protocol).
            if not args.skip_track_b:
                photo_fixed_sec, photo_fixed_stats = run_photometric_ba(
                    args.colmap_path,
                    tri_dir,
                    image_dir,
                    photo_ba_fixed_dir,
                    ws / f"logs/photo_ba_fixed_{args.tag}.log",
                    refine_poses=False,
                    refine_intrinsics=False,
                    pyramid_levels=args.pyramid_levels,
                    patch_size=args.patch_size,
                    max_image_size=args.max_image_size,
                    warmup_iters=0,
                    max_num_iterations=args.max_num_iterations,
                    min_patch_std=args.min_patch_std,
                    loss_type=args.loss_type,
                    loss_scale=args.loss_scale,
                    geo_weight=args.geo_weight,
                    num_threads=args.num_threads,
                    overwrite=args.overwrite_all or args.overwrite_photo_ba,
                )
                scene_res["photo_ba_fixed_sec"] = photo_fixed_sec
                scene_res["photo_ba_fixed_stats"] = photo_fixed_stats
                if has_eval_tool and scan_mlp.exists():
                    pending_3d_evals.append(
                        (
                            feature,
                            scene,
                            "tri_gt_3d",
                            tri_dir,
                            scan_mlp,
                            ws / "logs/eval_3d_tri_gt.log",
                            args.overwrite_all,
                        )
                    )
                    pending_3d_evals.append(
                        (
                            feature,
                            scene,
                            "photo_ba_fixed_3d",
                            photo_ba_fixed_dir,
                            scan_mlp,
                            ws / f"logs/eval_3d_photo_fixed_{args.tag}.log",
                            args.overwrite_all or args.overwrite_photo_ba,
                        )
                    )

            results["results"][feature][scene] = scene_res
            print(
                f"  GeoBA -> PhotoBA | "
                f"pos_med: {geo_metrics['abs_pos_median_mm']:.2f} -> "
                f"{photo_joint_metrics['abs_pos_median_mm']:.2f} mm | "
                f"rot_med: {geo_metrics['abs_rot_median_deg']:.4f} -> "
                f"{photo_joint_metrics['abs_rot_median_deg']:.4f} deg | "
                f"focal_err: {geo_metrics['focal_rel_err_pct']:.3f}% -> "
                f"{photo_joint_metrics['focal_rel_err_pct']:.3f}% | "
                f"grid_px: {geo_metrics['intrinsics_grid_err_px']:.2f} -> "
                f"{photo_joint_metrics['intrinsics_grid_err_px']:.2f} px | "
                f"time: {photo_joint_sec:.1f}s",
                flush=True,
            )

    if pending_3d_evals:
        print(
            f"\nRunning {len(pending_3d_evals)} ETH3D 3D scan evaluations "
            f"(workers={args.eval_workers}, cached skipped)...",
            flush=True,
        )

        def _run_eval(item: tuple[str, str, str, Path, Path, Path, bool]):
            feat, scn, key, rdir, mlp, lpath, ow = item
            assert eval_tool_path is not None
            val = eval_eth3d_point_cloud(
                eval_tool_path,
                rdir,
                mlp,
                DEFAULT_TOLERANCES,
                lpath,
                overwrite=ow,
            )
            return feat, scn, key, val

        with ThreadPoolExecutor(max_workers=max(1, args.eval_workers)) as pool:
            for feat, scn, key, val in pool.map(_run_eval, pending_3d_evals):
                results["results"][feat][scn][key] = val

        for feature in args.features:
            for scene in args.scenes:
                sr = results["results"][feature][scene]
                parts = []
                if "photo_ba_joint_3d" in sr and "geo_ba_3d" in sr:
                    j0 = sr["geo_ba_3d"]["accuracy_pct"]
                    j1 = sr["photo_ba_joint_3d"]["accuracy_pct"]
                    parts.append(
                        f"Joint 3D@1/2/5cm: "
                        f"[{j0[0]:.1f},{j0[1]:.1f},{j0[2]:.1f}] -> "
                        f"[{j1[0]:.1f},{j1[1]:.1f},{j1[2]:.1f}]%"
                    )
                if "photo_ba_fixed_3d" in sr and "tri_gt_3d" in sr:
                    t3 = sr["tri_gt_3d"]["accuracy_pct"]
                    p3 = sr["photo_ba_fixed_3d"]["accuracy_pct"]
                    parts.append(
                        f"Fixed 3D@1/2/5cm: "
                        f"[{t3[0]:.1f},{t3[1]:.1f},{t3[2]:.1f}] -> "
                        f"[{p3[0]:.1f},{p3[1]:.1f},{p3[2]:.1f}]%"
                    )
                if parts:
                    print(
                        f"  [{feature}] {scene} | {' | '.join(parts)}",
                        flush=True,
                    )

    out_json = args.run_dir / f"summary_{args.tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved full results to {out_json}")


if __name__ == "__main__":
    main()
