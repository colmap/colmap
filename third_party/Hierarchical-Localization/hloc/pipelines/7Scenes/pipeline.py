import argparse
from pathlib import Path

from ... import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    triangulation,
)
from ..Cambridge.utils import create_query_list_with_intrinsics, evaluate
from .create_gt_sfm import correct_sfm_with_gt_depth
from .utils import create_reference_sfm

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]


def run_scene(
    images,
    gt_dir,
    retrieval,
    outputs,
    results,
    num_covis,
    use_dense_depth,
    depth_dir=None,
):
    outputs.mkdir(exist_ok=True, parents=True)
    ref_sfm_sift = outputs / "sfm_sift"
    ref_sfm = outputs / "sfm_superpoint+superglue"
    query_list = outputs / "query_list_with_intrinsics.txt"

    feature_conf = {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "globs": ["*.color.png"],
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    matcher_conf = match_features.confs["superglue"]
    matcher_conf["model"]["sinkhorn_iterations"] = 5

    test_list = gt_dir / "list_test.txt"
    create_reference_sfm(gt_dir, ref_sfm_sift, test_list)
    create_query_list_with_intrinsics(gt_dir, query_list, test_list)

    features = extract_features.main(feature_conf, images, outputs, as_half=True)

    sfm_pairs = outputs / f"pairs-db-covis{num_covis}.txt"
    pairs_from_covisibility.main(ref_sfm_sift, sfm_pairs, num_matched=num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    if not (use_dense_depth and ref_sfm.exists()):
        triangulation.main(
            ref_sfm, ref_sfm_sift, images, sfm_pairs, features, sfm_matches
        )
    if use_dense_depth:
        assert depth_dir is not None
        ref_sfm_fix = outputs / "sfm_superpoint+superglue+depth"
        correct_sfm_with_gt_depth(ref_sfm, depth_dir, ref_sfm_fix)
        ref_sfm = ref_sfm_fix

    loc_matches = match_features.main(
        matcher_conf, retrieval, feature_conf["output"], outputs
    )

    localize_sfm.main(
        ref_sfm,
        query_list,
        retrieval,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default=SCENES, choices=SCENES, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/7scenes",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/7scenes",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument("--use_dense_depth", action="store_true")
    parser.add_argument(
        "--num_covis",
        type=int,
        default=30,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    args = parser.parse_args()

    gt_dirs = args.dataset / "7scenes_sfm_triangulated/{scene}/triangulated"
    retrieval_dirs = args.dataset / "7scenes_densevlad_retrieval_top_10"

    all_results = {}
    for scene in args.scenes:
        logger.info(f'Working on scene "{scene}".')
        results = (
            args.outputs
            / scene
            / "results_{}.txt".format("dense" if args.use_dense_depth else "sparse")
        )
        if args.overwrite or not results.exists():
            run_scene(
                args.dataset / scene,
                Path(str(gt_dirs).format(scene=scene)),
                retrieval_dirs / f"{scene}_top10.txt",
                args.outputs / scene,
                results,
                args.num_covis,
                args.use_dense_depth,
                depth_dir=args.dataset / f"depth/7scenes_{scene}/train/depth",
            )
        all_results[scene] = results

    for scene in args.scenes:
        logger.info(f'Evaluate scene "{scene}".')
        gt_dir = Path(str(gt_dirs).format(scene=scene))
        evaluate(gt_dir, all_results[scene], gt_dir / "list_test.txt")
