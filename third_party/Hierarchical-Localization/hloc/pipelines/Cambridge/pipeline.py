import argparse
from pathlib import Path

from ... import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)
from .utils import create_query_list_with_intrinsics, evaluate, scale_sfm_images

SCENES = ["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch", "GreatCourt"]


def run_scene(images, gt_dir, outputs, results, num_covis, num_loc):
    ref_sfm_sift = gt_dir / "model_train"
    test_list = gt_dir / "list_query.txt"

    outputs.mkdir(exist_ok=True, parents=True)
    ref_sfm = outputs / "sfm_superpoint+superglue"
    ref_sfm_scaled = outputs / "sfm_sift_scaled"
    query_list = outputs / "query_list_with_intrinsics.txt"
    sfm_pairs = outputs / f"pairs-db-covis{num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{num_loc}.txt"

    feature_conf = {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    matcher_conf = match_features.confs["superglue"]
    retrieval_conf = extract_features.confs["netvlad"]

    create_query_list_with_intrinsics(
        gt_dir / "empty_all", query_list, test_list, ext=".txt", image_dir=images
    )
    with open(test_list, "r") as f:
        query_seqs = {q.split("/")[0] for q in f.read().rstrip().split("\n")}

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        num_loc,
        db_model=ref_sfm_sift,
        query_prefix=query_seqs,
    )

    features = extract_features.main(feature_conf, images, outputs, as_half=True)
    pairs_from_covisibility.main(ref_sfm_sift, sfm_pairs, num_matched=num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    scale_sfm_images(ref_sfm_sift, ref_sfm_scaled, images)
    triangulation.main(
        ref_sfm, ref_sfm_scaled, images, sfm_pairs, features, sfm_matches
    )

    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    localize_sfm.main(
        ref_sfm,
        query_list,
        loc_pairs,
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
        default="datasets/cambridge",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/cambridge",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=10,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()

    gt_dirs = args.dataset / "CambridgeLandmarks_Colmap_Retriangulated_1024px"

    all_results = {}
    for scene in args.scenes:
        logger.info(f'Working on scene "{scene}".')
        results = args.outputs / scene / "results.txt"
        if args.overwrite or not results.exists():
            run_scene(
                args.dataset / scene,
                gt_dirs / scene,
                args.outputs / scene,
                results,
                args.num_covis,
                args.num_loc,
            )
        all_results[scene] = results

    for scene in args.scenes:
        logger.info(f'Evaluate scene "{scene}".')
        evaluate(
            gt_dirs / scene / "empty_all",
            all_results[scene],
            gt_dirs / scene / "list_query.txt",
            ext=".txt",
        )
