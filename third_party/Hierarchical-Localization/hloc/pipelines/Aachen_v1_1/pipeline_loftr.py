import argparse
from pathlib import Path
from pprint import pformat

from ... import (
    extract_features,
    localize_sfm,
    logger,
    match_dense,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)


def run(args):
    # Setup the paths
    dataset = args.dataset
    images = dataset / "images_upright/"
    sift_sfm = dataset / "3D-models/aachen_v_1_1"

    outputs = args.outputs  # where everything will be saved
    outputs.mkdir()
    reference_sfm = outputs / "sfm_loftr"  # the SfM model we will build
    sfm_pairs = (
        outputs / f"pairs-db-covis{args.num_covis}.txt"
    )  # top-k most covisible in SIFT model
    loc_pairs = (
        outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    )  # top-k retrieved by NetVLAD
    results = outputs / f"Aachen-v1.1_hloc_loftr_netvlad{args.num_loc}.txt"

    # list the standard configurations available
    logger.info("Configs for dense feature matchers:\n%s", pformat(match_dense.confs))

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    matcher_conf = match_dense.confs["loftr_aachen"]

    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    features, sfm_matches = match_dense.main(
        matcher_conf, sfm_pairs, images, outputs, max_kps=8192, overwrite=False
    )

    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )
    features, loc_matches = match_dense.main(
        matcher_conf,
        loc_pairs,
        images,
        outputs,
        features=features,
        max_kps=None,
        matches=sfm_matches,
    )

    localize_sfm.main(
        reference_sfm,
        dataset / "queries/*_time_queries_with_intrinsics.txt",
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
    )  # not required with loftr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/aachen_v1.1",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/aachen_v1.1",
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
        default=50,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()
