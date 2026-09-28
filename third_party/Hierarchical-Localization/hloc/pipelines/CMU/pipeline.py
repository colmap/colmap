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

TEST_SLICES = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def generate_query_list(dataset, path, slice_):
    cameras = {}
    with open(dataset / "intrinsics.txt", "r") as f:
        for line in f.readlines():
            if line[0] == "#" or line == "\n":
                continue
            data = line.split()
            cameras[data[0]] = data[1:]
    assert len(cameras) == 2

    queries = dataset / f"{slice_}/test-images-{slice_}.txt"
    with open(queries, "r") as f:
        queries = [q.rstrip("\n") for q in f.readlines()]

    out = [[q] + cameras[q.split("_")[2]] for q in queries]
    with open(path, "w") as f:
        f.write("\n".join(map(" ".join, out)))


def run_slice(slice_, root, outputs, num_covis, num_loc):
    dataset = root / slice_
    ref_images = dataset / "database"
    query_images = dataset / "query"
    sift_sfm = dataset / "sparse"

    outputs = outputs / slice_
    outputs.mkdir(exist_ok=True, parents=True)
    query_list = dataset / "queries_with_intrinsics.txt"
    sfm_pairs = outputs / f"pairs-db-covis{num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{num_loc}.txt"
    ref_sfm = outputs / "sfm_superpoint+superglue"
    results = outputs / f"CMU_hloc_superpoint+superglue_netvlad{num_loc}.txt"

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=num_covis)
    features = extract_features.main(feature_conf, ref_images, outputs, as_half=True)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    triangulation.main(ref_sfm, sift_sfm, ref_images, sfm_pairs, features, sfm_matches)

    generate_query_list(root, query_list, slice_)
    global_descriptors = extract_features.main(retrieval_conf, ref_images, outputs)
    global_descriptors = extract_features.main(retrieval_conf, query_images, outputs)
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, num_loc, query_list=query_list, db_model=ref_sfm
    )

    features = extract_features.main(feature_conf, query_images, outputs, as_half=True)
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    localize_sfm.main(
        ref_sfm,
        dataset / "queries/*_time_queries_with_intrinsics.txt",
        loc_pairs,
        features,
        loc_matches,
        results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slices",
        type=str,
        default="*",
        help="a single number, an interval (e.g. 2-6), "
        "or a Python-style list or int (e.g. [2, 3, 4]",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/cmu_extended",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/aachen_extended",
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

    if args.slice == "*":
        slices = TEST_SLICES
    if "-" in args.slices:
        min_, max_ = args.slices.split("-")
        slices = list(range(int(min_), int(max_) + 1))
    else:
        slices = eval(args.slices)
        if isinstance(slices, int):
            slices = [slices]

    for slice_ in slices:
        logger.info("Working on slice %s.", slice_)
        run_slice(
            f"slice{slice_}", args.dataset, args.outputs, args.num_covis, args.num_loc
        )
