import argparse
from pathlib import Path

from ... import extract_features, match_features, pairs_from_poses, triangulation
from .utils import build_empty_colmap_model, delete_unused_images, get_timestamps

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=Path,
    default="datasets/4Seasons",
    help="Path to the dataset, default: %(default)s",
)
parser.add_argument(
    "--outputs",
    type=Path,
    default="outputs/4Seasons",
    help="Path to the output directory, default: %(default)s",
)
args = parser.parse_args()

ref_dir = args.dataset / "reference"
assert ref_dir.exists(), f"{ref_dir} does not exist"
ref_images = ref_dir / "undistorted_images"

output_dir = args.outputs
output_dir.mkdir(exist_ok=True, parents=True)
ref_sfm_empty = output_dir / "sfm_reference_empty"
ref_sfm = output_dir / "sfm_superpoint+superglue"

num_ref_pairs = 20
ref_pairs = output_dir / f"pairs-db-dist{num_ref_pairs}.txt"

fconf = extract_features.confs["superpoint_max"]
mconf = match_features.confs["superglue"]

# Only reference images that have a pose are used in the pipeline.
# To save time in feature extraction, we delete unsused images.
delete_unused_images(ref_images, get_timestamps(ref_dir / "poses.txt", 0))

# Build an empty COLMAP model containing only camera and images
# from the provided poses and intrinsics.
build_empty_colmap_model(ref_dir, ref_sfm_empty)

# Match reference images that are spatially close.
pairs_from_poses.main(ref_sfm_empty, ref_pairs, num_ref_pairs)

# Extract, match, and triangulate the reference SfM model.
ffile = extract_features.main(fconf, ref_images, output_dir)
mfile = match_features.main(mconf, ref_pairs, fconf["output"], output_dir)
triangulation.main(ref_sfm, ref_sfm_empty, ref_images, ref_pairs, ffile, mfile)
