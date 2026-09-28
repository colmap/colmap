import argparse
from pathlib import Path

from ... import extract_features, localize_sfm, logger, match_features
from .utils import (
    delete_unused_images,
    evaluate_submission,
    generate_localization_pairs,
    generate_query_lists,
    get_timestamps,
    prepare_submission,
)

relocalization_files = {
    "training": "RelocalizationFilesTrain//relocalizationFile_recording_2020-03-24_17-36-22.txt",  # noqa: E501
    "validation": "RelocalizationFilesVal/relocalizationFile_recording_2020-03-03_12-03-23.txt",  # noqa: E501
    "test0": "RelocalizationFilesTest/relocalizationFile_recording_2020-03-24_17-45-31_*.txt",  # noqa: E501
    "test1": "RelocalizationFilesTest/relocalizationFile_recording_2020-04-23_19-37-00_*.txt",  # noqa: E501
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sequence",
    type=str,
    required=True,
    choices=["training", "validation", "test0", "test1"],
    help="Sequence to be relocalized.",
)
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
sequence = args.sequence

data_dir = args.dataset
ref_dir = data_dir / "reference"
assert ref_dir.exists(), f"{ref_dir} does not exist"
seq_dir = data_dir / sequence
assert seq_dir.exists(), f"{seq_dir} does not exist"
seq_images = seq_dir / "undistorted_images"
reloc = ref_dir / relocalization_files[sequence]

output_dir = args.outputs
output_dir.mkdir(exist_ok=True, parents=True)
query_list = output_dir / f"{sequence}_queries_with_intrinsics.txt"
ref_pairs = output_dir / "pairs-db-dist20.txt"
ref_sfm = output_dir / "sfm_superpoint+superglue"
results_path = output_dir / f"localization_{sequence}_hloc+superglue.txt"
submission_dir = output_dir / "submission_hloc+superglue"

num_loc_pairs = 10
loc_pairs = output_dir / f"pairs-query-{sequence}-dist{num_loc_pairs}.txt"

fconf = extract_features.confs["superpoint_max"]
mconf = match_features.confs["superglue"]

# Not all query images that are used for the evaluation
# To save time in feature extraction, we delete unsused images.
timestamps = get_timestamps(reloc, 1)
delete_unused_images(seq_images, timestamps)

# Generate a list of query images with their intrinsics.
generate_query_lists(timestamps, seq_dir, query_list)

# Generate the localization pairs from the given reference frames.
generate_localization_pairs(sequence, reloc, num_loc_pairs, ref_pairs, loc_pairs)

# Extract, match, amd localize.
ffile = extract_features.main(fconf, seq_images, output_dir)
mfile = match_features.main(mconf, loc_pairs, fconf["output"], output_dir)
localize_sfm.main(ref_sfm, query_list, loc_pairs, ffile, mfile, results_path)

# Convert the absolute poses to relative poses with the reference frames.
submission_dir.mkdir(exist_ok=True)
prepare_submission(results_path, reloc, ref_dir / "poses.txt", submission_dir)

# If not a test sequence: evaluation the localization accuracy
if "test" not in sequence:
    logger.info("Evaluating the relocalization submission...")
    evaluate_submission(submission_dir, reloc)
