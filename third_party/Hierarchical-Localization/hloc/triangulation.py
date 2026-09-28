import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import logger
from .colmap_backend import (
    create_database,
    export_subset_model,
    fill_database_from_model,
    geometric_verification,
    guided_geometric_verification,
    triangulate,
    write_keypoints,
    write_matches,
)


class OutputCapture:
    """Kept so older call sites still have a context manager.

    The colmap binary writes its own log. There is no pycolmap logging flag
    to toggle.
    """

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *args):
        return False


def create_db_from_model(model_dir: Path, database_path: Path, image_names=None):
    """Create a database whose image ids match the reference model."""
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    txt_dir = database_path.parent / "reference_txt"
    export_subset = image_names is not None
    if export_subset:
        subset = database_path.parent / "reference_subset"
        model_dir = export_subset_model(model_dir, subset, image_names)
        txt_dir = subset / "txt"
    else:
        from .colmap_backend import model_to_txt

        model_to_txt(model_dir, txt_dir)
    create_database(database_path)
    return fill_database_from_model(database_path, txt_dir, image_names), model_dir


def import_features(database_path: Path, features_path: Path, image_ids: Dict[str, int]):
    logger.info("Importing features into the database...")
    write_keypoints(database_path, image_ids, features_path)


def import_matches(
    database_path: Path,
    image_ids: Dict[str, int],
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float] = None,
    skip_geometric_verification: bool = False,
):
    logger.info("Importing matches into the database...")
    written = write_matches(
        database_path, image_ids, pairs_path, matches_path, min_match_score
    )
    logger.info("Wrote %d match pairs.", written)
    return skip_geometric_verification


def estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """Verify matches that were not checked against a known pose.

    ``pairs_path`` is unused: the binary walks every match already stored in
    the database. The argument stays so existing call sites keep working.
    """
    del pairs_path, verbose
    logger.info("Performing geometric verification of the matches...")
    geometric_verification(database_path)


def run_triangulation(
    model_path: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: Path,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> Path:
    del verbose, options
    logger.info("Running 3D triangulation with the local COLMAP binary...")
    return triangulate(database_path, image_dir, reference_model, model_path)


def main(
    sfm_dir: Path,
    reference_model: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    skip_geometric_verification: bool = False,
    estimate_two_view_geometries: bool = False,
    min_match_score: Optional[float] = None,
    verbose: bool = False,
    mapper_options: Optional[Dict[str, Any]] = None,
    image_names: Optional[List[str]] = None,
) -> Path:
    assert reference_model.exists(), reference_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches
    del mapper_options

    sfm_dir = Path(sfm_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    image_ids, reference = create_db_from_model(reference_model, database, image_names)
    import_features(database, features, image_ids)
    import_matches(
        database,
        image_ids,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )
    if not skip_geometric_verification:
        if estimate_two_view_geometries:
            estimation_and_geometric_verification(database, pairs, verbose)
        else:
            logger.info("Verifying matches with the reference poses...")
            guided_geometric_verification(database, reference)
    reconstruction = run_triangulation(
        sfm_dir / "model", database, image_dir, reference, verbose
    )
    logger.info("Finished the triangulation: %s", reconstruction)
    return reconstruction


def parse_option_args(args: List[str], default_options) -> Dict[str, Any]:
    options = {}
    for arg in args:
        idx = arg.find("=")
        if idx == -1:
            raise ValueError("Options format: key1=value1 key2=value2 etc.")
        key, value = arg[:idx], arg[idx + 1 :]
        if not hasattr(default_options, key):
            raise ValueError(
                f'Unknown option "{key}", allowed options and default values'
                f" for {default_options.summary()}"
            )
        value = eval(value)
        target_type = type(getattr(default_options, key))
        if not isinstance(value, target_type):
            raise ValueError(
                f'Incorrect type for option "{key}":' f" {type(value)} vs {target_type}"
            )
        options[key] = value
    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--reference_sfm_model", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--estimate_two_view_geometries", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(**args.__dict__)
