import pytest

import pycolmap


def test_exhaustive_pairing_options_init() -> None:
    options = pycolmap.ExhaustivePairingOptions()
    assert options is not None


def test_exhaustive_pairing_options_block_size() -> None:
    options = pycolmap.ExhaustivePairingOptions()
    options.block_size = 100
    assert options.block_size == 100


def test_exhaustive_pairing_options_check() -> None:
    options = pycolmap.ExhaustivePairingOptions()
    assert options.check()


def test_spatial_pairing_options_init() -> None:
    options = pycolmap.SpatialPairingOptions()
    assert options is not None


def test_spatial_pairing_options_max_num_neighbors() -> None:
    options = pycolmap.SpatialPairingOptions()
    options.max_num_neighbors = 100
    assert options.max_num_neighbors == 100


def test_spatial_pairing_options_max_distance() -> None:
    options = pycolmap.SpatialPairingOptions()
    options.max_distance = 200.0
    assert options.max_distance == 200.0


def test_spatial_pairing_options_check() -> None:
    options = pycolmap.SpatialPairingOptions()
    assert options.check()


def test_retrieval_pairing_options_init() -> None:
    options = pycolmap.RetrievalPairingOptions()
    assert options is not None


def test_retrieval_pairing_options_method() -> None:
    options = pycolmap.RetrievalPairingOptions()
    assert options.method == pycolmap.RetrievalMethod.VOCAB_TREE
    options.method = pycolmap.RetrievalMethod.GLOBAL_DESCRIPTOR
    assert options.method == pycolmap.RetrievalMethod.GLOBAL_DESCRIPTOR


def test_retrieval_pairing_options_num_images() -> None:
    options = pycolmap.RetrievalPairingOptions()
    options.num_images = 50
    assert options.num_images == 50


def test_retrieval_pairing_options_num_nearest_neighbors() -> None:
    options = pycolmap.RetrievalPairingOptions()
    options.num_nearest_neighbors = 3
    assert options.num_nearest_neighbors == 3


def test_retrieval_pairing_options_check() -> None:
    options = pycolmap.RetrievalPairingOptions()
    assert options.check()


def test_sequential_pairing_options_init() -> None:
    options = pycolmap.SequentialPairingOptions()
    assert options is not None


def test_sequential_pairing_options_overlap() -> None:
    options = pycolmap.SequentialPairingOptions()
    options.overlap = 15
    assert options.overlap == 15


def test_sequential_pairing_options_quadratic_overlap() -> None:
    options = pycolmap.SequentialPairingOptions()
    options.quadratic_overlap = True
    assert options.quadratic_overlap is True


def test_sequential_pairing_options_loop_detection_options() -> None:
    options = pycolmap.SequentialPairingOptions()
    assert options.loop_detection_options.num_images == 50
    options.loop_detection_options.num_images = 100
    assert options.loop_detection_options.num_images == 100


def test_sequential_pairing_options_check() -> None:
    options = pycolmap.SequentialPairingOptions()
    assert options.check()


def test_imported_pairing_options_init() -> None:
    options = pycolmap.ImportedPairingOptions()
    assert options is not None


def test_imported_pairing_options_block_size() -> None:
    options = pycolmap.ImportedPairingOptions()
    options.block_size = 200
    assert options.block_size == 200


def test_imported_pairing_options_check() -> None:
    options = pycolmap.ImportedPairingOptions()
    assert options.check()


def test_existing_matched_pairing_options_init() -> None:
    options = pycolmap.ExistingMatchedPairingOptions()
    assert options is not None


def test_existing_matched_pairing_options_batch_size() -> None:
    options = pycolmap.ExistingMatchedPairingOptions()
    options.batch_size = 500
    assert options.batch_size == 500


def test_geometric_verifier_options_init() -> None:
    options = pycolmap.GeometricVerifierOptions()
    assert options is not None


def test_geometric_verifier_options_num_threads() -> None:
    options = pycolmap.GeometricVerifierOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_pair_generator_class_exists() -> None:
    assert hasattr(pycolmap, "PairGenerator")


def test_exhaustive_pair_generator_class_exists() -> None:
    assert hasattr(pycolmap, "ExhaustivePairGenerator")


@pytest.mark.parametrize(
    "name",
    [
        "match_exhaustive",
        "match_spatial",
        "verify_matches",
        "geometric_verification",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))
