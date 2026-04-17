import pycolmap


def test_exhaustive_pairing_options_init():
    options = pycolmap.ExhaustivePairingOptions()
    assert options is not None


def test_exhaustive_pairing_options_block_size():
    options = pycolmap.ExhaustivePairingOptions()
    options.block_size = 100
    assert options.block_size == 100


def test_exhaustive_pairing_options_check():
    options = pycolmap.ExhaustivePairingOptions()
    assert options.check()


def test_spatial_pairing_options_init():
    options = pycolmap.SpatialPairingOptions()
    assert options is not None


def test_spatial_pairing_options_max_num_neighbors():
    options = pycolmap.SpatialPairingOptions()
    options.max_num_neighbors = 100
    assert options.max_num_neighbors == 100


def test_spatial_pairing_options_max_distance():
    options = pycolmap.SpatialPairingOptions()
    options.max_distance = 200.0
    assert options.max_distance == 200.0


def test_spatial_pairing_options_check():
    options = pycolmap.SpatialPairingOptions()
    assert options.check()


def test_vocab_tree_pairing_options_init():
    options = pycolmap.VocabTreePairingOptions()
    assert options is not None


def test_vocab_tree_pairing_options_num_images():
    options = pycolmap.VocabTreePairingOptions()
    options.num_images = 50
    assert options.num_images == 50


def test_vocab_tree_pairing_options_num_nearest_neighbors():
    options = pycolmap.VocabTreePairingOptions()
    options.num_nearest_neighbors = 3
    assert options.num_nearest_neighbors == 3


def test_vocab_tree_pairing_options_check():
    options = pycolmap.VocabTreePairingOptions()
    assert options.check()


def test_sequential_pairing_options_init():
    options = pycolmap.SequentialPairingOptions()
    assert options is not None


def test_sequential_pairing_options_overlap():
    options = pycolmap.SequentialPairingOptions()
    options.overlap = 15
    assert options.overlap == 15


def test_sequential_pairing_options_quadratic_overlap():
    options = pycolmap.SequentialPairingOptions()
    options.quadratic_overlap = True
    assert options.quadratic_overlap is True


def test_sequential_pairing_options_vocab_tree_options():
    options = pycolmap.SequentialPairingOptions()
    vocab_tree_options = options.vocab_tree_options()
    assert vocab_tree_options is not None


def test_sequential_pairing_options_check():
    options = pycolmap.SequentialPairingOptions()
    assert options.check()


def test_imported_pairing_options_init():
    options = pycolmap.ImportedPairingOptions()
    assert options is not None


def test_imported_pairing_options_block_size():
    options = pycolmap.ImportedPairingOptions()
    options.block_size = 200
    assert options.block_size == 200


def test_imported_pairing_options_check():
    options = pycolmap.ImportedPairingOptions()
    assert options.check()


def test_existing_matched_pairing_options_init():
    options = pycolmap.ExistingMatchedPairingOptions()
    assert options is not None


def test_existing_matched_pairing_options_batch_size():
    options = pycolmap.ExistingMatchedPairingOptions()
    options.batch_size = 500
    assert options.batch_size == 500


def test_geometric_verifier_options_init():
    options = pycolmap.GeometricVerifierOptions()
    assert options is not None


def test_geometric_verifier_options_num_threads():
    options = pycolmap.GeometricVerifierOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_pair_generator_class_exists():
    assert hasattr(pycolmap, "PairGenerator")


def test_exhaustive_pair_generator_class_exists():
    assert hasattr(pycolmap, "ExhaustivePairGenerator")


def test_match_exhaustive_callable():
    assert callable(pycolmap.match_exhaustive)


def test_match_spatial_callable():
    assert callable(pycolmap.match_spatial)


def test_verify_matches_callable():
    assert callable(pycolmap.verify_matches)


def test_geometric_verification_callable():
    assert callable(pycolmap.geometric_verification)
