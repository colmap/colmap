import pycolmap


def test_image_pair_stat_init():
    stat = pycolmap.ImagePairStat()
    assert stat is not None


def test_image_pair_stat_num_tri_corrs():
    stat = pycolmap.ImagePairStat()
    stat.num_tri_corrs = 10
    assert stat.num_tri_corrs == 10


def test_image_pair_stat_num_total_corrs():
    stat = pycolmap.ImagePairStat()
    stat.num_total_corrs = 20
    assert stat.num_total_corrs == 20


def test_reprojection_error_type_pixel():
    assert pycolmap.ReprojectionErrorType.PIXEL is not None


def test_reprojection_error_type_normalized():
    assert pycolmap.ReprojectionErrorType.NORMALIZED is not None


def test_reprojection_error_type_angular():
    assert pycolmap.ReprojectionErrorType.ANGULAR is not None


def _make_observation_manager(reconstruction):
    correspondence_graph = pycolmap.CorrespondenceGraph()
    for image_id in reconstruction.images:
        image = reconstruction.image(image_id)
        correspondence_graph.add_image(image_id, image.num_points2D())
    correspondence_graph.finalize()
    return pycolmap.ObservationManager(reconstruction, correspondence_graph)


def test_observation_manager_init(synthetic_reconstruction):
    manager = _make_observation_manager(synthetic_reconstruction)
    assert manager is not None


def test_observation_manager_image_pairs(synthetic_reconstruction):
    manager = _make_observation_manager(synthetic_reconstruction)
    pairs = manager.image_pairs
    assert hasattr(pairs, "__len__")


def test_observation_manager_num_observations(synthetic_reconstruction):
    manager = _make_observation_manager(synthetic_reconstruction)
    image_ids = list(synthetic_reconstruction.images.keys())
    count = manager.num_observations(image_ids[0])
    assert isinstance(count, int)


def test_observation_manager_num_correspondences(synthetic_reconstruction):
    manager = _make_observation_manager(synthetic_reconstruction)
    image_ids = list(synthetic_reconstruction.images.keys())
    count = manager.num_correspondences(image_ids[0])
    assert isinstance(count, int)


def test_find_small_triangulation_angle_points_is_non_mutating(
    synthetic_reconstruction,
):
    manager = _make_observation_manager(synthetic_reconstruction)
    point_ids = set(synthetic_reconstruction.points3D.keys())
    before = synthetic_reconstruction.num_points3D()
    selected = manager.find_points3D_with_small_triangulation_angle(
        180.0, point_ids
    )
    assert set(selected) == point_ids
    assert synthetic_reconstruction.num_points3D() == before
