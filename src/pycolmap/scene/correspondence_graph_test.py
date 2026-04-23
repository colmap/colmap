import numpy as np
import pytest

import pycolmap


def test_correspondence_default_init():
    correspondence = pycolmap.Correspondence()
    assert correspondence is not None


def test_correspondence_init_with_args():
    correspondence = pycolmap.Correspondence(image_id=1, point2D_idx=5)
    assert correspondence.image_id == 1
    assert correspondence.point2D_idx == 5


def test_correspondence_image_id_readwrite():
    correspondence = pycolmap.Correspondence()
    correspondence.image_id = 10
    assert correspondence.image_id == 10


def test_correspondence_point2d_idx_readwrite():
    correspondence = pycolmap.Correspondence()
    correspondence.point2D_idx = 20
    assert correspondence.point2D_idx == 20


def test_correspondence_graph_default_init():
    graph = pycolmap.CorrespondenceGraph()
    assert graph.num_images() == 0


def test_correspondence_graph_add_image():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 10)
    assert graph.exists_image(1)


def test_correspondence_graph_exists_image():
    graph = pycolmap.CorrespondenceGraph()
    assert not graph.exists_image(99)
    graph.add_image(1, 10)
    assert graph.exists_image(1)


def test_correspondence_graph_num_images():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 10)
    graph.add_image(2, 10)
    assert graph.num_images() == 2


def test_correspondence_graph_finalize():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 5)
    graph.finalize()
    assert graph.num_images() == 1


def test_correspondence_graph_num_image_pairs():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 5)
    graph.add_image(2, 5)
    graph.finalize()
    assert graph.num_image_pairs() >= 0


def test_correspondence_graph_num_observations_for_image():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 5)
    graph.finalize()
    assert graph.num_observations_for_image(1) >= 0


def test_correspondence_graph_num_correspondences_for_image():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 5)
    graph.finalize()
    assert graph.num_correspondences_for_image(1) >= 0


def test_correspondence_graph_image_pairs():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 5)
    graph.add_image(2, 5)
    graph.finalize()
    pairs = graph.image_pairs()
    assert len(pairs) >= 0


@pytest.fixture
def graph_with_matches():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 5)
    graph.add_image(2, 5)
    two_view = pycolmap.TwoViewGeometry()
    two_view.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    two_view.inlier_matches = np.array([[0, 0], [1, 1]], dtype=np.uint32)
    graph.add_two_view_geometry(1, 2, two_view)
    graph.finalize()
    return graph


def test_correspondence_graph_with_two_view_geometry(graph_with_matches):
    assert graph_with_matches.num_matches_between_images(1, 2) == 2


def test_correspondence_graph_extract_matches(graph_with_matches):
    matches = graph_with_matches.extract_matches_between_images(1, 2)
    assert matches.shape[0] == 2


def test_correspondence_graph_has_correspondences(graph_with_matches):
    assert graph_with_matches.has_correspondences(1, 0)


def test_correspondence_graph_find_correspondences(graph_with_matches):
    result = graph_with_matches.find_correspondences(1, 0)
    assert result is not None


def test_correspondence_graph_is_two_view_observation():
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 5)
    graph.add_image(2, 5)
    two_view = pycolmap.TwoViewGeometry()
    two_view.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    two_view.inlier_matches = np.array([[0, 0]], dtype=np.uint32)
    graph.add_two_view_geometry(1, 2, two_view)
    graph.finalize()
    assert graph.is_two_view_observation(1, 0) is True
