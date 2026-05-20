import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "PoissonMeshingOptions"),
    reason="PoissonMeshingOptions not available",
)


def test_poisson_meshing_options_init():
    options = pycolmap.PoissonMeshingOptions()
    assert options is not None


def test_poisson_meshing_options_depth():
    options = pycolmap.PoissonMeshingOptions()
    options.depth = 10
    assert options.depth == 10


def test_poisson_meshing_options_num_threads():
    options = pycolmap.PoissonMeshingOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_poisson_meshing_options_check():
    options = pycolmap.PoissonMeshingOptions()
    assert options.check()


def test_mesh_simplification_options_init():
    options = pycolmap.MeshSimplificationOptions()
    assert options is not None


def test_mesh_simplification_options_target_face_ratio():
    options = pycolmap.MeshSimplificationOptions()
    options.target_face_ratio = 0.5
    assert options.target_face_ratio == pytest.approx(0.5)


def test_mesh_simplification_options_check():
    options = pycolmap.MeshSimplificationOptions()
    assert options.check()


@pytest.mark.skipif(
    not hasattr(pycolmap, "DelaunayMeshingOptions"),
    reason="DelaunayMeshingOptions not available",
)
def test_delaunay_meshing_options_init():
    options = pycolmap.DelaunayMeshingOptions()
    assert options is not None
