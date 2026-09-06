import numpy as np

import pycolmap


def test_two_view_geometry_configuration_undefined() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.UNDEFINED is not None


def test_two_view_geometry_configuration_degenerate() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.DEGENERATE is not None


def test_two_view_geometry_configuration_calibrated() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.CALIBRATED is not None


def test_two_view_geometry_configuration_uncalibrated() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED is not None


def test_two_view_geometry_configuration_planar() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.PLANAR is not None


def test_two_view_geometry_configuration_panoramic() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.PANORAMIC is not None


def test_two_view_geometry_configuration_planar_or_panoramic() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.PLANAR_OR_PANORAMIC is not None


def test_two_view_geometry_configuration_watermark() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.WATERMARK is not None


def test_two_view_geometry_configuration_multiple() -> None:
    assert pycolmap.TwoViewGeometryConfiguration.MULTIPLE is not None


def test_two_view_geometry_default_init() -> None:
    geometry = pycolmap.TwoViewGeometry()
    assert geometry is not None


def test_two_view_geometry_config_readwrite() -> None:
    geometry = pycolmap.TwoViewGeometry()
    geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    assert geometry.config == pycolmap.TwoViewGeometryConfiguration.CALIBRATED


def test_two_view_geometry_e_readwrite() -> None:
    geometry = pycolmap.TwoViewGeometry()
    essential = np.eye(3)
    geometry.E = essential
    np.testing.assert_array_almost_equal(geometry.E, essential)


def test_two_view_geometry_f_readwrite() -> None:
    geometry = pycolmap.TwoViewGeometry()
    fundamental = np.eye(3)
    geometry.F = fundamental
    np.testing.assert_array_almost_equal(geometry.F, fundamental)


def test_two_view_geometry_h_readwrite() -> None:
    geometry = pycolmap.TwoViewGeometry()
    homography = np.eye(3)
    geometry.H = homography
    np.testing.assert_array_almost_equal(geometry.H, homography)


def test_two_view_geometry_cam2_from_cam1_readwrite() -> None:
    geometry = pycolmap.TwoViewGeometry()
    rigid = pycolmap.Rigid3d()
    geometry.cam2_from_cam1 = rigid
    assert isinstance(geometry.cam2_from_cam1, pycolmap.Rigid3d)


def test_two_view_geometry_inlier_matches_readwrite() -> None:
    geometry = pycolmap.TwoViewGeometry()
    matches = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    geometry.inlier_matches = matches
    result = geometry.inlier_matches
    assert result.shape[0] == 2


def test_two_view_geometry_tri_angle_readwrite() -> None:
    geometry = pycolmap.TwoViewGeometry()
    geometry.tri_angle = 1.5
    assert geometry.tri_angle == 1.5


def test_two_view_geometry_invert() -> None:
    geometry = pycolmap.TwoViewGeometry()
    geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    geometry.invert()
    assert geometry is not None
