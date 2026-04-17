import pycolmap


def test_triangulation_residual_type_angular_error():
    assert pycolmap.TriangulationResidualType.ANGULAR_ERROR is not None


def test_triangulation_residual_type_reprojection_error():
    assert pycolmap.TriangulationResidualType.REPROJECTION_ERROR is not None


def test_estimate_triangulation_options_default_init():
    options = pycolmap.EstimateTriangulationOptions()
    assert options is not None


def test_estimate_triangulation_options_min_tri_angle_readwrite():
    options = pycolmap.EstimateTriangulationOptions()
    assert isinstance(options.min_tri_angle, float)
    options.min_tri_angle = 0.1
    assert options.min_tri_angle == 0.1


def test_estimate_triangulation_options_residual_type_readwrite():
    options = pycolmap.EstimateTriangulationOptions()
    options.residual_type = (
        pycolmap.TriangulationResidualType.REPROJECTION_ERROR
    )
    assert (
        options.residual_type
        == pycolmap.TriangulationResidualType.REPROJECTION_ERROR
    )


def test_estimate_triangulation_options_ransac_property():
    options = pycolmap.EstimateTriangulationOptions()
    ransac = options.ransac
    assert isinstance(ransac, pycolmap.RANSACOptions)


def test_estimate_triangulation_is_callable():
    assert callable(pycolmap.estimate_triangulation)
