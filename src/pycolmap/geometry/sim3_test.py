import numpy as np

import pycolmap


def test_sim3d_default_init():
    similarity = pycolmap.Sim3d()
    assert similarity is not None


def test_sim3d_init_scale_rotation_translation():
    rotation = pycolmap.Rotation3d()
    translation = np.array([1.0, 2.0, 3.0])
    similarity = pycolmap.Sim3d(
        scale=2.0, rotation=rotation, translation=translation
    )
    assert similarity is not None


def test_sim3d_init_from_matrix():
    matrix = np.eye(3, 4)
    similarity = pycolmap.Sim3d(matrix=matrix)
    assert similarity is not None


def test_sim3d_params_readwrite():
    similarity = pycolmap.Sim3d()
    params = similarity.params
    assert params.shape == (8,)
    new_params = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0])
    similarity.params = new_params
    np.testing.assert_array_almost_equal(similarity.params, new_params)


def test_sim3d_scale_readwrite():
    similarity = pycolmap.Sim3d()
    scale = float(np.array(similarity.scale))
    assert isinstance(scale, float)
    similarity.scale = 3.0
    assert float(np.array(similarity.scale)) == 3.0


def test_sim3d_rotation_readwrite():
    similarity = pycolmap.Sim3d()
    rotation = similarity.rotation
    assert isinstance(rotation, pycolmap.Rotation3d)
    new_rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    similarity.rotation = new_rotation
    assert similarity.rotation is not None


def test_sim3d_translation_readwrite():
    similarity = pycolmap.Sim3d()
    translation = similarity.translation
    assert np.array(translation).shape == (3,)
    similarity.translation = np.array([4.0, 5.0, 6.0])
    np.testing.assert_array_almost_equal(
        similarity.translation, [4.0, 5.0, 6.0]
    )


def test_sim3d_matrix():
    similarity = pycolmap.Sim3d()
    matrix = similarity.matrix()
    assert matrix.shape == (3, 4)


def test_sim3d_inverse():
    similarity = pycolmap.Sim3d(
        scale=2.0,
        rotation=pycolmap.Rotation3d(),
        translation=np.array([1.0, 0.0, 0.0]),
    )
    inverse = similarity.inverse()
    assert isinstance(inverse, pycolmap.Sim3d)


def test_sim3d_multiply_sim3d():
    similarity1 = pycolmap.Sim3d()
    similarity2 = pycolmap.Sim3d()
    result = similarity1 * similarity2
    assert isinstance(result, pycolmap.Sim3d)


def test_sim3d_multiply_vector3d():
    similarity = pycolmap.Sim3d()
    vector = np.array([1.0, 2.0, 3.0])
    result = similarity * vector
    assert result.shape == (3,)


def test_sim3d_multiply_nx3_matrix():
    similarity = pycolmap.Sim3d()
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = similarity * points
    assert result.shape == (2, 3)


def test_sim3d_transform_camera_world():
    similarity = pycolmap.Sim3d()
    rigid = pycolmap.Rigid3d()
    result = similarity.transform_camera_world(rigid)
    assert isinstance(result, pycolmap.Rigid3d)
