import numpy as np

import pycolmap


def test_rigid3d_default_init():
    rigid = pycolmap.Rigid3d()
    assert rigid is not None


def test_rigid3d_init_rotation_translation():
    rotation = pycolmap.Rotation3d()
    translation = np.array([1.0, 2.0, 3.0])
    rigid = pycolmap.Rigid3d(rotation=rotation, translation=translation)
    assert rigid is not None


def test_rigid3d_init_from_matrix():
    matrix = np.eye(3, 4)
    rigid = pycolmap.Rigid3d(matrix=matrix)
    assert rigid is not None


def test_rigid3d_params_readwrite():
    rigid = pycolmap.Rigid3d()
    params = rigid.params
    assert params.shape == (7,)
    new_params = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0])
    rigid.params = new_params
    np.testing.assert_array_almost_equal(rigid.params, new_params)


def test_rigid3d_rotation_property():
    rigid = pycolmap.Rigid3d()
    rotation = rigid.rotation
    assert isinstance(rotation, pycolmap.Rotation3d)


def test_rigid3d_translation_readwrite():
    rigid = pycolmap.Rigid3d()
    translation = rigid.translation
    assert np.array(translation).shape == (3,)
    rigid.translation = np.array([4.0, 5.0, 6.0])
    np.testing.assert_array_almost_equal(rigid.translation, [4.0, 5.0, 6.0])


def test_rigid3d_matrix():
    rigid = pycolmap.Rigid3d()
    matrix = rigid.matrix()
    assert matrix.shape == (3, 4)


def test_rigid3d_tgt_origin_in_src():
    rigid = pycolmap.Rigid3d()
    rigid.translation = np.array([1.0, 2.0, 3.0])
    result = rigid.tgt_origin_in_src()
    assert result.shape == (3,)


def test_rigid3d_adjoint():
    rigid = pycolmap.Rigid3d()
    adjoint = rigid.adjoint()
    assert adjoint.shape == (6, 6)


def test_rigid3d_adjoint_inverse():
    rigid = pycolmap.Rigid3d()
    adjoint_inverse = rigid.adjoint_inverse()
    assert adjoint_inverse.shape == (6, 6)


def test_rigid3d_inverse():
    rigid = pycolmap.Rigid3d()
    rigid.translation = np.array([1.0, 2.0, 3.0])
    inverse = rigid.inverse()
    assert isinstance(inverse, pycolmap.Rigid3d)


def test_rigid3d_multiply_rigid3d():
    rigid1 = pycolmap.Rigid3d()
    rigid2 = pycolmap.Rigid3d()
    result = rigid1 * rigid2
    assert isinstance(result, pycolmap.Rigid3d)


def test_rigid3d_multiply_vector3d():
    rigid = pycolmap.Rigid3d()
    rigid.translation = np.array([1.0, 0.0, 0.0])
    vector = np.array([0.0, 0.0, 0.0])
    result = rigid * vector
    assert result.shape == (3,)


def test_rigid3d_multiply_nx3_matrix():
    rigid = pycolmap.Rigid3d()
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = rigid * points
    assert result.shape == (2, 3)


def test_rigid3d_interpolate():
    rigid1 = pycolmap.Rigid3d()
    rigid2 = pycolmap.Rigid3d()
    rigid2.translation = np.array([2.0, 0.0, 0.0])
    result = pycolmap.Rigid3d.interpolate(rigid1, rigid2, 0.5)
    assert isinstance(result, pycolmap.Rigid3d)


def test_get_covariance_for_inverse():
    rigid = pycolmap.Rigid3d()
    covariance = np.eye(6)
    result = pycolmap.get_covariance_for_inverse(rigid, covariance)
    assert result.shape == (6, 6)


def test_get_covariance_for_composed_rigid3d():
    rigid = pycolmap.Rigid3d()
    joint_covariance = np.eye(12)
    result = pycolmap.get_covariance_for_composed_rigid3d(
        rigid, joint_covariance
    )
    assert result.shape == (6, 6)


def test_get_covariance_for_relative_rigid3d():
    rigid1 = pycolmap.Rigid3d()
    rigid2 = pycolmap.Rigid3d()
    joint_covariance = np.eye(12)
    result = pycolmap.get_covariance_for_relative_rigid3d(
        rigid1, rigid2, joint_covariance
    )
    assert result.shape == (6, 6)


def test_average_quaternions():
    rotation1 = pycolmap.Rotation3d()
    rotation2 = pycolmap.Rotation3d()
    result = pycolmap.average_quaternions([rotation1, rotation2], [1.0, 1.0])
    assert result is not None


def test_interpolate_camera_poses():
    rigid1 = pycolmap.Rigid3d()
    rigid2 = pycolmap.Rigid3d()
    rigid2.translation = np.array([2.0, 0.0, 0.0])
    result = pycolmap.interpolate_camera_poses(rigid1, rigid2, 0.5)
    assert isinstance(result, pycolmap.Rigid3d)
