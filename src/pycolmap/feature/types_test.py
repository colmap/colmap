import math

import numpy as np

import pycolmap


def test_feature_extractor_type_enum():
    assert pycolmap.FeatureExtractorType.UNDEFINED is not None
    assert pycolmap.FeatureExtractorType.SIFT is not None


def test_feature_extractor_type_int_construction():
    sift = pycolmap.FeatureExtractorType.SIFT
    value = int(sift)
    assert isinstance(value, int)


def test_feature_descriptors_default_init():
    descriptors = pycolmap.FeatureDescriptors()
    assert descriptors is not None


def test_feature_descriptors_type_readwrite():
    descriptors = pycolmap.FeatureDescriptors()
    descriptors.type = pycolmap.FeatureExtractorType.SIFT
    assert descriptors.type == pycolmap.FeatureExtractorType.SIFT


def test_feature_descriptors_data_readwrite():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    descriptors = pycolmap.FeatureDescriptors(
        type=pycolmap.FeatureExtractorType.SIFT, data=data
    )
    np.testing.assert_array_equal(descriptors.data, data)
    new_data = np.array([[10, 20, 30]], dtype=np.uint8)
    descriptors.data = new_data
    np.testing.assert_array_equal(descriptors.data, new_data)


def test_feature_descriptors_float_default_init():
    descriptors = pycolmap.FeatureDescriptorsFloat()
    assert descriptors is not None


def test_feature_descriptors_float_type_readwrite():
    descriptors = pycolmap.FeatureDescriptorsFloat()
    descriptors.type = pycolmap.FeatureExtractorType.SIFT
    assert descriptors.type == pycolmap.FeatureExtractorType.SIFT


def test_feature_descriptors_float_data_readwrite():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    descriptors = pycolmap.FeatureDescriptorsFloat(
        type=pycolmap.FeatureExtractorType.SIFT, data=data
    )
    np.testing.assert_array_almost_equal(descriptors.data, data)
    new_data = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    descriptors.data = new_data
    np.testing.assert_array_almost_equal(descriptors.data, new_data)


def test_feature_keypoint_default_init():
    keypoint = pycolmap.FeatureKeypoint()
    assert keypoint is not None


def test_feature_keypoint_xy_readwrite():
    keypoint = pycolmap.FeatureKeypoint()
    keypoint.x = 10.5
    keypoint.y = 20.5
    assert keypoint.x == 10.5
    assert keypoint.y == 20.5


def test_feature_keypoint_affine_readwrite():
    keypoint = pycolmap.FeatureKeypoint()
    keypoint.a11 = 1.0
    keypoint.a12 = 0.5
    keypoint.a21 = -0.5
    keypoint.a22 = 1.0
    assert keypoint.a11 == 1.0
    assert keypoint.a12 == 0.5
    assert keypoint.a21 == -0.5
    assert keypoint.a22 == 1.0


def test_feature_keypoint_compute_scale():
    keypoint = pycolmap.FeatureKeypoint()
    scale = keypoint.compute_scale()
    assert isinstance(scale, float)


def test_feature_keypoint_compute_scale_x():
    keypoint = pycolmap.FeatureKeypoint()
    scale_x = keypoint.compute_scale_x()
    assert isinstance(scale_x, float)


def test_feature_keypoint_compute_scale_y():
    keypoint = pycolmap.FeatureKeypoint()
    scale_y = keypoint.compute_scale_y()
    assert isinstance(scale_y, float)


def test_feature_keypoint_compute_orientation():
    keypoint = pycolmap.FeatureKeypoint()
    orientation = keypoint.compute_orientation()
    assert isinstance(orientation, float)


def test_feature_keypoint_compute_shear():
    keypoint = pycolmap.FeatureKeypoint()
    shear = keypoint.compute_shear()
    assert isinstance(shear, float)


def test_feature_keypoint_rescale():
    keypoint = pycolmap.FeatureKeypoint()
    keypoint.x = 10.0
    keypoint.y = 20.0
    keypoint.a11 = 1.0
    keypoint.a12 = 0.0
    keypoint.a21 = 0.0
    keypoint.a22 = 1.0
    keypoint.rescale(2.0, 2.0)
    assert keypoint.x == 20.0
    assert keypoint.y == 40.0


def test_feature_keypoint_from_shape_parameters():
    keypoint = pycolmap.FeatureKeypoint.from_shape_parameters(
        1.0, 2.0, 2.0, 2.0, math.pi, 0.0
    )
    assert isinstance(keypoint, pycolmap.FeatureKeypoint)
    assert keypoint.x == 1.0
    assert keypoint.y == 2.0


def test_feature_match_default_init():
    match = pycolmap.FeatureMatch()
    assert match is not None


def test_feature_match_readwrite():
    match = pycolmap.FeatureMatch()
    match.point2D_idx1 = 5
    match.point2D_idx2 = 10
    assert match.point2D_idx1 == 5
    assert match.point2D_idx2 == 10


def test_feature_keypoints_vector():
    keypoints = pycolmap.FeatureKeypoints()
    keypoint = pycolmap.FeatureKeypoint()
    keypoint.x = 1.0
    keypoint.y = 2.0
    keypoints.append(keypoint)
    keypoints.append(pycolmap.FeatureKeypoint())
    assert len(keypoints) == 2
    count = 0
    for keypoint in keypoints:
        assert isinstance(keypoint, pycolmap.FeatureKeypoint)
        count += 1
    assert count == 2


def test_feature_matches_vector():
    matches = pycolmap.FeatureMatches()
    matches.append(pycolmap.FeatureMatch(0, 1))
    matches.append(pycolmap.FeatureMatch(2, 3))
    assert len(matches) == 2


def test_keypoints_to_from_matrix_roundtrip():
    keypoints = pycolmap.FeatureKeypoints()
    keypoint = pycolmap.FeatureKeypoint()
    keypoint.x = 10.0
    keypoint.y = 20.0
    keypoint.a11 = 1.0
    keypoint.a12 = 0.0
    keypoint.a21 = 0.0
    keypoint.a22 = 1.0
    keypoints.append(keypoint)
    matrix = pycolmap.keypoints_to_matrix(keypoints)
    assert matrix.shape[0] == 1
    assert matrix.shape[1] == 4
    recovered = pycolmap.keypoints_from_matrix(matrix)
    assert len(recovered) == 1
    assert recovered[0].x == keypoint.x
    assert recovered[0].y == keypoint.y


def test_matches_to_from_matrix_roundtrip():
    matches = pycolmap.FeatureMatches()
    matches.append(pycolmap.FeatureMatch(0, 5))
    matches.append(pycolmap.FeatureMatch(1, 3))
    matrix = pycolmap.matches_to_matrix(matches)
    assert matrix.shape == (2, 2)
    recovered = pycolmap.matches_from_matrix(matrix)
    assert len(recovered) == 2
    assert recovered[0].point2D_idx1 == 0
    assert recovered[0].point2D_idx2 == 5
    assert recovered[1].point2D_idx1 == 1
    assert recovered[1].point2D_idx2 == 3
