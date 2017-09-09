set(FOLDER_NAME "estimators")

COLMAP_ADD_LIBRARY(estimators
    absolute_pose.h absolute_pose.cc
    affine_transform.h affine_transform.cc
    coordinate_frame.h coordinate_frame.cc
    essential_matrix.h essential_matrix.cc
    euclidean_transform.h
    fundamental_matrix.h fundamental_matrix.cc
    generalized_absolute_pose.h generalized_absolute_pose.cc
    generalized_absolute_pose_coeffs.h generalized_absolute_pose_coeffs.cc
    generalized_relative_pose.h generalized_relative_pose.cc
    homography_matrix.h homography_matrix.cc
    pose.h pose.cc
    similarity_transform.h
    translation_transform.h
    triangulation.h triangulation.cc
    two_view_geometry.h two_view_geometry.cc
    utils.h utils.cc
)

COLMAP_ADD_TEST(absolute_pose_test absolute_pose_test.cc)
COLMAP_ADD_TEST(affine_transform_test affine_transform_test.cc)
COLMAP_ADD_TEST(coordinate_frame_test coordinate_frame_test.cc)
COLMAP_ADD_TEST(essential_matrix_test essential_matrix_test.cc)
COLMAP_ADD_TEST(estimators_utils_test utils_test.cc)
COLMAP_ADD_TEST(fundamental_matrix_test fundamental_matrix_test.cc)
COLMAP_ADD_TEST(generalized_absolute_pose_test generalized_absolute_pose_test.cc)
COLMAP_ADD_TEST(generalized_relative_pose_test generalized_relative_pose_test.cc)
COLMAP_ADD_TEST(homography_matrix_test homography_matrix_test.cc)
COLMAP_ADD_TEST(translation_transform_test translation_transform_test.cc)
