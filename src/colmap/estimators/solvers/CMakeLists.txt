# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "estimators_solvers")

COLMAP_ADD_LIBRARY(
    NAME colmap_estimators_solvers
    SRCS
        absolute_pose.h absolute_pose.cc
        affine_transform.h affine_transform.cc
        essential_matrix.h essential_matrix.cc
        essential_matrix_coeffs.h
        essential_matrix_poly.h
        euclidean_transform.h
        fundamental_matrix.h fundamental_matrix.cc
        generalized_absolute_pose.h generalized_absolute_pose.cc
        generalized_relative_pose.h generalized_relative_pose.cc
        homography_matrix.h homography_matrix.cc
        poselib_utils.h poselib_utils.cc
        relpose_one_sided_focal.h relpose_one_sided_focal.cc
        relpose_shared_focal.h relpose_shared_focal.cc
        similarity_transform.h similarity_transform.cc
        translation_transform.h
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_math
        colmap_geometry
        colmap_scene
        colmap_optim
        colmap_estimators_cost_functions
        Eigen3::Eigen
        Ceres::ceres
        PoseLib::PoseLib
)

COLMAP_ADD_TEST(
    NAME absolute_pose_test
    SRCS absolute_pose_test.cc
    LINK_LIBS colmap_estimators_solvers colmap_estimators
)
COLMAP_ADD_TEST(
    NAME affine_transform_test
    SRCS affine_transform_test.cc
    LINK_LIBS colmap_estimators_solvers
)
COLMAP_ADD_TEST(
    NAME essential_matrix_test
    SRCS essential_matrix_test.cc
    LINK_LIBS colmap_estimators_solvers colmap_estimators
)
COLMAP_ADD_TEST(
    NAME fundamental_matrix_test
    SRCS fundamental_matrix_test.cc
    LINK_LIBS colmap_estimators_solvers colmap_estimators
)
COLMAP_ADD_TEST(
    NAME generalized_absolute_pose_test
    SRCS generalized_absolute_pose_test.cc
    LINK_LIBS colmap_estimators_solvers colmap_estimators
)
COLMAP_ADD_TEST(
    NAME generalized_relative_pose_test
    SRCS generalized_relative_pose_test.cc
    LINK_LIBS colmap_estimators_solvers colmap_estimators
)
COLMAP_ADD_TEST(
    NAME homography_matrix_test
    SRCS homography_matrix_test.cc
    LINK_LIBS colmap_estimators_solvers
)
COLMAP_ADD_TEST(
    NAME similarity_transform_test
    SRCS similarity_transform_test.cc
    LINK_LIBS colmap_estimators_solvers
)
COLMAP_ADD_TEST(
    NAME translation_transform_test
    SRCS translation_transform_test.cc
    LINK_LIBS colmap_estimators_solvers
)
COLMAP_ADD_TEST(
    NAME poselib_utils_test
    SRCS poselib_utils_test.cc
    LINK_LIBS colmap_estimators_solvers
)
COLMAP_ADD_TEST(
    NAME relpose_one_sided_focal_test
    SRCS relpose_one_sided_focal_test.cc
    LINK_LIBS colmap_estimators_solvers colmap_estimators
)
COLMAP_ADD_TEST(
    NAME relpose_shared_focal_test
    SRCS relpose_shared_focal_test.cc
    LINK_LIBS colmap_estimators_solvers colmap_estimators
)
