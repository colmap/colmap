# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "optim")

COLMAP_ADD_LIBRARY(
    NAME colmap_optim
    SRCS
        combination_sampler.h combination_sampler.cc
        least_absolute_deviations.h least_absolute_deviations.cc
        progressive_sampler.h progressive_sampler.cc
        random_sampler.h random_sampler.cc
        sparse_cholesky.h sparse_cholesky.cc
        sprt.h sprt.cc
        support_measurement.h support_measurement.cc
        tiny_solver.h
    PUBLIC_LINK_LIBS
        colmap_math
        colmap_util
        Eigen3::Eigen
        CHOLMOD::CHOLMOD
)
if(OPENMP_FOUND)
    target_link_libraries(colmap_optim PUBLIC OpenMP::OpenMP_CXX)
endif()

COLMAP_ADD_TEST(
    NAME combination_sampler_test
    SRCS combination_sampler_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME least_absolute_deviations_test
    SRCS least_absolute_deviations_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME loransac_test
    SRCS loransac_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME progressive_sampler_test
    SRCS progressive_sampler_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME random_sampler_test
    SRCS random_sampler_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME ransac_test
    SRCS ransac_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME sparse_cholesky_test
    SRCS sparse_cholesky_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME sprt_test
    SRCS sprt_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME support_measurement_test
    SRCS support_measurement_test.cc
    LINK_LIBS colmap_optim
)
COLMAP_ADD_TEST(
    NAME tiny_solver_test
    SRCS tiny_solver_test.cc
    LINK_LIBS
        colmap_optim
        colmap_estimators_cost_functions
        Ceres::ceres
        Eigen3::Eigen
)
