# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "sfm")

COLMAP_ADD_LIBRARY(
    NAME colmap_sfm
    SRCS
        global_mapper.h global_mapper.cc
        incremental_mapper_impl.h incremental_mapper_impl.cc
        incremental_mapper.h incremental_mapper.cc
        incremental_triangulator.h incremental_triangulator.cc
        observation_manager.h observation_manager.cc
    PUBLIC_LINK_LIBS
        colmap_scene
        colmap_estimators
        colmap_util
    PRIVATE_LINK_LIBS
        colmap_geometry
        colmap_image
)

if(MSVC)
    target_compile_options(colmap_sfm PRIVATE /bigobj)
endif()

COLMAP_ADD_TEST(
    NAME global_mapper_test
    SRCS global_mapper_test.cc
    LINK_LIBS colmap_sfm
)
COLMAP_ADD_TEST(
    NAME incremental_triangulator_test
    SRCS incremental_triangulator_test.cc
    LINK_LIBS colmap_sfm
)
COLMAP_ADD_TEST(
    NAME observation_manager_test
    SRCS observation_manager_test.cc
    LINK_LIBS colmap_sfm
)
COLMAP_ADD_TEST(
    NAME incremental_mapper_test
    SRCS incremental_mapper_test.cc
    LINK_LIBS colmap_sfm
)
