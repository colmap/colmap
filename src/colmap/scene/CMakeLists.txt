# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "scene")

COLMAP_ADD_LIBRARY(
    NAME colmap_scene_types
    SRCS
        camera.h camera.cc
        frame.h frame.cc
        image.h image.cc
        point2d.h point2d.cc
        two_view_geometry.h two_view_geometry.cc
    PUBLIC_LINK_LIBS
        colmap_sensor
        colmap_feature_types
        colmap_geometry
        colmap_util
        Eigen3::Eigen
)

COLMAP_ADD_LIBRARY(
    NAME colmap_scene
    SRCS
        correspondence_graph.h correspondence_graph.cc
        database.h database.cc
        database_sqlite.h database_sqlite.cc
        database_cache.h database_cache.cc
        point3d.h point3d.cc
        pose_graph.h pose_graph.cc
        projection.h projection.cc
        reconstruction.h reconstruction.cc
        reconstruction_clustering.h reconstruction_clustering.cc
        reconstruction_io.h reconstruction_io.cc
        reconstruction_io_binary.h reconstruction_io_binary.cc
        reconstruction_io_text.h reconstruction_io_text.cc
        reconstruction_io_utils.h reconstruction_io_utils.cc
        reconstruction_manager.h reconstruction_manager.cc
        reconstruction_pruning.h reconstruction_pruning.cc
        rig.h rig.cc
        scene_clustering.h scene_clustering.cc
        synthetic.h synthetic.cc
        track.h track.cc
        visibility_pyramid.h visibility_pyramid.cc
    PUBLIC_LINK_LIBS
        colmap_scene_types
        colmap_util
        Eigen3::Eigen
        SQLite3::SQLite3
    PRIVATE_LINK_LIBS
        colmap_optim
)

if(TESTS_ENABLED)
    target_sources(
        colmap_scene
        PRIVATE
            reconstruction_matchers.h
    )
endif()

COLMAP_ADD_TEST(
    NAME camera_test
    SRCS camera_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME correspondence_graph_test
    SRCS correspondence_graph_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME database_cache_test
    SRCS database_cache_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME database_test
    SRCS database_test.cc
    LINK_LIBS colmap_scene
)
if(TESTS_ENABLED)
    if(MSVC)
        target_compile_options(colmap_scene_database_test PRIVATE /utf-8)
    else()
        target_compile_options(colmap_scene_database_test PRIVATE -finput-charset=UTF-8 -fexec-charset=UTF-8)
    endif()
endif()
COLMAP_ADD_TEST(
    NAME frame_test
    SRCS frame_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME image_test
    SRCS image_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME point2d_test
    SRCS point2d_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME point3d_test
    SRCS point3d_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME projection_test
    SRCS projection_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME reconstruction_test
    SRCS reconstruction_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME reconstruction_io_test
    SRCS reconstruction_io_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME reconstruction_manager_test
    SRCS reconstruction_manager_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME reconstruction_matchers_test
    SRCS reconstruction_matchers_test.cc
    LINK_LIBS colmap_scene colmap_estimators
)
COLMAP_ADD_TEST(
    NAME reconstruction_pruning_test
    SRCS reconstruction_pruning_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME reconstruction_clustering_test
    SRCS reconstruction_clustering_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME rig_test
    SRCS rig_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME scene_clustering_test
    SRCS scene_clustering_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME synthetic_test
    SRCS synthetic_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME track_test
    SRCS track_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME two_view_geometry_test
    SRCS two_view_geometry_test.cc
    LINK_LIBS colmap_scene
)
COLMAP_ADD_TEST(
    NAME visibility_pyramid_test
    SRCS visibility_pyramid_test.cc
    LINK_LIBS colmap_scene
)

COLMAP_ADD_TEST(
    NAME pose_graph_test
    SRCS pose_graph_test.cc
    LINK_LIBS colmap_scene
)
