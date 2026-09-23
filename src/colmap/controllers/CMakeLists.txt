# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "controllers")

COLMAP_ADD_LIBRARY(
    NAME colmap_controllers
    SRCS
        automatic_reconstruction.h automatic_reconstruction.cc
        base_option_manager.h base_option_manager.cc
        bundle_adjustment.h bundle_adjustment.cc
        hierarchical_pipeline.h hierarchical_pipeline.cc
        feature_extraction.h feature_extraction.cc
        feature_matching.h feature_matching.cc
        feature_matching_utils.h feature_matching_utils.cc
        matcher_cache.h matcher_cache.cc
        pairing.h pairing.cc
        global_pipeline.h global_pipeline.cc
        image_reader.h image_reader.cc
        incremental_pipeline.h incremental_pipeline.cc
        option_manager.h option_manager.cc
        reconstruction_clustering.h reconstruction_clustering.cc
        rotation_averaging.h rotation_averaging.cc
        undistorters.h undistorters.cc
    PUBLIC_LINK_LIBS
        colmap_estimators
        colmap_feature
        colmap_retrieval
        colmap_geometry
        colmap_scene
        colmap_util
        Eigen3::Eigen
        Boost::program_options
    PRIVATE_LINK_LIBS
        colmap_image
        colmap_math
        colmap_sfm
        Ceres::ceres
        ${COLMAP_BOOST_HEADER_LIBS}
        faiss
)
if(MVS_ENABLED)
    target_link_libraries(colmap_controllers PRIVATE colmap_mvs)
endif()
if(CUDA_ENABLED)
    target_link_libraries(colmap_controllers PRIVATE colmap_util_cuda)
    if(MVS_ENABLED)
        target_link_libraries(colmap_controllers PRIVATE colmap_mvs_cuda)
    endif()
endif()

COLMAP_ADD_TEST(
    NAME automatic_reconstruction_test
    SRCS automatic_reconstruction_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME bundle_adjustment_test
    SRCS bundle_adjustment_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME hierarchical_pipeline_test
    SRCS hierarchical_pipeline_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME incremental_pipeline_test
    SRCS incremental_pipeline_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME image_reader_test
    SRCS image_reader_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME feature_extraction_test
    SRCS feature_extraction_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME feature_matching_test
    SRCS feature_matching_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME feature_matching_utils_test
    SRCS feature_matching_utils_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME global_pipeline_test
    SRCS global_pipeline_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME base_option_manager_test
    SRCS base_option_manager_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME option_manager_test
    SRCS option_manager_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME rotation_averaging_test
    SRCS rotation_averaging_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME reconstruction_clustering_test
    SRCS reconstruction_clustering_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME undistorters_test
    SRCS undistorters_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME matcher_cache_test
    SRCS matcher_cache_test.cc
    LINK_LIBS colmap_controllers
)
COLMAP_ADD_TEST(
    NAME pairing_test
    SRCS pairing_test.cc
    LINK_LIBS colmap_controllers
)
