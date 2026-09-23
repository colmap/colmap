# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "retrieval")

COLMAP_ADD_LIBRARY(
    NAME colmap_retrieval
    SRCS
        geometry.h geometry.cc
        inverted_file.h
        inverted_file_entry.h
        inverted_index.h
        resources.h resources.cc
        utils.h
        visual_index.h visual_index.cc
        vote_and_verify.h vote_and_verify.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_math
        colmap_feature_types
        ${COLMAP_BOOST_HEADER_LIBS}
        Eigen3::Eigen
    PRIVATE_LINK_LIBS
        colmap_estimators
        colmap_optim
        faiss
)
if(OPENMP_FOUND)
    target_link_libraries(colmap_retrieval PUBLIC OpenMP::OpenMP_CXX)
endif()

COLMAP_ADD_TEST(
    NAME geometry_test
    SRCS geometry_test.cc
    LINK_LIBS colmap_retrieval
)
COLMAP_ADD_TEST(
    NAME inverted_file_entry_test
    SRCS inverted_file_entry_test.cc
    LINK_LIBS colmap_retrieval
)
COLMAP_ADD_TEST(
    NAME utils_test
    SRCS utils_test.cc
    LINK_LIBS colmap_retrieval
)
COLMAP_ADD_TEST(
    NAME visual_index_test
    SRCS visual_index_test.cc
    LINK_LIBS colmap_retrieval
)
COLMAP_ADD_TEST(
    NAME vote_and_verify_test
    SRCS vote_and_verify_test.cc
    LINK_LIBS colmap_retrieval
)
