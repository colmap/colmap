# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "math")

COLMAP_ADD_LIBRARY(
    NAME colmap_math
    SRCS
        connected_components.h
        graph_cut.h graph_cut.cc
        math.h math.cc
        matrix.h
        polynomial.h polynomial.cc
        random.h random.cc
        random_eigen.h
        spanning_tree.h spanning_tree.cc
        union_find.h
    PUBLIC_LINK_LIBS
        colmap_util
        Boost::graph
        Eigen3::Eigen
    PRIVATE_LINK_LIBS
        ${COLMAP_BOOST_HEADER_LIBS}
        metis
)

COLMAP_ADD_TEST(
    NAME connected_components_test
    SRCS connected_components_test.cc
    LINK_LIBS colmap_math
)
COLMAP_ADD_TEST(
    NAME graph_cut_test
    SRCS graph_cut_test.cc
    LINK_LIBS colmap_math
)
COLMAP_ADD_TEST(
    NAME math_test
    SRCS math_test.cc
    LINK_LIBS colmap_math
)
COLMAP_ADD_TEST(
    NAME matrix_test
    SRCS matrix_test.cc
    LINK_LIBS colmap_math
)
COLMAP_ADD_TEST(
    NAME polynomial_test
    SRCS polynomial_test.cc
    LINK_LIBS colmap_math
)
COLMAP_ADD_TEST(
    NAME random_test
    SRCS random_test.cc
    LINK_LIBS colmap_math
)
COLMAP_ADD_TEST(
    NAME random_eigen_test
    SRCS random_eigen_test.cc
    LINK_LIBS colmap_math
)
COLMAP_ADD_TEST(
    NAME union_find_test
    SRCS union_find_test.cc
    LINK_LIBS colmap_math
)
COLMAP_ADD_TEST(
    NAME spanning_tree_test
    SRCS spanning_tree_test.cc
    LINK_LIBS colmap_math
)
