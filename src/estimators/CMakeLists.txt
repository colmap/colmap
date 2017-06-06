set(FOLDER_NAME "estimators")

COLMAP_ADD_LIBRARY(estimators
    affine_transform.h affine_transform.cc
    coordinate_frame.h coordinate_frame.cc
    epnp.h epnp.cc
    essential_matrix.h essential_matrix.cc
    euclidean_transform.h
    fundamental_matrix.h fundamental_matrix.cc
    gp3p.h gp3p.cc gp3p_coeffs.h gp3p_coeffs.cc
    homography_matrix.h homography_matrix.cc
    p3p.h p3p.cc
    pose.h pose.cc
    similarity_transform.h
    translation_transform.h
    triangulation.h triangulation.cc
    two_view_geometry.h two_view_geometry.cc
    utils.h utils.cc
)

COLMAP_ADD_TEST(affine_transform_test affine_transform_test.cc)
COLMAP_ADD_TEST(coordinate_frame_test coordinate_frame_test.cc)
COLMAP_ADD_TEST(epnp_test epnp_test.cc)
COLMAP_ADD_TEST(essential_matrix_test essential_matrix_test.cc)
COLMAP_ADD_TEST(fundamental_matrix_test fundamental_matrix_test.cc)
COLMAP_ADD_TEST(gp3p_test gp3p_test.cc)
COLMAP_ADD_TEST(homography_matrix_test homography_matrix_test.cc)
COLMAP_ADD_TEST(p3p_test p3p_test.cc)
COLMAP_ADD_TEST(translation_transform_test translation_transform_test.cc)
COLMAP_ADD_TEST(utils_test utils_test.cc)
