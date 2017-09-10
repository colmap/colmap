set(FOLDER_NAME "feature")

COLMAP_ADD_LIBRARY(feature
    extraction.h extraction.cc
    matching.h matching.cc
    sift.h sift.cc
    types.h types.cc
    utils.h utils.cc
)

COLMAP_ADD_TEST(feature_utils_test utils_test.cc)
COLMAP_ADD_TEST(sift_test sift_test.cc)
COLMAP_ADD_TEST(types_test types_test.cc)
