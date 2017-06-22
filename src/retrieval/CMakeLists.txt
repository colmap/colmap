set(FOLDER_NAME "retrieval")

COLMAP_ADD_LIBRARY(retrieval
    geometry.h geometry.cc
    inverted_file.h
    inverted_file_entry.h
    inverted_index.h
    utils.h
    visual_index.h
    vote_and_verify.h vote_and_verify.cc
)

COLMAP_ADD_TEST(geometry_test geometry_test.cc)
COLMAP_ADD_TEST(inverted_file_entry_test inverted_file_entry_test.cc)
COLMAP_ADD_TEST(visual_index_test visual_index_test.cc)
