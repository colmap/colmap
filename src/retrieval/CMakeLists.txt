set(FOLDER_NAME "retrieval")

COLMAP_ADD_LIBRARY(retrieval
    inverted_file.h
    inverted_file_entry.h
    inverted_index.h
    utils.h
    visual_index.h visual_index.cc
)

COLMAP_ADD_TEST(visual_index_test visual_index_test.cc)
