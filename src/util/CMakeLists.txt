set(FOLDER_NAME "util")

COLMAP_ADD_LIBRARY(util
    alignment.h
    bitmap.h bitmap.cc
    cache.h
    camera_specs.h camera_specs.cc
    logging.h logging.cc
    math.h math.cc
    matrix.h
    misc.h misc.cc
    opengl_utils.h opengl_utils.cc
    option_manager.h option_manager.cc
    random.h random.cc
    sqlite3_utils.h
    string.h string.cc
    threading.h threading.cc
    timer.h timer.cc
    testing.h
    types.h
    version.h version.cc
)

if(CUDA_ENABLED)
    COLMAP_CUDA_ADD_LIBRARY(util_cuda
        cuda.h cuda.cc
        cudacc.h cudacc.cc
    )
endif()

COLMAP_ADD_TEST(bitmap_test bitmap_test.cc)
COLMAP_ADD_TEST(cache_test cache_test.cc)
COLMAP_ADD_TEST(endian_test endian_test.cc)
COLMAP_ADD_TEST(math_test math_test.cc)
COLMAP_ADD_TEST(matrix_test matrix_test.cc)
COLMAP_ADD_TEST(misc_test misc_test.cc)
COLMAP_ADD_TEST(opengl_utils_test opengl_utils_test.cc)
COLMAP_ADD_TEST(random_test random_test.cc)
COLMAP_ADD_TEST(string_test string_test.cc)
COLMAP_ADD_TEST(threading_test threading_test.cc)
COLMAP_ADD_TEST(timer_test timer_test.cc)
