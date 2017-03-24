set(FOLDER_NAME "util")

COLMAP_ADD_LIBRARY(util
    camera_specs.h camera_specs.cc
    math.h math.cc
    misc.h misc.cc
    option_manager.h option_manager.cc
    bitmap.h bitmap.cc
    random.h random.cc
    string.h string.cc
    timer.h timer.cc
    threading.h threading.cc
    opengl_utils.h opengl_utils.cc
    logging.h logging.cc
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
COLMAP_ADD_TEST(math_test math_test.cc)
COLMAP_ADD_TEST(misc_test misc_test.cc)
COLMAP_ADD_TEST(opengl_utils_test opengl_utils_test.cc)
COLMAP_ADD_TEST(random_test random_test.cc)
COLMAP_ADD_TEST(string_test string_test.cc)
COLMAP_ADD_TEST(threading_test threading_test.cc)
COLMAP_ADD_TEST(timer_test timer_test.cc)
