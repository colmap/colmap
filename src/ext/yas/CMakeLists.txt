# The YAS library is header only. As such, we specify it as an interfece.
add_library(yas INTERFACE)
target_include_directories(yas INTERFACE "${PROJECT_SOURCE_DIR}/src/ext/yas/include")

add_definitions("-std=c++11")
COLMAP_ADD_TEST(yas_tests tests/unit_tests.cc)
target_include_directories(yas_tests PRIVATE
                           "${PROJECT_SOURCE_DIR}/src/ext/yas/tests/base/include")
