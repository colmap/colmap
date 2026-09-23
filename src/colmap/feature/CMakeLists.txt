# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "feature")

COLMAP_ADD_LIBRARY(
    NAME colmap_feature_types
    SRCS
        types.h types.cc
    PUBLIC_LINK_LIBS
        colmap_util
        Eigen3::Eigen
)

COLMAP_ADD_TEST(
    NAME types_test
    SRCS types_test.cc
    LINK_LIBS colmap_feature_types
)

COLMAP_ADD_LIBRARY(
    NAME colmap_feature
    SRCS
        aliked.h aliked.cc
        extractor.h extractor.cc
        index.h index.cc
        loma.h loma.cc
        matcher.h matcher.cc
        onnx_matchers.h onnx_matchers.cc
        onnx_utils.h onnx_utils.cc
        sift.h sift.cc
        types.h types.cc
        utils.h utils.cc
    PUBLIC_LINK_LIBS
        colmap_feature_types
        colmap_scene_types
        colmap_geometry
        colmap_util
    PRIVATE_LINK_LIBS
        colmap_math
        colmap_sensor
        colmap_vlfeat
        Eigen3::Eigen
        faiss
)

if(GPU_ENABLED)
    target_link_libraries(colmap_feature PRIVATE colmap_sift_gpu)
    if(NOT GUI_ENABLED)
        target_link_libraries(colmap_feature PRIVATE GLEW::GLEW)
    endif()
endif()

if(OPENMP_FOUND)
    target_link_libraries(colmap_feature PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ONNX_ENABLED)
    target_link_libraries(colmap_feature PRIVATE onnxruntime::onnxruntime)
endif()

if(ONNX_ENABLED)
    COLMAP_ADD_TEST(
        NAME aliked_test
        SRCS aliked_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
    COLMAP_ADD_TEST(
        NAME onnx_matchers_test
        SRCS onnx_matchers_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
    COLMAP_ADD_TEST(
        NAME onnx_utils_test
        SRCS onnx_utils_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
    COLMAP_ADD_TEST(
        NAME loma_test
        SRCS loma_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
endif()
COLMAP_ADD_TEST(
    NAME extractor_test
    SRCS extractor_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME index_test
    SRCS index_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME matcher_test
    SRCS matcher_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME utils_test
    SRCS utils_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME sift_test
    SRCS sift_test.cc
    LINK_LIBS
        colmap_feature
        colmap_util
)
if(TESTS_ENABLED AND GUI_ENABLED)
    target_link_libraries(colmap_feature_sift_test Qt${QT_VERSION_MAJOR}::Widgets)
endif()
