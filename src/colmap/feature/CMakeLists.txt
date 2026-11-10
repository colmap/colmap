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
        cusift.h cusift.cc
        cusift_matcher.h cusift_matcher.cc
        extractor.h extractor.cc
        flexglue.h flexglue.cc
        index.h index.cc
        lightglue.h lightglue.cc
        loma.h loma.cc
        matcher.h matcher.cc
        onnx_matchers.h onnx_matchers.cc
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

if(CUSIFT_ENABLED)
    # colmap_util_cuda for device selection (SetBestCudaDevice).
    target_link_libraries(colmap_feature PRIVATE colmap_cusift colmap_util_cuda)
endif()

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
    target_link_libraries(colmap_feature PRIVATE colmap_onnx_util)
    if(CUDA_ENABLED)
        target_link_libraries(colmap_feature PRIVATE colmap_util_cuda)
    endif()
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
        NAME lightglue_test
        SRCS lightglue_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
    COLMAP_ADD_TEST(
        NAME flexglue_test
        SRCS flexglue_test.cc
        LINK_LIBS
            colmap_feature
            colmap_sensor
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
# The option tests in cusift_test and cusift_matcher_test run in every build;
# their GPU tests are compiled only with cusift.
COLMAP_ADD_TEST(
    NAME cusift_test
    SRCS cusift_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME cusift_matcher_test
    SRCS cusift_matcher_test.cc
    LINK_LIBS colmap_feature
)
if(TESTS_ENABLED AND CUSIFT_ENABLED)
    # Plain signature, as for sift_test below.
    target_link_libraries(colmap_feature_cusift_test colmap_util_cuda)
    # colmap_cusift: GemmMatcherContract tests call the vendored GEMM matcher
    # directly.
    target_link_libraries(colmap_feature_cusift_matcher_test
        colmap_util_cuda colmap_cusift)
endif()
if(CUSIFT_ENABLED)
    COLMAP_ADD_TEST(
        NAME cusift_determinism_test
        SRCS cusift_determinism_test.cc
        LINK_LIBS colmap_feature colmap_util_cuda
    )
    # Builds two-view and pose problems, so it also needs the estimators. It
    # reads IncrementalMapper::Options defaults from the header only, so it
    # does not link colmap_sfm.
    COLMAP_ADD_TEST(
        NAME cusift_verification_test
        SRCS cusift_verification_test.cc
        LINK_LIBS colmap_feature colmap_estimators colmap_util_cuda
    )
    COLMAP_ADD_TEST(
        NAME cusift_memory_test
        SRCS cusift_memory_test.cc
        LINK_LIBS colmap_feature colmap_cusift colmap_util_cuda
    )
    COLMAP_ADD_TEST(
        NAME cusift_pool_fallback_test
        SRCS cusift_pool_fallback_test.cc
        LINK_LIBS colmap_cusift colmap_util_cuda
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
    LINK_LIBS colmap_feature colmap_sensor
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
if(TESTS_ENABLED AND (CUDA_ENABLED OR HIP_ENABLED))
    # NOTE: plain signature to match COLMAP_ADD_TEST() above; mixing
    # keyword and plain signatures for one target is a CMake error.
    target_link_libraries(colmap_feature_sift_test colmap_util_cuda)
endif()
