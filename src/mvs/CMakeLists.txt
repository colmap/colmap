set(FOLDER_NAME "mvs")

if(CUDA_ENABLED)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        --use_fast_math
        -std=c++11
        -Xcompiler
        -D__CORRECT_ISO_CPP11_MATH_H_PROTO)

    COLMAP_ADD_LIBRARY(mvs
        cuda_utils.h cuda_utils.cc
        depth_map.h depth_map.cc
        fusion.h fusion.cc
        image.h image.cc
        model.h model.cc
        normal_map.h normal_map.cc
        patch_match.h patch_match.cc
    )

    # Workaround for Visual Studio compiler error,
    # due problems with compiling too complex code.
    set(ZO_FLAG_ENABLED FALSE)
    if ("${CMAKE_CXX_FLAGS}" MATCHES "/Zo")
        set(ZO_FLAG_ENABLED TRUE)
        string(REPLACE "/Zo" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif()

    COLMAP_CUDA_ADD_LIBRARY(mvs_cuda
        gpu_mat_prng.h gpu_mat_prng.cu
        gpu_mat_ref_image.h gpu_mat_ref_image.cu
        patch_match_cuda.h patch_match_cuda.cu
    )

    if(ZO_FLAG_ENABLED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zo")
    endif()

    # MSVC/NVCC cannot compile the Boost unit test suite.
    if(NOT IS_MSVC)
        COLMAP_CUDA_ADD_TEST(gpu_mat_test gpu_mat_test.cu)
    endif()
endif()
