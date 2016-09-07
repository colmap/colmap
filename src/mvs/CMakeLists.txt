set(FOLDER_NAME "mvs")

COLMAP_ADD_LIBRARY(mvs
    depth_map.h depth_map.cc
    fusion.h fusion.cc
    image.h image.cc
    model.h model.cc
    normal_map.h normal_map.cc
)

if(CUDA_ENABLED)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")

    if(MSVC)
        # Workaround for Visual Studio compiler error,
        # due problems with compiling too complex code.
        string(REPLACE "/Zo" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    endif()

    COLMAP_CUDA_ADD_LIBRARY(mvs_cuda
        cuda_utils.h cuda_utils.cc
        gpu_mat_prng.h gpu_mat_prng.cu
        gpu_mat_ref_image.h gpu_mat_ref_image.cu
        patch_match.h patch_match.cc
        patch_match_cuda.h patch_match_cuda.cu
    )

    # MSVC/NVCC cannot compile the Boost unit test suite.
    if(NOT IS_MSVC)
        COLMAP_CUDA_ADD_TEST(gpu_mat_test gpu_mat_test.cu)
    endif()
endif()
