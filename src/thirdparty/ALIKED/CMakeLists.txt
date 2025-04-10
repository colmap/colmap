set(OPTIONAL_CUDA_SRCS)
if(CUDA_ENABLED)
    set(OPTIONAL_CUDA_SRCS
        cuda_helpers.h
        deform_conv2d_kernel.cu
        get_patches_cuda.h get_patches_cuda.cu
    )
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_aliked
    SRCS
        aliked.hpp aliked.cpp
        blocks.hpp blocks.cpp
        deform_conv2d.h deform_conv2d.cpp
        dkd.hpp dkd.cpp
        get_patches.hpp get_patches.cpp
        input_padder.hpp input_padder.cpp
        sddh.hpp sddh.cpp
        ${OPTIONAL_CUDA_SRCS}
    PUBLIC_LINK_LIBS
        torch::torch
        glog::glog
)
