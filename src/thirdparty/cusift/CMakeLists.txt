# cusift: CUDA SIFT extractor (cudaImage/cudaSift*) and GEMM matcher
# (matcher_gemm.{h,cu}), used by FeatureExtractorType::CUSIFT and
# FeatureMatcherType::CUSIFT_GEMM. MIT plus third-party notices; see LICENSE.

# Note: cudaSiftD.cu is deliberately NOT a source file. cudaSiftH.cu #includes
# it, so listing it here would compile every kernel twice and fail to link.
COLMAP_ADD_LIBRARY(
    NAME colmap_cusift
    TYPE STATIC
    SRCS
        src/cudaImage.h src/cudaImage.cu
        src/cudaSift.h
        src/cudaSiftD.h
        src/cudaSiftH.cu
        src/cudautils.h
        src/cusift_device.h src/cusift_device.cu
        src/matcher_gemm.h src/matcher_gemm.cu
    PRIVATE_LINK_LIBS
        CUDA::cudart
        CUDA::cublas
)

target_include_directories(colmap_cusift PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

# cusift builds for COLMAP's CMAKE_CUDA_ARCHITECTURES, like every other CUDA
# target; the kernels use no architecture-specific intrinsics.

# --use_fast_math is part of cusift's numerics contract, not an optimization
# choice: vl_fast_atan2_f/vl_fast_sqrt_f are replicated in CUDA and every
# VLFeat-parity measurement was made under it. Dropping it
# measurably *reduces* parity with COLMAP's own VLFeat backend.
target_compile_options(colmap_cusift PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    # Matches the blanket -w/W0 this directory applies to thirdparty C/C++.
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
)
