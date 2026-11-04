# cusift - commercially-clean GPU SIFT (MIT + third-party notices, see README.md).
# Vendored from the standalone cusift library: the extraction path (cudaImage/
# cudaSift*) plus the GEMM matcher (matcher.h/matcher_gemm.cu), wired into
# COLMAP as FeatureExtractorType::CUSIFT and FeatureMatcherType::CUSIFT_GEMM
# respectively. The brute-force warp matcher (matcher.cu) is NOT vendored: no
# COLMAP backend targets it, so importing it would just be unused build cost.

# Note: cudaSiftD.cu is deliberately NOT a source file. cudaSiftH.cu #includes
# it, so listing it here would compile every kernel twice and fail to link.
COLMAP_ADD_LIBRARY(
    NAME colmap_cusift
    TYPE STATIC
    SRCS
        src/cudaImage.h src/cudaImage.cu
        src/cudaSift.h
        src/cudaSiftD.h
        src/cudaSiftH.h src/cudaSiftH.cu
        src/cudautils.h
        src/cusift_device.h src/cusift_device.cu
        src/matcher.h src/matcher_gemm.cu src/matcher_fused.cu
    PRIVATE_LINK_LIBS
        CUDA::cudart
        CUDA::cublas
)

target_include_directories(colmap_cusift PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

# By default cusift builds for COLMAP's project-wide CMAKE_CUDA_ARCHITECTURES,
# like every other CUDA target, so -DCMAKE_CUDA_ARCHITECTURES=... and the
# toolkit's supported range apply to it too (a fixed "86;89" list broke CUDA
# toolkits older than 11.8, which cannot target sm_89). The kernels use no
# architecture-specific intrinsics. Set COLMAP_CUSIFT_CUDA_ARCHITECTURES to
# build cusift for a different list than the rest of COLMAP, e.g. "86;89" for
# sm_86/sm_89 machine code plus PTX that JIT-compiles forward to newer GPUs.
set(COLMAP_CUSIFT_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures for the vendored cusift library (empty: use CMAKE_CUDA_ARCHITECTURES)")
if(COLMAP_CUSIFT_CUDA_ARCHITECTURES)
    set_target_properties(colmap_cusift PROPERTIES
        CUDA_ARCHITECTURES "${COLMAP_CUSIFT_CUDA_ARCHITECTURES}")
endif()

# --use_fast_math is part of cusift's numerics contract, not an optimization
# choice: vl_fast_atan2_f/vl_fast_sqrt_f are replicated in CUDA and every
# VLFeat-parity measurement was made under it. Dropping it
# measurably *reduces* parity with COLMAP's own VLFeat backend.
# Never substitute a global -fmad=false; cudaSiftD.cu scopes FMA contraction off
# for RefineOneKeypoint alone via `#pragma nv_fmad off`.
target_compile_options(colmap_cusift PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    # Matches the blanket -w/W0 this directory applies to thirdparty C/C++.
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
)
