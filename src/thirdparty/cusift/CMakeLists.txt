# cusift - commercially-clean GPU SIFT (MIT + third-party notices, see README.md).
# Vendored from exovision-research/cusift: the extraction path (cudaImage/
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

# Pinned per-target rather than inheriting COLMAP's project-wide
# CMAKE_CUDA_ARCHITECTURES, which defaults to `native`. Two reasons:
#
#   1. `-arch=native` emits SASS for the build machine's GPU and NO PTX, so the
#      result runs only on that exact architecture -- a COLMAP built on an Ada
#      card dies with "no kernel image is available for execution on the device"
#      on Ampere, and a container image inherits whatever card built it. A plain
#      "86;89" list expands to arch=compute_NN,code=[compute_NN,sm_NN], i.e.
#      cubins for both plus PTX that JITs forward to newer cards.
#   2. cusift's deliverable is measured VLFeat parity, and --use_fast_math is
#      part of that numerics contract (docs/HANDOVER.md §5). Machine-dependent
#      codegen would make those numbers unreproducible across boxes.
#
# Mirrors upstream cusift's own pinned list; keep the two in sync. Note PTX only
# JITs forward, so building for a pre-Ampere GPU needs an explicit override,
# e.g. -DCOLMAP_CUSIFT_CUDA_ARCHITECTURES="75-virtual;86-real;89-real".
set(COLMAP_CUSIFT_CUDA_ARCHITECTURES "86;89" CACHE STRING
    "CUDA architectures for the vendored cusift extractor")
set_target_properties(colmap_cusift PROPERTIES
    CUDA_ARCHITECTURES "${COLMAP_CUSIFT_CUDA_ARCHITECTURES}")

# --use_fast_math is part of cusift's numerics contract, not an optimization
# choice: vl_fast_atan2_f/vl_fast_sqrt_f are replicated in CUDA and every
# VLFeat-parity gate was validated under it (docs/HANDOVER.md §5). Dropping it
# measurably *reduces* parity with COLMAP's own VLFeat backend.
# Never substitute a global -fmad=false; cudaSiftD.cu scopes FMA contraction off
# for RefineOneKeypoint alone via `#pragma nv_fmad off`.
target_compile_options(colmap_cusift PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    # Matches the blanket -w/W0 this directory applies to thirdparty C/C++.
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
)
