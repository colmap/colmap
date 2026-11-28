# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "mvs")

COLMAP_ADD_LIBRARY(
    NAME colmap_mvs
    SRCS
        advancing_front_meshing.h advancing_front_meshing.cc
        consistency_graph.h consistency_graph.cc
        delaunay_meshing.h delaunay_meshing.cc
        depth_map.h depth_map.cc
        fusion.h fusion.cc
        image.h image.cc
        mat.h mat.cc
        mesh_simplification.h mesh_simplification.cc
        model.h model.cc
        poisson_meshing.h poisson_meshing.cc
        normal_map.h normal_map.cc
        patch_match_options.h patch_match_options.cc
        texture_mapping.h texture_mapping.cc
        workspace.h workspace.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_scene
    PRIVATE_LINK_LIBS
        colmap_math
        colmap_sensor
        colmap_image
        colmap_poisson_recon
        Eigen3::Eigen
)
if(CGAL_ENABLED)
    target_link_libraries(colmap_mvs PRIVATE CGAL)
endif()
if(OPENMP_FOUND)
    target_link_libraries(colmap_mvs PRIVATE OpenMP::OpenMP_CXX)
endif()

COLMAP_ADD_TEST(
    NAME patch_match_test
    SRCS patch_match_test.cc
    LINK_LIBS colmap_mvs
)

COLMAP_ADD_TEST(
    NAME consistency_graph_test
    SRCS consistency_graph_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME depth_map_test
    SRCS depth_map_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME fusion_test
    SRCS fusion_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME image_test
    SRCS image_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME mat_test
    SRCS mat_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME mesh_simplification_test
    SRCS mesh_simplification_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME patch_match_options_test
    SRCS patch_match_options_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME poisson_meshing_test
    SRCS poisson_meshing_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME delaunay_meshing_test
    SRCS delaunay_meshing_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME advancing_front_meshing_test
    SRCS advancing_front_meshing_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME model_test
    SRCS model_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME normal_map_test
    SRCS normal_map_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME texture_mapping_test
    SRCS texture_mapping_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME workspace_test
    SRCS workspace_test.cc
    LINK_LIBS colmap_mvs
)

if(CUDA_ENABLED)
    COLMAP_ADD_LIBRARY(
        NAME colmap_mvs_cuda
        SRCS
            gpu_mat_prng.h gpu_mat_prng.cu
            gpu_mat_ref_image.h gpu_mat_ref_image.cu
            patch_match.h patch_match.cc
            patch_match_cuda.h patch_match_cuda.cu patch_match_geometry.cuh
            patch_match_sampling.cuh
            patch_match_rng_perimeter.h
        PUBLIC_LINK_LIBS
            colmap_mvs
            colmap_util_cuda
            CUDA::cudart
            CUDA::curand
    )

    # Work around NVCC miscompilation on Blackwell GPUs (sm_100+).
    # The PatchMatch kernels are miscompiled at -O2/-O3 when generating native
    # SASS for sm_100+. Instead, we emit sm_90 PTX which Blackwell GPUs
    # JIT-compile at first launch, bypassing the buggy codegen path.
    # See: https://github.com/colmap/colmap/issues/3514
    #      https://github.com/ggml-org/llama.cpp/issues/18331
    set(_COLMAP_BLACKWELL_TARGET FALSE)

    # Classify architecture keywords. "all"/"all-major" expand based on the
    # CUDA toolkit version, while "native" expands based on installed GPUs.
    set(_HAS_ALL_KEYWORD FALSE)
    set(_HAS_NATIVE_KEYWORD FALSE)
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(_arch STREQUAL "all" OR _arch STREQUAL "all-major")
            set(_HAS_ALL_KEYWORD TRUE)
        elseif(_arch STREQUAL "native")
            set(_HAS_NATIVE_KEYWORD TRUE)
        endif()
    endforeach()

    # Query installed GPUs only for "native", since it targets present hardware.
    if(_HAS_NATIVE_KEYWORD)
        execute_process(
            COMMAND nvidia-smi
                --query-gpu=compute_cap
                --format=csv,noheader
            OUTPUT_VARIABLE _GPU_COMPUTE_CAPS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _NVIDIA_SMI_RESULT
        )
        if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
            string(REPLACE "\n" ";" _GPU_COMPUTE_CAPS
                   "${_GPU_COMPUTE_CAPS}")
        endif()
    endif()

    # Detect Blackwell from explicit numeric architectures.
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        string(REGEX REPLACE "[-].*" "" _arch_num "${_arch}")
        if(_arch_num MATCHES "^[0-9]+$" AND _arch_num GREATER_EQUAL 100)
            set(_COLMAP_BLACKWELL_TARGET TRUE)
            break()
        endif()
    endforeach()

    # Detect Blackwell from "all"/"all-major": these expand based on the
    # toolkit, not installed GPUs. CUDA 12.8 added sm_100 support.
    if(NOT _COLMAP_BLACKWELL_TARGET AND _HAS_ALL_KEYWORD)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            set(_COLMAP_BLACKWELL_TARGET TRUE)
        endif()
    endif()

    # Detect Blackwell from "native" via nvidia-smi (installed GPU check).
    if(NOT _COLMAP_BLACKWELL_TARGET AND _HAS_NATIVE_KEYWORD)
        if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
            foreach(_cap IN LISTS _GPU_COMPUTE_CAPS)
                string(STRIP "${_cap}" _cap)
                string(REGEX MATCH "^([0-9]+)" _major "${_cap}")
                if(_major AND _major GREATER_EQUAL 10)
                    set(_COLMAP_BLACKWELL_TARGET TRUE)
                    break()
                endif()
            endforeach()
        endif()
    endif()

    if(_COLMAP_BLACKWELL_TARGET)
        message(WARNING
            "Blackwell GPU (sm_100+) detected. Replacing sm_100+ architectures "
            "with 90-virtual (PTX) for colmap_mvs_cuda to work around an NVCC "
            "compiler bug. Blackwell GPUs will JIT-compile from PTX at first "
            "launch (see https://github.com/colmap/colmap/issues/3514).")

        # Build filtered architecture list: keep archs < 100, drop >= 100.
        set(_COLMAP_MVS_CUDA_ARCHS)
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "[-].*" "" _arch_num "${_arch}")
            if(_arch_num MATCHES "^[0-9]+$")
                if(_arch_num LESS 100)
                    list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_arch}")
                endif()
            elseif(_arch STREQUAL "all")
                # "all"/"all-major" keywords require CMake 3.24+, which
                # provides CMAKE_CUDA_ARCHITECTURES_ALL[_MAJOR] variables.
                foreach(_exp IN LISTS CMAKE_CUDA_ARCHITECTURES_ALL)
                    string(REGEX REPLACE "[-].*" "" _exp_num "${_exp}")
                    if(_exp_num MATCHES "^[0-9]+$" AND _exp_num LESS 100)
                        list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_exp}")
                    endif()
                endforeach()
            elseif(_arch STREQUAL "all-major")
                foreach(_exp IN LISTS CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR)
                    string(REGEX REPLACE "[-].*" "" _exp_num "${_exp}")
                    if(_exp_num MATCHES "^[0-9]+$" AND _exp_num LESS 100)
                        list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_exp}")
                    endif()
                endforeach()
            elseif(_arch STREQUAL "native")
                # Replace with explicit non-Blackwell GPU architectures.
                if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
                    foreach(_cap IN LISTS _GPU_COMPUTE_CAPS)
                        string(STRIP "${_cap}" _cap)
                        string(REGEX MATCH
                               "^([0-9]+)\.([0-9]+)" _match "${_cap}")
                        if(_match)
                            math(EXPR _sm
                                 "${CMAKE_MATCH_1} * 10 + ${CMAKE_MATCH_2}")
                            if(_sm LESS 100)
                                list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_sm}")
                            endif()
                        endif()
                    endforeach()
                endif()
            else()
                list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_arch}")
            endif()
        endforeach()

        # Add PTX for the highest pre-Blackwell arch so Blackwell GPUs
        # can JIT-compile at runtime.
        list(APPEND _COLMAP_MVS_CUDA_ARCHS "90-virtual")
        list(REMOVE_DUPLICATES _COLMAP_MVS_CUDA_ARCHS)

        set_target_properties(colmap_mvs_cuda PROPERTIES
            CUDA_ARCHITECTURES "${_COLMAP_MVS_CUDA_ARCHS}"
        )
    endif()

    if(TESTS_ENABLED)
        target_compile_definitions(colmap_mvs_cuda PRIVATE COLMAP_PATCH_MATCH_TESTING)
    endif()

    COLMAP_ADD_TEST(
        NAME patch_match_cuda_test
        SRCS patch_match_cuda_test.cc patch_match_cuda_test.cu patch_match_cuda_test.h
        LINK_LIBS colmap_mvs_cuda
    )

    COLMAP_ADD_TEST(
        NAME gpu_mat_test
        SRCS gpu_mat_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    COLMAP_ADD_TEST(
        NAME patch_match_sampling_test
        SRCS patch_match_sampling_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
endif()

# ROCm HIP support for MVS (patch_match_stereo on AMD GPUs).
#
# enable_language(HIP) and find_package(hip/hiprand/rocrand) are handled
# centrally in cmake/FindDependencies.cmake. Here we just mark the three
# .cu translation units as LANGUAGE HIP so CMake hands them to the HIP
# toolchain (clang++ -x hip) instead of the CUDA one. The source itself
# is unmodified; symbol remapping happens in src/colmap/util/cuda_to_hip.h.
if(HIP_ENABLED)
    COLMAP_ADD_LIBRARY(
        NAME colmap_mvs_cuda
        SRCS
            gpu_mat_prng.h gpu_mat_prng.cu
            gpu_mat_ref_image.h gpu_mat_ref_image.cu
            patch_match.h patch_match.cc
            patch_match_cuda.h patch_match_cuda.cu patch_match_geometry.cuh
            patch_match_sampling.cuh
            patch_match_rng_perimeter.h
        PUBLIC_LINK_LIBS
            colmap_mvs
            colmap_util_cuda
            hip::host
            hip::hiprand
            roc::rocrand
    )
    set_source_files_properties(
        gpu_mat_prng.cu
        gpu_mat_ref_image.cu
        patch_match_cuda.cu
        PROPERTIES LANGUAGE HIP
    )

    COLMAP_ADD_TEST(
        NAME gpu_mat_test
        SRCS gpu_mat_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    COLMAP_ADD_TEST(
        NAME patch_match_sampling_test
        SRCS patch_match_sampling_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    set_source_files_properties(
        gpu_mat_test.cu
        patch_match_sampling_test.cu
        patch_match_rng_perimeter_test.cu
        PROPERTIES LANGUAGE HIP
    )
endif()

# The ordinary production controls compile and register on both GPU backends.
if(CUDA_ENABLED OR HIP_ENABLED)
    COLMAP_ADD_TEST(
        NAME patch_match_rng_perimeter_test
        SRCS patch_match_rng_perimeter_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    COLMAP_ADD_TEST(
        NAME patch_match_probability_test
        SRCS patch_match_probability_test.cc
        LINK_LIBS colmap_mvs_cuda
    )

    # Keep all wrap flags and definitions local to this test executable.
    # COLMAP_ADD_TEST creates no target when TESTS_ENABLED is OFF.
    if(TESTS_ENABLED)
        get_target_property(_COLMAP_MVS_OWNER_TYPE colmap_mvs_cuda TYPE)
        if(CUDA_ENABLED AND NOT HIP_ENABLED AND UNIX AND NOT APPLE AND NOT WIN32
           AND _COLMAP_MVS_OWNER_TYPE STREQUAL "STATIC_LIBRARY"
           AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
            # Compile AND link the real CUDA symbol/forwarding contract; do not
            # run it or rely solely on acceptance of a linker option. Function
            # scope prevents changes to any other capability check or target.
            function(colmap_mvs_check_cuda_wrap)
                include(CheckCXXSourceCompiles)
                set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
                set(CMAKE_REQUIRED_LIBRARIES CUDA::cudart)
                set(CMAKE_REQUIRED_LINK_OPTIONS
                    "LINKER:--wrap=cudaMallocPitch"
                    "LINKER:--wrap=cudaFree")
                check_cxx_source_compiles([=[
                    #include <cuda_runtime_api.h>
                    extern "C" cudaError_t __real_cudaMallocPitch(
                        void**, size_t*, size_t, size_t);
                    extern "C" cudaError_t __real_cudaFree(void*);
                    extern "C" cudaError_t __wrap_cudaMallocPitch(
                        void** p, size_t* s, size_t w, size_t h) {
                        return __real_cudaMallocPitch(p, s, w, h);
                    }
                    extern "C" cudaError_t __wrap_cudaFree(void* p) {
                        return __real_cudaFree(p);
                    }
                    int main() {
                        void* p = nullptr;
                        size_t s = 0;
                        cudaError_t status = cudaMallocPitch(&p, &s, 1, 1);
                        return static_cast<int>(status == cudaSuccess
                            ? cudaFree(p) : status);
                    }
                ]=] _COLMAP_MVS_HAS_CUDA_WRAP)
                if(_COLMAP_MVS_HAS_CUDA_WRAP)
                    target_compile_definitions(colmap_mvs_patch_match_probability_test
                        PRIVATE COLMAP_MVS_CUDA_ALLOCATION_WRAP=1)
                    target_link_options(colmap_mvs_patch_match_probability_test PRIVATE
                        "LINKER:--wrap=cudaMallocPitch"
                        "LINKER:--wrap=cudaFree")
                endif()
            endfunction()
            colmap_mvs_check_cuda_wrap()
        endif()
    endif()
endif()
