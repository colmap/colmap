# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


set(FOLDER_NAME "mvs")

COLMAP_ADD_LIBRARY(
    NAME colmap_mvs
    SRCS
        consistency_graph.h consistency_graph.cc
        depth_map.h depth_map.cc
        fusion.h fusion.cc
        image.h image.cc
        mat.h mat.cc
        mesh_simplification.h mesh_simplification.cc
        meshing.h meshing.cc
        model.h model.cc
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
    NAME meshing_test
    SRCS meshing_test.cc
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
            patch_match_cuda.h patch_match_cuda.cu
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
                               "^([0-9]+)\\.([0-9]+)" _match "${_cap}")
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

    COLMAP_ADD_TEST(
        NAME gpu_mat_test
        SRCS gpu_mat_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
endif()
