# SPDX-License-Identifier: BSD-3-Clause

if(IS_MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/W3>)
    if(WERROR_ENABLED)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/WX>)
    endif()
    # Avoid pulling in too many header files through <windows.h>
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
elseif(IS_GNU OR IS_CLANG)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wall>)
    if(WERROR_ENABLED)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
    endif()
    if(IS_GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15 AND
       CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
        # GCC 15 emits false positives in Eigen 3.4 and libstdc++ internals.
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=array-bounds>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=maybe-uninitialized>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=stringop-overflow>)
    endif()
endif()

if(CUDA_ENABLED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

    # Use a separate stream per thread to allow for concurrent kernel execution
    # between multiple threads on the same device.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

    # Suppress warnings:
    # ptxas warning : Stack size for entry function X cannot be statically determined.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-suppress-stack-size-warning")
endif()

if(COVERAGE_ENABLED)
    add_compile_options(-coverage -fprofile-update=atomic)
    add_link_options(-coverage)
endif()

add_subdirectory(controllers)
add_subdirectory(estimators)
add_subdirectory(exe)
add_subdirectory(feature)
add_subdirectory(geometry)
add_subdirectory(image)
add_subdirectory(math)
if(MVS_ENABLED)
    add_subdirectory(mvs)
endif()
add_subdirectory(optim)
add_subdirectory(retrieval)
add_subdirectory(scene)
add_subdirectory(sensor)
add_subdirectory(sfm)
add_subdirectory(tools)
add_subdirectory(util)
if (GUI_ENABLED)
    add_subdirectory(ui)
endif()
