# SPDX-License-Identifier: BSD-3-Clause

# Run without building COLMAP:
# cmake -DTEST_BINARY_DIR=/tmp/colmap-onnx-test -P cmake/tests/FindonnxruntimeTest.cmake
cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED TEST_BINARY_DIR)
    message(FATAL_ERROR "Set TEST_BINARY_DIR to a scratch directory")
endif()
get_filename_component(MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(fallback_only_selections hints_lib64 partial_hints split_hints
    sibling_hints explicit_partial cuda_toggle)

foreach(mode config fallback)
    foreach(selection hints system missing_hints explicit
            ${fallback_only_selections})
        if(mode STREQUAL "config" AND selection IN_LIST fallback_only_selections)
            continue()
        endif()
        foreach(cuda_enabled ON OFF)
            if(mode STREQUAL "config" AND NOT cuda_enabled)
                continue()
            endif()
            set(case_dir "${TEST_BINARY_DIR}/${mode}_${selection}_${cuda_enabled}")
            # The cuda_toggle case reconfigures one build directory with CUDA
            # toggled, so both runs share a case directory.
            set(second_configure OFF)
            if(selection STREQUAL "cuda_toggle")
                set(case_dir "${TEST_BINARY_DIR}/${mode}_${selection}")
                if(NOT cuda_enabled)
                    set(second_configure ON)
                endif()
            endif()
            file(MAKE_DIRECTORY "${case_dir}/src")
            file(WRITE "${case_dir}/src/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.22)
project(FindonnxruntimeTest NONE)

# Isolate discovery from packages and registries on the machine running the test.
set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_PACKAGE_ROOT_PATH FALSE)
set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
# Replace system prefixes rather than disabling install-prefix searches.
# CMake records the original install-prefix occurrences during project().
# With this flag OFF, CMake 3.31 tries to remove those entries from our replaced
# list even though they are absent, which can discard the fixture prefix.
# Keeping the flag ON uses our replacement list without that removal step.
set(CMAKE_FIND_USE_INSTALL_PREFIX TRUE)
set(CMAKE_SYSTEM_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/system")
set(CMAKE_SYSTEM_INCLUDE_PATH "")
set(CMAKE_SYSTEM_LIBRARY_PATH "")
set(CMAKE_PREFIX_PATH "")
set(CMAKE_MODULE_PATH "${MODULE_DIR}")

# Re-running the test must perform discovery again rather than use its old cache,
# except on the second configure of the cuda_toggle case: the stale provider
# cache left by the first configure is what the module itself must handle.
if(NOT SECOND_CONFIGURE)
    foreach(var onnxruntime_DIR onnxruntime_INCLUDE_DIRS onnxruntime_LIBRARIES
            onnxruntime_PROVIDERS_SHARED_LIBRARY onnxruntime_PROVIDERS_CUDA_LIBRARY)
        unset(${var} CACHE)
    endforeach()
endif()

foreach(prefix system hinted explicit)
    set(library_dir lib)
    if(SELECTION STREQUAL "hints_lib64" AND prefix STREQUAL "hinted")
        set(library_dir lib64)
    endif()
    set(root "${CMAKE_CURRENT_BINARY_DIR}/${prefix}")
    file(MAKE_DIRECTORY "${root}/include/onnxruntime" "${root}/${library_dir}")
    file(WRITE "${root}/include/onnxruntime/onnxruntime_cxx_api.h" "")
    foreach(lib onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)
        # With partial hints, the hinted installation ships only the core
        # library; the provider libraries exist only in the system prefix.
        # With split hints, the provider libraries live in a second hinted
        # directory instead (created below).
        if((SELECTION STREQUAL "partial_hints" OR SELECTION STREQUAL "split_hints")
                AND prefix STREQUAL "hinted" AND NOT lib STREQUAL "onnxruntime")
            continue()
        endif()
        # With a partial explicit prefix, the explicit installation ships only
        # the core library; the hinted one must not supply its providers.
        if(SELECTION STREQUAL "explicit_partial" AND prefix STREQUAL "explicit"
                AND NOT lib STREQUAL "onnxruntime")
            continue()
        endif()
        # Discovery-only fixtures for Unix and Windows; no linker is invoked.
        file(WRITE "${root}/${library_dir}/lib${lib}.so" "")
        file(WRITE "${root}/${library_dir}/lib${lib}.dylib" "")
        file(WRITE "${root}/${library_dir}/${lib}.lib" "")
    endforeach()
    if(MODE STREQUAL "config")
        file(MAKE_DIRECTORY "${root}/share/onnxruntime/cmake")
        file(WRITE "${root}/share/onnxruntime/cmake/onnxruntimeConfig.cmake"
            "add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)\n"
            "set_property(TARGET onnxruntime::onnxruntime PROPERTY TEST_PREFIX ${prefix})\n")
    endif()
endforeach()
# A second hinted directory with only provider libraries.
if(SELECTION STREQUAL "split_hints" OR SELECTION STREQUAL "sibling_hints")
    set(root "${CMAKE_CURRENT_BINARY_DIR}/hinted2")
    file(MAKE_DIRECTORY "${root}/lib")
    foreach(lib onnxruntime_providers_shared onnxruntime_providers_cuda)
        file(WRITE "${root}/lib/lib${lib}.so" "")
        file(WRITE "${root}/lib/lib${lib}.dylib" "")
        file(WRITE "${root}/lib/${lib}.lib" "")
    endforeach()
endif()

set(expected system)
if(SELECTION STREQUAL "hints" OR SELECTION STREQUAL "explicit"
        OR SELECTION STREQUAL "hints_lib64" OR SELECTION STREQUAL "partial_hints"
        OR SELECTION STREQUAL "split_hints" OR SELECTION STREQUAL "sibling_hints"
        OR SELECTION STREQUAL "explicit_partial" OR SELECTION STREQUAL "cuda_toggle")
    set(hint "${CMAKE_CURRENT_BINARY_DIR}/hinted")
    set(expected hinted)
elseif(SELECTION STREQUAL "missing_hints")
    set(hint "${CMAKE_CURRENT_BINARY_DIR}/missing")
endif()
if(DEFINED hint)
    set(onnxruntime_CONFIG_DIR_HINTS "${hint}/share/onnxruntime/cmake")
    set(onnxruntime_INCLUDE_DIR_HINTS "${hint}/include")
    set(onnxruntime_LIBRARY_DIR_HINTS "${hint}/lib")
    if(SELECTION STREQUAL "hints_lib64")
        set(onnxruntime_LIBRARY_DIR_HINTS "${hint}/lib64")
    endif()
    if(SELECTION STREQUAL "split_hints")
        set(onnxruntime_LIBRARY_DIR_HINTS
            "${hint}/lib;${CMAKE_CURRENT_BINARY_DIR}/hinted2/lib")
    endif()
    # The providers-only directory comes first, but the providers beside the
    # core library must still win.
    if(SELECTION STREQUAL "sibling_hints")
        set(onnxruntime_LIBRARY_DIR_HINTS
            "${CMAKE_CURRENT_BINARY_DIR}/hinted2/lib;${hint}/lib")
    endif()
endif()
if(SELECTION STREQUAL "explicit" OR SELECTION STREQUAL "explicit_partial")
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/explicit")
    set(expected explicit)
endif()

find_package(onnxruntime REQUIRED MODULE)
if(MODE STREQUAL "config")
    get_target_property(actual onnxruntime::onnxruntime TEST_PREFIX)
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "Expected ${expected} config, got ${actual}")
    endif()
else()
    if(NOT onnxruntime_INCLUDE_DIRS STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/${expected}/include")
        message(FATAL_ERROR "Wrong headers: ${onnxruntime_INCLUDE_DIRS}")
    endif()
    list(LENGTH onnxruntime_LIBRARIES library_count)
    set(expected_library_count 2)
    if(CUDA_ENABLED)
        set(expected_library_count 3)
    endif()
    if(SELECTION STREQUAL "partial_hints" OR SELECTION STREQUAL "explicit_partial")
        # The providers must not be mixed in from another installation.
        set(expected_library_count 1)
    endif()
    if(NOT library_count EQUAL expected_library_count)
        message(FATAL_ERROR "Expected ${expected_library_count} libraries, got ${library_count}")
    endif()
    set(expected_library_dir lib)
    if(SELECTION STREQUAL "hints_lib64")
        set(expected_library_dir lib64)
    endif()
    foreach(lib IN LISTS onnxruntime_LIBRARIES)
        get_filename_component(dir "${lib}" DIRECTORY)
        get_filename_component(name "${lib}" NAME_WE)
        set(expected_dir "${CMAKE_CURRENT_BINARY_DIR}/${expected}/${expected_library_dir}")
        # With split hints, providers come from the second hint directory.
        if(SELECTION STREQUAL "split_hints" AND NOT name MATCHES "^(lib)?onnxruntime$")
            set(expected_dir "${CMAKE_CURRENT_BINARY_DIR}/hinted2/lib")
        endif()
        if(NOT dir STREQUAL expected_dir)
            message(FATAL_ERROR "Wrong library: ${lib}")
        endif()
    endforeach()
endif()
]=])
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -S "${case_dir}/src" -B "${case_dir}/build"
                    "-DMODULE_DIR=${MODULE_DIR}" "-DMODE=${mode}" "-DSELECTION=${selection}"
                    "-DCUDA_ENABLED=${cuda_enabled}" "-DSECOND_CONFIGURE=${second_configure}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
            if(NOT result EQUAL 0)
                message(FATAL_ERROR "${mode}/${selection}/CUDA=${cuda_enabled} failed:\n${output}\n${error}")
            endif()
            message(STATUS "Passed ${mode}/${selection}/CUDA=${cuda_enabled}")
        endforeach()
    endforeach()
endforeach()
