# SPDX-License-Identifier: BSD-3-Clause

# Run without building COLMAP:
# cmake -DTEST_BINARY_DIR=/tmp/colmap-onnx-test -P cmake/tests/FindonnxruntimeTest.cmake
cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED TEST_BINARY_DIR)
    message(FATAL_ERROR "Set TEST_BINARY_DIR to a scratch directory")
endif()
get_filename_component(MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

foreach(mode config fallback)
    foreach(selection hints system missing_hints explicit)
        set(case_dir "${TEST_BINARY_DIR}/${mode}_${selection}")
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
set(CMAKE_FIND_USE_INSTALL_PREFIX FALSE)
set(CMAKE_SYSTEM_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/system")
set(CMAKE_PREFIX_PATH "")
set(CMAKE_MODULE_PATH "${MODULE_DIR}")
set(CUDA_ENABLED ON)

# Re-running the test must perform discovery again rather than use its old cache.
foreach(var onnxruntime_DIR onnxruntime_INCLUDE_DIRS onnxruntime_LIBRARIES
        onnxruntime_PROVIDERS_SHARED_LIBRARY onnxruntime_PROVIDERS_CUDA_LIBRARY)
    unset(${var} CACHE)
endforeach()

foreach(prefix system hinted explicit)
    set(root "${CMAKE_CURRENT_BINARY_DIR}/${prefix}")
    file(MAKE_DIRECTORY "${root}/include/onnxruntime" "${root}/lib")
    file(WRITE "${root}/include/onnxruntime/onnxruntime_cxx_api.h" "")
    foreach(lib onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)
        # Discovery-only fixtures for Unix and Windows; no linker is invoked.
        file(WRITE "${root}/lib/lib${lib}.so" "")
        file(WRITE "${root}/lib/lib${lib}.dylib" "")
        file(WRITE "${root}/lib/${lib}.lib" "")
    endforeach()
    if(MODE STREQUAL "config")
        file(MAKE_DIRECTORY "${root}/share/onnxruntime/cmake")
        file(WRITE "${root}/share/onnxruntime/cmake/onnxruntimeConfig.cmake"
            "add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)\n"
            "set_property(TARGET onnxruntime::onnxruntime PROPERTY TEST_PREFIX ${prefix})\n")
    endif()
endforeach()

set(expected system)
if(SELECTION STREQUAL "hints" OR SELECTION STREQUAL "explicit")
    set(hint "${CMAKE_CURRENT_BINARY_DIR}/hinted")
    set(expected hinted)
elseif(SELECTION STREQUAL "missing_hints")
    set(hint "${CMAKE_CURRENT_BINARY_DIR}/missing")
endif()
if(DEFINED hint)
    set(onnxruntime_CONFIG_DIR_HINTS "${hint}/share/onnxruntime/cmake")
    set(onnxruntime_INCLUDE_DIR_HINTS "${hint}/include")
    set(onnxruntime_LIBRARY_DIR_HINTS "${hint}/lib")
endif()
if(SELECTION STREQUAL "explicit")
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
    if(NOT library_count EQUAL 3)
        message(FATAL_ERROR "Expected runtime and both provider libraries")
    endif()
    foreach(lib IN LISTS onnxruntime_LIBRARIES)
        get_filename_component(dir "${lib}" DIRECTORY)
        if(NOT dir STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/${expected}/lib")
            message(FATAL_ERROR "Wrong library: ${lib}")
        endif()
    endforeach()
endif()
]=])
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -S "${case_dir}/src" -B "${case_dir}/build"
                "-DMODULE_DIR=${MODULE_DIR}" "-DMODE=${mode}" "-DSELECTION=${selection}"
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "${mode}/${selection} failed:\n${output}\n${error}")
        endif()
        message(STATUS "Passed ${mode}/${selection}")
    endforeach()
endforeach()
