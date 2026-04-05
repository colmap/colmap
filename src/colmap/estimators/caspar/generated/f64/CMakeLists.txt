cmake_minimum_required(VERSION 3.16)

project(caspar_library LANGUAGES CXX)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(CUDA REQUIRED)
include(FetchContent)
find_package(pybind11 2.13.6 QUIET)
if (NOT pybind11_FOUND)
  message(STATUS "pybind11 not found, adding with FetchContent")
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.13.6.zip
    URL_HASH SHA256=d0a116e91f64a4a2d8fb7590c34242df92258a61ec644b79127951e821b47be6
  )
  FetchContent_MakeAvailable(pybind11)
else()
  message(STATUS "pybind11 found")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math -arch=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")

file(GLOB CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu" "*.cc")
list(FILTER CUDA_SOURCES EXCLUDE REGEX "pybind.*")
file(GLOB CUDA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cuh" "*.h")
list(FILTER CUDA_HEADERS EXCLUDE REGEX "pybind.*")

cuda_add_library(caspar_lib_core STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})
target_include_directories(caspar_lib_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(caspar_lib_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

file(GLOB CPP_SOURCES "pybind*.cc")
file(GLOB CPP_HEADERS "*.h")

pybind11_add_module(caspar_lib ${CPP_SOURCES} ${CPP_HEADERS})
target_link_libraries(caspar_lib PRIVATE caspar_lib_core)

set_target_properties(caspar_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)