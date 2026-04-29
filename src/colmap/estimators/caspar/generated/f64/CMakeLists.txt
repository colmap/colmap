cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDACXX})
  set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler")
endif()

project(caspar_library LANGUAGES CXX CUDA)

# SASS for Turing/Ampere/Ada (sm_75–89) + PTX fallback for forward JIT-compatibility. sm_75 minimum required by cooperative_groups::reduce.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 75-virtual)
  endif()
endif()

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
find_package(pybind11 2.13.6 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found, adding with FetchContent")
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.13.6.zip
    URL_HASH SHA256=d0a116e91f64a4a2d8fb7590c34242df92258a61ec644b79127951e821b47be6
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(pybind11)
else()
  message(STATUS "pybind11 found")
endif()

file(GLOB CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu" "*.cc")
list(FILTER CUDA_SOURCES EXCLUDE REGEX "pybind.*")
file(GLOB CUDA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cuh" "*.h")
list(FILTER CUDA_HEADERS EXCLUDE REGEX "pybind.*")

add_library(caspar_lib_core STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})
target_include_directories(caspar_lib_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(caspar_lib_core PUBLIC CUDA::cudart)
target_compile_options(caspar_lib_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
set_target_properties(caspar_lib_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

file(GLOB CPP_SOURCES "pybind*.cc")
file(GLOB CPP_HEADERS "*.h")

pybind11_add_module(caspar_lib ${CPP_SOURCES} ${CPP_HEADERS})
target_link_libraries(caspar_lib PRIVATE caspar_lib_core)

set_target_properties(caspar_lib PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)