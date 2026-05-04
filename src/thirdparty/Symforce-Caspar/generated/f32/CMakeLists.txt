cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDACXX})
  set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler")
endif()

project(caspar_library LANGUAGES CXX CUDA)

# SASS for Turing/Ampere/Ada (sm_75–89) + PTX fallback for forward JIT-compatibility. sm_75 minimum required by cooperative_groups::reduce.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 75-virtual)
endif()

find_package(CUDAToolkit REQUIRED)

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

