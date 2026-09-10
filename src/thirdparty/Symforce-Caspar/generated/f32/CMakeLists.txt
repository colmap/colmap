cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(USE_HIP)
  # project(... LANGUAGES ... HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise
  # auto-detects the host GPU and errors on a no-GPU build host.
  project(caspar_library LANGUAGES CXX HIP)

  find_package(hip REQUIRED)
  find_package(hipcub REQUIRED)
else()
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDACXX})
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler")
  endif()

  project(caspar_library LANGUAGES CXX CUDA)

  set(CASPAR_MIN_ARCH 75 CACHE INTERNAL "Minimum CUDA arch required by Caspar")

  # SASS for Turing/Ampere/Ada (sm_75–89) + PTX fallback for forward JIT-compatibility. sm_75 minimum required by cooperative_groups::reduce.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CASPAR_MIN_ARCH} 80-real 86-real 89-real)
  endif()

  find_package(CUDAToolkit REQUIRED)
endif()

file(GLOB CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu" "*.cc")
list(FILTER CUDA_SOURCES EXCLUDE REGEX "pybind.*")
file(GLOB CUDA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cuh" "*.h")
list(FILTER CUDA_HEADERS EXCLUDE REGEX "pybind.*")

add_library(caspar_lib_core STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})
target_include_directories(caspar_lib_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(USE_HIP)
  # Mark .cu files as HIP language
  set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
  target_compile_definitions(caspar_lib_core PUBLIC USE_HIP)
  # Link hip targets PRIVATE to avoid propagating HIP compile options to pure-CXX consumers.
  # The static library has the HIP code linked in; pybind consumers only need the includes.
  target_link_libraries(caspar_lib_core PRIVATE hip::host hip::hipcub)
  # Expose ROCm include directory for consumers that need hipError_t etc
  target_include_directories(caspar_lib_core PUBLIC ${hip_INCLUDE_DIRS})
  # Use -ffp-contract=fast -fno-math-errno instead of -ffast-math.
  # clang's -ffast-math enables -fassociative-math which can NaN backward passes.
  target_compile_options(caspar_lib_core PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=fast -fno-math-errno>
  )
  set_target_properties(caspar_lib_core PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  )
else()
  target_link_libraries(caspar_lib_core PUBLIC CUDA::cudart)
  target_compile_options(caspar_lib_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()

set_target_properties(caspar_lib_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

