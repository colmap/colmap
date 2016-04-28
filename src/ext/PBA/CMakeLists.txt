project(PBA)

if(NOT IS_MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -pthread -march=core2 -mfpmath=sse")
endif()

if(CUDA_ENABLED)
    # Fix for Ubuntu 16.04.
    add_definitions("-D_FORCE_INLINES")

    if(CUDA_VERSION_MAJOR LESS 7)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_12,code=sm_12)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
    endif()
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

    set(PBA_GPU_SOURCE_FILES
        ConfigBA.cpp
        ConfigBA.h
        CuTexImage.cpp
        CuTexImage.h
        DataInterface.h
        pba.cpp
        pba.h
        ProgramCU.cu
        ProgramCU.h
        SparseBundleCPU.cpp
        SparseBundleCPU.h
        SparseBundleCU.cpp
        SparseBundleCU.h
        util.h)
    COLMAP_CUDA_ADD_LIBRARY(pba ${PBA_GPU_SOURCE_FILES})
else()
    set(PBA_CPU_SOURCE_FILES
        ConfigBA.cpp
        ConfigBA.h
        DataInterface.h
        pba.cpp
        pba.h
        SparseBundleCPU.cpp
        SparseBundleCPU.h
        util.h)
    COLMAP_ADD_LIBRARY(pba ${PBA_CPU_SOURCE_FILES})
    set_target_properties(pba PROPERTIES COMPILE_FLAGS "-DPBA_NO_GPU")
endif()
