set(SIFT_GPU_ENABLE_OPENCL FALSE)
set(SIFT_GPU_ENABLE_SSE TRUE)
set(SIFT_GPU_ENABLE_DEVIL FALSE)
set(SIFT_GPU_PREFER_GLUT TRUE)

if(NOT IS_MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    if(SIFT_GPU_ENABLE_SSE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core2 -mfpmath=sse")
    endif()
endif()

if(SIFT_GPU_PREFER_GLUT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWINDOW_PREFER_GLUT")
endif()

if(SIFT_GPU_ENABLE_OPENCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCL_SIFTGPU_ENABLED")
endif()

if(SIFT_GPU_ENABLE_DEVIL)
    set(SIFT_GPU_LIBRARIES ${SIFT_GPU_LIBRARIES}
        ${IL_LIBRARIES}
        ${ILU_LIBRARIES}
        ${ILUT_LIBRARIES})
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSIFTGPU_NO_DEVIL")
endif()

set(SIFT_GPU_SOURCE_FILES
    FrameBufferObject.cpp
    FrameBufferObject.h
    GlobalUtil.cpp
    GlobalUtil.h
    GLTexImage.cpp
    GLTexImage.h
    ProgramGLSL.cpp
    ProgramGLSL.h
    ProgramGPU.h
    PyramidGL.cpp
    PyramidGL.h
    ShaderMan.cpp
    ShaderMan.h
    SiftGPU.cpp
    SiftGPU.h
    SiftMatch.cpp
    SiftMatch.h
    SiftPyramid.cpp
    SiftPyramid.h
)

if(CUDA_ENABLED)
    add_definitions("-DCUDA_SIFTGPU_ENABLED")
    # Fix for Ubuntu 16.04.
    add_definitions("-D_FORCE_INLINES")

    set(SIFT_GPU_SOURCE_FILES
        ${SIFT_GPU_SOURCE_FILES}
        CuTexImage.cpp
        CuTexImage.h
        ProgramCU.cu
        ProgramCU.h
        PyramidCU.cpp
        PyramidCU.h
        SiftMatchCU.cpp
        SiftMatchCU.h
    )

    if(CUDA_VERSION_MAJOR LESS 7)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_12,code=sm_12)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
    endif()
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

    COLMAP_CUDA_ADD_LIBRARY(sift_gpu ${SIFT_GPU_SOURCE_FILES})
else()
    COLMAP_ADD_LIBRARY(sift_gpu ${SIFT_GPU_SOURCE_FILES})
endif()

target_link_libraries(sift_gpu
    ${SIFT_GPU_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLUT_LIBRARY}
    ${OPENGL_LIBRARIES}
)
