add_definitions("-DSIFTGPU_NO_DEVIL")

if(SIMD_ENABLED AND IS_X86)
    add_definitions("-DUSE_SSE_FOR_SIFTGPU")
    if(MSVC)
        add_definitions("-D__SSE__")
    endif()
endif()

set(OPTIONAL_CUDA_SRCS)
set(OPTIONAL_CUDA_LINK_LIBS)
if(CUDA_ENABLED)
    add_definitions("-DCUDA_SIFTGPU_ENABLED")
    set(OPTIONAL_CUDA_SRCS
        CuTexImage.h CuTexImage.cpp
        ProgramCU.cu
        ProgramCU.h
        PyramidCU.h PyramidCU.cpp
        SiftMatchCU.h SiftMatchCU.cpp)
    set(OPTIONAL_CUDA_LINK_LIBS
        CUDA::cudart
        CUDA::curand
    )
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_sift_gpu
    SRCS
        FrameBufferObject.h FrameBufferObject.cpp
        GlobalUtil.h GlobalUtil.cpp
        GLTexImage.h GLTexImage.cpp
        ProgramGLSL.h ProgramGLSL.cpp
        ProgramGPU.h
        PyramidGL.h PyramidGL.cpp
        ShaderMan.h ShaderMan.cpp
        SiftGPU.h SiftGPU.cpp
        SiftMatch.h SiftMatch.cpp
        SiftPyramid.h SiftPyramid.cpp
        ${OPTIONAL_CUDA_SRCS}
    PRIVATE_LINK_LIBS
        OpenGL::GL
        GLEW::GLEW
        ${OPTIONAL_CUDA_LINK_LIBS}
)
