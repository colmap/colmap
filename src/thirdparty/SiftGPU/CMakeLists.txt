add_compile_definitions(SIFTGPU_NO_DEVIL)

if(SIMD_ENABLED AND IS_X86)
    add_compile_definitions(SIFTGPU_USE_SSE_FOR)
    if(MSVC)
        add_compile_definitions(__SSE__)
    endif()
endif()

set(OPTIONAL_CUDA_SRCS)
set(OPTIONAL_CUDA_LINK_LIBS)
if(CUDA_ENABLED)
    add_compile_definitions(SIFTGPU_CUDA_ENABLED)
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

# SiftGPU's GPU backend is gated by SIFTGPU_CUDA_ENABLED. On HIP we also
# turn it on - the source is CUDA-shaped but compiles unmodified against
# cuda_to_hip.h on ROCm. The legacy cudaGL* OpenGL interop calls inside
# CuTexImage.cpp are guarded with !COLMAP_HIP_ENABLED.
set(OPTIONAL_HIP_SRCS)
set(OPTIONAL_HIP_LINK_LIBS)
if(HIP_ENABLED)
    add_compile_definitions(SIFTGPU_CUDA_ENABLED)
    set(OPTIONAL_HIP_SRCS
        CuTexImage.h CuTexImage.cpp
        ProgramCU.cu
        ProgramCU.h
        PyramidCU.h PyramidCU.cpp
        SiftMatchCU.h SiftMatchCU.cpp)
    set(OPTIONAL_HIP_LINK_LIBS
        colmap_util_cuda
        hip::host
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
        ${OPTIONAL_HIP_SRCS}
    PRIVATE_LINK_LIBS
        OpenGL::GL
        GLEW::GLEW
        ${OPTIONAL_CUDA_LINK_LIBS}
        ${OPTIONAL_HIP_LINK_LIBS}
)

if(HIP_ENABLED)
    set_source_files_properties(ProgramCU.cu PROPERTIES LANGUAGE HIP)
endif()
