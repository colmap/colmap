add_compile_definitions(SIFTGPU_NO_DEVIL)

if(SIMD_ENABLED AND IS_X86)
    add_compile_definitions(SIFTGPU_USE_SSE_FOR)
    if(MSVC)
        add_compile_definitions(__SSE__)
    endif()
endif()

set(OPTIONAL_GPU_SRCS)
set(OPTIONAL_GPU_LINK_LIBS)
if(CUDA_ENABLED OR HIP_ENABLED)
    # The compute backend sources are written in the CUDA spelling for both
    # toolchains; on ROCm they are translated by colmap/util/cuda_to_hip.h.
    add_compile_definitions(SIFTGPU_CUDA_ENABLED)
    set(OPTIONAL_GPU_SRCS
        CuTexImage.h CuTexImage.cpp
        ProgramCU.cu
        ProgramCU.h
        PyramidCU.h PyramidCU.cpp
        SiftMatchCU.h SiftMatchCU.cpp)
endif()
if(CUDA_ENABLED)
    set(OPTIONAL_GPU_LINK_LIBS
        CUDA::cudart
        CUDA::curand
    )
elseif(HIP_ENABLED)
    set_source_files_properties(ProgramCU.cu PROPERTIES LANGUAGE HIP)
    set(OPTIONAL_GPU_LINK_LIBS
        hip::host
        hip::hiprand
        roc::rocrand
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
        ${OPTIONAL_GPU_SRCS}
    PRIVATE_LINK_LIBS
        OpenGL::GL
        GLEW::GLEW
        ${OPTIONAL_GPU_LINK_LIBS}
)
