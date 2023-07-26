add_definitions("-DSIFTGPU_NO_DEVIL")

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
    PRIVATE_LINK_LIBS
        OpenGL::GL
        GLEW::GLEW
)

if(CUDA_ENABLED)
    add_definitions("-DCUDA_SIFTGPU_ENABLED")

    COLMAP_ADD_LIBRARY(
        NAME colmap_sift_gpu_cuda
        SRCS
            CuTexImage.h CuTexImage.cpp
            ProgramCU.cu
            ProgramCU.h
            PyramidCU.h PyramidCU.cpp
            SiftMatchCU.h SiftMatchCU.cpp
        PRIVATE_LINK_LIBS
            CUDA::cudart
            CUDA::curand
    )

    target_link_libraries(colmap_sift_gpu PRIVATE colmap_sift_gpu_cuda)
endif()
