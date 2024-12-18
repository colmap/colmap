if(FETCH_LIBTORCH)
    include(FetchContent)
    # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
        cmake_policy(SET CMP0135 NEW)
    endif()
    set(LIBTORCH_VERSION "2.5.1")
    set(LIBTORCH_CUDA_VERSION "124")
    if(IS_MSVC)
        if(CUDA_ENABLED)
            FetchContent_Declare(libtorch
                URL https://download.pytorch.org/libtorch/cu${LIBTORCH_CUDA_VERSION}/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${LIBTORCH_CUDA_VERSION}.zip
                URL_HASH SHA256=8625bfd95ac62f260f114ac606d7f9d5acbf72855429dc085c2bc1e0e76fc0cf
            )
        else()
            FetchContent_Declare(libtorch
                URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip
                URL_HASH SHA256=176f3c501e50694cec2f23ca512ca36c0c268f523e90d91c8fca57bed56a6a65
            )
        endif()
    elseif(IS_MACOS)
        if(NOT IS_ARM)
            message(FATAL_ERROR "libtorch only supported with ARM-based Macs")
        endif()
        FetchContent_Declare(libtorch
            URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip
            URL_HASH SHA256=0822824c1df267159a649ad4701518217c60b8f75de056a26ab43958d8ab1622
        )
    elseif(IS_LINUX)
        if(NOT IS_X86)
            message(FATAL_ERROR "libtorch only supported with x86-based Linux")
        endif()
        if(CUDA_ENABLED)
            FetchContent_Declare(libtorch
                URL https://download.pytorch.org/libtorch/cu${LIBTORCH_CUDA_VERSION}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${LIBTORCH_CUDA_VERSION}.zip
                URL_HASH SHA256=470ab7f7f56e96d28d1dc9ae34ceb2e0d8723cc2899c5d0192f4cb12b8f7843b
            )
        else()
            FetchContent_Declare(libtorch
                URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip
                URL_HASH SHA256=618ca54eef82a1dca46ff1993d5807d9c0deb0bae147da4974166a147cb562fa
            )
        endif()
    endif()
    # find_package() targets are only visible in the current folder, so we need
    # to call it here. Once we require CMake 3.24, we can use the
    # find_package(GLOBAL) option and move this logic to the thirdparty folder.
    FetchContent_MakeAvailable(libtorch)
    if(NOT DEFINED MKL_ROOT)
        cmake_policy(SET CMP0074 NEW)
        if(IS_LINUX)
            set(DEBIAN_DEFAULT_MKL_LIB_PATH "/usr/lib/x86_64-linux-gnu/libmkl_core.so")
            if(EXISTS "${DEBIAN_DEFAULT_MKL_LIB_PATH}")
                set(MKL_INCLUDE_DIR "/usr/include")
                set(MKL_LIBRARIES "${DEBIAN_DEFAULT_MKL_LIB_PATH}")
            else()
                set(MKL_ROOT "/opt/intel/mkl")
            endif()
        elseif(IS_MSVC)
            set(DEFAULT_MKL_LIB_PATH "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64/mkl_core_dll.lib")
            if(EXISTS "${DEFAULT_MKL_LIB_PATH}")
                set(MKL_INCLUDE_DIR "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include")
                set(MKL_LIBRARIES "${DEFAULT_MKL_LIB_PATH}")
            endif()
        endif()
    endif()
    find_package(Torch REQUIRED PATHS "${libtorch_SOURCE_DIR}" NO_DEFAULT_PATH)
else()
    find_package(Torch REQUIRED)
endif()
