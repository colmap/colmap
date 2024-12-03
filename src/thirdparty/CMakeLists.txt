set(FOLDER_NAME "thirdparty")

# Only show moderate warnings for external library code.
if(IS_MSVC)
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(REMOVED_WARNING_LEVEL TRUE)
    elseif(CMAKE_CXX_FLAGS MATCHES "/Wall")
        string(REGEX REPLACE "/Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(REMOVED_WARNING_LEVEL TRUE)
    endif()
    if(CMAKE_C_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        set(REMOVED_WARNING_LEVEL TRUE)
    elseif(CMAKE_C_FLAGS MATCHES "/Wall")
        string(REGEX REPLACE "/Wall" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        set(REMOVED_WARNING_LEVEL TRUE)
    endif()
elseif(IS_GNU OR IS_CLANG)
    if(CMAKE_CXX_FLAGS MATCHES "-Wall")
        string(REGEX REPLACE "-Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(REMOVED_WARNING_LEVEL TRUE)
    endif()
    if(CMAKE_C_FLAGS MATCHES "-Wall")
        string(REGEX REPLACE "-Wall" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        set(REMOVED_WARNING_LEVEL TRUE)
    endif()
endif()

if(IS_MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
elseif(IS_GNU OR IS_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

set(CLANG_TIDY_EXE_TEMP ${CLANG_TIDY_EXE})
set(CLANG_TIDY_EXE "")
if(LSD_ENABLED)
    add_subdirectory(LSD)
endif()
add_subdirectory(PoissonRecon)
if(GPU_ENABLED)
    add_subdirectory(SiftGPU)
endif()
add_subdirectory(VLFeat)

if(FETCH_POSELIB)
    include(FetchContent)
    # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
        cmake_policy(SET CMP0135 NEW)
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
        set(_fetch_content_declare_args "SYSTEM" "EXCLUDE_FROM_ALL")
    endif()
    FetchContent_Declare(poselib
        URL https://github.com/PoseLib/PoseLib/archive/8028473d92c9347794a0e3d3541863b5cbb15743.zip
        URL_HASH SHA256=20ff75d14c839dcab3d38fb51503c47088898f394b812472a758ec17720a54c2
        ${_fetch_content_declare_args}
    )
    message(STATUS "Configuring PoseLib...")
    set(MARCH_NATIVE OFF CACHE BOOL "")
    FetchContent_MakeAvailable(poselib)
    message(STATUS "Configuring PoseLib... done")
endif()

if(FETCH_LIBTORCH)
    include(FetchContent)
    # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
        cmake_policy(SET CMP0135 NEW)
    endif()
    # TODO: Specialize for CUDA vs. non-CUDA and for x86 vs. ARM.
    set(LIBTORCH_VERSION "2.5.1")
    if(IS_MSVC)
        FetchContent_Declare(libtorch
            URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip
            URL_HASH SHA256=176f3c501e50694cec2f23ca512ca36c0c268f523e90d91c8fca57bed56a6a65
        )
    elseif(IS_MACOS)
        FetchContent_Declare(libtorch
            URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip
            URL_HASH SHA256=0822824c1df267159a649ad4701518217c60b8f75de056a26ab43958d8ab1622
        )
    elseif(IS_LINUX)
        FetchContent_Declare(libtorch
            URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip
            URL_HASH SHA256=618ca54eef82a1dca46ff1993d5807d9c0deb0bae147da4974166a147cb562fa
        )
    endif()
    FetchContent_MakeAvailable(libtorch)
    list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
    find_package(Torch REQUIRED)
    add_library(Torch INTERFACE)
    target_include_directories(Torch INTERFACE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(Torch INTERFACE ${TORCH_LIBRARIES})
    target_compile_options(Torch INTERFACE ${TORCH_CXX_FLAGS})
    add_library(Torch::Torch ALIAS Torch)
endif()

set(CLANG_TIDY_EXE ${CLANG_TIDY_EXE_TEMP})
