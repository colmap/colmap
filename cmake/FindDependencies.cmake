if(COLMAP_FIND_QUIETLY)
    set(COLMAP_FIND_TYPE QUIET)
else()
    set(COLMAP_FIND_TYPE REQUIRED)
endif()

# Track all the compile definitions
set(COLMAP_COMPILE_DEFINITIONS)
if(LSD_ENABLED)
    list(APPEND COLMAP_COMPILE_DEFINITIONS COLMAP_LSD_ENABLED)
    message(STATUS "Enabling LSD support")
else()
    message(STATUS "Disabling LSD support")
endif()

find_package(OpenMP REQUIRED COMPONENTS C CXX)

find_package(Boost ${COLMAP_FIND_TYPE} COMPONENTS
             graph
             program_options
             OPTIONAL_COMPONENTS
             system)

find_package(Eigen3 ${COLMAP_FIND_TYPE})

find_package(FreeImage ${COLMAP_FIND_TYPE})

find_package(Metis ${COLMAP_FIND_TYPE})

find_package(Glog ${COLMAP_FIND_TYPE})
if(DEFINED glog_VERSION_MAJOR)
  # Older versions of glog don't export version variables.
  list(APPEND COLMAP_COMPILE_DEFINITIONS GLOG_VERSION_MAJOR=${glog_VERSION_MAJOR})
  list(APPEND COLMAP_COMPILE_DEFINITIONS GLOG_VERSION_MINOR=${glog_VERSION_MINOR})
endif()

find_package(SQLite3 ${COLMAP_FIND_TYPE})

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL ${COLMAP_FIND_TYPE})

find_package(Glew ${COLMAP_FIND_TYPE})

find_package(Git)

find_package(Ceres ${COLMAP_FIND_TYPE})
if(NOT TARGET Ceres::ceres)
    # Older Ceres versions don't come with an imported interface target.
    add_library(Ceres::ceres INTERFACE IMPORTED)
    target_include_directories(
        Ceres::ceres INTERFACE ${CERES_INCLUDE_DIRS})
    target_link_libraries(
        Ceres::ceres INTERFACE ${CERES_LIBRARIES})
endif()

if(TESTS_ENABLED)
    find_package(GTest ${COLMAP_FIND_TYPE})
endif()

if(CGAL_ENABLED)
    set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
    # We do not use CGAL data. This prevents an unnecessary warning by CMake.
    set(CGAL_DATA_DIR "unused")
    find_package(CGAL ${COLMAP_FIND_TYPE})
endif()

if(CGAL_FOUND)
    list(APPEND COLMAP_COMPILE_DEFINITIONS COLMAP_CGAL_ENABLED)
    list(APPEND CGAL_LIBRARY ${CGAL_LIBRARIES})
    message(STATUS "Found CGAL")
    message(STATUS "  Includes : ${CGAL_INCLUDE_DIRS}")
    message(STATUS "  Libraries : ${CGAL_LIBRARY}")
    if(NOT TARGET CGAL)
        # Older CGAL versions don't come with an imported interface target.
        add_library(CGAL INTERFACE IMPORTED)
        target_include_directories(
            CGAL INTERFACE ${CGAL_INCLUDE_DIRS} ${GMP_INCLUDE_DIR})
        target_link_libraries(
            CGAL INTERFACE ${CGAL_LIBRARY} ${GMP_LIBRARIES})
    endif()
    list(APPEND COLMAP_LINK_DIRS ${CGAL_LIBRARIES_DIR})
endif()

if(DOWNLOAD_ENABLED)
    # The OpenSSL package in vcpkg seems broken under Windows and leads to
    # missing certificate verification when connecting to SSL servers. We
    # therefore use curl[schannel] (i.e., native Windows SSL/TLS) under Windows
    # and curl[openssl] otherwise.
    find_package(CURL QUIET)
    set(CRYPTO_FOUND FALSE)
    if(IS_MSVC AND IS_ARM64)
        # OpenSSL crashes for ARM64 under Windows. We therefore fall back to
        # CryptoPP as an alternative to OpenSSL for SHA256 computation.
        find_package(CryptoPP QUIET)
        if(CryptoPP_FOUND)
            set(CRYPTO_FOUND TRUE)
        else()
            message(STATUS "CryptoPP not found")
        endif()
    else()
        find_package(OpenSSL QUIET COMPONENTS Crypto)
        if(OpenSSL_FOUND)
            set(CRYPTO_FOUND TRUE)
        else()
            message(STATUS "OpenSSL::Crypto not found")
        endif()
    endif()
    if(CURL_FOUND AND CRYPTO_FOUND)
        message(STATUS "Enabling download support")
        list(APPEND COLMAP_COMPILE_DEFINITIONS COLMAP_DOWNLOAD_ENABLED)
    else()
        set(DOWNLOAD_ENABLED OFF)
        message(STATUS "Disabling download support (Curl/Crypto not found)")
    endif()
else()
    message(STATUS "Disabling download support")
endif()

if(NOT FETCH_POSELIB)
    find_package(PoseLib ${COLMAP_FIND_TYPE})
endif()

if(NOT FETCH_FAISS)
    find_package(faiss ${COLMAP_FIND_TYPE})
endif()

set(COLMAP_LINK_DIRS ${Boost_LIBRARY_DIRS})

set(CUDA_MIN_VERSION "7.0")
if(CUDA_ENABLED)
    if(CMAKE_VERSION VERSION_LESS 3.17)
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "Found CUDA version ${CUDA_VERSION} installed in "
                    "${CUDA_TOOLKIT_ROOT_DIR} via legacy CMake (<3.17) module. "
                    "Using the legacy CMake module means that any installation of "
                    "COLMAP will require that the CUDA libraries are "
                    "available under LD_LIBRARY_PATH.")
            message(STATUS "Found CUDA ")
            message(STATUS "  Includes : ${CUDA_INCLUDE_DIRS}")
            message(STATUS "  Libraries : ${CUDA_LIBRARIES}")

            enable_language(CUDA)

            macro(declare_imported_cuda_target module)
                add_library(CUDA::${module} INTERFACE IMPORTED)
                target_include_directories(
                    CUDA::${module} INTERFACE ${CUDA_INCLUDE_DIRS})
                target_link_libraries(
                    CUDA::${module} INTERFACE ${CUDA_${module}_LIBRARY} ${ARGN})
            endmacro()

            declare_imported_cuda_target(cudart ${CUDA_LIBRARIES})
            declare_imported_cuda_target(curand ${CUDA_LIBRARIES})

            set(CUDAToolkit_VERSION "${CUDA_VERSION_STRING}")
            set(CUDAToolkit_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
        else()
            message(STATUS "Disabling CUDA support (not found)")
        endif()
    else()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_FOUND ON)
            enable_language(CUDA)
        else()
            message(STATUS "Disabling CUDA support (not found)")
        endif()
    endif()
endif()

if(CUDA_ENABLED AND CUDA_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    list(APPEND COLMAP_COMPILE_DEFINITIONS COLMAP_CUDA_ENABLED)

    # Do not show warnings if the architectures are deprecated.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    # Suppress warnings related to Eigen:
    # Calling a constexpr __host__ function from a __host__ __device__ function is not allowed.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    # Explicitly set PIC flags for CUDA targets.
    if(NOT IS_MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    endif()

    message(STATUS "Enabling CUDA support (version: ${CUDAToolkit_VERSION}, "
                    "archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(CUDA_ENABLED OFF)
    message(STATUS "Disabling CUDA support")
endif()

if(ONNX_ENABLED)
    if(FETCH_ONNX AND IS_WINDOWS)
        message(WARNING "Fetching ONNX is not supported under Windows, "
            "because the provided binary onnxruntime package does not ship with CMake configs. "
            "Assuming it was installed manually, e.g., using vcpkg's onnxruntime-gpu package.")
        set(FETCH_ONNX OFF)
    endif()
    if(FETCH_ONNX)
        include(FetchContent)

        set(ONNX_VERSION "1.22.0")
        if(IS_MACOS)
            FetchContent_Declare(onnxruntime
                URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-arm64-${ONNX_VERSION}.tgz
                URL_HASH SHA256=cab6dcbd77e7ec775390e7b73a8939d45fec3379b017c7cb74f5b204c1a1cc07
                ${_fetch_content_declare_args}
            )
        elseif(IS_LINUX)
            FetchContent_Declare(onnxruntime
                URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz
                URL_HASH SHA256=2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55
                ${_fetch_content_declare_args}
            )
        endif()

        message(STATUS "Configuring onnxruntime...")
        FetchContent_MakeAvailable(onnxruntime)
        message(STATUS "Configuring onnxruntime... done")

        if(IS_LINUX)
            set(ONNX_LIB_DIR_NAME lib64)
        else()
            set(ONNX_LIB_DIR_NAME lib)
        endif()

        set(ONNX_INCLUDE_DIR ${onnxruntime_BINARY_DIR}/include/onnxruntime)
        if(NOT EXISTS ${ONNX_INCLUDE_DIR})
            file(MAKE_DIRECTORY ${ONNX_INCLUDE_DIR})
            file(COPY ${onnxruntime_SOURCE_DIR}/include/ DESTINATION ${ONNX_INCLUDE_DIR}/)
        endif()
        set(ONNX_LIB_DIR ${onnxruntime_BINARY_DIR}/${ONNX_LIB_DIR_NAME})
        if(NOT EXISTS ${ONNX_LIB_DIR})
            file(MAKE_DIRECTORY ${ONNX_LIB_DIR})
            file(COPY ${onnxruntime_SOURCE_DIR}/lib/ DESTINATION ${ONNX_LIB_DIR}/)
            file(REMOVE_RECURSE ${ONNX_LIB_DIR}/cmake)
        endif()
        set(ONNX_DATA_DIR ${onnxruntime_BINARY_DIR}/share/onnxruntime)
        if(NOT EXISTS ${ONNX_DATA_DIR})
            file(MAKE_DIRECTORY ${ONNX_DATA_DIR})
            file(COPY ${onnxruntime_SOURCE_DIR}/lib/cmake/onnxruntime/ DESTINATION ${ONNX_DATA_DIR}/cmake/)
            file(REMOVE_RECURSE ${onnxruntime_SOURCE_DIR}/lib/cmake)
        endif()

        find_package(onnxruntime REQUIRED PATHS ${ONNX_DATA_DIR}/cmake NO_DEFAULT_PATH)

        install(
            DIRECTORY "${onnxruntime_BINARY_DIR}/include/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
        install(
            DIRECTORY "${onnxruntime_BINARY_DIR}/${ONNX_LIB_DIR_NAME}/"
            DESTINATION "${ONNX_LIB_DIR_NAME}")
        install(
            DIRECTORY "${onnxruntime_BINARY_DIR}/share/"
            DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}")
    else()
        find_package(onnxruntime ${COLMAP_FIND_TYPE})
        if(NOT onnxruntime_FOUND)
            message(STATUS "Disabling ONNX support (not found)")
        endif()
    endif()
else()
    message(STATUS "Disabling ONNX support")
endif()

if(TARGET onnxruntime::onnxruntime)
    list(APPEND COLMAP_COMPILE_DEFINITIONS COLMAP_ONNX_ENABLED)
    message(STATUS "Enabling ONNX support")
endif()

if(GUI_ENABLED)
    find_package(QT NAMES Qt5 Qt6 REQUIRED)
    set(COLMAP_QT_COMPONENTS Core OpenGL Widgets)
    if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
        list(APPEND COLMAP_QT_COMPONENTS OpenGLWidgets)
    endif()
    find_package(Qt${QT_VERSION_MAJOR} ${COLMAP_FIND_TYPE} ${COLMAP_QT_COMPONENTS})
    message(STATUS "Found Qt")
    message(STATUS "  Module : ${Qt${QT_VERSION_MAJOR}Core_DIR}")
    message(STATUS "  Module : ${Qt${QT_VERSION_MAJOR}OpenGL_DIR}")
    message(STATUS "  Module : ${Qt${QT_VERSION_MAJOR}Widgets_DIR}")
    if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
        message(STATUS "  Module : ${Qt${QT_VERSION_MAJOR}OpenGLWidgets_DIR}")
    endif()
    if(Qt5_FOUND)
        # Qt5 was built with -reduce-relocations.
        if(Qt5_POSITION_INDEPENDENT_CODE)
            set(CMAKE_POSITION_INDEPENDENT_CODE ON)
            # Workaround for Qt5 CMake config bug under Ubuntu 20.04: https://gitlab.kitware.com/cmake/cmake/-/issues/16915
            if(TARGET Qt5::Core)
                get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
                string(REPLACE "-fPIC" "" new_qt5_core_options "${core_options}")
                set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_qt5_core_options})
                set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
                if(NOT IS_MSVC)
                    set(CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC")
                endif()
            endif()
        endif()
    endif()

    if(QT_FOUND)
        # Enable automatic compilation of Qt resource files.
        set(CMAKE_AUTORCC ON)
    endif()
endif()

if(GUI_ENABLED AND Qt${QT_VERSION_MAJOR}_FOUND)
    list(APPEND COLMAP_COMPILE_DEFINITIONS COLMAP_GUI_ENABLED)
    message(STATUS "Enabling GUI support")
else()
    set(GUI_ENABLED OFF)
    message(STATUS "Disabling GUI support")
endif()

if(OPENGL_ENABLED)
    if(NOT GUI_ENABLED)
        message(STATUS "Disabling GUI also disables OpenGL")
        set(OPENGL_ENABLED OFF)
    else()
        message(STATUS "Enabling OpenGL support")
    endif()
else()
    message(STATUS "Disabling OpenGL support")
endif()

set(GPU_ENABLED OFF)
if(OPENGL_ENABLED OR CUDA_ENABLED)
    list(APPEND COLMAP_COMPILE_DEFINITIONS COLMAP_GPU_ENABLED)
    message(STATUS "Enabling GPU support (OpenGL: ${OPENGL_ENABLED}, CUDA: ${CUDA_ENABLED})")
    set(GPU_ENABLED ON)
endif()
