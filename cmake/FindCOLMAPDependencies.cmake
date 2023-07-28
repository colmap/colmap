if(COLMAP_FIND_QUIETLY)
    set(COLMAP_FIND_TYPE QUIET)
else()
    set(COLMAP_FIND_TYPE REQUIRED)
endif()

find_package(Boost ${COLMAP_FIND_TYPE} COMPONENTS
             program_options
             filesystem
             graph
             system)

find_package(Eigen3 ${COLMAP_FIND_TYPE})

find_package(FreeImage ${COLMAP_FIND_TYPE})

find_package(FLANN ${COLMAP_FIND_TYPE})
find_package(LZ4 ${COLMAP_FIND_TYPE})

find_package(Metis ${COLMAP_FIND_TYPE})

find_package(Glog ${COLMAP_FIND_TYPE})

find_package(SQLite3 ${COLMAP_FIND_TYPE})

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL ${COLMAP_FIND_TYPE})

find_package(Glew ${COLMAP_FIND_TYPE})
if(NOT TARGET GLEW::GLEW)
    # vcpkg's glew CMake config defines an imported interface target.
    # Only define it here, if it doesn't already exist.
    add_library(GLEW::GLEW INTERFACE IMPORTED)
    target_include_directories(
        GLEW::GLEW INTERFACE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(
        GLEW::GLEW INTERFACE ${GLEW_LIBRARIES})
endif()

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

if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()

if(OPENMP_ENABLED AND OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DCOLMAP_OPENMP_ENABLED")
else()
    message(STATUS "Disabling OpenMP support")
endif()

if(CGAL_ENABLED)
    set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
    # We do not use CGAL data. This prevents an unnecessary warning by CMake.
    set(CGAL_DATA_DIR "unused")
    find_package(CGAL ${COLMAP_FIND_TYPE})
endif()

if(CGAL_FOUND)
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
        endif()
    else()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_FOUND ON)
            enable_language(CUDA)
        endif()
    endif()
endif()

if(CUDA_ENABLED AND CUDA_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(
            FATAL_ERROR "You must set CMAKE_CUDA_ARCHITECTURES to e.g. 'native', 'all-major', '70', etc. "
            "More information at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html")
    endif()

    add_definitions("-DCOLMAP_CUDA_ENABLED")

    # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
    # Do not show warnings if the architectures are deprecated.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
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

if(GUI_ENABLED)
    find_package(Qt5 5.4 ${COLMAP_FIND_TYPE} COMPONENTS Core OpenGL Widgets)
    message(STATUS "Found Qt")
    message(STATUS "  Module : ${Qt5Core_DIR}")
    message(STATUS "  Module : ${Qt5OpenGL_DIR}")
    message(STATUS "  Module : ${Qt5Widgets_DIR}")
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

        # Enable automatic compilation of Qt resource files.
        set(CMAKE_AUTORCC ON)
    endif()
endif()

if(GUI_ENABLED AND Qt5_FOUND)
    add_definitions("-DCOLMAP_GUI_ENABLED")
    message(STATUS "Enabling GUI support")
else()
    message(STATUS "Disabling GUI support")
endif()

if(OPENGL_ENABLED)
    if(NOT GUI_ENABLED)
        message(STATUS "Disabling GUI also disables OpenGL")
        set(OPENGL_ENABLED OFF)
    else()
        add_definitions("-DCOLMAP_OPENGL_ENABLED")
        message(STATUS "Enabling OpenGL support")
    endif()
else()
    message(STATUS "Disabling OpenGL support")
endif()
