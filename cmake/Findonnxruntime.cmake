# SPDX-License-Identifier: BSD-3-Clause

# Find package module for onnxruntime library.
#
# The following variables are set by this module:
#
#   onnxruntime_FOUND: TRUE if onnxruntime is found.
#   onnxruntime::onnxruntime: Imported target to link against.
#
# The following variables control the behavior of this module:
#
# onnxruntime_CONFIG_DIR_HINTS: List of additional directories in which to
#                         search for onnxruntime CMake configs.
# onnxruntime_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for onnxruntime includes.
# onnxruntime_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for onnxruntime libraries.
#
# Hint directories take precedence over system installations, while explicit
# CMake package/prefix overrides (onnxruntime_DIR, CMAKE_PREFIX_PATH,
# <PackageName>_ROOT) are honored first. A stale onnxruntime_DIR cache entry
# from a previous configuration therefore still wins over the hints and must
# be cleared, or a fresh build directory must be used.
#
# The provider library cache entries (onnxruntime_PROVIDERS_SHARED_LIBRARY,
# onnxruntime_PROVIDERS_CUDA_LIBRARY) are recomputed on every configure so they
# always match the selected core library; they cannot be overridden manually.

set(onnxruntime_CONFIG_DIR_HINTS "" CACHE PATH "onnxruntime config directory")
set(onnxruntime_INCLUDE_DIR_HINTS "" CACHE PATH "onnxruntime include directory")
set(onnxruntime_LIBRARY_DIR_HINTS "" CACHE PATH "onnxruntime library directory")

unset(onnxruntime_FOUND)
unset(onnxruntime_INCLUDE_DIRS)
unset(onnxruntime_LIBRARIES)

# Computed hints (including the fetched runtime) take precedence over system
# installations, while explicit CMake package/prefix overrides remain honored.
find_package(onnxruntime CONFIG QUIET HINTS ${onnxruntime_CONFIG_DIR_HINTS})
if(TARGET onnxruntime::onnxruntime)
    set(onnxruntime_FOUND TRUE)
    message(STATUS "Found onnxruntime")
    message(STATUS "  Target : onnxruntime::onnxruntime")
else()
    find_path(onnxruntime_INCLUDE_DIRS
        NAMES
        onnxruntime/onnxruntime_cxx_api.h
        HINTS
        ${onnxruntime_INCLUDE_DIR_HINTS}
        PATHS
        /usr/include
        /usr/local/include
        /sw/include
        /opt/include
        /opt/local/include)
    set(_onnxruntime_library_paths
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/lib
        /opt/local/lib)
    find_library(onnxruntime_LIBRARIES
        NAMES onnxruntime libonnxruntime
        HINTS ${onnxruntime_LIBRARY_DIR_HINTS}
        PATHS ${_onnxruntime_library_paths})
    # Clear stale provider results first: find_library does not re-search
    # while a cache entry exists, so a provider found for a previous
    # configuration (e.g. with CUDA enabled, or with different hints) would
    # otherwise be reused.
    unset(onnxruntime_PROVIDERS_SHARED_LIBRARY CACHE)
    unset(onnxruntime_PROVIDERS_CUDA_LIBRARY CACHE)
    if(onnxruntime_LIBRARIES AND onnxruntime_LIBRARY_DIR_HINTS)
        # Provider libraries must come from the same installation as the
        # core library: search the core library's directory first and, only
        # if the core library came from the hints, the other hint
        # directories. Never fall back to other installations. The hint
        # directories must not be passed as HINTS here: HINTS are searched
        # before PATHS, so a hinted provider could otherwise win over one
        # beside the core library.
        get_filename_component(_onnxruntime_core_dir
            "${onnxruntime_LIBRARIES}" DIRECTORY)
        get_filename_component(_onnxruntime_core_dir_real
            "${_onnxruntime_core_dir}" REALPATH)
        set(_onnxruntime_provider_paths ${_onnxruntime_core_dir})
        foreach(_onnxruntime_hint ${onnxruntime_LIBRARY_DIR_HINTS})
            get_filename_component(_onnxruntime_hint_real
                "${_onnxruntime_hint}" REALPATH)
            if(_onnxruntime_hint_real STREQUAL _onnxruntime_core_dir_real)
                list(APPEND _onnxruntime_provider_paths
                    ${onnxruntime_LIBRARY_DIR_HINTS})
                break()
            endif()
        endforeach()
        find_library(onnxruntime_PROVIDERS_SHARED_LIBRARY
            NAMES onnxruntime_providers_shared libonnxruntime_providers_shared
            PATHS ${_onnxruntime_provider_paths}
            NO_DEFAULT_PATH)
        if(CUDA_ENABLED)
            find_library(onnxruntime_PROVIDERS_CUDA_LIBRARY
                NAMES onnxruntime_providers_cuda libonnxruntime_providers_cuda
                PATHS ${_onnxruntime_provider_paths}
                NO_DEFAULT_PATH)
        endif()
        unset(_onnxruntime_core_dir)
        unset(_onnxruntime_core_dir_real)
        unset(_onnxruntime_hint)
        unset(_onnxruntime_hint_real)
        unset(_onnxruntime_provider_paths)
    else()
        find_library(onnxruntime_PROVIDERS_SHARED_LIBRARY
            NAMES onnxruntime_providers_shared libonnxruntime_providers_shared
            HINTS ${onnxruntime_LIBRARY_DIR_HINTS}
            PATHS ${_onnxruntime_library_paths})
        if(CUDA_ENABLED)
            find_library(onnxruntime_PROVIDERS_CUDA_LIBRARY
                NAMES onnxruntime_providers_cuda libonnxruntime_providers_cuda
                HINTS ${onnxruntime_LIBRARY_DIR_HINTS}
                PATHS ${_onnxruntime_library_paths})
        endif()
    endif()

    unset(_onnxruntime_library_paths)

    # A missing providers_shared library is normal on some platforms (e.g.
    # the macOS package ships none), but a missing CUDA provider would
    # otherwise silently disable GPU inference.
    if(onnxruntime_LIBRARIES AND onnxruntime_LIBRARY_DIR_HINTS
            AND CUDA_ENABLED AND NOT onnxruntime_PROVIDERS_CUDA_LIBRARY)
        message(WARNING "onnxruntime_providers_cuda not found in the same "
            "installation as the core onnxruntime library; linking without "
            "the CUDA execution provider.")
    endif()

    if(onnxruntime_INCLUDE_DIRS AND onnxruntime_LIBRARIES)
        if(onnxruntime_PROVIDERS_SHARED_LIBRARY)
            list(APPEND onnxruntime_LIBRARIES ${onnxruntime_PROVIDERS_SHARED_LIBRARY})
        endif()
        if(onnxruntime_PROVIDERS_CUDA_LIBRARY)
            list(APPEND onnxruntime_LIBRARIES ${onnxruntime_PROVIDERS_CUDA_LIBRARY})
        endif()
        set(onnxruntime_FOUND TRUE)
        message(STATUS "Found onnxruntime")
        message(STATUS "  Includes : ${onnxruntime_INCLUDE_DIRS}")
        message(STATUS "  Libraries : ${onnxruntime_LIBRARIES}")
    else()
        set(onnxruntime_FOUND FALSE)
    endif()

    add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)
    target_include_directories(
        onnxruntime::onnxruntime INTERFACE ${onnxruntime_INCLUDE_DIRS}/onnxruntime)
    target_link_libraries(
        onnxruntime::onnxruntime INTERFACE ${onnxruntime_LIBRARIES})
    # This is a hack to make sure that the onnxruntime dll is copied to the output directory,
    # since vcpkg's custom add_library/add_executable macros copy any dependencies from vcpkg's
    # installed directory to the output directory.
    # See: https://github.com/microsoft/vcpkg/blob/fb7ba3b89b0d8e3e56b0508a144fe85015edfab6/scripts/buildsystems/vcpkg.cmake#L607
    if(IS_WINDOWS AND VCPKG_INSTALLED_DIR)
        foreach(_lib ${onnxruntime_LIBRARIES})
            get_filename_component(_lib_dir "${_lib}" DIRECTORY)
            get_filename_component(_lib_name "${_lib}" NAME_WE)
            set(_dlls
                "${_lib_dir}/${_lib_name}.dll"
                "${_lib_dir}/../bin/${_lib_name}.dll")
            foreach(_dll ${_dlls})
                if( EXISTS "${_dll}" )
                    file(COPY "${_dll}" DESTINATION "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
                endif()
            endforeach()
        endforeach()
    endif()
endif()

if(NOT onnxruntime_FOUND AND onnxruntime_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find onnxruntime")
endif()
