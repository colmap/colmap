# SPDX-License-Identifier: BSD-3-Clause

if (DEFINED GIT_COMMIT_ID OR DEFINED GIT_COMMIT_DATE)
    message(STATUS "Using custom-defined GIT_COMMIT_ID (${GIT_COMMIT_ID}) "
            "and GIT_COMMIT_DATE (${GIT_COMMIT_DATE})")
elseif(Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(COMMAND
        "${GIT_EXECUTABLE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_COMMIT_ID
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(COMMAND
        "${GIT_EXECUTABLE}" log -1 --format=%ad --date=short
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_COMMIT_DATE
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Re-generate version.cc if the git index changes.
    set_property(
        DIRECTORY APPEND 
        PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/.git/index"
    )
else()
    set(GIT_COMMIT_ID "Unknown")
    set(GIT_COMMIT_DATE "Unknown")
endif()

# Parse COLMAP_VERSION to extract MAJOR, MINOR, PATCH components.
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _version_match "${COLMAP_VERSION}")
set(COLMAP_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(COLMAP_VERSION_MINOR "${CMAKE_MATCH_2}")
set(COLMAP_VERSION_PATCH "${CMAKE_MATCH_3}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/colmap/util/version.cc.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/src/colmap/util/version.cc")
