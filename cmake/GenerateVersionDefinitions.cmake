if(Git_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(COMMAND
        "${GIT_EXECUTABLE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_COMMIT_ID
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(COMMAND
        "${GIT_EXECUTABLE}" log -1 --format=%ad --date=short
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_COMMIT_DATE
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    set(GIT_COMMIT_ID "Unkown")
    set(GIT_COMMIT_DATE "Unkown")
endif()

configure_file("${CMAKE_SOURCE_DIR}/src/util/version.h.in"
               "${CMAKE_SOURCE_DIR}/src/util/version.h")
