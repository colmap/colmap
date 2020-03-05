if(POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
endif()

if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Determine project compiler.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(IS_MSVC TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(IS_GNU TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(IS_CLANG TRUE)
endif()

# Determine project architecture.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "[ix].?86|amd64|AMD64")
    set(IS_X86 TRUE)
endif()

# Determine project operating system.
string(REGEX MATCH "Linux" IS_LINUX ${CMAKE_SYSTEM_NAME})
string(REGEX MATCH "DragonFly|BSD" IS_BSD ${CMAKE_SYSTEM_NAME})
string(REGEX MATCH "SunOS" IS_SOLARIS ${CMAKE_SYSTEM_NAME})
if(WIN32)
    SET(IS_WINDOWS TRUE BOOL INTERNAL)
endif()
if(APPLE)
    SET(IS_MACOS TRUE BOOL INTERNAL)
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "debug"
   OR CMAKE_BUILD_TYPE_LOWER STREQUAL "relwithdebinfo")
    set(IS_DEBUG TRUE)
endif()

# Enable solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_TARGETS_ROOT_FOLDER "cmake")
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER
             ${CMAKE_TARGETS_ROOT_FOLDER})
set(COLMAP_TARGETS_ROOT_FOLDER "colmap_targets")
set(COLMAP_SRC_ROOT_FOLDER "colmap_sources")

# This macro will search for source files in a given directory, will add them
# to a source group (folder within a project), and will then return paths to
# each of the found files. The usage of the macro is as follows:
# COLMAP_ADD_SOURCE_DIR(
#     <source directory to search>
#     <output variable with found source files>
#     <search expressions such as *.h *.cc>)
macro(COLMAP_ADD_SOURCE_DIR SRC_DIR SRC_VAR)
    # Create the list of expressions to be used in the search.
    set(GLOB_EXPRESSIONS "")
    foreach(ARG ${ARGN})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/${ARG})
    endforeach()
    # Perform the search for the source files.
    file(GLOB ${SRC_VAR} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         ${GLOB_EXPRESSIONS})
    # Create the source group.
    string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})
    source_group(${GROUP_NAME} FILES ${${SRC_VAR}})
    # Clean-up.
    unset(GLOB_EXPRESSIONS)
    unset(ARG)
    unset(GROUP_NAME)
endmacro(COLMAP_ADD_SOURCE_DIR)

# Macro to add source files to COLMAP library.
macro(COLMAP_ADD_SOURCES)
    set(SOURCE_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else()
            list(APPEND SOURCE_FILES
                 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
    endforeach()
    set(COLMAP_SOURCES ${COLMAP_SOURCES} ${SOURCE_FILES} PARENT_SCOPE)
endmacro(COLMAP_ADD_SOURCES)

# Macro to add CUDA source files to COLMAP library.
macro(COLMAP_ADD_CUDA_SOURCES)
    set(SOURCE_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            # Absolute path.
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else()
            # Relative path.
            list(APPEND SOURCE_FILES
                 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
    endforeach()

    set(COLMAP_CUDA_SOURCES
        ${COLMAP_CUDA_SOURCES}
        ${SOURCE_FILES}
        PARENT_SCOPE)
endmacro(COLMAP_ADD_CUDA_SOURCES)

# Replacement for the normal add_library() command. The syntax remains the same
# in that the first argument is the target name, and the following arguments
# are the source files to use when building the target.
macro(COLMAP_ADD_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/colmap/)
endmacro(COLMAP_ADD_LIBRARY)
macro(COLMAP_ADD_STATIC_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} STATIC ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/colmap)
endmacro(COLMAP_ADD_STATIC_LIBRARY)

# Replacement for the normal cuda_add_library() command. The syntax remains the
# same in that the first argument is the target name, and the following
# arguments are the source files to use when building the target.
macro(COLMAP_ADD_CUDA_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    cuda_add_library(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/colmap/)
endmacro(COLMAP_ADD_CUDA_LIBRARY)
macro(COLMAP_ADD_STATIC_CUDA_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    cuda_add_library(${TARGET_NAME} STATIC ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/colmap/)
endmacro(COLMAP_ADD_STATIC_CUDA_LIBRARY)

# Replacement for the normal add_executable() command. The syntax remains the
# same in that the first argument is the target name, and the following
# arguments are the source files to use when building the target.
macro(COLMAP_ADD_EXECUTABLE TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_executable(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(${TARGET_NAME} colmap)
    if(VCPKG_BUILD)
        install(TARGETS ${TARGET_NAME} DESTINATION tools/)
    else()
        install(TARGETS ${TARGET_NAME} DESTINATION bin/)
    endif()
endmacro(COLMAP_ADD_EXECUTABLE)

# Wrapper for test executables.
macro(COLMAP_ADD_TEST TARGET_NAME)
    if(TESTS_ENABLED)
        # ${ARGN} will store the list of source files passed to this function.
        add_executable(${TARGET_NAME} ${ARGN})
        set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(${TARGET_NAME} colmap
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        add_test("${FOLDER_NAME}/${TARGET_NAME}" ${TARGET_NAME})
        if(IS_MSVC)
            install(TARGETS ${TARGET_NAME} DESTINATION bin/)
        endif()
    endif()
endmacro(COLMAP_ADD_TEST)

# Wrapper for CUDA test executables.
macro(COLMAP_ADD_CUDA_TEST TARGET_NAME)
    if(TESTS_ENABLED)
        # ${ARGN} will store the list of source files passed to this function.
        cuda_add_executable(${TARGET_NAME} ${ARGN})
        set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${COLMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(${TARGET_NAME} colmap
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        add_test("${FOLDER_NAME}/${TARGET_NAME}" ${TARGET_NAME})
        if(IS_MSVC)
            install(TARGETS ${TARGET_NAME} DESTINATION bin/)
        endif()
    endif()
endmacro(COLMAP_ADD_CUDA_TEST)
