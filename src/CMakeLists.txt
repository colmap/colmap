include_directories(
    .
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${FREEIMAGE_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5OpenGL_INCLUDE_DIRS}
)

link_directories(
    ${Boost_LIBRARY_DIRS}
)

set(COLMAP_LIBRARIES
    mvs
    sfm
    base
    util
    estimators
    base
    util
    estimators
    optim
    retrieval
    base
    util
    flann
    lsd
    pba
    poisson_recon
    sqlite3
    sift_gpu
    vlfeat
    ${Boost_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${FREEIMAGE_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OPENGL_LIBRARIES})

if(CUDA_ENABLED)
    set(COLMAP_LIBRARIES mvs mvs_cuda ${COLMAP_LIBRARIES})
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(COLMAP_LIBRARIES ${COLMAP_LIBRARIES} pthread dl)
endif()

set(COLMAP_UI_LIBRARIES ui)

set(COLMAP_QT_MODULES Core Widgets OpenGL)

add_subdirectory(base)
add_subdirectory(estimators)
add_subdirectory(exe)
add_subdirectory(ext)
add_subdirectory(mvs)
add_subdirectory(optim)
add_subdirectory(retrieval)
add_subdirectory(sfm)
add_subdirectory(tools)
add_subdirectory(ui)
add_subdirectory(util)

# This macro will search for source files in a given directory, will add them
# to a source group (folder within a project), and will then return paths to
# each of the found files. The usage of the macro is as follows:
# ADD_SOURCE_DIR(
#     <source directory to search>
#     <output variable with found source files>
#     <search expressions such as *.h *.cc>)
macro(ADD_SOURCE_DIR SRC_DIR SRC_VAR)
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
endmacro(ADD_SOURCE_DIR)

ADD_SOURCE_DIR(base BASE_SRC *.h *.cc)
ADD_SOURCE_DIR(estimators ESTIMATORS_SRC *.h *.cc)
ADD_SOURCE_DIR(exe EXE_SRC *.h *.cc)
ADD_SOURCE_DIR(ext/FLANN EXT_FLANN_SRC *.h *.cpp *.hpp *.cu)
ADD_SOURCE_DIR(ext/LSD EXT_LSD_SRC *.h *.c)
ADD_SOURCE_DIR(ext/PBA EXT_PBA_SRC *.h *.cpp *.cu)
ADD_SOURCE_DIR(ext/SiftGPU EXT_SIFTGPU_SRC *.h *.cpp *.cu)
ADD_SOURCE_DIR(ext/SQLite EXT_SQLITE_SRC *.h *.c)
ADD_SOURCE_DIR(ext/VLFeat EXT_VLFEAT_SRC *.h *.c *.tc)
ADD_SOURCE_DIR(mvs MVS_SRC *.h *.cc *.cu)
ADD_SOURCE_DIR(optim OPTIM_SRC *.h *.cc)
ADD_SOURCE_DIR(retrieval RETRIEVAL_SRC *.h *.cc)
ADD_SOURCE_DIR(sfm SFM_SRC *.h *.cc)
ADD_SOURCE_DIR(tools TOOLS_SRC *.h *.cc)
ADD_SOURCE_DIR(ui UI_SRC *.h *.cc)
ADD_SOURCE_DIR(util UTIL_SRC *.h *.cc)

# Add all of the source files to a regular library target, as using a custom
# target does not allow us to set its C++ include directories (and thus
# intellisense can't find any of the included files).
add_library(
    ${COLMAP_SRC_ROOT_FOLDER}
    ${BASE_SRC}
    ${ESTIMATORS_SRC}
    ${EXE_SRC}
    ${EXT_FLANN_SRC}
    ${EXT_LSD_SRC}
    ${EXT_PBA_SRC}
    ${EXT_SIFTGPU_SRC}
    ${EXT_SQLITE_SRC}
    ${EXT_VLFEAT_SRC}
    ${MVS_SRC}
    ${OPTIM_SRC}
    ${RETRIEVAL_SRC}
    ${SFM_SRC}
    ${TOOLS_SRC}
    ${UI_SRC}
    ${UTIL_SRC})

# Prevent the library from being compiled automatically.
set_target_properties(
    ${COLMAP_SRC_ROOT_FOLDER} PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1)
