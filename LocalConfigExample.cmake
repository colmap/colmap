# This is an example of a LocalConfig.cmake file. This is useful on Windows
# platforms, where CMake cannot find many of the libraries automatically and
# it is tedious to set their paths manually in the GUI. To use this file, copy
# and rename it to LocalConfig.cmake

message(STATUS "NOTE: Using LocalConfig.cmake to override CMake settings in the GUI")

set(Ceres_DIR "<ceres-solver-dir>/build/install/CMake" CACHE PATH "" FORCE)

set(EIGEN3_INCLUDE_DIRS "<Eigen-dir>" CACHE PATH "" FORCE)

set(GLOG_INCLUDE_DIR_HINTS "<glog-dir>/src/windows" CACHE PATH "" FORCE)
set(GLOG_LIBRARY_DIR_HINTS "<glog-dir>/x64/Release" CACHE FILEPATH "" FORCE)

set(BOOST_ROOT "<boost-dir>" CACHE PATH "" FORCE)
set(BOOST_LIBRARYDIR "<boost-lib-dir>" CACHE PATH "" FORCE)

set(OPENGL_gl_LIBRARY "opengl32" CACHE STRING "" FORCE)
set(OPENGL_glu_LIBRARY "glu32" CACHE STRING "" FORCE)

set(GLEW_INCLUDE_DIR_HINTS "<glew-dir>/include" CACHE PATH "" FORCE)
set(GLEW_LIBRARY_DIR_HINTS "<glew-dir>/lib/Release/x64" CACHE FILEPATH "" FORCE)

set(Qt5_CMAKE_DIR "<Qt-precompiled-dir>/lib/cmake" CACHE PATH "" FORCE)
set(Qt5Core_DIR ${Qt5_CMAKE_DIR}/Qt5Core CACHE PATH "" FORCE)
set(Qt5OpenGL_DIR ${Qt5_CMAKE_DIR}/Qt5OpenGL CACHE PATH "" FORCE)

set(FREEIMAGE_INCLUDE_DIR_HINTS "<FreeImage-dir>/Dist/x64" CACHE PATH "" FORCE)
set(FREEIMAGE_LIBRARY_DIR_HINTS "<FreeImage-dir>/Dist/x64" CACHE PATH "" FORCE)
