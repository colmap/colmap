# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Authors: pablo.speciale@gmail.com (Pablo Speciale)
#          alexs.mac@gmail.com (Alex Stewart)
#

# Config file for Ceres Solver - Find Ceres & dependencies.
#
# This file is used by CMake when find_package(Ceres) is invoked and either
# the directory containing this file either is present in CMAKE_MODULE_PATH
# (if Ceres was installed), or exists in the local CMake package registry if
# the Ceres build directory was exported.
#
# This module defines the following variables:
#
# Ceres_FOUND / CERES_FOUND: True if Ceres has been successfully
#                            found. Both variables are set as although
#                            FindPackage() only references Ceres_FOUND
#                            in Config mode, given the conventions for
#                            <package>_FOUND when FindPackage() is
#                            called in Module mode, users could
#                            reasonably expect to use CERES_FOUND
#                            instead.
#
# CERES_VERSION: Version of Ceres found.
#
# CERES_LIBRARIES: Libraries for Ceres and all
#                  dependencies against which Ceres was
#                  compiled. This will not include any optional
#                  dependencies that were disabled when Ceres was
#                  compiled.
#
# NOTE: There is no equivalent of CERES_INCLUDE_DIRS as the exported
#       CMake target already includes the definition of its public
#       include directories.

include(CMakeFindDependencyMacro)

# Called if we failed to find Ceres or any of its required dependencies,
# unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
macro(CERES_REPORT_NOT_FOUND REASON_MSG)
  # FindPackage() only references Ceres_FOUND, and requires it to be
  # explicitly set FALSE to denote not found (not merely undefined).
  set(Ceres_FOUND FALSE)
  set(CERES_FOUND FALSE)
  unset(CERES_INCLUDE_DIR)
  unset(CERES_INCLUDE_DIRS)
  unset(CERES_LIBRARIES)

  # Reset the CMake module path to its state when this script was called.
  set(CMAKE_MODULE_PATH ${CALLERS_CMAKE_MODULE_PATH})

  # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by
  # FindPackage() use the camelcase library name, not uppercase.
  if (Ceres_FIND_QUIETLY)
    message(STATUS "Failed to find Ceres - " ${REASON_MSG} ${ARGN})
  elseif (Ceres_FIND_REQUIRED)
    message(FATAL_ERROR "Failed to find Ceres - " ${REASON_MSG} ${ARGN})
  else()
    # Neither QUIETLY nor REQUIRED, use SEND_ERROR which emits an error
    # that prevents generation, but continues configuration.
    message(SEND_ERROR "Failed to find Ceres - " ${REASON_MSG} ${ARGN})
  endif ()
  return()
endmacro(CERES_REPORT_NOT_FOUND)


# ceres_message([mode] "message text")
#
# Wraps the standard cmake 'message' command, but suppresses output
# if the QUIET flag was passed to the find_package(Ceres ...) call.
function(ceres_message)
  if (NOT Ceres_FIND_QUIETLY)
    message(${ARGN})
  endif()
endfunction()


# ceres_pretty_print_cmake_list( OUTPUT_VAR [item1 [item2 ... ]] )
#
# Sets ${OUTPUT_VAR} in the caller's scope to a human-readable string
# representation of the list passed as the remaining arguments formed
# as: "[item1, item2, ..., itemN]".
function(ceres_pretty_print_cmake_list OUTPUT_VAR)
  string(REPLACE ";" ", " PRETTY_LIST_STRING "[${ARGN}]")
  set(${OUTPUT_VAR} "${PRETTY_LIST_STRING}" PARENT_SCOPE)
endfunction()

# The list of (optional) components this version of Ceres was compiled with.
set(CERES_COMPILED_COMPONENTS "EigenSparse;SparseLinearAlgebraLibrary;SchurSpecializations;Multithreading")

# If Ceres was not installed, then by definition it was exported
# from a build directory.
set(CERES_WAS_INSTALLED TRUE)

# Record the state of the CMake module path when this script was
# called so that we can ensure that we leave it in the same state on
# exit as it was on entry, but modify it locally.
set(CALLERS_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})

# Get the (current, i.e. installed) directory containing this file.
get_filename_component(CERES_CURRENT_CONFIG_DIR
  "${CMAKE_CURRENT_LIST_FILE}" PATH)

if (CERES_WAS_INSTALLED)
  # Reset CMake module path to the installation directory of this
  # script, thus we will use the FindPackage() scripts shipped with
  # Ceres to find Ceres' dependencies, even if the user has equivalently
  # named FindPackage() scripts in their project.
  set(CMAKE_MODULE_PATH ${CERES_CURRENT_CONFIG_DIR})

  # Build the absolute root install directory as a relative path
  # (determined when Ceres was configured & built) from the current
  # install directory for this this file.  This allows for the install
  # tree to be relocated, after Ceres was built, outside of CMake.
  get_filename_component(CURRENT_ROOT_INSTALL_DIR
    "${CERES_CURRENT_CONFIG_DIR}/../../../"
    ABSOLUTE)
  if (NOT EXISTS ${CURRENT_ROOT_INSTALL_DIR})
    ceres_report_not_found(
      "Ceres install root: ${CURRENT_ROOT_INSTALL_DIR}, "
      "determined from relative path from CeresConfig.cmake install location: "
      "${CERES_CURRENT_CONFIG_DIR}, does not exist. Either the install "
      "directory was deleted, or the install tree was only partially relocated "
      "outside of CMake after Ceres was built.")
  endif (NOT EXISTS ${CURRENT_ROOT_INSTALL_DIR})

else(CERES_WAS_INSTALLED)
  # Ceres was exported from the build tree.
  set(CERES_EXPORTED_BUILD_DIR ${CERES_CURRENT_CONFIG_DIR})
  get_filename_component(CERES_EXPORTED_SOURCE_DIR
    "${CERES_EXPORTED_BUILD_DIR}/../../../"
    ABSOLUTE)
  if (NOT EXISTS ${CERES_EXPORTED_SOURCE_DIR})
    ceres_report_not_found(
      "Ceres exported source directory: ${CERES_EXPORTED_SOURCE_DIR}, "
      "determined from relative path from CeresConfig.cmake exported build "
      "directory: ${CERES_EXPORTED_BUILD_DIR} does not exist.")
  endif()

  # Reset CMake module path to the cmake directory in the Ceres source
  # tree which was exported, thus we will use the FindPackage() scripts shipped
  # with Ceres to find Ceres' dependencies, even if the user has equivalently
  # named FindPackage() scripts in their project.
  set(CMAKE_MODULE_PATH ${CERES_EXPORTED_SOURCE_DIR}/cmake)
endif(CERES_WAS_INSTALLED)

# Set the version.
set(CERES_VERSION 2.1.0)

include(CMakeFindDependencyMacro)
find_dependency(Threads)

# Optional dependencies



# As imported CMake targets are not re-exported when a dependent target is
# exported, we must invoke find_package(XXX) here to reload the definition
# of their targets.  Without this, the dependency target names (e.g.
# 'gflags-shared') which will be present in the ceres target would not be
# defined, and so CMake will assume that they refer to a library name and
# fail to link correctly.

# Eigen.
# Flag set during configuration and build of Ceres.
set(CERES_EIGEN_VERSION 3.4.0)
# Search quietly to control the timing of the error message if not found. The
# search should be for an exact match, but for usability reasons do a soft
# match and reject with an explanation below.
find_package(Eigen3 ${CERES_EIGEN_VERSION} QUIET)
if (Eigen3_FOUND)
  if (NOT Eigen3_VERSION VERSION_EQUAL CERES_EIGEN_VERSION)
    # CMake's VERSION check in FIND_PACKAGE() will accept any version >= the
    # specified version. However, only version = is supported. Improve
    # usability by explaining why we don't accept non-exact version matching.
    ceres_report_not_found("Found Eigen dependency, but the version of Eigen "
      "found (${Eigen3_VERSION}) does not exactly match the version of Eigen "
      "Ceres was compiled with (${CERES_EIGEN_VERSION}). This can cause subtle "
      "bugs by triggering violations of the One Definition Rule. See the "
      "Wikipedia article http://en.wikipedia.org/wiki/One_Definition_Rule "
      "for more details")
  endif ()
  ceres_message(STATUS "Found required Ceres dependency: "
    "Eigen version ${CERES_EIGEN_VERSION} in ${Eigen3_DIR}")
else (Eigen3_FOUND)
  ceres_report_not_found("Missing required Ceres "
    "dependency: Eigen version ${CERES_EIGEN_VERSION}, please set "
    "Eigen3_DIR.")
endif (Eigen3_FOUND)

# glog (and maybe gflags).
#
# Flags set during configuration and build of Ceres.
set(CERES_USES_MINIGLOG ON)
set(CERES_GLOG_VERSION )
set(CERES_GLOG_WAS_BUILT_WITH_CMAKE )

set(CERES_USES_GFLAGS OFF)
set(CERES_GFLAGS_VERSION )

if (CERES_USES_MINIGLOG)
  # Output message at standard log level (not the lower STATUS) so that
  # the message is output in GUI during configuration to warn user.
  ceres_message("-- Found Ceres compiled with miniglog substitute "
    "for glog, beware this will likely cause problems if glog is later linked.")
else(CERES_USES_MINIGLOG)
  if (CERES_GLOG_WAS_BUILT_WITH_CMAKE)
    find_package(glog ${CERES_GLOG_VERSION} CONFIG QUIET)
    set(GLOG_FOUND ${glog_FOUND})
  else()
    # Version of glog against which Ceres was built was not built with CMake,
    # use the exported glog find_package() module from Ceres to find it again.
    # Append the locations of glog when Ceres was built to the search path hints.
    list(APPEND GLOG_INCLUDE_DIR_HINTS "")
    get_filename_component(CERES_BUILD_GLOG_LIBRARY_DIR "" PATH)
    list(APPEND GLOG_LIBRARY_DIR_HINTS ${CERES_BUILD_GLOG_LIBRARY_DIR})

    # Search quietly s/t we control the timing of the error message if not found.
    find_package(Glog QUIET)
  endif()

  if (GLOG_FOUND)
    ceres_message(STATUS "Found required Ceres dependency: glog")
  else()
    ceres_report_not_found("Missing required Ceres dependency: glog.")
  endif()

  # gflags is only a public dependency of Ceres via glog, thus is not required
  # if Ceres was built with MINIGLOG.
  if (CERES_USES_GFLAGS)
    # Search quietly s/t we control the timing of the error message if not found.
    find_package(gflags ${CERES_GFLAGS_VERSION} QUIET)
    if (gflags_FOUND AND TARGET gflags)
      ceres_message(STATUS "Found required Ceres dependency: gflags")
    else()
      ceres_report_not_found("Missing required Ceres "
        "dependency: gflags (not found, or not found as exported CMake target).")
    endif()
  endif()
endif(CERES_USES_MINIGLOG)

# Import exported Ceres targets, if they have not already been imported.
if (NOT TARGET ceres AND NOT Ceres_BINARY_DIR)
  include(${CERES_CURRENT_CONFIG_DIR}/CeresTargets.cmake)
endif (NOT TARGET ceres AND NOT Ceres_BINARY_DIR)
# Set the expected XX_LIBRARIES variable for FindPackage().
set(CERES_LIBRARIES Ceres::ceres)

# Reset CMake module path to its state when this script was called.
set(CMAKE_MODULE_PATH ${CALLERS_CMAKE_MODULE_PATH})

# Build the detected Ceres version string to correctly capture whether it
# was installed, or exported.
ceres_pretty_print_cmake_list(CERES_COMPILED_COMPONENTS_STRING
  ${CERES_COMPILED_COMPONENTS})
if (CERES_WAS_INSTALLED)
  set(CERES_DETECTED_VERSION_STRING "Ceres version: ${CERES_VERSION} "
    "installed in: ${CURRENT_ROOT_INSTALL_DIR} with components: "
    "${CERES_COMPILED_COMPONENTS_STRING}")
else (CERES_WAS_INSTALLED)
  set(CERES_DETECTED_VERSION_STRING "Ceres version: ${CERES_VERSION} "
    "exported from build directory: ${CERES_EXPORTED_BUILD_DIR} with "
    "components: ${CERES_COMPILED_COMPONENTS_STRING}")
endif()

# If the user called this script through find_package() whilst specifying
# particular Ceres components that should be found via:
# find_package(Ceres COMPONENTS XXX YYY), check the requested components against
# those with which Ceres was compiled.  In this case, we should only report
# Ceres as found if all the requested components have been found.
if (Ceres_FIND_COMPONENTS)
  foreach (REQUESTED_COMPONENT ${Ceres_FIND_COMPONENTS})
    list(FIND CERES_COMPILED_COMPONENTS ${REQUESTED_COMPONENT} HAVE_REQUESTED_COMPONENT)
    # list(FIND ..) returns -1 if the element was not in the list, but CMake
    # interprets if (VAR) to be true if VAR is any non-zero number, even
    # negative ones, hence we have to explicitly check for >= 0.
    if (HAVE_REQUESTED_COMPONENT EQUAL -1)
      # Check for the presence of all requested components before reporting
      # not found, such that we report all of the missing components rather
      # than just the first.
      list(APPEND MISSING_CERES_COMPONENTS ${REQUESTED_COMPONENT})
    endif()
  endforeach()
  if (MISSING_CERES_COMPONENTS)
    ceres_pretty_print_cmake_list(REQUESTED_CERES_COMPONENTS_STRING
      ${Ceres_FIND_COMPONENTS})
    ceres_pretty_print_cmake_list(MISSING_CERES_COMPONENTS_STRING
      ${MISSING_CERES_COMPONENTS})
    ceres_report_not_found("Missing requested Ceres components: "
      "${MISSING_CERES_COMPONENTS_STRING} (components requested: "
      "${REQUESTED_CERES_COMPONENTS_STRING}). Detected "
      "${CERES_DETECTED_VERSION_STRING}.")
  endif()
endif()

# As we use CERES_REPORT_NOT_FOUND() to abort, if we reach this point we have
# found Ceres and all required dependencies.
ceres_message(STATUS "Found " ${CERES_DETECTED_VERSION_STRING})

# Set CERES_FOUND to be equivalent to Ceres_FOUND, which is set to
# TRUE by FindPackage() if this file is found and run, and after which
# Ceres_FOUND is not (explicitly, i.e. undefined does not count) set
# to FALSE.
set(CERES_FOUND TRUE)

if (NOT TARGET ceres)
  # For backwards compatibility, create a local 'alias' target with the
  # non-namespace-qualified Ceres target name. Note that this is not a
  # true ALIAS library in CMake terms as they cannot point to imported targets.
  add_library(ceres INTERFACE IMPORTED)
  set_target_properties(ceres PROPERTIES INTERFACE_LINK_LIBRARIES Ceres::ceres)
endif()
