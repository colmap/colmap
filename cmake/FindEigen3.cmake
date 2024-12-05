# Try to find Eigen3
# Defines:
#   Eigen3::Eigen  - imported target
#   EIGEN3_INCLUDE_DIR

if(Eigen3_FOUND)
  return()
endif()

find_path(EIGEN3_INCLUDE_DIR
  NAMES Eigen/Core
  PATH_SUFFIXES eigen3 eigen
  DOC "Eigen3 include directory"
)

if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find Eigen3: set EIGEN3_INCLUDE_DIR")
endif()

# Create interface library
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set(Eigen3_FOUND TRUE)
set_target_properties(Eigen3::Eigen PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
)

message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")