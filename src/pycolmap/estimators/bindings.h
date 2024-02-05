#pragma once

#include "pycolmap/estimators/absolute_pose.h"
#include "pycolmap/estimators/alignment.h"
#include "pycolmap/estimators/cost_functions.h"
#include "pycolmap/estimators/essential_matrix.h"
#include "pycolmap/estimators/fundamental_matrix.h"
#include "pycolmap/estimators/generalized_absolute_pose.h"
#include "pycolmap/estimators/homography_matrix.h"
#include "pycolmap/estimators/triangulation.h"
#include "pycolmap/estimators/two_view_geometry.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindEstimators(py::module& m) {
  BindAbsolutePoseEstimator(m);
  BindAlignmentEstimator(m);
  BindCostFunctions(m);
  BindEssentialMatrixEstimator(m);
  BindFundamentalMatrixEstimator(m);
  BindGeneralizedAbsolutePoseEstimator(m);
  BindHomographyMatrixEstimator(m);
  BindTriangulationEstimator(m);
  BindTwoViewGeometryEstimator(m);
}
