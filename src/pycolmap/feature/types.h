#pragma once

#include "colmap/feature/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(colmap::FeatureKeypoints);

PYBIND11_MAKE_OPAQUE(colmap::FeatureMatches);
