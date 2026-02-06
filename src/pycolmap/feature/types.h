#pragma once

#include "colmap/feature/types.h"

#include <pybind11/pybind11.h>

// PYBIND11_MAKE_OPAQUE must be declared before including pybind11/stl.h
// to avoid ODR violations with the default type_caster.
PYBIND11_MAKE_OPAQUE(colmap::FeatureKeypoints);
PYBIND11_MAKE_OPAQUE(colmap::FeatureMatches);

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
