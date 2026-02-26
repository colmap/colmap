#pragma once

// This header declares opaque types for pybind11 vector bindings.
// It MUST be included before <pybind11/stl.h> in any translation unit
// that uses FeatureKeypoints or FeatureMatches to avoid ODR violations.

#include "colmap/feature/types.h"

#include <pybind11/pybind11.h>

PYBIND11_MAKE_OPAQUE(colmap::FeatureKeypoints);
PYBIND11_MAKE_OPAQUE(colmap::FeatureMatches);
