#pragma once

#include "pycolmap/sfm/incremental_mapper.h"
#include "pycolmap/sfm/incremental_triangulator.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindSfMObjects(py::module& m) {
  BindIncrementalTriangulator(m);
  BindIncrementalMapper(m);
}
