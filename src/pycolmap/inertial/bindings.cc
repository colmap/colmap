// SPDX-License-Identifier: BSD-3-Clause

#include "pycolmap/helpers.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindImuPreintegration(py::module& m);
void BindImuPreintegrationCosts(py::module& m);

void BindInertial(py::module& m_parent) {
  py::module_ m = m_parent.def_submodule("inertial");
  IsPyceresAvailable();  // Try to import pyceres to populate the docstrings.
  BindImuPreintegration(m);
  BindImuPreintegrationCosts(m);
}
