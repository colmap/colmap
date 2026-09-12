// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cancellation.h"

#include <memory>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace colmap;

void BindCancellation(py::module& m) {
  py::class_<CancellationToken, std::shared_ptr<CancellationToken>>(
      m, "CancellationToken")
      .def(py::init<>())
      .def("cancel", &CancellationToken::Cancel)
      .def_property_readonly("is_cancelled", &CancellationToken::IsCancelled);
}
