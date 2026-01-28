#include "pycolmap/glomap/types.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindPoseGraph(py::module& m);
void BindGlomapEstimators(py::module& m);

void BindGlomap(py::module& m) {
  BindPoseGraph(m);
  BindGlomapEstimators(m);
}
