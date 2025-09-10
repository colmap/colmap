#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindMvsModel(py::module& m);

void BindMvs(py::module& m) {
  BindMvsModel(m);
}
