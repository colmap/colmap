#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindDepthMap(py::module& m);
void BindNormalMap(py::module& m);
void BindMVSModel(py::module& m);

void BindMvs(py::module& m) {
  BindDepthMap(m);
  BindNormalMap(m);
  BindMVSModel(m);
}
