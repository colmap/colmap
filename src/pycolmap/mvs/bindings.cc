#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindMVSModel(py::module& m);

void BindMvs(py::module& m) { BindMVSModel(m); }
