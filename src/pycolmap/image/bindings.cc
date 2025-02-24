#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindUndistortion(py::module& m);

void BindImage(py::module& m) { BindUndistortion(m); }
