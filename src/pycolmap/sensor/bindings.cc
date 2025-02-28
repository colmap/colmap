#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindBitmap(py::module& m);

void BindSensor(py::module& m) { BindBitmap(m); }
