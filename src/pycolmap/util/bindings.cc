#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindLogging(py::module& m);
void BindTimer(py::module& m);
void BindUtilTypes(py::module& m);

void BindUtil(py::module& m) {
  BindUtilTypes(m);
  BindLogging(m);
  BindTimer(m);
}
