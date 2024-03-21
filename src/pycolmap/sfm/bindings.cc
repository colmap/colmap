#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindIncrementalTriangulator(py::module& m);
void BindIncrementalMapper(py::module& m);

void BindSfMObjects(py::module& m) {
  BindIncrementalTriangulator(m);
  BindIncrementalMapper(m);
}
