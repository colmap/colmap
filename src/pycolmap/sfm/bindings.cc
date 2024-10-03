#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindObservationManager(py::module& m);
void BindIncrementalTriangulator(py::module& m);
void BindIncrementalMapper(py::module& m);

void BindSfMObjects(py::module& m) {
  BindObservationManager(m);
  BindIncrementalTriangulator(m);
  BindIncrementalMapper(m);
}
