#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindObservationManager(py::module& m);
void BindIncrementalTriangulator(py::module& m);
void BindIncrementalMapper(py::module& m);
void BindHierarchicalMapper(py::module& m);
void BindGlobalMapper(py::module& m);

void BindSfm(py::module& m) {
  BindObservationManager(m);
  BindIncrementalTriangulator(m);
  BindIncrementalMapper(m);
  BindHierarchicalMapper(m);
  // Must be bound after BindIncrementalTriangulator, because
  // GlobalMapperOptions has an IncrementalTriangulatorOptions member.
  BindGlobalMapper(m);
}
