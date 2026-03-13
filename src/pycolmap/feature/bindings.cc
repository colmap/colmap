#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindFeatureExtraction(py::module& m);
void BindFeatureMatching(py::module& m);

void BindFeature(py::module& m) {
  BindFeatureExtraction(m);
  BindFeatureMatching(m);
}
