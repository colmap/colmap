#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindFeatureTypes(py::module& m);
void BindFeatureExtraction(py::module& m);
void BindFeatureMatching(py::module& m);

void BindFeature(py::module& m) {
  BindFeatureTypes(m);
  BindFeatureExtraction(m);
  BindFeatureMatching(m);
}
