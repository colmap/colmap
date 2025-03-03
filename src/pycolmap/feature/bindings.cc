#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindFeatureTypes(py::module& m);
void BindSift(py::module& m);

void BindFeature(py::module& m) {
  BindFeatureTypes(m);
  BindSift(m);
}
