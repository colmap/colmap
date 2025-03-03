#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindVisualIndex(py::module& m);

void BindRetrieval(py::module& m) { BindVisualIndex(m); }
