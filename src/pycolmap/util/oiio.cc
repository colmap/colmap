#include "colmap/util/oiio_utils.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindOpenImageIO(py::module& m) {
    colmap::InitializeOpenImageIO();
}
