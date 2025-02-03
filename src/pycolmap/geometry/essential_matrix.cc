#include "colmap/geometry/essential_matrix.h"

#include "colmap/util/logging.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindEssentialMatrixGeometry(py::module& m) {
  m.def("essential_matrix_from_pose",
        &EssentialMatrixFromPose,
        "cam2_from_cam1"_a,
        "Construct essential matrix from relative pose.");
}
