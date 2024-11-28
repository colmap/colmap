#include "colmap/estimators/manifold.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindCustomizedManifold(py::module& m) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  py::class_<PositiveExponentialManifold<ceres::DYNAMIC>, ceres::Manifold>(
      m, "PositiveExponentialManifold")
      .def(py::init<int>());
#endif
}

void BindManifold(py::module& m_parent) {
  py::module_ m = m_parent.def_submodule("manifold");
  if (IsPyceresAvailable()) {
    BindCustomizedManifold(m);
  }
}
