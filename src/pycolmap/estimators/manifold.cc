#include "colmap/estimators/manifold.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

bool IsPyceresAvailable() {
  py::module importlib = py::module::import("importlib");
  py::object find_spec_func = importlib.attr("util").attr("find_spec");
  py::object module_spec = find_spec_func("pyceres");
  return !module_spec.is_none();
}

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
  bool is_pyceres_available = IsPyceresAvailable();
  if (is_pyceres_available) {
    BindCustomizedManifold(m);
  }
}
