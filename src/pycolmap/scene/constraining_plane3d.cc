#include "colmap/scene/constraining_plane3d.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <memory>
#include <sstream>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
namespace py = pybind11;

void BindConstrainingPlane3D(py::module& m) {
  py::class_ext_<ConstrainingPlane3D, std::shared_ptr<ConstrainingPlane3D>>
      PyConstrainingPlane3D(m, "ConstrainingPlane3D");
  PyConstrainingPlane3D.def(py::init<>())
      .def(py::init<const Eigen::Vector3d&, double>(), "normal"_a, "offset"_a)
      .def_readwrite("normal", &ConstrainingPlane3D::normal)
      .def_readwrite("offset", &ConstrainingPlane3D::offset)
      .def_readwrite("is_fixed", &ConstrainingPlane3D::is_fixed)
      .def_readwrite("prior_normal", &ConstrainingPlane3D::prior_normal)
      .def_readwrite("prior_normal_sigma_deg",
                     &ConstrainingPlane3D::prior_normal_sigma_deg)
      .def("has_normal_prior", &ConstrainingPlane3D::HasNormalPrior)
      .def("signed_distance", &ConstrainingPlane3D::SignedDistance, "xyz"_a)
      .def("normalize", &ConstrainingPlane3D::Normalize);
  MakeDataclass(PyConstrainingPlane3D);

  py::bind_map<std::unordered_map<int, ConstrainingPlane3D>>(
      m, "MapPlane3DIdToConstrainingPlane3D");
}
