#include "colmap/geometry/rigid3.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindRigid3(py::module& m) {
  py::class_ext_<Rigid3d> PyRigid3d(m, "Rigid3d");
  PyRigid3d.def(py::init<>())
      .def(py::init<const Eigen::Quaterniond&, const Eigen::Vector3d&>(),
           "rotation"_a,
           "translation"_a)
      .def(py::init(&Rigid3d::FromMatrix),
           "matrix"_a,
           "3x4 transformation matrix.")
      .def_readwrite("rotation", &Rigid3d::rotation)
      .def_readwrite("translation", &Rigid3d::translation)
      .def("matrix", &Rigid3d::ToMatrix)
      .def("adjoint", &Rigid3d::Adjoint)
      .def("adjoint_inverse", &Rigid3d::AdjointInverse)
      .def(py::self * Rigid3d())
      .def(py::self * Eigen::Vector3d())
      .def("__mul__",
           [](const Rigid3d& t,
              const py::EigenDRef<const Eigen::MatrixX3d>& points)
               -> Eigen::MatrixX3d {
             return (points * t.rotation.toRotationMatrix().transpose())
                        .rowwise() +
                    t.translation.transpose();
           })
      .def("inverse", static_cast<Rigid3d (*)(const Rigid3d&)>(&Inverse))
      .def_static("interpolate",
                  &InterpolateCameraPoses,
                  "cam_from_world1"_a,
                  "cam_from_world2"_a,
                  "t"_a);
  py::implicitly_convertible<py::array, Rigid3d>();
  MakeDataclass(PyRigid3d);

  m.def("get_covariance_for_inverse",
        &GetCovarianceForRigid3dInverse,
        "rigid3d"_a,
        "covar"_a);
  m.def("get_covariance_for_composed_rigid3d",
        &GetCovarianceForComposedRigid3d,
        "left_rigid3d"_a,
        "joint_covar"_a);
  m.def("get_covariance_for_relative_rigid3d",
        &GetCovarianceForRelativeRigid3d,
        "base_rigid3d"_a,
        "target_rigid3d"_a,
        "joint_covar"_a);
}
