#pragma once

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/sim3.h"

#include "pycolmap/geometry/homography_matrix.h"
#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

void BindGeometry(py::module& m) {
  BindHomographyGeometry(m);

  py::class_ext_<Eigen::Quaterniond> PyRotation3d(m, "Rotation3d");
  PyRotation3d.def(py::init([]() { return Eigen::Quaterniond::Identity(); }))
      .def(py::init<const Eigen::Vector4d&>(),
           "xyzw"_a,
           "Quaternion in [x,y,z,w] format.")
      .def(py::init<const Eigen::Matrix3d&>(),
           "rotmat"_a,
           "3x3 rotation matrix.")
      .def(py::init([](const Eigen::Vector3d& vec) {
             return Eigen::Quaterniond(
                 Eigen::AngleAxis<double>(vec.norm(), vec.normalized()));
           }),
           "axis_angle"_a,
           "Axis-angle 3D vector.")
      .def_property(
          "quat",
          py::overload_cast<>(&Eigen::Quaterniond::coeffs),
          [](Eigen::Quaterniond& self, const Eigen::Vector4d& quat) {
            self.coeffs() = quat;
          },
          "Quaternion in [x,y,z,w] format.")
      .def(py::self * Eigen::Quaterniond())
      .def(py::self * Eigen::Vector3d())
      .def("__mul__",
           [](const Eigen::Quaterniond& self,
              const py::EigenDRef<const Eigen::MatrixX3d>& points)
               -> Eigen::MatrixX3d {
             return points * self.toRotationMatrix().transpose();
           })
      .def("normalize", &Eigen::Quaterniond::normalize)
      .def("matrix", &Eigen::Quaterniond::toRotationMatrix)
      .def("norm", &Eigen::Quaterniond::norm)
      .def("angle",
           [](const Eigen::Quaterniond& self) {
             return Eigen::AngleAxis<double>(self).angle();
           })
      .def("angle_to",
           [](const Eigen::Quaterniond& self, const Eigen::Quaterniond& other) {
             return self.angularDistance(other);
           })
      .def("inverse", &Eigen::Quaterniond::inverse)
      .def("__repr__", [](const Eigen::Quaterniond& self) {
        std::stringstream ss;
        ss << "Rotation3d(quat_xyzw=[" << self.coeffs().format(vec_fmt) << "])";
        return ss.str();
      });
  py::implicitly_convertible<py::array, Eigen::Quaterniond>();
  MakeDataclass(PyRotation3d);

  py::class_ext_<Rigid3d> PyRigid3d(m, "Rigid3d");
  PyRigid3d.def(py::init<>())
      .def(py::init<const Eigen::Quaterniond&, const Eigen::Vector3d&>())
      .def(py::init([](const Eigen::Matrix3x4d& matrix) {
        return Rigid3d(Eigen::Quaterniond(matrix.leftCols<3>()), matrix.col(3));
      }))
      .def_readwrite("rotation", &Rigid3d::rotation)
      .def_readwrite("translation", &Rigid3d::translation)
      .def("matrix", &Rigid3d::ToMatrix)
      .def("essential_matrix", &EssentialMatrixFromPose)
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
      .def_static("interpolate", &InterpolateCameraPoses)
      .def("__repr__", [](const Rigid3d& self) {
        std::stringstream ss;
        ss << "Rigid3d("
           << "quat_xyzw=[" << self.rotation.coeffs().format(vec_fmt) << "], "
           << "t=[" << self.translation.format(vec_fmt) << "])";
        return ss.str();
      });
  py::implicitly_convertible<py::array, Rigid3d>();
  MakeDataclass(PyRigid3d);

  py::class_ext_<Sim3d> PySim3d(m, "Sim3d");
  PySim3d.def(py::init<>())
      .def(
          py::init<double, const Eigen::Quaterniond&, const Eigen::Vector3d&>())
      .def(py::init(&Sim3d::FromMatrix))
      .def_readwrite("scale", &Sim3d::scale)
      .def_readwrite("rotation", &Sim3d::rotation)
      .def_readwrite("translation", &Sim3d::translation)
      .def("matrix", &Sim3d::ToMatrix)
      .def(py::self * Sim3d())
      .def(py::self * Eigen::Vector3d())
      .def("__mul__",
           [](const Sim3d& t,
              const py::EigenDRef<const Eigen::MatrixX3d>& points)
               -> Eigen::MatrixX3d {
             return (t.scale *
                     (points * t.rotation.toRotationMatrix().transpose()))
                        .rowwise() +
                    t.translation.transpose();
           })
      .def("transform_camera_world", &TransformCameraWorld)
      .def("inverse", static_cast<Sim3d (*)(const Sim3d&)>(&Inverse))
      .def("__repr__", [](const Sim3d& self) {
        std::stringstream ss;
        ss << "Sim3d("
           << "scale=" << self.scale << ", "
           << "quat_xyzw=[" << self.rotation.coeffs().format(vec_fmt) << "], "
           << "t=[" << self.translation.format(vec_fmt) << "])";
        return ss.str();
      });
  py::implicitly_convertible<py::array, Sim3d>();
  MakeDataclass(PySim3d);
}
