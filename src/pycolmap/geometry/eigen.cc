#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <Eigen/Geometry>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindEigenGeometry(py::module& m) {
  py::class_ext_<Eigen::Quaterniond> PyRotation3d(m, "Rotation3d");
  PyRotation3d.def(py::init([]() { return Eigen::Quaterniond::Identity(); }))
      .def(py::init<const Eigen::Vector4d&>(),
           "xyzw"_a,
           "Quaternion in [x,y,z,w] format.")
      .def(py::init<const Eigen::Matrix3d&>(),
           "matrix"_a,
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
      .def(
          "angle_to",
          [](const Eigen::Quaterniond& self, const Eigen::Quaterniond& other) {
            return self.angularDistance(other);
          },
          "other"_a)
      .def("inverse", &Eigen::Quaterniond::inverse)
      .def("__repr__", [](const Eigen::Quaterniond& self) {
        std::ostringstream ss;
        ss << "Rotation3d(xyzw=[" << self.coeffs().format(vec_fmt) << "])";
        return ss.str();
      });
  py::implicitly_convertible<py::array, Eigen::Quaterniond>();
  MakeDataclass(PyRotation3d);

  py::class_ext_<Eigen::AlignedBox3d> PyAlignedBox3d(m, "AlignedBox3d");
  PyAlignedBox3d.def(py::init<>())
      .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&>(),
           "min"_a,
           "max"_a)
      .def_property("min",
                    py::overload_cast<>(&Eigen::AlignedBox3d::min),
                    [](Eigen::AlignedBox3d& self, const Eigen::Vector3d& min) {
                      self.min() = min;
                    })
      .def_property("max",
                    py::overload_cast<>(&Eigen::AlignedBox3d::max),
                    [](Eigen::AlignedBox3d& self, const Eigen::Vector3d& max) {
                      self.max() = max;
                    })
      .def("diagonal", &Eigen::AlignedBox3d::diagonal)
      .def(
          "contains_point",
          [](const Eigen::AlignedBox3d& self, const Eigen::Vector3d& point) {
            return self.contains(point);
          },
          "point"_a)
      .def(
          "contains_bbox",
          [](const Eigen::AlignedBox3d& self,
             const Eigen::AlignedBox3d& other) { return self.contains(other); },
          "other"_a)
      .def("__repr__", [](const Eigen::AlignedBox3d& self) {
        std::ostringstream ss;
        ss << "AlignedBox3d(min=[" << self.min().format(vec_fmt) << "], max=["
           << self.max().format(vec_fmt) << "])";
        return ss.str();
      });
  MakeDataclass(PyAlignedBox3d);
}
