#include "colmap/geometry/gps.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/sim3.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindHomographyMatrixGeometry(py::module& m);
void BindEssentialMatrixGeometry(py::module& m);

void BindGeometry(py::module& m) {
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
      .def("get_covariance_for_inverse",
           static_cast<Eigen::Matrix6d (*)(const Rigid3d&,
                                           const Eigen::Matrix6d&)>(
               &GetCovarianceForRigid3dInverse),
           py::arg("covar"))
      .def_static("interpolate",
                  &InterpolateCameraPoses,
                  "cam_from_world1"_a,
                  "cam_from_world2"_a,
                  "t"_a);
  py::implicitly_convertible<py::array, Rigid3d>();
  MakeDataclass(PyRigid3d);

  py::class_ext_<Sim3d> PySim3d(m, "Sim3d");
  PySim3d.def(py::init<>())
      .def(
          py::init<double, const Eigen::Quaterniond&, const Eigen::Vector3d&>(),
          "scale"_a,
          "rotation"_a,
          "translation"_a)
      .def(py::init(&Sim3d::FromMatrix),
           "matrix"_a,
           "3x4 transformation matrix.")
      .def_property(
          "scale",
          [](Sim3d& self) {
            return py::array({}, {}, &self.scale, py::cast(self));
          },
          [](Sim3d& self, double scale) { self.scale = scale; })
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
      .def("transform_camera_world", &TransformCameraWorld, "cam_from_world"_a)
      .def("inverse", static_cast<Sim3d (*)(const Sim3d&)>(&Inverse));
  py::implicitly_convertible<py::array, Sim3d>();
  MakeDataclass(PySim3d);

  using PPCoordinateSystem = PosePrior::CoordinateSystem;
  py::enum_<PPCoordinateSystem> PyCoordinateSystem(m,
                                                   "PosePriorCoordinateSystem");
  PyCoordinateSystem.value("UNDEFINED", PPCoordinateSystem::UNDEFINED)
      .value("WGS84", PPCoordinateSystem::WGS84)
      .value("CARTESIAN", PPCoordinateSystem::CARTESIAN);
  AddStringToEnumConstructor(PyCoordinateSystem);

  py::class_ext_<PosePrior> PyPosePrior(m, "PosePrior");
  PyPosePrior.def(py::init<>())
      .def(py::init<const Eigen::Vector3d&>(), "position"_a)
      .def(py::init<const Eigen::Vector3d&, const PPCoordinateSystem>(),
           "position"_a,
           "coordinate_system"_a)
      .def(py::init<const Eigen::Vector3d&, const Eigen::Matrix3d&>(),
           "position"_a,
           "position_covariance"_a)
      .def(py::init<const Eigen::Vector3d&,
                    const Eigen::Matrix3d&,
                    const PPCoordinateSystem>(),
           "position"_a,
           "position_covariance"_a,
           "coordinate_system"_a)
      .def_readwrite("position", &PosePrior::position)
      .def_readwrite("position_covariance", &PosePrior::position_covariance)
      .def_readwrite("coordinate_system", &PosePrior::coordinate_system)
      .def("is_valid", &PosePrior::IsValid)
      .def("is_covariance_valid", &PosePrior::IsCovarianceValid);
  MakeDataclass(PyPosePrior);

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

  BindHomographyMatrixGeometry(m);
  BindEssentialMatrixGeometry(m);
}
