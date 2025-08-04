#include "colmap/geometry/pose_prior.h"

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;              // NOLINT
using namespace pybind11::literals;  // NOLINT
namespace py = pybind11;

void BindPosePrior(py::module& m) {
  using PosePriorCoordinateSystem = PosePrior::CoordinateSystem;

  py::enum_<PosePriorCoordinateSystem> PyCoordinateSystem(
      m, "PosePriorCoordinateSystem");
  PyCoordinateSystem.value("UNDEFINED", PosePriorCoordinateSystem::UNDEFINED)
      .value("WGS84", PosePriorCoordinateSystem::WGS84)
      .value("CARTESIAN", PosePriorCoordinateSystem::CARTESIAN);
  AddStringToEnumConstructor(PyCoordinateSystem);

  py::class_ext_<PosePrior> PyPosePrior(m, "PosePrior");
  PyPosePrior.def(py::init<>())
      .def(py::init<const Eigen::Vector3d&>(), "position"_a)
      .def(py::init<const Eigen::Vector3d&, const PosePriorCoordinateSystem>(),
           "coordinate_system"_a,
           "position"_a)
      .def(py::init<const Eigen::Vector3d&, const Eigen::Matrix3d&>(),
           "position"_a,
           "position_covariance"_a)
      .def(py::init<const Eigen::Vector3d&,
                    const Eigen::Matrix3d&,
                    const PosePriorCoordinateSystem>(),
           "coordinate_system"_a,
           "position"_a,
           "position_covariance"_a)

      .def(py::init<const Eigen::Vector3d&, const Eigen::Quaterniond&>(),
           "position"_a,
           "rotation"_a)
      .def(py::init<const Eigen::Vector3d&,
                    const Eigen::Quaterniond&,
                    const Eigen::Matrix3d&,
                    const Eigen::Matrix3d&>(),
           "position"_a,
           "rotation"_a,
           "position_covariance"_a,
           "rotation_covariance"_a)

      .def_readwrite("coordinate_system", &PosePrior::coordinate_system)
      .def_readwrite("world_from_cam", &PosePrior::world_from_cam)
      .def_readwrite("position_covariance", &PosePrior::position_covariance)
      .def_readwrite("rotation_covariance", &PosePrior::rotation_covariance)

      .def("cam_from_world", &PosePrior::CamFromWorld)

      .def("world_from_cam_covariance", &PosePrior::WorldFromCamCovariance)
      .def("set_world_from_cam_covariance",
           &PosePrior::SetWorldFromCamCovariance)
      .def("cam_from_world_covariance", &PosePrior::CamFromWorldCovariance)
      .def("set_cam_from_world_covariance",
           &PosePrior::SetCamFromWorldCovariance)

      .def("has_valid_rotation", &PosePrior::HasValidRotation)
      .def("has_valid_rotation_covariance",
           &PosePrior::HasValidRotationCovariance)
      .def("has_valid_position", &PosePrior::HasValidPosition)
      .def("has_valid_position_covariance",
           &PosePrior::HasValidPositionCovariance)
      .def("has_valid_world_from_cam", &PosePrior::HasValidWorldFromCam)
      .def("has_valid_world_from_cam_covariance",
           &PosePrior::HasValidWorldFromCamCovariance)

      .def("__eq__", &PosePrior::operator==)
      .def("__ne__", &PosePrior::operator!=);

  MakeDataclass(PyPosePrior);
}
