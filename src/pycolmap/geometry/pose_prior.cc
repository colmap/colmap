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
           "position"_a,
           "coordinate_system"_a)
      .def(py::init<const Eigen::Vector3d&, const Eigen::Matrix3d&>(),
           "position"_a,
           "position_covariance"_a)
      .def(py::init<const Eigen::Vector3d&,
                    const Eigen::Matrix3d&,
                    const PosePriorCoordinateSystem>(),
           "position"_a,
           "position_covariance"_a,
           "coordinate_system"_a)

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
      .def(py::init<const Rigid3d&>(), "cam_from_world"_a)
      .def(py::init<const Rigid3d&,
                    const Eigen::Matrix3d&,
                    const Eigen::Matrix3d&>(),
           "cam_from_world"_a,
           "position_covariance"_a,
           "rotation_covariance"_a)

      .def_readwrite("coordinate_system", &PosePrior::coordinate_system)
      .def_readwrite("cam_from_world", &PosePrior::cam_from_world)

      .def("position", &PosePrior::Position)
      .def("set_position", &PosePrior::SetPosition)
      .def("rotation", &PosePrior::Rotation)
      .def("set_rotation", &PosePrior::SetRotation)
      .def("set_rotation_from_coeffs", &PosePrior::SetRotationFromCoeffs)
      .def("translation", &PosePrior::Translation)
      .def("set_translation", &PosePrior::SetTranslation)

      .def("translation_covariance", &PosePrior::TranslationCovariance)
      .def("set_translation_covariance", &PosePrior::SetTranslationCovariance)
      .def("rotation_covariance", &PosePrior::RotationCovariance)
      .def("set_rotation_covariance", &PosePrior::SetRotationCovariance)
      .def("position_covariance", &PosePrior::PositionCovariance)
      .def("set_position_covariance", &PosePrior::SetPositionCovariance)
      .def("pose_covariance", &PosePrior::PoseCovariance)
      .def("set_pose_covariance", &PosePrior::SetPoseCovariance)

      .def("has_valid_translation", &PosePrior::HasValidTranslation)
      .def("has_valid_translation_covariance",
           &PosePrior::HasValidTranslationCovariance)
      .def("has_valid_rotation", &PosePrior::HasValidRotation)
      .def("has_valid_rotation_covariance",
           &PosePrior::HasValidRotationCovariance)
      .def("has_valid_position", &PosePrior::HasValidPosition)
      .def("has_valid_position_covariance",
           &PosePrior::HasValidPositionCovariance)
      .def("has_valid_pose", &PosePrior::HasValidPose)
      .def("has_valid_pose_covariance", &PosePrior::HasValidPoseCovariance)

      .def("__eq__", &PosePrior::operator==)
      .def("__ne__", &PosePrior::operator!=);

  MakeDataclass(PyPosePrior);
}
