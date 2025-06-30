#include "colmap/geometry/pose_prior.h"

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
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
      .def_readwrite("position", &PosePrior::position)
      .def_readwrite("position_covariance", &PosePrior::position_covariance)
      .def_readwrite("coordinate_system", &PosePrior::coordinate_system)
      .def("is_valid", &PosePrior::IsValid)
      .def("is_covariance_valid", &PosePrior::IsCovarianceValid);
  MakeDataclass(PyPosePrior);
}
