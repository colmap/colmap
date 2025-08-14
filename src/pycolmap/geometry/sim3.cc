#include "colmap/geometry/sim3.h"

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

void BindSim3(py::module& m) {
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
      .def("inverse", static_cast<Sim3d (*)(const Sim3d&)>(&Inverse));
  py::implicitly_convertible<py::array, Sim3d>();
  MakeDataclass(PySim3d);

  m.def("transform_to_cam_from_new_world",
        &TransformToCamFromNewWorld,
        "new_from_old_world"_a,
        "cam_from_world"_a);
  m.def("propagate_covariance_to_cam_from_new_world",
        &PropagateCovarianceToCamFromNewWorld);

  m.def(
      "propagate_covariance_for_inverse",
      pybind11::overload_cast<const Sim3d&, const Eigen::Matrix<double, 7, 7>&>(
          &PropagateCovarianceForInverse),
      "sim3d"_a,
      "covar"_a);

  m.def("propagate_covariance_for_compose",
        pybind11::overload_cast<const Sim3d&,
                                const Eigen::Matrix<double, 14, 14>&>(
            &PropagateCovarianceForCompose),
        "left_sim3d"_a,
        "joint_covar"_a);

  m.def("propagate_covariance_for_relative",
        pybind11::overload_cast<const Sim3d&,
                                const Sim3d&,
                                const Eigen::Matrix<double, 14, 14>&>(
            &PropagateCovarianceForRelative),
        "base_sim3d"_a,
        "target_sim3d"_a,
        "joint_covar"_a);
  m.def("propagate_covariance_for_transform_point",
        pybind11::overload_cast<const Sim3d&, const Eigen::Matrix3d&>(
            &PropagateCovarianceForTransformPoint),
        "sim3"_a,
        "covar"_a);
}
