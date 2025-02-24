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
      .def("transform_camera_world", &TransformCameraWorld, "cam_from_world"_a)
      .def("inverse", static_cast<Sim3d (*)(const Sim3d&)>(&Inverse));
  py::implicitly_convertible<py::array, Sim3d>();
  MakeDataclass(PySim3d);
}
