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
  using Rotation3dWrapper = pycolmap::Rotation3dWrapper;
  py::classh_ext<Sim3d> PySim3d(m, "Sim3d");
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
          "params",
          [](Sim3d& self) -> Eigen::Vector8d& { return self.params; },
          [](Sim3d& self, const Eigen::Vector8d& params) {
            self.params = params;
          })
      .def_property(
          "scale",
          [](py::object self) {
            Sim3d& sim3 = self.cast<Sim3d&>();
            return py::array_t<double>({}, {}, sim3.params.data() + 7, self);
          },
          [](Sim3d& self, const py::object& value) {
            if (py::isinstance<py::array>(value)) {
              auto arr = value.cast<py::array_t<double>>();
              THROW_CHECK_EQ(arr.size(), 1);
              self.scale() = arr.at(0);
            } else {
              self.scale() = value.cast<double>();
            }
          })
      .def_property(
          "rotation",
          [](py::object self) {
            Sim3d& sim3 = self.cast<Sim3d&>();
            py::array_t<double> arr(
                {4}, {sizeof(double)}, sim3.params.data(), self);
            return Rotation3dWrapper(std::move(arr));
          },
          [](Sim3d& self, const Eigen::Quaterniond& q) { self.rotation() = q; })
      .def_property(
          "translation",
          [](py::object self) {
            Sim3d& sim3 = self.cast<Sim3d&>();
            return py::array_t<double>(
                {3}, {sizeof(double)}, sim3.params.data() + 4, self);
          },
          [](Sim3d& self, const Eigen::Vector3d& t) { self.translation() = t; })
      .def("matrix", &Sim3d::ToMatrix)
      .def(py::self * Sim3d())
      .def(py::self * Eigen::Vector3d())
      .def("__mul__",
           [](const Sim3d& t,
              const py::EigenDRef<const Eigen::MatrixX3d>& points)
               -> Eigen::MatrixX3d {
             return (t.scale() *
                     (points * t.rotation().toRotationMatrix().transpose()))
                        .rowwise() +
                    t.translation().transpose();
           })
      .def("transform_camera_world", &TransformCameraWorld, "cam_from_world"_a)
      .def("inverse", static_cast<Sim3d (*)(const Sim3d&)>(&Inverse));
  py::implicitly_convertible<py::array, Sim3d>();
  MakeDataclass(PySim3d);
}
