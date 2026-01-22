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
  using Rotation3dWrapper = pycolmap::Rotation3dWrapper;
  py::classh_ext<Rotation3dWrapper> PyRotation3d(m, "Rotation3d");
  PyRotation3d.def(py::init<>())
      .def(py::init<const Eigen::Vector4d&>(),
           "xyzw"_a,
           "Quaternion in [x,y,z,w] format.")
      .def(py::init<const Eigen::Matrix3d&>(),
           "matrix"_a,
           "3x3 rotation matrix.")
      .def(py::init<const Eigen::Vector3d&>(),
           "axis_angle"_a,
           "Axis-angle 3D vector.")
      .def_static(
          "from_buffer",
          [](py::array_t<double> arr) {
            THROW_CHECK_EQ(arr.size(), 4);
            return Rotation3dWrapper(std::move(arr));
          },
          "array"_a,
          "Create from numpy array view (zero-copy if contiguous).")
      .def_property(
          "quat",
          [](Rotation3dWrapper& self) { return self.data; },
          [](Rotation3dWrapper& self, const Eigen::Vector4d& quat) {
            self.map().coeffs() = quat;
          },
          "Quaternion in [x,y,z,w] format.")
      .def("__mul__",
           [](const Rotation3dWrapper& self, const Rotation3dWrapper& other) {
             return Rotation3dWrapper(self.map() * other.map());
           })
      .def("__mul__",
           [](const Rotation3dWrapper& self, const Eigen::Vector3d& v) {
             return Eigen::Vector3d(self.map() * v);
           })
      .def("__mul__",
           [](const Rotation3dWrapper& self,
              const py::EigenDRef<const Eigen::MatrixX3d>& points)
               -> Eigen::MatrixX3d {
             return points * self.map().toRotationMatrix().transpose();
           })
      .def("normalize", [](Rotation3dWrapper& self) { self.map().normalize(); })
      .def("matrix",
           [](const Rotation3dWrapper& self) {
             return self.map().toRotationMatrix();
           })
      .def("norm",
           [](const Rotation3dWrapper& self) { return self.map().norm(); })
      .def("angle",
           [](const Rotation3dWrapper& self) {
             return Eigen::AngleAxis<double>(self.map()).angle();
           })
      .def(
          "angle_to",
          [](const Rotation3dWrapper& self, const Rotation3dWrapper& other) {
            return self.map().angularDistance(other.map());
          },
          "other"_a)
      .def("inverse",
           [](const Rotation3dWrapper& self) {
             return Rotation3dWrapper(self.map().inverse());
           })
      .def("__repr__", [](const Rotation3dWrapper& self) {
        std::ostringstream ss;
        ss << "Rotation3d(xyzw=[" << self.map().coeffs().format(vec_fmt)
           << "])";
        return ss.str();
      });
  py::implicitly_convertible<py::array, Rotation3dWrapper>();
  // Define deepcopy before MakeDataclass to ensure proper deep copy of array
  // data
  PyRotation3d.def("__deepcopy__",
                   [](const Rotation3dWrapper& self, const py::dict&) {
                     Rotation3dWrapper copy;
                     copy.map() = self.map();
                     return copy;
                   });
  MakeDataclass(PyRotation3d);

  py::classh_ext<Eigen::AlignedBox3d> PyAlignedBox3d(m, "AlignedBox3d");
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
