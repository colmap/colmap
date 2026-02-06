#include "colmap/feature/types.h"

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"

#include <cstring>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindFeatureTypes(py::module& m) {
  py::enum_<FeatureExtractorType>(m, "FeatureExtractorType")
      .value("UNDEFINED", FeatureExtractorType::UNDEFINED)
      .value("SIFT", FeatureExtractorType::SIFT)
      .value("ALIKED_N16ROT", FeatureExtractorType::ALIKED_N16ROT)
      .value("ALIKED_N32", FeatureExtractorType::ALIKED_N32);

  // Bind FeatureDescriptorsFloat first so it's available for to_float method
  auto PyFeatureDescriptorsFloat =
      py::classh<FeatureDescriptorsFloat>(m, "FeatureDescriptorsFloat")
          .def(py::init<>())
          .def(py::init<FeatureExtractorType, FeatureDescriptorsFloatData>(),
               "type"_a,
               "data"_a)
          .def_readwrite("type", &FeatureDescriptorsFloat::type)
          .def_readwrite("data", &FeatureDescriptorsFloat::data);
  MakeDataclass(PyFeatureDescriptorsFloat);

  auto PyFeatureDescriptors =
      py::classh<FeatureDescriptors>(m, "FeatureDescriptors")
          .def(py::init<>())
          .def(py::init<FeatureExtractorType, FeatureDescriptorsData>(),
               "type"_a,
               "data"_a)
          .def(py::init([](FeatureExtractorType type,
                           const FeatureDescriptorsFloatData& float_data) {
                 // Reinterpret float32 data as uint8, increasing columns by 4x.
                 const Eigen::Index rows = float_data.rows();
                 const Eigen::Index float_cols = float_data.cols();
                 const Eigen::Index uint8_cols = float_cols * sizeof(float);
                 FeatureDescriptorsData uint8_data(rows, uint8_cols);
                 std::memcpy(uint8_data.data(),
                             float_data.data(),
                             rows * float_cols * sizeof(float));
                 return FeatureDescriptors(type, std::move(uint8_data));
               }),
               "type"_a,
               "float_data"_a,
               "Construct from float data by reinterpreting as uint8 bytes.")
          .def(
              "to_float",
              [](const FeatureDescriptors& self) {
                // Reinterpret uint8 data as float32, decreasing columns by 4x
                const Eigen::Index rows = self.data.rows();
                const Eigen::Index uint8_cols = self.data.cols();
                THROW_CHECK_EQ(uint8_cols % sizeof(float), 0);
                const Eigen::Index float_cols = uint8_cols / sizeof(float);
                FeatureDescriptorsFloatData float_data(rows, float_cols);
                std::memcpy(
                    float_data.data(), self.data.data(), rows * uint8_cols);
                return FeatureDescriptorsFloat(self.type,
                                               std::move(float_data));
              },
              "Reinterpret uint8 data as float32, returning "
              "FeatureDescriptorsFloat.")
          .def_readwrite("type", &FeatureDescriptors::type)
          .def_readwrite("data", &FeatureDescriptors::data);
  MakeDataclass(PyFeatureDescriptors);

  auto PyFeatureKeypoint =
      py::classh<FeatureKeypoint>(m, "FeatureKeypoint")
          .def(py::init<>())
          .def_readwrite("x", &FeatureKeypoint::x)
          .def_readwrite("y", &FeatureKeypoint::y)
          .def_readwrite("a11", &FeatureKeypoint::a11)
          .def_readwrite("a12", &FeatureKeypoint::a12)
          .def_readwrite("a21", &FeatureKeypoint::a21)
          .def_readwrite("a22", &FeatureKeypoint::a22)
          .def_static("from_shape_parameters",
                      &FeatureKeypoint::FromShapeParameters)
          .def("rescale", py::overload_cast<float>(&FeatureKeypoint::Rescale))
          .def("rescale",
               py::overload_cast<float, float>(&FeatureKeypoint::Rescale))
          .def("compute_scale", &FeatureKeypoint::ComputeScale)
          .def("compute_scale_x", &FeatureKeypoint::ComputeScaleX)
          .def("compute_scale_y", &FeatureKeypoint::ComputeScaleY)
          .def("compute_orientation", &FeatureKeypoint::ComputeOrientation)
          .def("compute_shear", &FeatureKeypoint::ComputeShear)
          .def("__repr__", [](const FeatureKeypoint& keypoint) {
            std::ostringstream ss;
            ss << "FeatureKeypoint(x=" << keypoint.x << ", y=" << keypoint.y
               << ")";
            return ss.str();
          });
  MakeDataclass(PyFeatureKeypoint);
  py::bind_vector<FeatureKeypoints>(m, "FeatureKeypoints");
  py::implicitly_convertible<py::iterable, FeatureKeypoints>();

  auto PyFeatureMatch =
      py::classh<FeatureMatch>(m, "FeatureMatch")
          .def(py::init<>())
          .def(py::init<const point2D_t, const point2D_t>())
          .def_readwrite("point2D_idx1", &FeatureMatch::point2D_idx1)
          .def_readwrite("point2D_idx2", &FeatureMatch::point2D_idx2)
          .def("__repr__", [](const FeatureMatch& match) {
            std::ostringstream ss;
            ss << "FeatureMatch(idx1=" << match.point2D_idx1
               << ", idx2=" << match.point2D_idx2 << ")";
            return ss.str();
          });
  MakeDataclass(PyFeatureMatch);
  py::bind_vector<FeatureMatches>(m, "FeatureMatches");
  py::implicitly_convertible<py::iterable, FeatureMatches>();
}
