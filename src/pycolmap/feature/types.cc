#include "colmap/feature/types.h"

#include "pycolmap/feature/types.h"
#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

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

  // Define both classes first without cross-referencing methods.
  auto PyFeatureDescriptors =
      py::classh<FeatureDescriptors>(m, "FeatureDescriptors")
          .def(py::init<>())
          .def(py::init<FeatureExtractorType, FeatureDescriptorsData>(),
               "type"_a,
               "data"_a)
          .def_readwrite("type", &FeatureDescriptors::type)
          .def_readwrite("data", &FeatureDescriptors::data);
  auto PyFeatureDescriptorsFloat =
      py::classh<FeatureDescriptorsFloat>(m, "FeatureDescriptorsFloat")
          .def(py::init<>())
          .def(py::init<FeatureExtractorType, FeatureDescriptorsFloatData>(),
               "type"_a,
               "data"_a)
          .def_readwrite("type", &FeatureDescriptorsFloat::type)
          .def_readwrite("data", &FeatureDescriptorsFloat::data);

  // Add cross-referencing methods after both classes are defined.
  PyFeatureDescriptors
      .def_static("from_float",
                  &FeatureDescriptors::FromFloat,
                  "float_desc"_a,
                  "Create from float descriptors by reinterpreting float32 "
                  "data as uint8 bytes.")
      .def("to_float",
           &FeatureDescriptors::ToFloat,
           "Convert to float descriptors by reinterpreting uint8 data as "
           "float32.");
  PyFeatureDescriptorsFloat
      .def_static("from_bytes",
                  &FeatureDescriptorsFloat::FromBytes,
                  "byte_desc"_a,
                  "Create from byte descriptors by reinterpreting uint8 "
                  "data as float32.")
      .def("to_bytes",
           &FeatureDescriptorsFloat::ToBytes,
           "Convert to byte descriptors by reinterpreting float32 data as "
           "uint8.");

  MakeDataclass(PyFeatureDescriptors);
  MakeDataclass(PyFeatureDescriptorsFloat);

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

  m.def("keypoints_to_matrix",
        &KeypointsToMatrix,
        "keypoints"_a,
        "Convert FeatureKeypoints to an Nx4 matrix "
        "[x, y, scale, orientation].");

  m.def("keypoints_from_matrix",
        &KeypointsFromMatrix,
        "keypoints"_a,
        "Convert an Nx4 matrix [x, y, scale, orientation] to "
        "FeatureKeypoints.");

  m.def("matches_to_matrix",
        &MatchesToMatrix,
        "matches"_a,
        "Convert FeatureMatches to an Nx2 matrix of point2D indices.");

  m.def("matches_from_matrix",
        &MatchesFromMatrix,
        "matches"_a,
        "Convert an Nx2 matrix of point2D indices to FeatureMatches.");
}
