#include "colmap/util/types.h"

#include <pybind11/pybind11.h>

using namespace colmap;
namespace py = pybind11;

void BindConstants(py::module& m) {
  m.attr("INVALID_CAMERA_ID") = colmap::kInvalidCameraId;
  m.attr("INVALID_IMAGE_ID") = colmap::kInvalidImageId;
  m.attr("INVALID_IMAGE_PAIR_ID") = colmap::kInvalidImagePairId;
  m.attr("INVALID_POINT2D_IDX") = colmap::kInvalidPoint2DIdx;
  m.attr("INVALID_POINT3D_ID") = colmap::kInvalidPoint3DId;
}
