#include "colmap/util/types.h"

#include <pybind11/pybind11.h>

using namespace colmap;
namespace py = pybind11;

void BindConstants(py::module& m) {
  m.attr("invalid_camera_id") = colmap::kInvalidCameraId;
  m.attr("invalid_image_id") = colmap::kInvalidImageId;
  m.attr("invalid_image_pair_id") = colmap::kInvalidImagePairId;
  m.attr("invalid_point2D_idx") = colmap::kInvalidPoint2DIdx;
  m.attr("invalid_point3D_id") = colmap::kInvalidPoint3DId;
}
