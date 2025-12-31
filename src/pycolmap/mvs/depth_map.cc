#include "colmap/mvs/depth_map.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace colmap::mvs;
using namespace pybind11::literals;
namespace py = pybind11;

#ifdef _MSC_VER  // If compiling with MSVC
#include <stddef.h>
typedef ptrdiff_t ssize_t;
#endif

py::array_t<float> ArrayFromDepthMap(const DepthMap& self) {
  const std::vector<ssize_t> shape = {static_cast<ssize_t>(self.GetHeight()),
                                      static_cast<ssize_t>(self.GetWidth())};
  py::array_t<float> output(shape);
  py::buffer_info output_info = output.request();
  float* output_ptr = reinterpret_cast<float*>(output_info.ptr);
  std::memcpy(output_ptr, self.GetPtr(), self.GetNumBytes());
  return output;
}

DepthMap DepthMapFromArray(py::array_t<float, py::array::c_style> array,
                           float depth_min,
                           float depth_max) {
  if (array.ndim() != 2) {
    throw std::runtime_error("Input array must have 2 dimensions!");
  }

  const int width = array.shape(1);
  const int height = array.shape(0);
  if (width == 0 || height == 0) {
    throw std::runtime_error(
        "Input array must have positive width and height!");
  }

  DepthMap output(width, height, depth_min, depth_max);

  const float* input_ptr = static_cast<float*>(array.request().ptr);
  std::memcpy(output.GetPtr(), input_ptr, output.GetNumBytes());
  return output;
}

void BindDepthMap(pybind11::module& m) {
  py::classh<DepthMap>(m, "DepthMap")
      .def(py::init<>())
      .def(py::init<size_t, size_t, float, float>(),
           "width"_a,
           "height"_a,
           "depth_min"_a,
           "depth_max"_a)
      .def("to_array", &ArrayFromDepthMap)
      .def_static("from_array",
                  &DepthMapFromArray,
                  "array"_a,
                  "depth_min"_a,
                  "depth_max"_a,
                  "Create depth map as a copy of array. Returns depth map "
                  "with shape (H, W).")
      .def("rescale", &DepthMap::Rescale, "factor"_a, "Rescale depth map.")
      .def("downsize",
           &DepthMap::Downsize,
           "max_width"_a,
           "max_height"_a,
           "Downsize depth map to fit maximum dimensions.")
      .def("to_bitmap",
           &DepthMap::ToBitmap,
           "min_percentile"_a,
           "max_percentile"_a,
           "Convert depth map to bitmap for visualization.")
      .def_property_readonly(
          "width", &DepthMap::GetWidth, "Width of the depth map.")
      .def_property_readonly(
          "height", &DepthMap::GetHeight, "Height of the depth map.")
      .def_property_readonly(
          "depth_min", &DepthMap::GetDepthMin, "Minimum depth value.")
      .def_property_readonly(
          "depth_max", &DepthMap::GetDepthMax, "Maximum depth value.")
      .def("read",
           &DepthMap::Read,
           "path"_a,
           "Read depth map from file at given path.")
      .def("write",
           &DepthMap::Write,
           "path"_a,
           "Write depth map to file at given path.")
      .def("__repr__", &CreateRepresentation<DepthMap>);
}
