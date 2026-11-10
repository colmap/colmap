// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/normal_map.h"

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

py::array_t<float> ArrayFromNormalMap(const NormalMap& self) {
  const size_t width = self.GetWidth();
  const size_t height = self.GetHeight();
  const size_t num_channels = self.GetDepth();
  const std::vector<ssize_t> shape = {static_cast<ssize_t>(height),
                                      static_cast<ssize_t>(width),
                                      static_cast<ssize_t>(num_channels)};
  py::array_t<float> output(shape);
  float* output_ptr = output.mutable_data();
  for (size_t d = 0; d < num_channels; ++d) {
    size_t output_index = d;
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        output_ptr[output_index] = self.Get(row, col, d);
        output_index += num_channels;
      }
    }
  }
  return output;
}

NormalMap NormalMapFromArray(py::array_t<float, py::array::c_style> array) {
  if (array.ndim() != 3) {
    throw std::runtime_error("Input array must have 3 dimensions!");
  }

  const int width = array.shape(1);
  const int height = array.shape(0);
  const int depth = array.shape(2);

  if (width == 0 || height == 0) {
    throw std::runtime_error(
        "Input array must have positive width and height!");
  }

  if (depth != 3) {
    throw std::runtime_error("Input array must have 3 channels for normals!");
  }

  NormalMap output(width, height);

  const float* input_ptr = array.data();
  for (size_t row = 0; row < output.GetHeight(); ++row) {
    for (size_t col = 0; col < output.GetWidth(); ++col) {
      for (size_t d = 0; d < 3; ++d) {
        output.Set(row, col, d, *input_ptr++);
      }
    }
  }
  return output;
}

void BindNormalMap(pybind11::module& m) {
  py::classh<NormalMap>(m, "NormalMap")
      .def(py::init<>())
      .def(py::init<size_t, size_t>(), "width"_a, "height"_a)
      .def("to_array", &ArrayFromNormalMap)
      .def_static("from_array",
                  &NormalMapFromArray,
                  "array"_a,
                  "Create normal map as a copy of array. Returns normal map "
                  "with shape (H, W, 3) where the 3 channels represent the "
                  "x, y, z components of the normal vectors.")
      .def("rescale", &NormalMap::Rescale, "factor"_a, "Rescale normal map.")
      .def("downsize",
           &NormalMap::Downsize,
           "max_width"_a,
           "max_height"_a,
           "Downsize normal map to fit maximum dimensions.")
      .def("to_bitmap",
           &NormalMap::ToBitmap,
           "Convert normal map to bitmap for visualization.")
      .def_property_readonly(
          "width", &NormalMap::GetWidth, "Width of the normal map.")
      .def_property_readonly(
          "height", &NormalMap::GetHeight, "Height of the normal map.")
      .def("read",
           &NormalMap::Read,
           "path"_a,
           "Read normal map from file at given path.")
      .def("write",
           &NormalMap::Write,
           "path"_a,
           "Write normal map to file at given path.")
      .def("__repr__", &CreateRepresentation<NormalMap>);
}
