#include "colmap/sensor/bitmap.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

#ifdef _MSC_VER  // If compiling with MSVC
#include <stddef.h>
typedef ptrdiff_t ssize_t;
#endif

py::array_t<uint8_t> ArrayFromBitmap(const Bitmap& self) {
  std::vector<ssize_t> shape = {static_cast<ssize_t>(self.Height()),
                                static_cast<ssize_t>(self.Width())};
  const auto channels = static_cast<ssize_t>(self.Channels());
  const bool is_rgb = self.IsRGB();
  if (channels != 1) {
    if (channels != 3) {
      throw std::runtime_error(
          "Can only convert grayscale or 3-channel RGB image to "
          "array");
    }
    shape.push_back(channels);
  }
  py::array_t<uint8_t> output(shape);
  py::buffer_info output_into = output.request();
  uint8_t* output_row_ptr = reinterpret_cast<uint8_t*>(output.request().ptr);
  std::memcpy(output_row_ptr, self.RowMajorData().data(), self.NumBytes());
  return output;
}

Bitmap BitmapFromArray(py::array_t<uint8_t, py::array::c_style> array,
                       bool linear_colorspace) {
  int channels = 1;
  if (array.ndim() == 3) {
    channels = array.shape(2);
  } else if (array.ndim() != 2) {
    throw std::runtime_error("Input array must have 2 or 3 dimensions!");
  }

  const int width = array.shape(1);
  const int height = array.shape(0);
  if (width == 0 || height == 0) {
    throw std::runtime_error(
        "Input array must have positive width and height!");
  }

  if (channels != 1 && channels != 3) {
    throw std::runtime_error("Input array must have 1 or 3 channels!");
  }

  const bool as_rgb = channels != 1;
  const size_t pitch = width * channels;

  Bitmap output(width,
                height,
                /*as_rgb=*/as_rgb,
                /*linear_colorspace=*/linear_colorspace);

  const uint8_t* input_row_ptr = static_cast<uint8_t*>(array.request().ptr);
  std::memcpy(output.RowMajorData().data(), input_row_ptr, output.NumBytes());
  return output;
}

void BindBitmap(pybind11::module& m) {
  using BitmapRescaleFilter = Bitmap::RescaleFilter;
  py::enum_<BitmapRescaleFilter> PyRescaleFilter(m, "BitmapRescaleFilter");
  PyRescaleFilter.value("BILINEAR", BitmapRescaleFilter::kBilinear)
      .value("BOX", BitmapRescaleFilter::kBox);
  AddStringToEnumConstructor(PyRescaleFilter);

  py::classh<Bitmap>(m, "Bitmap")
      .def(py::init<>())
      .def(py::init<>(
               [](int width, int height, bool as_rgb, bool linear_colorspace) {
                 return Bitmap(width,
                               height,
                               /*as_rgb=*/as_rgb,
                               /*linear_colorspace=*/linear_colorspace);
               }),
           "width"_a,
           "height"_a,
           "as_rgb"_a,
           "linear_colorspace"_a = false)
      .def("to_array", &ArrayFromBitmap)
      .def_static("from_array",
                  &BitmapFromArray,
                  "array"_a,
                  "linear_colorspace"_a = false,
                  "Create bitmap as a copy of array. Returns RGB bitmap, "
                  "if array has shape (H, W, 3), or grayscale bitmap, if "
                  "array has shape (H, W[, 1]).")
      .def("write",
           &Bitmap::Write,
           "path"_a,
           "delinearize_colorspace"_a = true,
           "Write bitmap to file at given path. Defaults to converting to "
           "sRGB colorspace for file storage.")
      .def_static(
          "read",
          [](const std::string& path,
             bool as_rgb,
             bool linearize_colorspace) -> py::typing::Optional<Bitmap> {
            Bitmap bitmap;
            if (!bitmap.Read(path,
                             /*as_rgb=*/as_rgb,
                             /*linearize_colorspace=*/linearize_colorspace)) {
              return py::none();
            }
            return py::cast(bitmap);
          },
          "path"_a,
          "as_rgb"_a,
          "linearize_colorspace"_a = false,
          "Read bitmap at given path and convert to grey- or colorscale. "
          "Defaults to keeping the original colorspace (potentially "
          "non-linear) for image processing.")
      .def("rescale",
           &Bitmap::Rescale,
           "new_width"_a,
           "new_height"_a,
           "filter"_a = BitmapRescaleFilter::kBilinear,
           "Rescale image to the new dimensions.")
      .def_property_readonly("width", &Bitmap::Width, "Width of the image.")
      .def_property_readonly("height", &Bitmap::Height, "Height of the image.")
      .def_property_readonly(
          "channels", &Bitmap::Channels, "Number of channels of the image.")
      .def_property_readonly(
          "is_rgb", &Bitmap::IsRGB, "Whether the image is colorscale.")
      .def_property_readonly(
          "is_grey", &Bitmap::IsGrey, "Whether the image is greyscale.")
      .def("__repr__", &CreateRepresentation<Bitmap>);
}
