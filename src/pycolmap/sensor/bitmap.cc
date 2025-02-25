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

void BindBitmap(pybind11::module& m) {
  py::class_<Bitmap>(m, "Bitmap")
      .def(py::init<>())
      .def("to_array",
           [](const Bitmap& self) {
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
             uint8_t* output_row_ptr =
                 reinterpret_cast<uint8_t*>(output.request().ptr);
             const size_t output_pitch =
                 (output_into.shape.size() == 2)
                     ? output_into.shape[1]
                     : (output_into.shape[1] * output_into.shape[2]);
             for (ssize_t y = 0; y < output_into.shape[0]; ++y) {
               if (is_rgb) {
                 for (ssize_t x = 0; x < output_into.shape[1]; ++x) {
                   // Notice that the underlying FreeImage buffer may order
                   // the channels as BGR or in any other format and with
                   // different striding, so we have to set each pixel
                   // separately.
                   // We always return the array in the order R, G, B.
                   BitmapColor<uint8_t> color;
                   THROW_CHECK(self.GetPixel(x, y, &color));
                   output_row_ptr[3 * x] = color.r;
                   output_row_ptr[3 * x + 1] = color.g;
                   output_row_ptr[3 * x + 2] = color.b;
                 }
               } else {
                 // Copy (guaranteed contiguous) row memory directly.
                 std::memcpy(const_cast<uint8_t*>(self.GetScanline(y)),
                             output_row_ptr,
                             output_into.shape[1]);
               }
               output_row_ptr += output_pitch;
             }
             return output;
           })
      .def_static(
          "from_array",
          [](py::array_t<uint8_t, py::array::c_style> array) -> Bitmap {
            int channels = 1;
            if (array.ndim() == 3) {
              channels = array.shape(2);
            } else if (array.ndim() != 2) {
              throw std::runtime_error(
                  "Input array must have 2 or 3 dimensions!");
            }

            const int width = array.shape(1);
            const int height = array.shape(0);
            if (width == 0 || height == 0) {
              throw std::runtime_error(
                  "Input array must have positive width and height");
            }

            if (channels != 1 && channels != 3 && channels != 4) {
              throw std::runtime_error(
                  "Input array must have 1, 3, or 4 channels!");
            }

            const bool is_rgb = channels != 1;
            const size_t pitch = width * channels;

            Bitmap output;
            output.Allocate(width, height, is_rgb);

            const uint8_t* input_row_ptr =
                static_cast<uint8_t*>(array.request().ptr);

            for (int y = 0; y < height; ++y) {
              if (is_rgb) {
                for (int x = 0; x < width; ++x) {
                  // We assume that provided array dimensions are R, G, B.
                  // Notice that the underlying FreeImage buffer may order
                  // the channels as BGR or in any other format and with
                  // different striding, so we have to set each pixel
                  // separately.
                  output.SetPixel(
                      x,
                      y,
                      BitmapColor<uint8_t>(input_row_ptr[channels * x],
                                           input_row_ptr[channels * x + 1],
                                           input_row_ptr[channels * x + 2]));
                }
              } else {
                // Copy (guaranteed contiguous) row memory directly.
                std::memcpy(const_cast<uint8_t*>(output.GetScanline(y)),
                            input_row_ptr,
                            width);
              }

              input_row_ptr += pitch;
            }

            return output;
          },
          "array"_a,
          "Create bitmap as a copy of array. Returns RGB bitmap, if array has "
          "shape (H, W, 3), or grayscale bitmap, if array has shape (H, W[, "
          "1]).")
      .def("write",
           &Bitmap::Write,
           "path"_a,
           "flags"_a = 0,
           "Write bitmap to file.")
      .def("__repr__", &CreateRepresentation<Bitmap>)
      .def_static(
          "read",
          [](const std::string& path, bool as_rgb) {
            Bitmap bitmap;
            bitmap.Read(path, as_rgb);
            return bitmap;
          },
          "path"_a,
          "as_rgb"_a,
          "Read bitmap from file.");
}
