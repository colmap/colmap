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
  py::classh<Bitmap>(m, "Bitmap")
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
             std::memcpy(
                 output_row_ptr, self.RowMajorData().data(), self.NumBytes());
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
                  "Input array must have positive width and height!");
            }

            if (channels != 1 && channels != 3) {
              throw std::runtime_error(
                  "Input array must have 1 or 3 channels!");
            }

            const bool is_rgb = channels != 1;

            Bitmap output(width, height, is_rgb);

            const uint8_t* input_row_ptr =
                static_cast<uint8_t*>(array.request().ptr);

            std::memcpy(
                output.RowMajorData().data(), input_row_ptr, output.NumBytes());

            return output;
          },
          "array"_a,
          "Create bitmap as a copy of array. Returns RGB bitmap, if array has "
          "shape (H, W, 3), or grayscale bitmap, if array has shape (H, W[, "
          "1]).")
      .def("write",
           &Bitmap::Write,
           "path"_a,
           "delinearize_colorspace"_a,
           "Write bitmap to file.")
      .def("__repr__", &CreateRepresentation<Bitmap>)
      .def_static(
          "read",
          [](const std::string& path,
             bool as_rgb) -> py::typing::Optional<Bitmap> {
            Bitmap bitmap;
            if (!bitmap.Read(path, as_rgb)) {
              return py::none();
            }
            return py::cast(bitmap);
          },
          "path"_a,
          "as_rgb"_a,
          "Read bitmap from file.");
}
