#pragma once

#include "colmap/sensor/bitmap.h"

#include <pybind11/numpy.h>

namespace py = pybind11;

colmap::Bitmap BitmapFromArray(py::array_t<uint8_t, py::array::c_style> array,
                               bool linear_colorspace = false);
