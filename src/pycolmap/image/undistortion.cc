// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/undistortion.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindUndistortion(py::module& m) {
  auto PyWarpImageOptions =
      py::classh<WarpImageOptions>(m, "WarpImageOptions").def(py::init<>());
  auto PyInterpolation =
      py::enum_<WarpImageOptions::Interpolation>(PyWarpImageOptions,
                                                 "Interpolation")
          .value("NEAREST_NEIGHBOR",
                 WarpImageOptions::Interpolation::kNearestNeighbor)
          .value("BILINEAR", WarpImageOptions::Interpolation::kBilinear);
  AddStringToEnumConstructor(PyInterpolation);
  PyWarpImageOptions
      .def_readwrite("interpolation", &WarpImageOptions::interpolation)
      .def_readwrite("direct_warp_min_scale",
                     &WarpImageOptions::direct_warp_min_scale);
  MakeDataclass(PyWarpImageOptions);

  m.def(
      "warp_image_between_cameras",
      [](const WarpImageOptions& options,
         const Camera& source_camera,
         const Camera& target_camera,
         const Bitmap& source_image) -> Bitmap {
        py::gil_scoped_release release;
        Bitmap target_image;
        WarpImageBetweenCameras(
            options, source_camera, target_camera, source_image, &target_image);
        return target_image;
      },
      "options"_a,
      "source_camera"_a,
      "target_camera"_a,
      "source_image"_a,
      "Warp the image of the source camera to the target camera by inverse "
      "mapping the target pixels through the target and source cameras. "
      "Target pixels that project outside the source image are black.");

  auto PyUndistortCameraOptions =
      py::classh<UndistortCameraOptions>(m, "UndistortCameraOptions")
          .def(py::init<>())
          .def_readwrite("blank_pixels", &UndistortCameraOptions::blank_pixels)
          .def_readwrite("min_scale", &UndistortCameraOptions::min_scale)
          .def_readwrite("max_scale", &UndistortCameraOptions::max_scale)
          .def_readwrite("max_image_size",
                         &UndistortCameraOptions::max_image_size)
          .def_readwrite("roi_min_x", &UndistortCameraOptions::roi_min_x)
          .def_readwrite("roi_min_y", &UndistortCameraOptions::roi_min_y)
          .def_readwrite("roi_max_x", &UndistortCameraOptions::roi_max_x)
          .def_readwrite("roi_max_y", &UndistortCameraOptions::roi_max_y)
          .def_readwrite("max_cam_point_norm",
                         &UndistortCameraOptions::max_cam_point_norm)
          .def_readwrite("warp_options", &UndistortCameraOptions::warp_options);
  MakeDataclass(PyUndistortCameraOptions);

  m.def("undistort_camera",
        &UndistortCamera,
        "options"_a,
        "camera"_a,
        "Undistort camera.");

  m.def(
      "undistort_image",
      [](const UndistortCameraOptions& options,
         const Bitmap& distorted_image,
         const Camera& distorted_camera) -> std::pair<Bitmap, Camera> {
        py::gil_scoped_release release;
        std::pair<Bitmap, Camera> result;
        UndistortImage(options,
                       distorted_image,
                       distorted_camera,
                       &result.first,
                       &result.second);
        return result;
      },
      "options"_a,
      "distorted_image"_a,
      "distorted_camera"_a,
      "Undistort image and corresponding camera.");
}
