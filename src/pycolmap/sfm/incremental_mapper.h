#pragma once

#include "colmap/sfm/incremental_mapper.h"

#include "pycolmap/helpers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
namespace py = pybind11;

void BindIncrementalMapper(py::module& m) {
  using ImageSelection = IncrementalMapper::Options::ImageSelectionMethod;
  auto PyImageSelectionMethod =
      py::enum_<ImageSelection>(m, "ImageSelectionMethod")
          .value("MAX_VISIBLE_POINTS_NUM",
                 ImageSelection::MAX_VISIBLE_POINTS_NUM)
          .value("MAX_VISIBLE_POINTS_RATIO",
                 ImageSelection::MAX_VISIBLE_POINTS_RATIO)
          .value("MIN_UNCERTAINTY", ImageSelection::MIN_UNCERTAINTY);
  AddStringToEnumConstructor(PyImageSelectionMethod);

  using Opts = IncrementalMapper::Options;
  auto PyOpts = py::class_<Opts>(m, "IncrementalMapperOptions");
  PyOpts.def(py::init<>())
      .def_readwrite("init_min_num_inliers",
                     &Opts::init_min_num_inliers,
                     "Minimum number of inliers for initial image pair.")
      .def_readwrite("init_max_error",
                     &Opts::init_max_error,
                     "Maximum error in pixels for two-view geometry estimation "
                     "for initial image pair.")
      .def_readwrite("init_max_forward_motion",
                     &Opts::init_max_forward_motion,
                     "Maximum forward motion for initial image pair.")
      .def_readwrite("init_min_tri_angle",
                     &Opts::init_min_tri_angle,
                     "Minimum triangulation angle for initial image pair.")
      .def_readwrite(
          "init_max_reg_trials",
          &Opts::init_max_reg_trials,
          "Maximum number of trials to use an image for initialization.")
      .def_readwrite("abs_pose_max_error",
                     &Opts::abs_pose_max_error,
                     "Maximum reprojection error in absolute pose estimation.")
      .def_readwrite("abs_pose_min_num_inliers",
                     &Opts::abs_pose_min_num_inliers,
                     "Minimum number of inliers in absolute pose estimation.")
      .def_readwrite("abs_pose_min_inlier_ratio",
                     &Opts::abs_pose_min_inlier_ratio,
                     "Minimum inlier ratio in absolute pose estimation.")
      .def_readwrite(
          "abs_pose_refine_focal_length",
          &Opts::abs_pose_refine_focal_length,
          "Whether to estimate the focal length in absolute pose estimation.")
      .def_readwrite("abs_pose_refine_extra_params",
                     &Opts::abs_pose_refine_extra_params,
                     "Whether to estimate the extra parameters in absolute "
                     "pose estimation.")
      .def_readwrite("local_ba_num_images",
                     &Opts::local_ba_num_images,
                     "Number of images to optimize in local bundle adjustment.")
      .def_readwrite("local_ba_min_tri_angle",
                     &Opts::local_ba_min_tri_angle,
                     "Minimum triangulation for images to be chosen in local "
                     "bundle adjustment.")
      .def_readwrite("min_focal_length_ratio",
                     &Opts::min_focal_length_ratio,
                     "The threshold used to filter and ignore images with "
                     "degenerate intrinsics.")
      .def_readwrite("max_focal_length_ratio",
                     &Opts::max_focal_length_ratio,
                     "The threshold used to filter and ignore images with "
                     "degenerate intrinsics.")
      .def_readwrite("max_extra_param",
                     &Opts::max_extra_param,
                     "The threshold used to filter and ignore images with "
                     "degenerate intrinsics.")
      .def_readwrite("filter_max_reproj_error",
                     &Opts::filter_max_reproj_error,
                     "Maximum reprojection error in pixels for observations.")
      .def_readwrite(
          "filter_min_tri_angle",
          &Opts::filter_min_tri_angle,
          "Minimum triangulation angle in degrees for stable 3D points.")
      .def_readwrite("max_reg_trials",
                     &Opts::max_reg_trials,
                     "Maximum number of trials to register an image.")
      .def_readwrite("fix_existing_images",
                     &Opts::fix_existing_images,
                     "If reconstruction is provided as input, fix the existing "
                     "image poses.")
      .def_readwrite("num_threads", &Opts::num_threads, "Number of threads.")
      .def_readwrite("image_selection_method",
                     &Opts::image_selection_method,
                     "Method to find and select next best image to register.");
  MakeDataclass(PyOpts);
}
