#pragma once

#include "colmap/sfm/incremental_triangulator.h"

#include "pycolmap/helpers.h"

#include <memory>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
namespace py = pybind11;

void BindIncrementalTriangulator(py::module& m) {
  using Opts = IncrementalTriangulator::Options;
  auto PyOpts = py::class_<Opts>(m, "IncrementalTriangulatorOptions");
  PyOpts.def(py::init<>())
      .def_readwrite("max_transitivity",
                     &Opts::max_transitivity,
                     "Maximum transitivity to search for correspondences.")
      .def_readwrite("create_max_angle_error",
                     &Opts::create_max_angle_error,
                     "Maximum angular error to create new triangulations.")
      .def_readwrite(
          "continue_max_angle_error",
          &Opts::continue_max_angle_error,
          "Maximum angular error to continue existing triangulations.")
      .def_readwrite(
          "merge_max_reproj_error",
          &Opts::merge_max_reproj_error,
          "Maximum reprojection error in pixels to merge triangulations.")
      .def_readwrite(
          "complete_max_reproj_error",
          &Opts::complete_max_reproj_error,
          "Maximum reprojection error to complete an existing triangulation.")
      .def_readwrite("complete_max_transitivity",
                     &Opts::complete_max_transitivity,
                     "Maximum transitivity for track completion.")
      .def_readwrite("re_max_angle_error",
                     &Opts::re_max_angle_error,
                     "Maximum angular error to re-triangulate "
                     "under-reconstructed image pairs.")
      .def_readwrite("re_min_ratio",
                     &Opts::re_min_ratio,
                     "Minimum ratio of common triangulations between an image "
                     "pair over the number of correspondences between that "
                     "image pair to be considered as under-reconstructed.")
      .def_readwrite(
          "re_max_trials",
          &Opts::re_max_trials,
          "Maximum number of trials to re-triangulate an image pair.")
      .def_readwrite(
          "min_angle",
          &Opts::min_angle,
          "Minimum pairwise triangulation angle for a stable triangulation.")
      .def_readwrite("ignore_two_view_tracks",
                     &Opts::ignore_two_view_tracks,
                     "Whether to ignore two-view tracks.")
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
                     "degenerate intrinsics.");
  MakeDataclass(PyOpts);

  // TODO: Add bindings for GetModifiedPoints3D.
  // TODO: Add bindings for Find, Create, Continue, Merge, Complete,
  // HasCameraBogusParams once they become public.
  py::class_<IncrementalTriangulator, std::shared_ptr<IncrementalTriangulator>>(
      m, "IncrementalTriangulator")
      .def(py::init<std::shared_ptr<CorrespondenceGraph>,
                    std::shared_ptr<Reconstruction>>())
      .def("triangulate_image", &IncrementalTriangulator::TriangulateImage)
      .def("complete_image", &IncrementalTriangulator::CompleteImage)
      .def("complete_all_tracks", &IncrementalTriangulator::CompleteAllTracks)
      .def("merge_all_tracks", &IncrementalTriangulator::MergeAllTracks)
      .def("retriangulate", &IncrementalTriangulator::Retriangulate)
      .def("add_modified_point3D", &IncrementalTriangulator::AddModifiedPoint3D)
      .def("clear_modified_points3D",
           &IncrementalTriangulator::ClearModifiedPoints3D)
      .def("merge_tracks", &IncrementalTriangulator::MergeTracks)
      .def("complete_tracks", &IncrementalTriangulator::CompleteTracks)
      .def("__copy__",
           [](const IncrementalTriangulator& self) {
             return IncrementalTriangulator(self);
           })
      .def("__deepcopy__",
           [](const IncrementalTriangulator& self, const py::dict&) {
             return IncrementalTriangulator(self);
           })
      .def("__repr__", [](const IncrementalTriangulator& self) {
        // TODO: Print reconstruction and correspondence_graph once public.
        return "IncrementalTriangulator()";
      });
}
