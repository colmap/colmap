#include "colmap/sfm/incremental_triangulator.h"

#include "pycolmap/helpers.h"

#include <memory>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindIncrementalTriangulator(py::module& m) {
  using Opts = IncrementalTriangulator::Options;
  auto PyOpts = py::classh<Opts>(m, "IncrementalTriangulatorOptions");
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
                     "degenerate intrinsics.")
      .def_readwrite(
          "random_seed",
          &Opts::random_seed,
          "PRNG seed for all stochastic methods during triangulation.")
      .def("check", &Opts::Check);
  MakeDataclass(PyOpts);

  py::classh<IncrementalTriangulator>(
      m,
      "IncrementalTriangulator",
      "Class that triangulates points during the incremental reconstruction. "
      "It holds the state and provides all functionality for triangulation.")
      .def(py::init<std::shared_ptr<const CorrespondenceGraph>,
                    Reconstruction&,
                    std::shared_ptr<ObservationManager>>(),
           "correspondence_graph"_a,
           "reconstruction"_a,
           "observation_manager"_a = py::none(),
           py::keep_alive<1, 3>(),
           "Create new incremental triangulator. Note that both the "
           "correspondence graph and the reconstruction objects must live "
           "as long as the triangulator.")
      .def("triangulate_image",
           &IncrementalTriangulator::TriangulateImage,
           "options"_a,
           "image_id"_a,
           "Triangulate observations of image. Triangulation includes "
           "creation of new points, continuation of existing points, and "
           "merging of separate points if given image bridges tracks. Note "
           "that the given image must be registered and its pose must be set "
           "in the associated reconstruction.")
      .def("complete_image",
           &IncrementalTriangulator::CompleteImage,
           "options"_a,
           "image_id"_a,
           "Complete triangulations for image. Tries to create new tracks "
           "for not yet triangulated observations and tries to complete "
           "existing tracks. Returns the number of completed observations.")
      .def("complete_all_tracks",
           &IncrementalTriangulator::CompleteAllTracks,
           "options"_a,
           "Complete tracks of all 3D points. Returns the number of "
           "completed observations.")
      .def("merge_all_tracks",
           &IncrementalTriangulator::MergeAllTracks,
           "options"_a,
           "Merge tracks of all 3D points. Returns the number of merged "
           "observations.")
      .def("retriangulate",
           &IncrementalTriangulator::Retriangulate,
           "options"_a,
           "Perform retriangulation for under-reconstructed image pairs. "
           "Under-reconstruction usually occurs in the case of a drifting "
           "reconstruction.")
      .def("add_modified_point3D",
           &IncrementalTriangulator::AddModifiedPoint3D,
           "point3D_id"_a,
           "Indicate that a 3D point has been modified.")
      .def("clear_modified_points3D",
           &IncrementalTriangulator::ClearModifiedPoints3D,
           "Clear the collection of changed 3D points.")
      .def("get_modified_points3D",
           &IncrementalTriangulator::GetModifiedPoints3D,
           "Get changed 3D points, since the last call to "
           "clear_modified_points3D.")
      .def("merge_tracks",
           &IncrementalTriangulator::MergeTracks,
           "options"_a,
           "point3D_ids"_a,
           "Merge tracks for specific 3D points. Returns the number of "
           "merged observations.")
      .def("complete_tracks",
           &IncrementalTriangulator::CompleteTracks,
           "options"_a,
           "point3D_ids"_a,
           "Complete tracks for specific 3D points. Completion tries to "
           "recursively add observations to a track that might have failed "
           "to triangulate before due to inaccurate poses, etc. Returns "
           "the number of completed observations.")
      .def("__copy__",
           [](const IncrementalTriangulator& self) {
             return IncrementalTriangulator(self);
           })
      .def("__deepcopy__",
           [](const IncrementalTriangulator& self, const py::dict&) {
             return IncrementalTriangulator(self);
           })
      .def("__repr__", &CreateRepresentation<IncrementalTriangulator>);
}
