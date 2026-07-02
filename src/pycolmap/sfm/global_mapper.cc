#include "colmap/sfm/global_mapper.h"

#include "colmap/controllers/global_pipeline.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_manager.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <memory>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindGlobalMapper(py::module& m) {
  {
    using Opts = GlobalMapperOptions;
    auto PyOpts = py::classh<Opts>(m, "GlobalMapperOptions");
    PyOpts.def(py::init<>())
        .def_readwrite("num_threads", &Opts::num_threads)
        .def_readwrite("random_seed", &Opts::random_seed)
        .def_readwrite(
            "image_path",
            &Opts::image_path,
            "The image path at which to find the images to extract point "
            "colors.")
        .def_readwrite("refine_sensor_from_rig",
                       &Opts::refine_sensor_from_rig,
                       "When False, treat each non-ref sensor's "
                       "cam_from_rig as a pre-calibrated constant across "
                       "rotation averaging, global positioning and "
                       "bundle adjustment.")
        .def_readwrite("rotation_averaging", &Opts::rotation_averaging)
        .def_readwrite("global_positioning", &Opts::global_positioning)
        .def_readwrite("bundle_adjustment", &Opts::bundle_adjustment)
        .def_readwrite("retriangulation", &Opts::retriangulation)
        .def_readwrite("track_intra_image_consistency_threshold",
                       &Opts::track_intra_image_consistency_threshold)
        .def_readwrite("track_required_tracks_per_view",
                       &Opts::track_required_tracks_per_view)
        .def_readwrite("track_min_num_views_per_track",
                       &Opts::track_min_num_views_per_track)
        .def_readwrite("keep_max_num_tracks", &Opts::keep_max_num_tracks)
        .def_readwrite("max_angular_reproj_error_deg",
                       &Opts::max_angular_reproj_error_deg)
        .def_readwrite("max_normalized_reproj_error",
                       &Opts::max_normalized_reproj_error)
        .def_readwrite("min_tri_angle_deg", &Opts::min_tri_angle_deg)
        .def_readwrite("ba_gpu_index", &Opts::ba_gpu_index)
        .def_readwrite("ba_num_iterations", &Opts::ba_num_iterations)
        .def_readwrite("ba_skip_fixed_rotation_stage",
                       &Opts::ba_skip_fixed_rotation_stage)
        .def_readwrite("ba_skip_joint_optimization_stage",
                       &Opts::ba_skip_joint_optimization_stage)
        .def_readwrite("skip_rotation_averaging",
                       &Opts::skip_rotation_averaging)
        .def_readwrite("skip_track_establishment",
                       &Opts::skip_track_establishment)
        .def_readwrite("skip_global_positioning",
                       &Opts::skip_global_positioning)
        .def_readwrite("skip_bundle_adjustment", &Opts::skip_bundle_adjustment)
        .def_readwrite("skip_retriangulation", &Opts::skip_retriangulation)
        .def("get_rotation_averaging",
             &Opts::RotationAveraging,
             "Get rotation averaging options with shared settings applied.")
        .def("get_global_positioning",
             &Opts::GlobalPositioning,
             "Get global positioning options with shared settings applied.")
        .def("get_bundle_adjustment",
             &Opts::BundleAdjustment,
             "Get bundle adjustment options with shared settings applied.")
        .def("get_retriangulation",
             &Opts::Retriangulation,
             "Get retriangulation options with shared settings applied.");
    MakeDataclass(PyOpts);
  }

  {
    using Opts = GlobalPipelineOptions;
    auto PyOpts = py::classh<Opts>(m, "GlobalPipelineOptions");
    PyOpts.def(py::init<>())
        .def_readwrite("min_num_matches", &Opts::min_num_matches)
        .def_readwrite("ignore_watermarks", &Opts::ignore_watermarks)
        .def_readwrite("image_names", &Opts::image_names)
        .def_readwrite(
            "image_path",
            &Opts::image_path,
            "The image path at which to find the images to extract point "
            "colors.")
        .def_readwrite("num_threads", &Opts::num_threads)
        .def_readwrite("random_seed", &Opts::random_seed)
        .def_readwrite("decompose_relative_pose",
                       &Opts::decompose_relative_pose)
        .def_readwrite("reconstruct_all_components",
                       &Opts::reconstruct_all_components)
        .def_readwrite("min_num_frames", &Opts::min_num_frames)
        .def_readwrite("mapper", &Opts::mapper);
    MakeDataclass(PyOpts);
  }

  using CallbackType = GlobalPipeline::CallbackType;
  auto PyCallbackType =
      py::enum_<CallbackType>(m, "GlobalPipelineCallback")
          .value("MODEL_UPDATE_CALLBACK", CallbackType::MODEL_UPDATE_CALLBACK);
  AddStringToEnumConstructor(PyCallbackType);

  py::classh<GlobalPipeline>(
      m,
      "GlobalPipeline",
      "Class that controls the global mapping procedure (GLOMAP) by jointly "
      "estimating all camera poses from the view graph using rotation "
      "averaging and global positioning, followed by bundle adjustment and "
      "retriangulation.")
      .def(py::init<GlobalPipelineOptions,
                    std::shared_ptr<Database>,
                    std::shared_ptr<ReconstructionManager>>(),
           "options"_a,
           "database"_a,
           "reconstruction_manager"_a,
           py::call_guard<py::gil_scoped_release>())
      .def("add_callback",
           &GlobalPipeline::AddCallback,
           "id"_a,
           "func"_a,
           "Add a callback function for the given callback type.")
      .def("callback",
           &GlobalPipeline::Callback,
           "id"_a,
           "Invoke the callback for the given callback type.")
      .def("run",
           &GlobalPipeline::Run,
           py::call_guard<py::gil_scoped_release>(),
           "Run the full global mapping pipeline.");
}
