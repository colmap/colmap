#include "colmap/exe/sfm.h"

#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/estimators/view_graph_calibration.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <filesystem>
#include <memory>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

std::map<size_t, std::shared_ptr<Reconstruction>> ReconstructionManagerToMap(
    const std::shared_ptr<ReconstructionManager>& reconstruction_manager) {
  std::map<size_t, std::shared_ptr<Reconstruction>> reconstructions;
  for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
    reconstructions[i] = reconstruction_manager->Get(i);
  }
  return reconstructions;
}

}  // namespace

std::shared_ptr<Reconstruction> TriangulatePoints(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const bool clear_points,
    const IncrementalPipelineOptions& options,
    const bool refine_intrinsics) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  RunPointTriangulatorImpl(reconstruction,
                           database_path,
                           image_path,
                           output_path,
                           options,
                           clear_points,
                           refine_intrinsics);
  return reconstruction;
}

std::map<size_t, std::shared_ptr<Reconstruction>> IncrementalMapping(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const IncrementalPipelineOptions& options,
    const std::filesystem::path& input_path,
    std::function<void()> initial_image_pair_callback,
    std::function<void()> next_image_callback) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    reconstruction_manager->Read(input_path);
  }
  auto options_ = std::make_shared<IncrementalPipelineOptions>(options);

  PyInterrupt py_interrupt(1.0);  // Check for interrupts every second
  auto next_image_callback_py_interruptible =
      [&py_interrupt, next_image_callback = std::move(next_image_callback)]() {
        if (py_interrupt.Raised()) {
          throw py::error_already_set();
        }
        if (next_image_callback) {
          next_image_callback();
        }
      };

  if (!RunIncrementalMapperImpl(database_path,
                                image_path,
                                output_path,
                                options_,
                                reconstruction_manager,
                                initial_image_pair_callback,
                                next_image_callback_py_interruptible)) {
    return {};
  }

  return ReconstructionManagerToMap(reconstruction_manager);
}

std::map<size_t, std::shared_ptr<Reconstruction>> GlobalMapping(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    GlobalPipelineOptions options) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options_ = std::make_shared<GlobalPipelineOptions>(std::move(options));
  if (!RunGlobalMapperImpl(database_path,
                           image_path,
                           output_path,
                           options_,
                           reconstruction_manager)) {
    return {};
  }

  return ReconstructionManagerToMap(reconstruction_manager);
}

void BundleAdjustment(const std::shared_ptr<Reconstruction>& reconstruction,
                      const BundleAdjustmentOptions& options) {
  py::gil_scoped_release release;
  OptionManager option_manager;
  option_manager.bundle_adjustment =
      std::make_shared<BundleAdjustmentOptions>(options);
  BundleAdjustmentController controller(option_manager, reconstruction);
  controller.Run();
}

bool ViewGraphCalibration(const std::filesystem::path& database_path,
                          const ViewGraphCalibrationOptions& options) {
  THROW_CHECK_FILE_EXISTS(database_path);
  py::gil_scoped_release release;
  auto database = Database::Open(database_path);
  return CalibrateViewGraph(options, database.get());
}

void BindSfM(py::module& m) {
  // ViewGraphCalibrationOptions
  {
    using Opts = ViewGraphCalibrationOptions;
    auto PyOpts =
        py::classh<Opts>(m, "ViewGraphCalibrationOptions")
            .def(py::init<>())
            .def_readwrite("random_seed", &Opts::random_seed)
            .def_readwrite("cross_validate_prior_focal_lengths",
                           &Opts::cross_validate_prior_focal_lengths)
            .def_readwrite("min_calibrated_pair_ratio",
                           &Opts::min_calibrated_pair_ratio)
            .def_readwrite("reestimate_relative_pose",
                           &Opts::reestimate_relative_pose)
            .def_readwrite("min_focal_length_ratio",
                           &Opts::min_focal_length_ratio)
            .def_readwrite("max_focal_length_ratio",
                           &Opts::max_focal_length_ratio)
            .def_readwrite("max_calibration_error",
                           &Opts::max_calibration_error)
            .def_readwrite("loss_function_scale", &Opts::loss_function_scale)
            .def_readwrite("relpose_max_error", &Opts::relpose_max_error)
            .def_readwrite("relpose_min_num_inliers",
                           &Opts::relpose_min_num_inliers)
            .def_readwrite("relpose_min_inlier_ratio",
                           &Opts::relpose_min_inlier_ratio);
    MakeDataclass(PyOpts);
  }

  // GlobalMapperOptions
  {
    using Opts = GlobalMapperOptions;
    auto PyOpts =
        py::classh<Opts>(m, "GlobalMapperOptions")
            .def(py::init<>())
            .def_readwrite("num_threads", &Opts::num_threads)
            .def_readwrite("random_seed", &Opts::random_seed)
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
            .def_readwrite("max_angular_reproj_error_deg",
                           &Opts::max_angular_reproj_error_deg)
            .def_readwrite("max_normalized_reproj_error",
                           &Opts::max_normalized_reproj_error)
            .def_readwrite("min_tri_angle_deg", &Opts::min_tri_angle_deg)
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
            .def_readwrite("skip_bundle_adjustment",
                           &Opts::skip_bundle_adjustment)
            .def_readwrite("skip_retriangulation", &Opts::skip_retriangulation);
    MakeDataclass(PyOpts);
  }

  // GlobalPipelineOptions
  {
    using Opts = GlobalPipelineOptions;
    auto PyOpts =
        py::classh<Opts>(m, "GlobalPipelineOptions")
            .def(py::init<>())
            .def_readwrite("min_num_matches", &Opts::min_num_matches)
            .def_readwrite("ignore_watermarks", &Opts::ignore_watermarks)
            .def_readwrite("image_names", &Opts::image_names)
            .def_readwrite("num_threads", &Opts::num_threads)
            .def_readwrite("random_seed", &Opts::random_seed)
            .def_readwrite("decompose_relative_pose",
                           &Opts::decompose_relative_pose)
            .def_readwrite("mapper", &Opts::mapper);
    MakeDataclass(PyOpts);
  }

  m.def("triangulate_points",
        &TriangulatePoints,
        "reconstruction"_a,
        "database_path"_a,
        "image_path"_a,
        "output_path"_a,
        "clear_points"_a = true,
        py::arg_v("options",
                  IncrementalPipelineOptions(),
                  "IncrementalPipelineOptions()"),
        "refine_intrinsics"_a = false,
        "Triangulate 3D points from known camera poses");

  m.def("incremental_mapping",
        &IncrementalMapping,
        "database_path"_a,
        "image_path"_a,
        "output_path"_a,
        py::arg_v("options",
                  IncrementalPipelineOptions(),
                  "IncrementalPipelineOptions()"),
        "input_path"_a = py::str(""),
        "initial_image_pair_callback"_a = py::none(),
        "next_image_callback"_a = py::none(),
        "Recover 3D points and unknown camera poses");

  m.def(
      "global_mapping",
      &GlobalMapping,
      "database_path"_a,
      "image_path"_a,
      "output_path"_a,
      py::arg_v("options", GlobalPipelineOptions(), "GlobalPipelineOptions()"),
      "Recover 3D points and camera poses using global SfM (GLOMAP)");

  m.def("calibrate_view_graph",
        &ViewGraphCalibration,
        "database_path"_a,
        py::arg_v("options",
                  ViewGraphCalibrationOptions(),
                  "ViewGraphCalibrationOptions()"),
        "Calibrate focal lengths from fundamental matrices and upgrade "
        "two-view geometries to CALIBRATED in the database. Run before "
        "global_mapping when reliable intrinsics are unavailable.");

  m.def("bundle_adjustment",
        &BundleAdjustment,
        "reconstruction"_a,
        py::arg_v(
            "options", BundleAdjustmentOptions(), "BundleAdjustmentOptions()"),
        "Jointly refine 3D points and camera poses");
}
