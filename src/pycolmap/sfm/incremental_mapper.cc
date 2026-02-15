#include "colmap/sfm/incremental_mapper.h"

#include "colmap/controllers/incremental_pipeline.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindIncrementalPipeline(py::module& m) {
  using Opts = IncrementalPipelineOptions;
  auto PyOpts = py::classh<Opts>(m, "IncrementalPipelineOptions");
  PyOpts.def(py::init<>())
      .def_readwrite(
          "min_num_matches",
          &Opts::min_num_matches,
          "The minimum number of matches for inlier matches to be considered.")
      .def_readwrite(
          "ignore_watermarks",
          &Opts::ignore_watermarks,
          "Whether to ignore the inlier matches of watermark image pairs.")
      .def_readwrite("multiple_models",
                     &Opts::multiple_models,
                     "Whether to reconstruct multiple sub-models.")
      .def_readwrite("max_num_models",
                     &Opts::max_num_models,
                     "The number of sub-models to reconstruct.")
      .def_readwrite(
          "max_model_overlap",
          &Opts::max_model_overlap,
          "The maximum number of overlapping images between sub-models. If the "
          "current sub-models shares more than this number of images with "
          "another model, then the reconstruction is stopped.")
      .def_readwrite("min_model_size",
                     &Opts::min_model_size,
                     "The minimum number of registered images of a sub-model, "
                     "otherwise the sub-model is discarded. Note that the "
                     "first sub-model is always kept independent of size. If "
                     "the model contains at least half of the total number of "
                     "images, we also always keep it.")
      .def_readwrite("init_image_id1",
                     &Opts::init_image_id1,
                     "The image identifier of the first image used to "
                     "initialize the reconstruction.")
      .def_readwrite(
          "init_image_id2",
          &Opts::init_image_id2,
          "The image identifier of the second image used to initialize the "
          "reconstruction. Determined automatically if left unspecified.")
      .def_readwrite("init_num_trials",
                     &Opts::init_num_trials,
                     "The number of trials to initialize the reconstruction.")
      .def_readwrite("structure_less_registration_fallback",
                     &Opts::structure_less_registration_fallback,
                     "Enable fallback to structure-less image registration "
                     "using 2D-2D correspondences, if structured-based "
                     "registration fails using 2D-3D correspondences.")
      .def_readwrite("structure_less_registration_only",
                     &Opts::structure_less_registration_only,
                     "Only use structure-less and skip structure-based image "
                     "registration.")
      .def_readwrite("extract_colors",
                     &Opts::extract_colors,
                     "Whether to extract colors for reconstructed points.")
      .def_readwrite("num_threads",
                     &Opts::num_threads,
                     "The number of threads to use during reconstruction.")
      .def_readwrite(
          "random_seed",
          &Opts::random_seed,
          "PRNG seed for all stochastic methods during reconstruction.")
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
          "ba_refine_focal_length",
          &Opts::ba_refine_focal_length,
          "Whether to refine the focal length during the reconstruction.")
      .def_readwrite(
          "ba_refine_principal_point",
          &Opts::ba_refine_principal_point,
          "Whether to refine the principal point during the reconstruction.")
      .def_readwrite(
          "ba_refine_extra_params",
          &Opts::ba_refine_extra_params,
          "Whether to refine extra parameters during the reconstruction.")
      .def_readwrite("ba_refine_sensor_from_rig",
                     &Opts::ba_refine_sensor_from_rig,
                     "Whether to refine rig poses during the reconstruction.")
      .def_readwrite(
          "ba_min_num_residuals_for_cpu_multi_threading",
          &Opts::ba_min_num_residuals_for_cpu_multi_threading,
          "The minimum number of residuals per bundle adjustment problem to "
          "enable multi-threading solving of the problems.")
      .def_readwrite(
          "ba_local_function_tolerance",
          &Opts::ba_local_function_tolerance,
          "Ceres solver function tolerance for local bundle adjustment.")
      .def_readwrite(
          "ba_local_max_num_iterations",
          &Opts::ba_local_max_num_iterations,
          "The maximum number of local bundle adjustment iterations.")
      .def_readwrite(
          "ba_global_frames_ratio",
          &Opts::ba_global_frames_ratio,
          "The growth rates after which to perform global bundle adjustment.")
      .def_readwrite(
          "ba_global_points_ratio",
          &Opts::ba_global_points_ratio,
          "The growth rates after which to perform global bundle adjustment.")
      .def_readwrite(
          "ba_global_frames_freq",
          &Opts::ba_global_frames_freq,
          "The growth rates after which to perform global bundle adjustment.")
      .def_readwrite(
          "ba_global_points_freq",
          &Opts::ba_global_points_freq,
          "The growth rates after which to perform global bundle adjustment.")
      .def_readwrite(
          "ba_global_function_tolerance",
          &Opts::ba_global_function_tolerance,
          "Ceres solver function tolerance for global bundle adjustment.")
      .def_readwrite(
          "ba_global_max_num_iterations",
          &Opts::ba_global_max_num_iterations,
          "The maximum number of global bundle adjustment iterations.")
      .def_readwrite(
          "ba_local_max_refinements",
          &Opts::ba_local_max_refinements,
          "The thresholds for iterative bundle adjustment refinements.")
      .def_readwrite(
          "ba_local_max_refinement_change",
          &Opts::ba_local_max_refinement_change,
          "The thresholds for iterative bundle adjustment refinements.")
      .def_readwrite(
          "ba_global_max_refinements",
          &Opts::ba_global_max_refinements,
          "The thresholds for iterative bundle adjustment refinements.")
      .def_readwrite(
          "ba_global_max_refinement_change",
          &Opts::ba_global_max_refinement_change,
          "The thresholds for iterative bundle adjustment refinements.")
      .def_readwrite("ba_use_gpu",
                     &IncrementalPipelineOptions::ba_use_gpu,
                     "Whether to use Ceres' CUDA sparse linear algebra "
                     "library, if available.")
      .def_readwrite("ba_gpu_index",
                     &IncrementalPipelineOptions::ba_gpu_index,
                     "Index of CUDA GPU to use for BA, if available.")
      .def_readwrite("use_prior_position",
                     &Opts::use_prior_position,
                     "Whether to use priors on the camera positions.")
      .def_readwrite("use_robust_loss_on_prior_position",
                     &Opts::use_robust_loss_on_prior_position,
                     "Whether to use a robust loss on prior camera positions.")
      .def_readwrite("prior_position_loss_scale",
                     &Opts::prior_position_loss_scale,
                     "Threshold on the residual for the robust position prior "
                     "loss (chi2 for 3DOF at 95% = 7.815).")
      .def_readwrite("snapshot_path",
                     &Opts::snapshot_path,
                     "Path to a folder in which reconstruction snapshots will "
                     "be saved during incremental reconstruction.")
      .def_readwrite("snapshot_frames_freq",
                     &Opts::snapshot_frames_freq,
                     "Frequency of registered images according to which "
                     "reconstruction snapshots will be saved.")
      .def_readwrite(
          "image_path",
          &Opts::image_path,
          "The image path at which to find the images to extract point colors.")
      .def_readwrite(
          "image_names",
          &Opts::image_names,
          "Optional list of image names to reconstruct. If no images are "
          "specified, all images will be reconstructed by default.")
      .def_readwrite("fix_existing_frames",
                     &Opts::fix_existing_frames,
                     "If reconstruction is provided as input, fix the existing "
                     "frame poses.")
      .def_readwrite(
          "constant_rigs",
          &Opts::constant_rigs,
          "List of rigs for which to fix the sensor_from_rig transformation, "
          "independent of ba_refine_sensor_from_rig.")
      .def_readwrite("constant_cameras",
                     &Opts::constant_cameras,
                     "List of cameras for which to fix the camera parameters "
                     "independent of refine_focal_length, "
                     "refine_principal_point, and refine_extra_params.")
      .def_readwrite(
          "max_runtime_seconds",
          &Opts::max_runtime_seconds,
          "Maximum runtime in seconds for the reconstruction process. If set "
          "to a non-positive value, the process will run until completion.")
      .def_readwrite(
          "mapper", &Opts::mapper, "Options of the IncrementalMapper.")
      .def_readwrite("triangulation",
                     &Opts::triangulation,
                     "Options of the IncrementalTriangulator.")
      .def("get_mapper",
           &Opts::Mapper,
           "Get mapper options with shared settings applied.")
      .def("get_triangulation",
           &Opts::Triangulation,
           "Get triangulation options with shared settings applied.")
      .def("get_local_bundle_adjustment",
           &Opts::LocalBundleAdjustment,
           "Get local bundle adjustment options.")
      .def("get_global_bundle_adjustment",
           &Opts::GlobalBundleAdjustment,
           "Get global bundle adjustment options.")
      .def("is_initial_pair_provided",
           &Opts::IsInitialPairProvided,
           "Check whether both initial image identifiers are provided.")
      .def("check", &Opts::Check);
  MakeDataclass(PyOpts);

  using CallbackType = IncrementalPipeline::CallbackType;
  auto PyCallbackType =
      py::enum_<CallbackType>(m, "IncrementalPipelineCallback")
          .value("INITIAL_IMAGE_PAIR_REG_CALLBACK",
                 CallbackType::INITIAL_IMAGE_PAIR_REG_CALLBACK)
          .value("NEXT_IMAGE_REG_CALLBACK",
                 CallbackType::NEXT_IMAGE_REG_CALLBACK)
          .value("LAST_IMAGE_REG_CALLBACK",
                 CallbackType::LAST_IMAGE_REG_CALLBACK);
  AddStringToEnumConstructor(PyCallbackType);

  using Status = IncrementalPipeline::Status;
  auto PyStatus =
      py::enum_<Status>(m, "IncrementalPipelineStatus")
          .value("SUCCESS", Status::SUCCESS)
          .value("INTERRUPTED", Status::INTERRUPTED)
          .value("CONTINUE", Status::CONTINUE)
          .value("STOP", Status::STOP)
          .value("UNKNOWN_SENSOR_FROM_RIG", Status::UNKNOWN_SENSOR_FROM_RIG)
          .value("NO_INITIAL_PAIR", Status::NO_INITIAL_PAIR)
          .value("BAD_INITIAL_PAIR", Status::BAD_INITIAL_PAIR);
  AddStringToEnumConstructor(PyStatus);

  py::classh<IncrementalPipeline>(
      m,
      "IncrementalPipeline",
      "Class that controls the incremental mapping procedure by iteratively "
      "initializing reconstructions from the same scene graph.")
      .def(py::init<std::shared_ptr<IncrementalPipelineOptions>,
                    std::shared_ptr<Database>,
                    std::shared_ptr<ReconstructionManager>>(),
           "options"_a,
           "database"_a,
           "reconstruction_manager"_a)
      .def(py::init<std::shared_ptr<IncrementalPipelineOptions>,
                    std::shared_ptr<DatabaseCache>,
                    std::shared_ptr<ReconstructionManager>>(),
           "options"_a,
           "database_cache"_a,
           "reconstruction_manager"_a)
      .def_property_readonly("options", &IncrementalPipeline::Options)
      .def_property_readonly("reconstruction_manager",
                             &IncrementalPipeline::ReconstructionManager)
      .def_property_readonly("database_cache",
                             &IncrementalPipeline::DatabaseCache)
      .def("add_callback",
           &IncrementalPipeline::AddCallback,
           "id"_a,
           "func"_a,
           "Add a callback function for the given callback type.")
      .def("callback",
           &IncrementalPipeline::Callback,
           "id"_a,
           "Invoke the callback for the given callback type.")
      .def("reconstruct",
           &IncrementalPipeline::Reconstruct,
           "mapper"_a,
           "mapper_options"_a,
           "continue_reconstruction"_a,
           "Reconstruct the scene using the given mapper and options.")
      .def("reconstruct_sub_model",
           &IncrementalPipeline::ReconstructSubModel,
           "mapper"_a,
           "mapper_options"_a,
           "reconstruction"_a,
           "Reconstruct a sub-model using the given mapper and options.")
      .def("initialize_reconstruction",
           &IncrementalPipeline::InitializeReconstruction,
           "mapper"_a,
           "mapper_options"_a,
           "reconstruction"_a,
           "Initialize the reconstruction by finding and registering an "
           "initial image pair.")
      .def("run",
           &IncrementalPipeline::Run,
           "Run the full incremental mapping pipeline.")
      .def("check_run_global_refinement",
           &IncrementalPipeline::CheckRunGlobalRefinement,
           "reconstruction"_a,
           "ba_prev_num_reg_images"_a,
           "ba_prev_num_points"_a,
           "Check whether global bundle adjustment should be run based on "
           "the growth of registered images and points.")
      .def("check_reached_max_runtime",
           &IncrementalPipeline::CheckReachedMaxRuntime,
           "Check whether the maximum runtime has been reached.");
}

void BindIncrementalMapperOptions(py::module& m) {
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
  auto PyOpts = py::classh<Opts>(m, "IncrementalMapperOptions");
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
      .def_readwrite("ba_local_num_images",
                     &Opts::ba_local_num_images,
                     "Number of images to optimize in local bundle adjustment.")
      .def_readwrite("ba_local_min_tri_angle",
                     &Opts::ba_local_min_tri_angle,
                     "Minimum triangulation for images to be chosen in local "
                     "bundle adjustment.")
      .def_readwrite(
          "ba_global_ignore_redundant_points3D",
          &Opts::ba_global_ignore_redundant_points3D,
          "Whether to ignore redundant 3D points in bundle adjustment when "
          "jointly optimizing all parameters. If this is enabled, then the "
          "bundle adjustment problem is first solved with a reduced set of 3D "
          "points and then the remaining 3D points are optimized in a second "
          "step with all other parameters fixed. Points excplicitly configured "
          "as constant or variable are not ignored. This is only activated "
          "when the reconstruction has reached sufficient size with at least "
          "10 registered frames.")
      .def_readwrite(
          "ba_global_prune_points_min_coverage_gain",
          &Opts::ba_global_ignore_redundant_points3D_min_coverage_gain,
          "The minimum coverage gain for any 3D point to be "
          "included in the optimization. A larger value means "
          "more 3D points are ignored.")
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
      .def_readwrite("fix_existing_frames",
                     &Opts::fix_existing_frames,
                     "If reconstruction is provided as input, fix the existing "
                     "frame poses.")
      .def_readwrite(
          "constant_rigs",
          &Opts::constant_rigs,
          "List of rigs for which to fix the sensor_from_rig transformation, "
          "independent of ba_refine_sensor_from_rig.")
      .def_readwrite("constant_cameras",
                     &Opts::constant_cameras,
                     "List of cameras for which to fix the camera parameters "
                     "independent of refine_focal_length, "
                     "refine_principal_point, and refine_extra_params.")
      .def_readwrite("num_threads", &Opts::num_threads, "Number of threads.")
      .def_readwrite(
          "random_seed",
          &Opts::random_seed,
          "PRNG seed for all stochastic methods during reconstruction.")
      .def_readwrite("image_selection_method",
                     &Opts::image_selection_method,
                     "Method to find and select next best image to register.")
      .def("check", &Opts::Check);
  MakeDataclass(PyOpts);
}

void BindIncrementalMapperImpl(py::module& m) {
  BindIncrementalMapperOptions(m);

  // bind local bundle adjustment report
  using LocalBAReport = IncrementalMapper::LocalBundleAdjustmentReport;
  auto PyLocalBAReport =
      py::classh<LocalBAReport>(m, "LocalBundleAdjustmentReport");
  PyLocalBAReport.def(py::init<>())
      .def_readwrite("num_merged_observations",
                     &LocalBAReport::num_merged_observations)
      .def_readwrite("num_completed_observations",
                     &LocalBAReport::num_completed_observations)
      .def_readwrite("num_filtered_observations",
                     &LocalBAReport::num_filtered_observations)
      .def_readwrite("num_adjusted_observations",
                     &LocalBAReport::num_adjusted_observations);
  MakeDataclass(PyLocalBAReport);

  // bind incremental mapper
  py::classh<IncrementalMapper>(
      m,
      "IncrementalMapper",
      "Class that provides all functionality for the incremental "
      "reconstruction procedure.")
      .def(py::init<std::shared_ptr<const DatabaseCache>>(),
           "database_cache"_a,
           "Create incremental mapper. The database cache must live for the "
           "entire life-time of the incremental mapper.")
      .def("begin_reconstruction",
           &IncrementalMapper::BeginReconstruction,
           "reconstruction"_a,
           "Prepare the mapper for a new reconstruction, which might have "
           "existing registered images (in which case register_next_image "
           "must be called) or which is empty (in which case "
           "register_initial_image_pair must be called).")
      .def("end_reconstruction",
           &IncrementalMapper::EndReconstruction,
           "discard"_a,
           "Cleanup the mapper after the current reconstruction is done. If "
           "the model is discarded, the number of total and shared registered "
           "images will be updated accordingly.")
      .def(
          "find_initial_image_pair",
          [](IncrementalMapper& self,
             const IncrementalMapper::Options& options,
             int image_id1,
             int image_id2)
              -> py::typing::Optional<
                  py::typing::Tuple<py::typing::Tuple<image_t, image_t>,
                                    Rigid3d>> {
            // Explicitly handle the conversion
            // from -1 (int) to kInvalidImageId (uint32_t).
            image_t image_id1_cast = image_id1;
            image_t image_id2_cast = image_id2;
            Rigid3d cam2_from_cam1;
            const bool success = self.FindInitialImagePair(
                options, image_id1_cast, image_id2_cast, cam2_from_cam1);
            if (success) {
              const auto pair = std::make_pair(image_id1_cast, image_id2_cast);
              return py::cast(std::make_pair(pair, cam2_from_cam1));
            } else {
              return py::none();
            }
          },
          "options"_a,
          "image_id1"_a,
          "image_id2"_a,
          "Find initial image pair to seed the incremental reconstruction. "
          "Returns a tuple of ((image_id1, image_id2), cam2_from_cam1) on "
          "success, or None on failure. This function automatically ignores "
          "image pairs that failed to register previously.")
      .def(
          "estimate_initial_two_view_geometry",
          [](IncrementalMapper& self,
             const IncrementalMapper::Options& options,
             const image_t image_id1,
             const image_t image_id2) -> py::typing::Optional<Rigid3d> {
            Rigid3d cam2_from_cam1;
            const bool success = self.EstimateInitialTwoViewGeometry(
                options, image_id1, image_id2, cam2_from_cam1);
            if (success)
              return py::cast(cam2_from_cam1);
            else
              return py::none();
          },
          "options"_a,
          "image_id1"_a,
          "image_id2"_a,
          "Estimate two-view geometry and check if it is suitable for "
          "initialization. Returns the relative pose on success, or None "
          "on failure.")
      .def("register_initial_image_pair",
           &IncrementalMapper::RegisterInitialImagePair,
           "options"_a,
           "two_view_geometry"_a,
           "image_id1"_a,
           "image_id2"_a,
           "Attempt to seed the reconstruction from an image pair.")
      .def("find_next_images",
           &IncrementalMapper::FindNextImages,
           "options"_a,
           "structure_less"_a,
           "Find best next images to register in the incremental "
           "reconstruction. This function automatically ignores images that "
           "failed to register for max_reg_trials.")
      .def("register_next_image",
           &IncrementalMapper::RegisterNextImage,
           "options"_a,
           "image_id"_a,
           "Attempt to register image to the existing model. This requires "
           "that a previous call to register_initial_image_pair was "
           "successful.")
      .def("register_next_structure_less_image",
           &IncrementalMapper::RegisterNextStructureLessImage,
           "options"_a,
           "image_id"_a,
           "Attempt to register image using structure-less resectioning.")
      .def("triangulate_image",
           &IncrementalMapper::TriangulateImage,
           "tri_options"_a,
           "image_id"_a,
           "Triangulate observations of image.")
      .def("retriangulate",
           &IncrementalMapper::Retriangulate,
           "tri_options"_a,
           "Retriangulate image pairs that should have common observations "
           "according to the scene graph but don't due to drift, etc.")
      .def("complete_tracks",
           &IncrementalMapper::CompleteTracks,
           "tri_options"_a,
           "Complete tracks by transitively following the scene graph "
           "correspondences. This is especially effective after bundle "
           "adjustment, since many cameras and point locations might have "
           "improved.")
      .def("merge_tracks",
           &IncrementalMapper::MergeTracks,
           "tri_options"_a,
           "Merge tracks by using scene graph correspondences. Similar to "
           "complete_tracks, this is effective after bundle adjustment and "
           "improves the redundancy in subsequent bundle adjustments.")
      .def("complete_and_merge_tracks",
           &IncrementalMapper::CompleteAndMergeTracks,
           "tri_options"_a,
           "Globally complete and merge tracks.")
      .def("adjust_local_bundle",
           &IncrementalMapper::AdjustLocalBundle,
           "options"_a,
           "ba_options"_a,
           "tri_options"_a,
           "image_id"_a,
           "point3D_ids"_a,
           "Adjust locally connected images and points of a reference image. "
           "In addition, refine the provided 3D points. Only images connected "
           "to the reference image are optimized. If the provided 3D points "
           "are not locally connected to the reference image, their observing "
           "images are set as constant in the adjustment.")
      .def("iterative_local_refinement",
           &IncrementalMapper::IterativeLocalRefinement,
           "max_num_refinements"_a,
           "max_refinement_change"_a,
           "options"_a,
           "ba_options"_a,
           "tri_options"_a,
           "image_id"_a,
           "Perform multiple rounds of local bundle adjustment.")
      .def("find_local_bundle",
           &IncrementalMapper::FindLocalBundle,
           "options"_a,
           "image_id"_a,
           "Find local bundle for given image in the reconstruction. The "
           "local bundle is defined as the images that are most connected, "
           "i.e. maximum number of shared 3D points, to the given image.")
      .def("adjust_global_bundle",
           &IncrementalMapper::AdjustGlobalBundle,
           "options"_a,
           "ba_options"_a,
           "Global bundle adjustment using Ceres Solver.")
      .def("iterative_global_refinement",
           &IncrementalMapper::IterativeGlobalRefinement,
           "max_num_refinements"_a,
           "max_refinement_change"_a,
           "options"_a,
           "ba_options"_a,
           "tri_options"_a,
           "normalize_reconstruction"_a = true,
           "Perform multiple rounds of global bundle adjustment.")
      .def("filter_frames",
           &IncrementalMapper::FilterFrames,
           "options"_a,
           "Filter frames with degenerate camera parameters.")
      .def("filter_points",
           &IncrementalMapper::FilterPoints,
           "options"_a,
           "Filter points with large reprojection errors or small "
           "triangulation angles.")
      .def_property_readonly("reconstruction",
                             &IncrementalMapper::Reconstruction)
      .def_property_readonly("observation_manager",
                             &IncrementalMapper::ObservationManager)
      .def_property_readonly("triangulator", &IncrementalMapper::Triangulator)
      .def_property_readonly("filtered_frames",
                             &IncrementalMapper::FilteredFrames)
      .def_property_readonly("existing_frame_ids",
                             &IncrementalMapper::ExistingFrameIds)
      .def("reset_initialization_stats",
           &IncrementalMapper::ResetInitializationStats,
           "Reset registration statistics for initialization. This can be "
           "used when relaxing the initialization thresholds, such that "
           "previously tried pairs will be tried again.")
      .def_property_readonly("num_reg_frames_per_rig",
                             &IncrementalMapper::NumRegFramesPerRig)
      .def_property_readonly("num_reg_images_per_camera",
                             &IncrementalMapper::NumRegImagesPerCamera)
      .def("num_total_reg_images",
           &IncrementalMapper::NumTotalRegImages,
           "Number of images that are registered in at least one "
           "reconstruction.")
      .def("num_shared_reg_images",
           &IncrementalMapper::NumSharedRegImages,
           "Number of shared images between current reconstruction and all "
           "other previous reconstructions.")
      .def("get_modified_points3D",
           &IncrementalMapper::GetModifiedPoints3D,
           "Get changed 3D points, since the last call to "
           "clear_modified_points3D.")
      .def("clear_modified_points3D",
           &IncrementalMapper::ClearModifiedPoints3D,
           "Clear the collection of changed 3D points.");
}

void BindIncrementalMapper(py::module& m) {
  BindIncrementalMapperImpl(m);
  BindIncrementalPipeline(m);
}
