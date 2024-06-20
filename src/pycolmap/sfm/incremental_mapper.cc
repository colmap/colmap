#include "colmap/controllers/incremental_mapper.h"

#include "colmap/sfm/incremental_mapper.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
namespace py = pybind11;

void BindIncrementalMapperController(py::module& m) {
  // bind callbacks
  using CallbackType = IncrementalMapperController::CallbackType;
  auto PyCallbackType =
      py::enum_<CallbackType>(m, "IncrementalMapperCallback")
          .value("INITIAL_IMAGE_PAIR_REG_CALLBACK",
                 CallbackType::INITIAL_IMAGE_PAIR_REG_CALLBACK)
          .value("NEXT_IMAGE_REG_CALLBACK",
                 CallbackType::NEXT_IMAGE_REG_CALLBACK)
          .value("LAST_IMAGE_REG_CALLBACK",
                 CallbackType::LAST_IMAGE_REG_CALLBACK);
  AddStringToEnumConstructor(PyCallbackType);

  // bind status
  using Status = IncrementalMapperController::Status;
  auto PyStatus = py::enum_<Status>(m, "IncrementalMapperStatus")
                      .value("NO_INITIAL_PAIR", Status::NO_INITIAL_PAIR)
                      .value("BAD_INITIAL_PAIR", Status::BAD_INITIAL_PAIR)
                      .value("SUCCESS", Status::SUCCESS)
                      .value("INTERRUPTED", Status::INTERRUPTED);
  AddStringToEnumConstructor(PyStatus);

  // bind controller
  py::class_<IncrementalMapperController,
             std::shared_ptr<IncrementalMapperController>>(
      m, "IncrementalMapperController")
      .def(py::init<std::shared_ptr<const IncrementalMapperOptions>,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<ReconstructionManager>>(),
           "options"_a,
           "image_path"_a,
           "database_path"_a,
           "reconstruction_manager"_a)
      .def_property_readonly("options", &IncrementalMapperController::Options)
      .def_property_readonly("image_path",
                             &IncrementalMapperController::ImagePath)
      .def_property_readonly("database_path",
                             &IncrementalMapperController::DatabasePath)
      .def_property_readonly(
          "reconstruction_manager",
          &IncrementalMapperController::ReconstructionManager)
      .def_property_readonly("database_cache",
                             &IncrementalMapperController::DatabaseCache)
      .def("add_callback",
           &IncrementalMapperController::AddCallback,
           "id"_a,
           "func"_a)
      .def("callback", &IncrementalMapperController::Callback, "id"_a)
      .def("load_database", &IncrementalMapperController::LoadDatabase)
      .def("check_run_global_refinement",
           &IncrementalMapperController::CheckRunGlobalRefinement,
           "reconstruction"_a,
           "ba_prev_num_reg_images"_a,
           "ba_prev_num_points"_a)
      .def("reconstruct",
           &IncrementalMapperController::Reconstruct,
           "mapper_options"_a)
      .def("reconstruct_sub_model",
           &IncrementalMapperController::ReconstructSubModel,
           "core_mapper"_a,
           "mapper_options"_a,
           "reconstruction"_a)
      .def("initialize_reconstruction",
           &IncrementalMapperController::InitializeReconstruction,
           "core_mapper"_a,
           "mapper_options"_a,
           "reconstruction"_a)
      .def("run", &IncrementalMapperController::Run);
}

void BindIncrementalMapperOptions(py::module& m) {
  using Opts = IncrementalMapper::Options;
  auto PyOpts =
      py::class_<Opts, std::shared_ptr<Opts>>(m, "IncrementalMapperOptions");
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

void BindIncrementalMapperImpl(py::module& m) {
  // bind options
  BindIncrementalMapperOptions(m);

  // bind ImageSelectionMethod
  using ImageSelection = IncrementalMapper::Options::ImageSelectionMethod;
  auto PyImageSelectionMethod =
      py::enum_<ImageSelection>(m, "ImageSelectionMethod")
          .value("MAX_VISIBLE_POINTS_NUM",
                 ImageSelection::MAX_VISIBLE_POINTS_NUM)
          .value("MAX_VISIBLE_POINTS_RATIO",
                 ImageSelection::MAX_VISIBLE_POINTS_RATIO)
          .value("MIN_UNCERTAINTY", ImageSelection::MIN_UNCERTAINTY);
  AddStringToEnumConstructor(PyImageSelectionMethod);

  // bind local bundle adjustment report
  using LocalBAReport = IncrementalMapper::LocalBundleAdjustmentReport;
  auto PyLocalBAReport =
      py::class_<LocalBAReport>(m, "LocalBundleAdjustmentReport");
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
  // TODO: migrate comments. improve formatting
  py::class_<IncrementalMapper, std::shared_ptr<IncrementalMapper>>(
      m, "IncrementalMapper")
      .def(py::init<std::shared_ptr<const DatabaseCache>>())
      .def("begin_reconstruction",
           &IncrementalMapper::BeginReconstruction,
           "reconstruction"_a)
      .def("end_reconstruction",
           &IncrementalMapper::EndReconstruction,
           "discard"_a)
      .def(
          "find_initial_image_pair",
          [](IncrementalMapper& self,
             const IncrementalMapper::Options& options,
             int image_id1,
             int image_id2) -> py::object {
            // Explicitly handle the conversion
            // from -1 (int) to kInvalidImageId (uint32_t).
            image_t image_id1_cast = image_id1;
            image_t image_id2_cast = image_id2;
            TwoViewGeometry two_view_geometry;
            const bool success = self.FindInitialImagePair(
                options, two_view_geometry, image_id1_cast, image_id2_cast);
            if (success) {
              const auto pair = std::make_pair(image_id1_cast, image_id2_cast);
              return py::cast(std::make_pair(pair, two_view_geometry));
            } else {
              return py::none();
            }
          },
          "options"_a,
          "image_id1"_a,
          "image_id2"_a)
      .def(
          "estimate_initial_two_view_geometry",
          [](IncrementalMapper& self,
             const IncrementalMapper::Options& options,
             const image_t image_id1,
             const image_t image_id2) -> py::object {
            TwoViewGeometry two_view_geometry;
            const bool success = self.EstimateInitialTwoViewGeometry(
                options, two_view_geometry, image_id1, image_id2);
            if (success)
              return py::cast(two_view_geometry);
            else
              return py::none();
          },
          "options"_a,
          "image_id1"_a,
          "image_id2"_a)
      .def("register_initial_image_pair",
           &IncrementalMapper::RegisterInitialImagePair,
           "options"_a,
           "two_view_geometry"_a,
           "image_id1"_a,
           "image_id2"_a)
      .def("find_next_images", &IncrementalMapper::FindNextImages, "options"_a)
      .def("register_next_image",
           &IncrementalMapper::RegisterNextImage,
           "options"_a,
           "image_id"_a)
      .def("triangulate_image",
           &IncrementalMapper::TriangulateImage,
           "tri_options"_a,
           "image_id"_a)
      .def("retriangulate", &IncrementalMapper::Retriangulate, "tri_options"_a)
      .def("complete_tracks",
           &IncrementalMapper::CompleteTracks,
           "tri_options"_a)
      .def("merge_tracks", &IncrementalMapper::MergeTracks, "tri_options"_a)
      .def("complete_and_merge_tracks",
           &IncrementalMapper::CompleteAndMergeTracks,
           "tri_options"_a)
      .def("adjust_local_bundle",
           &IncrementalMapper::AdjustLocalBundle,
           "options"_a,
           "ba_options"_a,
           "tri_options"_a,
           "image_id"_a,
           "point3D_ids"_a)
      .def("iterative_local_refinement",
           &IncrementalMapper::IterativeLocalRefinement,
           "max_num_refinements"_a,
           "max_refinement_change"_a,
           "options"_a,
           "ba_options"_a,
           "tri_options"_a,
           "image_id"_a)
      .def("find_local_bundle",
           &IncrementalMapper::FindLocalBundle,
           "options"_a,
           "image_id"_a)
      .def("adjust_global_bundle",
           &IncrementalMapper::AdjustGlobalBundle,
           "options"_a,
           "ba_options"_a)
      .def("iterative_global_refinement",
           &IncrementalMapper::IterativeGlobalRefinement,
           "max_num_refinements"_a,
           "max_refinement_change"_a,
           "options"_a,
           "ba_options"_a,
           "tri_options"_a,
           "normalize_reconstruction"_a = true)
      .def("filter_images", &IncrementalMapper::FilterImages, "options"_a)
      .def("filter_points", &IncrementalMapper::FilterPoints, "options"_a)
      .def_property_readonly("reconstruction",
                             &IncrementalMapper::Reconstruction)
      .def_property_readonly("observation_manager",
                             &IncrementalMapper::ObservationManager)
      .def_property_readonly("triangulator", &IncrementalMapper::Triangulator)
      .def_property_readonly("filtered_images",
                             &IncrementalMapper::FilteredImages)
      .def_property_readonly("existing_image_ids",
                             &IncrementalMapper::ExistingImageIds)
      .def_property_readonly("num_reg_images_per_camera",
                             &IncrementalMapper::NumRegImagesPerCamera)
      .def("num_total_reg_images", &IncrementalMapper::NumTotalRegImages)
      .def("num_shared_reg_images", &IncrementalMapper::NumSharedRegImages)
      .def("get_modified_points3D", &IncrementalMapper::GetModifiedPoints3D)
      .def("clear_modified_points3D",
           &IncrementalMapper::ClearModifiedPoints3D);
}

void BindIncrementalMapper(py::module& m) {
  BindIncrementalMapperController(m);
  BindIncrementalMapperImpl(m);
}
