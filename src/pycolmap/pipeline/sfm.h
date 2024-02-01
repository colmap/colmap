#pragma once

#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/controllers/incremental_mapper.h"
#include "colmap/exe/sfm.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"
#include "colmap/util/misc.h"

#include "pycolmap/helpers.h"

#include <memory>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

std::shared_ptr<Reconstruction> TriangulatePoints(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::string& database_path,
    const std::string& image_path,
    const std::string& output_path,
    const bool clear_points,
    const IncrementalMapperOptions& options,
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
    const std::string& database_path,
    const std::string& image_path,
    const std::string& output_path,
    const IncrementalMapperOptions& options,
    const std::string& input_path,
    const std::function<void()>& initial_image_pair_callback,
    const std::function<void()>& next_image_callback) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    reconstruction_manager->Read(input_path);
  }
  auto options_ = std::make_shared<IncrementalMapperOptions>(options);
  IncrementalMapperController mapper(
      options_, image_path, database_path, reconstruction_manager);

  PyInterrupt py_interrupt(1.0);  // Check for interrupts every second
  mapper.AddCallback(IncrementalMapperController::NEXT_IMAGE_REG_CALLBACK,
                     [&]() {
                       if (py_interrupt.Raised()) {
                         throw py::error_already_set();
                       }
                       if (next_image_callback) {
                         next_image_callback();
                       }
                     });
  if (initial_image_pair_callback) {
    mapper.AddCallback(
        IncrementalMapperController::INITIAL_IMAGE_PAIR_REG_CALLBACK,
        initial_image_pair_callback);
  }

  mapper.Start();
  mapper.Wait();

  reconstruction_manager->Write(output_path);
  std::map<size_t, std::shared_ptr<Reconstruction>> reconstructions;
  for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
    reconstructions[i] = reconstruction_manager->Get(i);
  }
  return reconstructions;
}

void BundleAdjustment(const std::shared_ptr<Reconstruction>& reconstruction,
                      const BundleAdjustmentOptions& options) {
  py::gil_scoped_release release;
  OptionManager option_manager;
  *option_manager.bundle_adjustment = options;
  BundleAdjustmentController controller(option_manager, reconstruction);
  controller.Start();
  PyWait(&controller);
}

void BindSfM(py::module& m) {
  using MapperOpts = IncrementalMapperOptions;
  auto PyMapperOpts = py::class_<MapperOpts>(m, "IncrementalPipelineOptions");
  PyMapperOpts.def(py::init<>())
      .def_readwrite(
          "min_num_matches",
          &MapperOpts::min_num_matches,
          "The minimum number of matches for inlier matches to be considered.")
      .def_readwrite(
          "ignore_watermarks",
          &MapperOpts::ignore_watermarks,
          "Whether to ignore the inlier matches of watermark image pairs.")
      .def_readwrite("multiple_models",
                     &MapperOpts::multiple_models,
                     "Whether to reconstruct multiple sub-models.")
      .def_readwrite("max_num_models",
                     &MapperOpts::max_num_models,
                     "The number of sub-models to reconstruct.")
      .def_readwrite(
          "max_model_overlap",
          &MapperOpts::max_model_overlap,
          "The maximum number of overlapping images between sub-models. If the "
          "current sub-models shares more than this number of images with "
          "another model, then the reconstruction is stopped.")
      .def_readwrite("min_model_size",
                     &MapperOpts::min_model_size,
                     "The minimum number of registered images of a sub-model, "
                     "otherwise the sub-model is discarded. Note that the "
                     "first sub-model is always kept independent of size.")
      .def_readwrite("init_image_id1",
                     &MapperOpts::init_image_id1,
                     "The image identifier of the first image used to "
                     "initialize the reconstruction.")
      .def_readwrite(
          "init_image_id2",
          &MapperOpts::init_image_id2,
          "The image identifier of the second image used to initialize the "
          "reconstruction. Determined automatically if left unspecified.")
      .def_readwrite("init_num_trials",
                     &MapperOpts::init_num_trials,
                     "The number of trials to initialize the reconstruction.")
      .def_readwrite("extract_colors",
                     &MapperOpts::extract_colors,
                     "Whether to extract colors for reconstructed points.")
      .def_readwrite("num_threads",
                     &MapperOpts::num_threads,
                     "The number of threads to use during reconstruction.")
      .def_readwrite("min_focal_length_ratio",
                     &MapperOpts::min_focal_length_ratio,
                     "The threshold used to filter and ignore images with "
                     "degenerate intrinsics.")
      .def_readwrite("max_focal_length_ratio",
                     &MapperOpts::max_focal_length_ratio,
                     "The threshold used to filter and ignore images with "
                     "degenerate intrinsics.")
      .def_readwrite("max_extra_param",
                     &MapperOpts::max_extra_param,
                     "The threshold used to filter and ignore images with "
                     "degenerate intrinsics.")
      .def_readwrite(
          "ba_refine_focal_length",
          &MapperOpts::ba_refine_focal_length,
          "Which intrinsic parameters to optimize during the reconstruction.")
      .def_readwrite(
          "ba_refine_principal_point",
          &MapperOpts::ba_refine_principal_point,
          "Which intrinsic parameters to optimize during the reconstruction.")
      .def_readwrite(
          "ba_refine_extra_params",
          &MapperOpts::ba_refine_extra_params,
          "Which intrinsic parameters to optimize during the reconstruction.")
      .def_readwrite(
          "ba_min_num_residuals_for_multi_threading",
          &MapperOpts::ba_min_num_residuals_for_multi_threading,
          "The minimum number of residuals per bundle adjustment problem to "
          "enable multi-threading solving of the problems.")
      .def_readwrite(
          "ba_local_num_images",
          &MapperOpts::ba_local_num_images,
          "The number of images to optimize in local bundle adjustment.")
      .def_readwrite(
          "ba_local_function_tolerance",
          &MapperOpts::ba_local_function_tolerance,
          "Ceres solver function tolerance for local bundle adjustment.")
      .def_readwrite(
          "ba_local_max_num_iterations",
          &MapperOpts::ba_local_max_num_iterations,
          "The maximum number of local bundle adjustment iterations.")
      .def_readwrite(
          "ba_global_images_ratio",
          &MapperOpts::ba_global_images_ratio,
          "The growth rates after which to perform global bundle adjustment.")
      .def_readwrite(
          "ba_global_points_ratio",
          &MapperOpts::ba_global_points_ratio,
          "The growth rates after which to perform global bundle adjustment.")
      .def_readwrite(
          "ba_global_images_freq",
          &MapperOpts::ba_global_images_freq,
          "The growth rates after which to perform global bundle adjustment.")
      .def_readwrite(
          "ba_global_points_freq",
          &MapperOpts::ba_global_points_freq,
          "The growth rates after which to perform global bundle adjustment.")
      .def_readwrite(
          "ba_global_function_tolerance",
          &MapperOpts::ba_global_function_tolerance,
          "Ceres solver function tolerance for global bundle adjustment.")
      .def_readwrite(
          "ba_global_max_num_iterations",
          &MapperOpts::ba_global_max_num_iterations,
          "The maximum number of global bundle adjustment iterations.")
      .def_readwrite(
          "ba_local_max_refinements",
          &MapperOpts::ba_local_max_refinements,
          "The thresholds for iterative bundle adjustment refinements.")
      .def_readwrite(
          "ba_local_max_refinement_change",
          &MapperOpts::ba_local_max_refinement_change,
          "The thresholds for iterative bundle adjustment refinements.")
      .def_readwrite(
          "ba_global_max_refinements",
          &MapperOpts::ba_global_max_refinements,
          "The thresholds for iterative bundle adjustment refinements.")
      .def_readwrite(
          "ba_global_max_refinement_change",
          &MapperOpts::ba_global_max_refinement_change,
          "The thresholds for iterative bundle adjustment refinements.")
      .def_readwrite("snapshot_path",
                     &MapperOpts::snapshot_path,
                     "Path to a folder in which reconstruction snapshots will "
                     "be saved during incremental reconstruction.")
      .def_readwrite("snapshot_images_freq",
                     &MapperOpts::snapshot_images_freq,
                     "Frequency of registered images according to which "
                     "reconstruction snapshots will be saved.")
      .def_readwrite("image_names",
                     &MapperOpts::image_names,
                     "Which images to reconstruct. If no images are specified, "
                     "all images will be reconstructed by default.")
      .def_readwrite("fix_existing_images",
                     &MapperOpts::fix_existing_images,
                     "If reconstruction is provided as input, fix the existing "
                     "image poses.")
      .def_readwrite(
          "mapper", &MapperOpts::mapper, "Options of the IncrementalMapper.")
      .def_readwrite("triangulation",
                     &MapperOpts::triangulation,
                     "Options of the IncrementalTriangulator.")
      .def("get_mapper", &MapperOpts::Mapper)
      .def("get_triangulation", &MapperOpts::Triangulation);
  MakeDataclass(PyMapperOpts);
  auto mapper_options = PyMapperOpts().cast<MapperOpts>();

  using BAOpts = BundleAdjustmentOptions;
  auto PyBALossFunctionType =
      py::enum_<BAOpts::LossFunctionType>(m, "LossFunctionType")
          .value("TRIVIAL", BAOpts::LossFunctionType::TRIVIAL)
          .value("SOFT_L1", BAOpts::LossFunctionType::SOFT_L1)
          .value("CAUCHY", BAOpts::LossFunctionType::CAUCHY);
  AddStringToEnumConstructor(PyBALossFunctionType);
  using CSOpts = ceres::Solver::Options;
  auto PyCeresSolverOptions =
      py::class_<CSOpts>(
          m,
          "CeresSolverOptions",
          // If ceres::Solver::Options is registered by pycolmap AND a
          // downstream library, importing the downstream library results in
          // error:
          //   ImportError: generic_type: type "CeresSolverOptions" is already
          //   registered!
          // Adding a `py::module_local()` fixes this.
          // https://github.com/pybind/pybind11/issues/439#issuecomment-1338251822
          py::module_local())
          .def(py::init<>())
          .def_readwrite("function_tolerance", &CSOpts::function_tolerance)
          .def_readwrite("gradient_tolerance", &CSOpts::gradient_tolerance)
          .def_readwrite("parameter_tolerance", &CSOpts::parameter_tolerance)
          .def_readwrite("minimizer_progress_to_stdout",
                         &CSOpts::minimizer_progress_to_stdout)
          .def_readwrite("minimizer_progress_to_stdout",
                         &CSOpts::minimizer_progress_to_stdout)
          .def_readwrite("max_num_iterations", &CSOpts::max_num_iterations)
          .def_readwrite("max_linear_solver_iterations",
                         &CSOpts::max_linear_solver_iterations)
          .def_readwrite("max_num_consecutive_invalid_steps",
                         &CSOpts::max_num_consecutive_invalid_steps)
          .def_readwrite("max_consecutive_nonmonotonic_steps",
                         &CSOpts::max_consecutive_nonmonotonic_steps)
          .def_readwrite("num_threads", &CSOpts::num_threads);
  MakeDataclass(PyCeresSolverOptions);
  auto PyBundleAdjustmentOptions =
      py::class_<BAOpts>(m, "BundleAdjustmentOptions")
          .def(py::init<>())
          .def_readwrite("loss_function_type",
                         &BAOpts::loss_function_type,
                         "Loss function types: Trivial (non-robust) and Cauchy "
                         "(robust) loss.")
          .def_readwrite("loss_function_scale",
                         &BAOpts::loss_function_scale,
                         "Scaling factor determines residual at which "
                         "robustification takes place.")
          .def_readwrite("refine_focal_length",
                         &BAOpts::refine_focal_length,
                         "Whether to refine the focal length parameter group.")
          .def_readwrite(
              "refine_principal_point",
              &BAOpts::refine_principal_point,
              "Whether to refine the principal point parameter group.")
          .def_readwrite("refine_extra_params",
                         &BAOpts::refine_extra_params,
                         "Whether to refine the extra parameter group.")
          .def_readwrite("refine_extrinsics",
                         &BAOpts::refine_extrinsics,
                         "Whether to refine the extrinsic parameter group.")
          .def_readwrite("print_summary",
                         &BAOpts::print_summary,
                         "Whether to print a final summary.")
          .def_readwrite("min_num_residuals_for_multi_threading",
                         &BAOpts::min_num_residuals_for_multi_threading,
                         "Minimum number of residuals to enable "
                         "multi-threading. Note that "
                         "single-threaded is typically better for small bundle "
                         "adjustment problems "
                         "due to the overhead of threading. ")
          .def_readwrite("solver_options",
                         &BAOpts::solver_options,
                         "Ceres-Solver options.");
  MakeDataclass(PyBundleAdjustmentOptions);
  auto ba_options = PyBundleAdjustmentOptions().cast<BAOpts>();

  m.def("triangulate_points",
        &TriangulatePoints,
        "reconstruction"_a,
        "database_path"_a,
        "image_path"_a,
        "output_path"_a,
        "clear_points"_a = true,
        "options"_a = mapper_options,
        "refine_intrinsics"_a = false,
        "Triangulate 3D points from known camera poses");

  m.def("incremental_mapping",
        &IncrementalMapping,
        "database_path"_a,
        "image_path"_a,
        "output_path"_a,
        "options"_a = mapper_options,
        "input_path"_a = py::str(""),
        "initial_image_pair_callback"_a = py::none(),
        "next_image_callback"_a = py::none(),
        "Recover 3D points and unknown camera poses");

  m.def("bundle_adjustment",
        &BundleAdjustment,
        "reconstruction"_a,
        "options"_a = ba_options,
        "Jointly refine 3D points and camera poses");
}
