// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/mvs.h"

#include "colmap/mvs/fusion.h"
#include "colmap/mvs/mvs_estimator_controller.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/controller_thread.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/patch_match.h"
#endif

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void PatchMatchStereo(
    const std::filesystem::path& workspace_path,
    std::string workspace_format,
    const std::string& pmvs_option_name,
    const mvs::PatchMatchOptions& options,
    const std::filesystem::path& config_path,
    const std::shared_ptr<CancellationToken>& cancellation_token) {
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  StringToLower(&workspace_format);
  THROW_CHECK(workspace_format == "colmap" || workspace_format == "pmvs")
      << "Invalid `workspace_format` " << workspace_format
      << " - supported values are 'COLMAP' or 'PMVS'.";

  auto controller = std::make_shared<mvs::PatchMatchController>(
      options, workspace_path, workspace_format, pmvs_option_name, config_path);
  ControllerThread<mvs::PatchMatchController> thread(std::move(controller));
  thread.Start();
  PyWait(&thread, cancellation_token);
#else
  static_cast<void>(workspace_path);
  static_cast<void>(workspace_format);
  static_cast<void>(pmvs_option_name);
  static_cast<void>(options);
  static_cast<void>(config_path);
  static_cast<void>(cancellation_token);
  LOG(FATAL_THROW) << "Dense stereo reconstruction requires CUDA or HIP, "
                      "neither of which is available on your system.";
#endif
}

void EstimateMVSDepth(
    const std::filesystem::path& workspace_path,
    std::string workspace_format,
    const std::string& pmvs_option_name,
    const mvs::MVSEstimator::Options& options,
    const std::filesystem::path& config_path,
    const std::shared_ptr<CancellationToken>& cancellation_token) {
  StringToLower(&workspace_format);
  THROW_CHECK(workspace_format == "colmap" || workspace_format == "pmvs")
      << "Invalid `workspace_format` " << workspace_format;
  auto controller = std::make_shared<mvs::MVSEstimatorController>(
      options, workspace_path, workspace_format, pmvs_option_name, config_path);
  ControllerThread<mvs::MVSEstimatorController> thread(std::move(controller));
  thread.Start();
  PyWait(&thread, cancellation_token);
}

Reconstruction StereoFusion(
    const std::filesystem::path& output_path,
    const std::filesystem::path& workspace_path,
    std::string workspace_format,
    const std::string& pmvs_option_name,
    std::string input_type,
    const mvs::StereoFusionOptions& options,
    std::string output_type,
    const std::shared_ptr<CancellationToken>& cancellation_token) {
  py::gil_scoped_release release;
  PyInterruptChecker interrupt_checker(cancellation_token);
  Reconstruction reconstruction =
      RunStereoFuserImpl(output_path,
                         workspace_path,
                         std::move(workspace_format),
                         pmvs_option_name,
                         std::move(input_type),
                         options,
                         std::move(output_type),
                         interrupt_checker.Callback());
  interrupt_checker.CheckAndThrow();
  return reconstruction;
}

void BindMVS(py::module& m) {
  py::enum_<mvs::MVSEstimator::Type>(m, "MVSEstimatorType")
      .value("PATCH_MATCH", mvs::MVSEstimator::Type::PATCH_MATCH)
      .value("MVSFORMER_PP", mvs::MVSEstimator::Type::MVSFORMER_PP)
      .export_values();

  using MFOpts = mvs::MVSFormerPlusPlus::Options;
  auto PyMVSFormerOptions =
      py::classh<MFOpts>(m, "MVSFormerPlusPlusOptions")
          .def(py::init<>())
          .def_readwrite("model_path", &MFOpts::model_path)
          .def_readwrite("num_views", &MFOpts::num_views)
          .def_readwrite("max_image_size", &MFOpts::max_image_size)
          .def_readwrite("depth_min", &MFOpts::depth_min)
          .def_readwrite("depth_max", &MFOpts::depth_max)
          .def_readwrite("min_confidence", &MFOpts::min_confidence)
          .def_readwrite("filter_max_reproj_error",
                         &MFOpts::filter_max_reproj_error)
          .def_readwrite("filter_max_depth_error",
                         &MFOpts::filter_max_depth_error)
          .def_readwrite("filter_max_normal_error",
                         &MFOpts::filter_max_normal_error)
          .def_readwrite("filter_min_num_consistent",
                         &MFOpts::filter_min_num_consistent)
          .def_readwrite("cache_size", &MFOpts::cache_size)
          .def_readwrite("use_gpu", &MFOpts::use_gpu)
          .def_readwrite("gpu_index", &MFOpts::gpu_index)
          .def_readwrite("num_threads", &MFOpts::num_threads)
          .def_readwrite("geom_consistency", &MFOpts::geom_consistency)
          .def_readwrite("allow_missing_files", &MFOpts::allow_missing_files)
          .def_readwrite("write_consistency_graph",
                         &MFOpts::write_consistency_graph)
          .def("check", &MFOpts::Check);
  MakeDataclass(PyMVSFormerOptions);

  using PMOpts = mvs::PatchMatchOptions;
  auto PyPatchMatchOptions =
      py::classh<PMOpts>(m, "PatchMatchOptions")
          .def(py::init<>())
          .def_readwrite("max_image_size",
                         &PMOpts::max_image_size,
                         "Maximum image size in either dimension.")
          .def_readwrite(
              "gpu_index",
              &PMOpts::gpu_index,
              "Index of the GPU used for patch match. For multi-GPU usage, "
              "you should separate multiple GPU indices by comma, e.g., "
              "\"0,1,2,3\".")
          .def_readwrite("depth_min", &PMOpts::depth_min)
          .def_readwrite("depth_max", &PMOpts::depth_max)
          .def_readwrite(
              "window_radius",
              &PMOpts::window_radius,
              "Half window size to compute NCC photo-consistency cost.")
          .def_readwrite("window_step",
                         &PMOpts::window_step,
                         "Number of pixels to skip when computing NCC.")
          .def_readwrite("sigma_spatial",
                         &PMOpts::sigma_spatial,
                         "Spatial sigma for bilaterally weighted NCC.")
          .def_readwrite("sigma_color",
                         &PMOpts::sigma_color,
                         "Color sigma for bilaterally weighted NCC.")
          .def_readwrite(
              "num_samples",
              &PMOpts::num_samples,
              "Number of random samples to draw in Monte Carlo sampling.")
          .def_readwrite("ncc_sigma",
                         &PMOpts::ncc_sigma,
                         "Spread of the NCC likelihood function.")
          .def_readwrite("min_triangulation_angle",
                         &PMOpts::min_triangulation_angle,
                         "Minimum triangulation angle in degrees.")
          .def_readwrite("incident_angle_sigma",
                         &PMOpts::incident_angle_sigma,
                         "Spread of the incident angle likelihood function.")
          .def_readwrite("num_iterations",
                         &PMOpts::num_iterations,
                         "Number of coordinate descent iterations.")
          .def_readwrite("geom_consistency",
                         &PMOpts::geom_consistency,
                         "Whether to add a regularized geometric consistency "
                         "term to the cost function. If true, the "
                         "`depth_maps` and `normal_maps` must not be null.")
          .def_readwrite("geom_consistency_regularizer",
                         &PMOpts::geom_consistency_regularizer,
                         "The relative weight of the geometric consistency "
                         "term w.r.t. to the photo-consistency term.")
          .def_readwrite("geom_consistency_max_cost",
                         &PMOpts::geom_consistency_max_cost,
                         "Maximum geometric consistency cost in terms of the "
                         "forward-backward reprojection error in pixels.")
          .def_readwrite(
              "filter", &PMOpts::filter, "Whether to enable filtering.")
          .def_readwrite(
              "filter_min_ncc",
              &PMOpts::filter_min_ncc,
              "Minimum NCC coefficient for pixel to be photo-consistent.")
          .def_readwrite("filter_min_triangulation_angle",
                         &PMOpts::filter_min_triangulation_angle,
                         "Minimum triangulation angle to be stable.")
          .def_readwrite(
              "filter_min_num_consistent",
              &PMOpts::filter_min_num_consistent,
              "Minimum number of source images have to be consistent "
              "for pixel not to be filtered.")
          .def_readwrite(
              "filter_geom_consistency_max_cost",
              &PMOpts::filter_geom_consistency_max_cost,
              "Maximum forward-backward reprojection error for pixel "
              "to be geometrically consistent.")
          .def_readwrite("cache_size",
                         &PMOpts::cache_size,
                         "Cache size in gigabytes for patch match.")
          .def_readwrite(
              "allow_missing_files",
              &PMOpts::allow_missing_files,
              "Whether to tolerate missing images/maps in the problem setup")
          .def_readwrite("write_consistency_graph",
                         &PMOpts::write_consistency_graph,
                         "Whether to write the consistency graph.")
          .def_readwrite("num_threads",
                         &PMOpts::num_threads,
                         "Number of threads for processing. "
                         "-1 uses all available threads.")
          .def("check", &PMOpts::Check);
  MakeDataclass(PyPatchMatchOptions);

  using MVSEOpts = mvs::MVSEstimator::Options;
  auto PyMVSEstimatorOptions =
      py::classh<MVSEOpts>(m, "MVSEstimatorOptions")
          .def(py::init<mvs::MVSEstimator::Type>(),
               "type"_a = MVSEOpts::DefaultType())
          .def_readwrite("type", &MVSEOpts::type)
          .def_readwrite("patch_match", &MVSEOpts::patch_match)
          .def_readwrite("mvsformer_pp", &MVSEOpts::mvsformer_pp)
          .def("check", &MVSEOpts::Check);
  MakeDataclass(PyMVSEstimatorOptions);

  m.def("mvs_depth_estimation",
        &EstimateMVSDepth,
        "workspace_path"_a,
        "workspace_format"_a = "COLMAP",
        "pmvs_option_name"_a = "option-all",
        py::arg_v(
            "options", mvs::MVSEstimator::Options(), "MVSEstimatorOptions()"),
        "config_path"_a = "",
        "cancellation_token"_a = py::none(),
        "Runs the selected multi-view stereo depth estimator",
        py::call_guard<py::gil_scoped_release>());

  m.def("patch_match_stereo",
        &PatchMatchStereo,
        "workspace_path"_a,
        "workspace_format"_a = "COLMAP",
        "pmvs_option_name"_a = "option-all",
        py::arg_v("options", mvs::PatchMatchOptions(), "PatchMatchOptions()"),
        "config_path"_a = "",
        "cancellation_token"_a = py::none(),
        "Runs Patch-Match-Stereo (requires CUDA)",
        py::call_guard<py::gil_scoped_release>());

  using SFOpts = mvs::StereoFusionOptions;
  auto PyStereoFusionOptions =
      py::classh<SFOpts>(m, "StereoFusionOptions")
          .def(py::init<>())
          .def_readwrite("mask_path",
                         &SFOpts::mask_path,
                         "Path for PNG masks. Same format expected as "
                         "ImageReaderOptions.")
          .def_readwrite("num_threads",
                         &SFOpts::num_threads,
                         "The number of threads to use during fusion.")
          .def_readwrite("max_image_size",
                         &SFOpts::max_image_size,
                         "Maximum image size in either dimension.")
          .def_readwrite("min_num_pixels",
                         &SFOpts::min_num_pixels,
                         "Minimum number of fused pixels to produce a point.")
          .def_readwrite(
              "max_num_pixels",
              &SFOpts::max_num_pixels,
              "Maximum number of pixels to fuse into a single point.")
          .def_readwrite("max_traversal_depth",
                         &SFOpts::max_traversal_depth,
                         "Maximum depth in consistency graph traversal.")
          .def_readwrite("max_reproj_error",
                         &SFOpts::max_reproj_error,
                         "Maximum relative difference between measured and "
                         "projected pixel.")
          .def_readwrite("max_depth_error",
                         &SFOpts::max_depth_error,
                         "Maximum relative difference between measured and "
                         "projected depth.")
          .def_readwrite("max_normal_error",
                         &SFOpts::max_normal_error,
                         "Maximum angular difference in degrees of normals "
                         "of pixels to be fused.")
          .def_readwrite("check_num_images",
                         &SFOpts::check_num_images,
                         "Number of overlapping images to transitively check "
                         "for fusing points.")
          .def_readwrite(
              "use_cache",
              &SFOpts::use_cache,
              "Flag indicating whether to use LRU cache or pre-load all data")
          .def_readwrite("cache_size",
                         &SFOpts::cache_size,
                         "Cache size in gigabytes for fusion.")
          .def_readwrite("bounding_box",
                         &SFOpts::bounding_box,
                         "Bounding box Tuple[min, max]")
          .def("check", &SFOpts::Check);
  MakeDataclass(PyStereoFusionOptions);

  m.def(
      "stereo_fusion",
      &StereoFusion,
      "output_path"_a,
      "workspace_path"_a,
      "workspace_format"_a = "COLMAP",
      "pmvs_option_name"_a = "option-all",
      "input_type"_a = "geometric",
      py::arg_v("options", mvs::StereoFusionOptions(), "StereoFusionOptions()"),
      "output_type"_a = "bin",
      "cancellation_token"_a = py::none(),
      "Stereo Fusion");
}
