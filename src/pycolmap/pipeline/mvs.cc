#include "colmap/mvs/fusion.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/misc.h"

#ifdef COLMAP_CUDA_ENABLED
#include "colmap/mvs/patch_match.h"
#endif  // COLMAP_CUDA_ENABLED

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

#ifdef COLMAP_CUDA_ENABLED
void PatchMatchStereo(const std::string& workspace_path,
                      std::string workspace_format,
                      const std::string& pmvs_option_name,
                      const mvs::PatchMatchOptions& options,
                      const std::string& config_path) {
  THROW_CHECK_DIR_EXISTS(workspace_path);
  StringToLower(&workspace_format);
  THROW_CHECK(workspace_format == "colmap" || workspace_format == "pmvs")
      << "Invalid `workspace_format` - supported values are 'COLMAP' or "
         "'PMVS'.";

  py::gil_scoped_release release;
  mvs::PatchMatchController controller(
      options, workspace_path, workspace_format, pmvs_option_name, config_path);
  controller.Run();
}
#endif  // COLMAP_CUDA_ENABLED

Reconstruction StereoFusion(const std::string& output_path,
                            const std::string& workspace_path,
                            std::string workspace_format,
                            const std::string& pmvs_option_name,
                            std::string input_type,
                            const mvs::StereoFusionOptions& options) {
  THROW_CHECK_DIR_EXISTS(workspace_path);
  StringToLower(&workspace_format);
  THROW_CHECK(workspace_format == "colmap" || workspace_format == "pmvs")
      << "Invalid `workspace_format` - supported values are 'COLMAP' or "
         "'PMVS'.";

  StringToLower(&input_type);
  THROW_CHECK(input_type == "photometric" || input_type == "geometric")
      << "Invalid input type - supported values are 'photometric' and "
         "'geometric'.";

  py::gil_scoped_release release;
  mvs::StereoFusion fuser(
      options, workspace_path, workspace_format, pmvs_option_name, input_type);
  fuser.Run();

  Reconstruction reconstruction;
  // read data from sparse reconstruction
  if (workspace_format == "colmap") {
    reconstruction.Read(JoinPaths(workspace_path, "sparse"));
  }

  // overwrite sparse point cloud with dense point cloud from fuser
  reconstruction.ImportPLY(fuser.GetFusedPoints());

  if (ExistsDir(output_path)) {
    reconstruction.WriteBinary(output_path);
  } else {
    WriteBinaryPlyPoints(output_path, fuser.GetFusedPoints());
    mvs::WritePointsVisibility(output_path + ".vis",
                               fuser.GetFusedPointsVisibility());
  }

  return reconstruction;
}

void BindMVS(py::module& m) {
#ifdef COLMAP_CUDA_ENABLED
  using PMOpts = mvs::PatchMatchOptions;
  auto PyPatchMatchOptions =
      py::class_<PMOpts>(m, "PatchMatchOptions")
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
                         "Whether to write the consistency graph.");
  MakeDataclass(PyPatchMatchOptions);

  m.def("patch_match_stereo",
        &PatchMatchStereo,
        "workspace_path"_a,
        "workspace_format"_a = "COLMAP",
        "pmvs_option_name"_a = "option-all",
        py::arg_v("options", mvs::PatchMatchOptions(), "PatchMatchOptions()"),
        "config_path"_a = "",
        "Runs Patch-Match-Stereo (requires CUDA)");
#endif  // COLMAP_CUDA_ENABLED

  using SFOpts = mvs::StereoFusionOptions;
  auto PyStereoFusionOptions =
      py::class_<SFOpts>(m, "StereoFusionOptions")
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
                         "Bounding box Tuple[min, max]");
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
      "Stereo Fusion");
}
