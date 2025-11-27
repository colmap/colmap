#include "colmap/estimators/alignment.h"

#include "colmap/exe/model.h"
#include "colmap/geometry/sim3.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindAlignmentEstimator(py::module& m) {
  py::classh<ImageAlignmentError>(m, "ImageAlignmentError")
      .def(py::init<>())
      .def_readwrite("image_name", &ImageAlignmentError::image_name)
      .def_readwrite("rotation_error_deg",
                     &ImageAlignmentError::rotation_error_deg)
      .def_readwrite("proj_center_error",
                     &ImageAlignmentError::proj_center_error);

  m.def(
      "align_reconstructions_via_reprojections",
      [](const Reconstruction& src_reconstruction,
         const Reconstruction& tgt_reconstruction,
         const double min_inlier_observations,
         const double max_reproj_error) -> py::typing::Optional<Sim3d> {
        Sim3d tgt_from_src;
        if (!AlignReconstructionsViaReprojections(src_reconstruction,
                                                  tgt_reconstruction,
                                                  min_inlier_observations,
                                                  max_reproj_error,
                                                  &tgt_from_src)) {
          return py::none();
        }
        return py::cast(tgt_from_src);
      },
      "src_reconstruction"_a,
      "tgt_reconstruction"_a,
      "min_inlier_observations"_a = 0.3,
      "max_reproj_error"_a = 8.0);

  m.def(
      "align_reconstructions_via_proj_centers",
      [](const Reconstruction& src_reconstruction,
         const Reconstruction& tgt_reconstruction,
         const double max_proj_center_error) -> py::typing::Optional<Sim3d> {
        Sim3d tgt_from_src;
        if (!AlignReconstructionsViaProjCenters(src_reconstruction,
                                                tgt_reconstruction,
                                                max_proj_center_error,
                                                &tgt_from_src)) {
          return py::none();
        }
        return py::cast(tgt_from_src);
      },
      "src_reconstruction"_a,
      "tgt_reconstruction"_a,
      "max_proj_center_error"_a);

  m.def(
      "align_reconstructions_via_points",
      [](const Reconstruction& src_reconstruction,
         const Reconstruction& tgt_reconstruction,
         const size_t min_common_observations,
         const double max_error,
         const double min_inlier_ratio) -> py::typing::Optional<Sim3d> {
        Sim3d tgt_from_src;
        if (!AlignReconstructionsViaPoints(src_reconstruction,
                                           tgt_reconstruction,
                                           min_common_observations,
                                           max_error,
                                           min_inlier_ratio,
                                           &tgt_from_src)) {
          return py::none();
        }
        return py::cast(tgt_from_src);
      },
      "src_reconstruction"_a,
      "tgt_reconstruction"_a,
      "min_common_observations"_a = 3,
      "max_error"_a = 0.005,
      "min_inlier_ratio"_a = 0.9);

  m.def(
      "align_reconstruction_to_locations",
      [](const Reconstruction& src,
         const std::vector<std::string>& tgt_image_names,
         const std::vector<Eigen::Vector3d>& tgt_locations,
         const int min_common_images,
         const RANSACOptions& ransac_options) -> py::typing::Optional<Sim3d> {
        Sim3d locations_from_src;
        if (!AlignReconstructionToLocations(src,
                                            tgt_image_names,
                                            tgt_locations,
                                            min_common_images,
                                            ransac_options,
                                            &locations_from_src)) {
          return py::none();
        }
        return py::cast(locations_from_src);
      },
      "src"_a,
      "tgt_image_names"_a,
      "tgt_locations"_a,
      "min_common_images"_a,
      "ransac_options"_a);

  m.def(
      "compare_reconstructions",
      [](const Reconstruction& reconstruction1,
         const Reconstruction& reconstruction2,
         const std::string& alignment_error,
         double min_inlier_observations,
         double max_reproj_error,
         double max_proj_center_error) -> py::typing::Optional<py::dict> {
        std::vector<ImageAlignmentError> errors;
        Sim3d rec2_from_rec1;
        if (!CompareModels(reconstruction1,
                           reconstruction2,
                           alignment_error,
                           min_inlier_observations,
                           max_reproj_error,
                           max_proj_center_error,
                           errors,
                           rec2_from_rec1)) {
          return py::none();
        }
        return py::dict("rec2_from_rec1"_a = rec2_from_rec1,
                        "errors"_a = errors);
      },
      "reconstruction1"_a,
      "reconstruction2"_a,
      "alignment_error"_a = "reprojection",
      "min_inlier_observations"_a = 0.3,
      "max_reproj_error"_a = 8.0,
      "max_proj_center_error"_a = 0.1);

  using CameraMergeMethod = MergeReconstructionsOptions::CameraMergeMethod;
  auto PyCameraMergeMethod =
      py::enum_<CameraMergeMethod>(m, "CameraMergeMethod")
          .value("SOURCE", CameraMergeMethod::SOURCE)
          .value("TARGET", CameraMergeMethod::TARGET)
          .value("BETTER", CameraMergeMethod::BETTER)
          .value("REFINED", CameraMergeMethod::REFINED);
  AddStringToEnumConstructor(PyCameraMergeMethod);

  using MergeOpts = MergeReconstructionsOptions;
  auto PyMergeOpts = py::classh<MergeOpts>(m, "MergeReconstructionsOptions");
  PyMergeOpts.def(py::init<>())
      .def_readwrite("camera_merge_method",
                     &MergeOpts::camera_merge_method,
                     "Method for selecting or merging camera intrinsics.")
      .def_readwrite(
          "min_inlier_observations",
          &MergeOpts::min_inlier_observations,
          "Minimum required inlier ratio per overlapping image pair.")
      .def_readwrite(
          "max_reproj_error",
          &MergeOpts::max_reproj_error,
          "Maximum reprojection error for considering a point3D as inlier.")
      .def("check", &MergeOpts::Check);
  MakeDataclass(PyMergeOpts);

  m.def(
      "merge_reconstructions",
      [](const MergeReconstructionsOptions& options,
         const Reconstruction& src_reconstruction,
         const Reconstruction& tgt_reconstruction)
          -> py::typing::Optional<Reconstruction> {
        Reconstruction merged_reconstruction = tgt_reconstruction;
        if (!MergeReconstructions(
                options, src_reconstruction, merged_reconstruction)) {
          return py::none();
        }

        return py::cast(merged_reconstruction);
      },
      "options"_a,
      "src_reconstruction"_a,
      "tgt_reconstruction"_a);

  m.def("align_reconstruction_to_orig_rig_scales",
        &AlignReconstructionToOrigRigScales,
        "orig_rigs"_a,
        "reconstruction"_a);
}
