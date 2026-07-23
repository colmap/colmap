#include "colmap/estimators/alignment.h"

#include "colmap/exe/model.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/geometry/sim3.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"
#include "pycolmap/utils.h"

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
      "align_reconstruction_to_pose_priors",
      [](const Reconstruction& src_reconstruction,
         const std::vector<PosePrior>& tgt_pose_priors,
         const RANSACOptions& ransac_options) -> py::typing::Optional<Sim3d> {
        Sim3d tgt_from_src;
        if (!AlignReconstructionToPosePriors(src_reconstruction,
                                             tgt_pose_priors,
                                             ransac_options,
                                             &tgt_from_src)) {
          return py::none();
        }
        return py::cast(tgt_from_src);
      },
      "src_reconstruction"_a,
      "tgt_pose_priors"_a,
      "ransac_options"_a);

  py::classh<PosePriorAlignmentResult>(m, "PosePriorAlignmentResult")
      .def(py::init<>())
      .def_readwrite("success", &PosePriorAlignmentResult::success)
      .def_readwrite("tgt_from_src", &PosePriorAlignmentResult::tgt_from_src)
      .def_readwrite("correspondence_image_ids",
                     &PosePriorAlignmentResult::correspondence_image_ids)
      .def_readwrite("orientation_requested",
                     &PosePriorAlignmentResult::orientation_requested)
      .def_readwrite("orientation_engaged",
                     &PosePriorAlignmentResult::orientation_engaged)
      .def_readwrite("orientation_image_ids",
                     &PosePriorAlignmentResult::orientation_image_ids)
      .def_readwrite("orientation_residuals_deg",
                     &PosePriorAlignmentResult::orientation_residuals_deg)
      .def_property_readonly(
          "inlier_mask",
          [](const PosePriorAlignmentResult& self) -> PyInlierMask {
            return ToPythonMask(self.inlier_mask);
          })
      .def_property_readonly(
          "orientation_inlier_mask",
          [](const PosePriorAlignmentResult& self) -> PyInlierMask {
            return ToPythonMask(self.orientation_inlier_mask);
          });

  py::classh<AnisotropicPositionGate>(m, "AnisotropicPositionGate")
      .def(py::init<>())
      .def_readwrite("max_horizontal_error",
                     &AnisotropicPositionGate::max_horizontal_error)
      .def_readwrite("max_vertical_error",
                     &AnisotropicPositionGate::max_vertical_error)
      .def("is_set", &AnisotropicPositionGate::IsSet);

  m.def("align_reconstruction_to_pose_priors_robust",
        &AlignReconstructionToPosePriorsRobust,
        "src_reconstruction"_a,
        "tgt_pose_priors"_a,
        "ransac_options"_a,
        "anisotropic_gate"_a = AnisotropicPositionGate(),
        "Robustly align a reconstruction to pose priors, returning the "
        "RANSAC inlier mask and correspondence image ids alongside the "
        "similarity transform. If anisotropic_gate is set, RANSAC "
        "admission evaluates ENU horizontal/vertical residuals separately "
        "instead of the isotropic ransac_options.max_error gate.");

  m.def(
      "refine_pose_prior_alignment_with_orientations",
      [](const Reconstruction& src_reconstruction,
         const std::vector<PosePrior>& tgt_pose_priors,
         const double position_fallback_stddev,
         const double orientation_fallback_stddev_rad,
         const double orientation_max_error_deg,
         PosePriorAlignmentResult result) {
        RefinePosePriorAlignmentWithOrientations(
            src_reconstruction,
            tgt_pose_priors,
            position_fallback_stddev,
            orientation_fallback_stddev_rad,
            orientation_max_error_deg,
            &result);
        return result;
      },
      "src_reconstruction"_a,
      "tgt_pose_priors"_a,
      "position_fallback_stddev"_a,
      "orientation_fallback_stddev_rad"_a,
      "orientation_max_error_deg"_a,
      "result"_a,
      "Refine a successful position alignment with absolute orientation "
      "priors while preserving its position inlier set.");

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

  m.def("align_reconstruction_to_orig_rig_scales",
        &AlignReconstructionToOrigRigScales,
        "orig_rigs"_a,
        "reconstruction"_a);
}
