#pragma once

#include "colmap/estimators/alignment.h"
#include "colmap/exe/model.h"
#include "colmap/geometry/sim3.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindAlignmentEstimator(py::module& m) {
  py::class_<ImageAlignmentError>(m, "ImageAlignmentError")
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
         const double max_reproj_error) -> py::object {
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
         const double max_proj_center_error) -> py::object {
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
         const double min_inlier_ratio) -> py::object {
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
      "align_reconstrution_to_locations",
      [](const Reconstruction& src,
         const std::vector<std::string>& image_names,
         const std::vector<Eigen::Vector3d>& locations,
         const int min_common_images,
         const RANSACOptions& ransac_options) -> py::object {
        Sim3d locations_from_src;
        if (!AlignReconstructionToLocations(src,
                                            image_names,
                                            locations,
                                            min_common_images,
                                            ransac_options,
                                            &locations_from_src)) {
          return py::none();
        }
        return py::cast(locations_from_src);
      },
      "src"_a,
      "image_names"_a,
      "locations"_a,
      "min_common_points"_a,
      "ransac_options"_a);

  m.def(
      "compare_reconstructions",
      [](const Reconstruction& reconstruction1,
         const Reconstruction& reconstruction2,
         const std::string& alignment_error,
         double min_inlier_observations,
         double max_reproj_error,
         double max_proj_center_error) -> py::object {
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
}
