#pragma once

#include "colmap/estimators/alignment.h"
#include "colmap/exe/model.h"
#include "colmap/geometry/sim3.h"
#include "colmap/scene/reconstruction.h"

#include "pycolmap/log_exceptions.h"

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
         const double max_reproj_error) {
        THROW_CHECK_GE(min_inlier_observations, 0.0);
        THROW_CHECK_LE(min_inlier_observations, 1.0);
        Sim3d tgt_from_src;
        THROW_CHECK(
            AlignReconstructionsViaReprojections(src_reconstruction,
                                                 tgt_reconstruction,
                                                 min_inlier_observations,
                                                 max_reproj_error,
                                                 &tgt_from_src));
        return tgt_from_src;
      },
      "src_reconstruction"_a,
      "tgt_reconstruction"_a,
      "min_inlier_observations"_a = 0.3,
      "max_reproj_error"_a = 8.0);

  m.def(
      "align_reconstructions_via_proj_centers",
      [](const Reconstruction& src_reconstruction,
         const Reconstruction& tgt_reconstruction,
         const double max_proj_center_error) {
        THROW_CHECK_GT(max_proj_center_error, 0.0);
        Sim3d tgt_from_src;
        THROW_CHECK(AlignReconstructionsViaProjCenters(src_reconstruction,
                                                       tgt_reconstruction,
                                                       max_proj_center_error,
                                                       &tgt_from_src));
        return tgt_from_src;
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
         const double min_inlier_ratio) {
        THROW_CHECK_GT(min_common_observations, 0);
        THROW_CHECK_GT(max_error, 0.0);
        THROW_CHECK_GE(min_inlier_ratio, 0.0);
        THROW_CHECK_LE(min_inlier_ratio, 1.0);
        Sim3d tgt_from_src;
        THROW_CHECK(AlignReconstructionsViaPoints(src_reconstruction,
                                                  tgt_reconstruction,
                                                  min_common_observations,
                                                  max_error,
                                                  min_inlier_ratio,
                                                  &tgt_from_src));
        return tgt_from_src;
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
         const RANSACOptions& ransac_options) {
        THROW_CHECK_GE(min_common_images, 3);
        THROW_CHECK_EQ(image_names.size(), locations.size());
        Sim3d locationsFromSrc;
        THROW_CHECK(AlignReconstructionToLocations(src,
                                                   image_names,
                                                   locations,
                                                   min_common_images,
                                                   ransac_options,
                                                   &locationsFromSrc));
        return locationsFromSrc;
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
         double max_proj_center_error) {
        std::vector<ImageAlignmentError> errors;
        Sim3d rec2_from_rec1;
        THROW_CUSTOM_CHECK_MSG(CompareModels(reconstruction1,
                                             reconstruction2,
                                             alignment_error,
                                             min_inlier_observations,
                                             max_reproj_error,
                                             max_proj_center_error,
                                             errors,
                                             rec2_from_rec1),
                               std::runtime_error,
                               "=> Reconstruction alignment failed.");
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
