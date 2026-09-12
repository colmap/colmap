// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/sim3.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"

#include <vector>

namespace colmap {

// Robustly align reconstruction to given image locations (projection centers).
bool AlignReconstructionToLocations(
    const Reconstruction& src_reconstruction,
    const std::vector<std::string>& tgt_image_names,
    const std::vector<Eigen::Vector3d>& tgt_image_locations,
    int min_common_images,
    const RANSACOptions& ransac_options,
    Sim3d* tgt_from_src);

// Robustly align reconstruction to given pose priors. If max_error is not set
// in the RANSAC options, derive it from the median position covariance.
bool AlignReconstructionToPosePriors(
    const Reconstruction& src_reconstruction,
    const std::vector<PosePrior>& tgt_pose_priors,
    RANSACOptions ransac_options,
    double prior_position_fallback_stddev,
    Sim3d* tgt_from_src);

// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers. An alignment
// is verified by reprojecting common 3D point observations.
// The min_inlier_observations threshold determines how many observations
// in a common image must reproject within the given threshold.
bool AlignReconstructionsViaReprojections(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    double min_inlier_observations,
    double max_reproj_error,
    Sim3d* tgt_from_src);

// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers and by
// minimizing the Euclidean distance between them in world space.
bool AlignReconstructionsViaProjCenters(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    double max_proj_center_error,
    Sim3d* tgt_from_src);

// Robustly compute the alignment between reconstructions that share the
// same 2D points. It is estimated by minimizing the 3D distance between
// corresponding 3D points.
bool AlignReconstructionsViaPoints(const Reconstruction& src_reconstruction,
                                   const Reconstruction& tgt_reconstruction,
                                   size_t min_common_observations,
                                   double max_error,
                                   double min_inlier_ratio,
                                   Sim3d* tgt_from_src);

// Compute image alignment errors in the target coordinate frame.
struct ImageAlignmentError {
  std::string image_name;
  double rotation_error_deg = -1;
  double proj_center_error = -1;
};
std::vector<ImageAlignmentError> ComputeImageAlignmentError(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const Sim3d& tgt_from_src);

// Summary of alignment errors for image poses.
struct AlignmentErrorSummary {
  struct Statistics {
    double min = 0;
    double max = 0;
    double mean = 0;
    double median = 0;
    double p90 = 0;
    double p99 = 0;
  };

  Statistics rotation_errors_deg;
  Statistics proj_center_errors;

  static AlignmentErrorSummary Compute(
      const std::vector<ImageAlignmentError>& errors);
};

// Aligns the source to the target reconstruction and merges cameras, images,
// points3D into the target using the alignment. Returns false on failure.
bool MergeReconstructions(double max_reproj_error,
                          const Reconstruction& src_reconstruction,
                          Reconstruction& tgt_reconstruction);

// Align reconstruction to the original metric scales in rig extrinsics. Returns
// false if there is no available non-panoramic rig in the alignment process.
bool AlignReconstructionToOrigRigScales(
    const NodeHashMap<rig_t, Rig>& orig_rigs, Reconstruction* reconstruction);

}  // namespace colmap
