// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/geometry/sim3.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/reconstruction.h"

namespace colmap {

// Robustly align reconstruction to given image locations (projection centers).
bool AlignReconstructionToLocations(
    const Reconstruction& src_reconstruction,
    const std::vector<std::string>& tgt_image_names,
    const std::vector<Eigen::Vector3d>& tgt_image_locations,
    int min_common_images,
    const RANSACOptions& ransac_options,
    Sim3d* tgt_from_src);

// Robustly align reconstruction to given pose priors.
bool AlignReconstructionToPosePriors(
    const Reconstruction& src_reconstruction,
    const std::unordered_map<image_t, PosePrior>& tgt_pose_priors,
    const RANSACOptions& ransac_options,
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

// Aligns the source to the target reconstruction and merges cameras, images,
// points3D into the target using the alignment. Both reconstructions must come
// from the same database to keep identifiers consistent. Returns false on
// failure.
struct MergeReconstructionsOptions {
  // Method for selecting or merging camera intrinsics
  // SOURCE:  Use cameras from the source reconstruction
  // TARGET:  Use cameras from the target reconstruction
  // BETTER:  Choose the camera with the smaller reprojection error
  // REFINED: Merge cameras and refine by optimization
  MAKE_ENUM_CLASS(CameraMergeMethod, 0, SOURCE, TARGET, BETTER, REFINED);

  CameraMergeMethod camera_merge_method = CameraMergeMethod::REFINED;

  // Minimum required inlier ratio per overlapping image pair.
  double min_inlier_observations = 0.3;

  // Maximum reprojection error for considering a point3D as inlier.
  double max_reproj_error = 64;

  bool Check() const;
};
bool MergeReconstructions(const MergeReconstructionsOptions& options,
                          const Reconstruction& src_reconstruction,
                          Reconstruction& tgt_reconstruction);

// Align reconstruction to the original metric scales in rig extrinsics. Returns
// false if there is no available non-panoramic rig in the alignment process.
bool AlignReconstructionToOrigRigScales(
    const std::unordered_map<rig_t, Rig>& orig_rigs,
    Reconstruction* reconstruction);

}  // namespace colmap
