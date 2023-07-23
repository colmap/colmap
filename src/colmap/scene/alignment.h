// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/geometry/sim3.h"
#include "colmap/scene/reconstruction.h"

namespace colmap {

bool AlignReconstructionToLocations(
    const Reconstruction& reconstruction,
    const std::vector<std::string>& image_names,
    const std::vector<Eigen::Vector3d>& locations,
    int min_common_images,
    const RANSACOptions& ransac_options,
    Sim3d* tform);

// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers. An alignment
// is verified by reprojecting common 3D point observations.
// The min_inlier_observations threshold determines how many observations
// in a common image must reproject within the given threshold.
bool AlignReconstructions(const Reconstruction& src_reconstruction,
                          const Reconstruction& tgt_reconstruction,
                          double min_inlier_observations,
                          double max_reproj_error,
                          Sim3d* tgtFromSrc);

// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers and by
// minimizing the Euclidean distance between them in world space.
bool AlignReconstructions(const Reconstruction& src_reconstruction,
                          const Reconstruction& tgt_reconstruction,
                          double max_proj_center_error,
                          Sim3d* tgtFromSrc);

// Compute image alignment errors in the target coordinate frame.
struct ImageAlignmentError {
  image_t image_id = kInvalidImageId;
  double rotation_error_deg = -1;
  double proj_center_error = -1;
};
std::vector<ImageAlignmentError> ComputeImageAlignmentError(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const Sim3d& tgtFromSrc);

// Aligns the source to the target reconstruction and merges cameras, images,
// points3D into the target using the alignment. Returns false on failure.
bool MergeReconstructions(double max_reproj_error,
                          const Reconstruction& src_reconstruction,
                          Reconstruction* tgt_reconstruction);

}  // namespace colmap
