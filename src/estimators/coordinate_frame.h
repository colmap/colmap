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

#ifndef COLMAP_SRC_ESTIMATORS_COORDINATE_AXES_H_
#define COLMAP_SRC_ESTIMATORS_COORDINATE_AXES_H_

#include <Eigen/Core>

#include "base/reconstruction.h"

namespace colmap {

struct ManhattanWorldFrameEstimationOptions {
  // The maximum image size for line detection.
  int max_image_size = 1024;
  // The minimum length of line segments in pixels.
  double min_line_length = 3;
  // The tolerance for classifying lines into horizontal/vertical.
  double line_orientation_tolerance = 0.2;
  // The maximum distance in pixels between lines and the vanishing points.
  double max_line_vp_distance = 0.5;
  // The maximum cosine distance between estimated axes to be inliers.
  double max_axis_distance = 0.05;
};

// Estimate gravity vector by assuming gravity-aligned image orientation, i.e.
// the majority of images is assumed to have the gravity vector aligned with an
// upright image plane.
Eigen::Vector3d EstimateGravityVectorFromImageOrientation(
    const Reconstruction& reconstruction,
    const double max_axis_distance = 0.05);

// Estimate the coordinate frame of the reconstruction assuming a Manhattan
// world by finding the major vanishing points in each image. This function
// assumes that the majority of images is taken in upright direction, i.e.
// people are standing upright in the image. The orthonormal axes of the
// estimated coordinate frame will be given in the columns of the returned
// matrix. If one axis could not be determined, the respective column will be
// zero. The axes are specified in the world coordinate system in the order
// rightward, downward, forward.
Eigen::Matrix3d EstimateManhattanWorldFrame(
    const ManhattanWorldFrameEstimationOptions& options,
    const Reconstruction& reconstruction, const std::string& image_path);

// Aligns the reconstruction to the plane defined by running PCA on the 3D
// points. The model centroid is at the origin of the new coordinate system
// and the X axis is the first principal component with the Y axis being the
// second principal component
void AlignToPrincipalPlane(Reconstruction* recon, SimilarityTransform3* tform);

// Aligns the reconstruction to the local ENU plane orientation. Rotates the
// reconstruction such that the x-y plane aligns with the ENU tangent plane at
// the point cloud centroid and translates the origin to the centroid.
// If unscaled == true, then the original scale of the model remains unchanged.
void AlignToENUPlane(Reconstruction* recon, SimilarityTransform3* tform,
                     bool unscaled);

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_COORDINATE_AXES_H_
