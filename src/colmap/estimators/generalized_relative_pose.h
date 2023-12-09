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

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Solver for the Generalized Relative Pose problem using a minimal of 8 2D-2D
// correspondences. This implementation is based on:
//
//    "Efficient Computation of Relative Pose for Multi-Camera Systems",
//    Kneip and Li. CVPR 2014.
//
// Note that the solution to this problem is degenerate in the case of pure
// translation and when all correspondences are observed from the same cameras.
//
// The implementation is a modified and improved version of Kneip's original
// implementation in OpenGV licensed under the BSD license.
class GR6PEstimator {
 public:
  // The generalized image observations of the left camera, which is composed of
  // the relative pose of a camera in the generalized camera and a ray in the
  // camera frame.
  struct X_t {
    Rigid3d cam_from_rig;
    Eigen::Vector3d ray_in_cam;
  };

  // The normalized image feature points in the right camera.
  typedef X_t Y_t;
  // The estimated rig2_from_rig1 relative pose between the generalized cameras.
  typedef Rigid3d M_t;

  // The minimum number of samples needed to estimate a model. Note that in
  // theory the minimum required number of samples is 6 but Laurent Kneip showed
  // in his paper that using 8 samples is more stable.
  static const int kMinNumSamples = 8;

  // Estimate the most probable solution of the GR6P problem from a set of
  // six 2D-2D point correspondences.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // Calculate the squared Sampson error between corresponding points.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& rig2_from_rig1,
                        std::vector<double>* residuals);
};

}  // namespace colmap
