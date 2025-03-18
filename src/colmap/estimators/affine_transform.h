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

#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

class AffineTransformEstimator {
 public:
  typedef Eigen::Vector2d X_t;
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix2x3d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate the affine transformation from at least 3 correspondences.
  static void Estimate(const std::vector<X_t>& src,
                       const std::vector<Y_t>& tgt,
                       std::vector<M_t>* tgt_from_src);

  // Compute the squared transformation error.
  static void Residuals(const std::vector<X_t>& src,
                        const std::vector<Y_t>& tgt,
                        const M_t& tgt_from_src,
                        std::vector<double>* residuals);
};

bool EstimateAffine2d(const std::vector<Eigen::Vector2d>& src,
                      const std::vector<Eigen::Vector2d>& tgt,
                      Eigen::Matrix2x3d& tgt_from_src);

typename RANSAC<AffineTransformEstimator>::Report EstimateAffine2dRobust(
    const std::vector<Eigen::Vector2d>& src,
    const std::vector<Eigen::Vector2d>& tgt,
    const RANSACOptions& options,
    Eigen::Matrix2x3d& tgt_from_src);

}  // namespace colmap
