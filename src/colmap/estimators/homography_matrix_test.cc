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

#define TEST_NAME "estimators/homography_matrix"
#include "colmap/estimators/homography_matrix.h"

#include "colmap/util/testing.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEstimate) {
  for (double x = 0; x < 10; ++x) {
    Eigen::Matrix3d H0;
    H0 << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    std::vector<Eigen::Vector2d> src;
    src.emplace_back(x, 0);
    src.emplace_back(1, 0);
    src.emplace_back(2, 1);
    src.emplace_back(10, 30);

    std::vector<Eigen::Vector2d> dst;

    for (size_t i = 0; i < 4; ++i) {
      const Eigen::Vector3d dsth = H0 * src[i].homogeneous();
      dst.push_back(dsth.hnormalized());
    }

    HomographyMatrixEstimator est_tform;
    const auto models = est_tform.Estimate(src, dst);

    std::vector<double> residuals;
    est_tform.Residuals(src, dst, models[0], &residuals);

    for (size_t i = 0; i < 4; ++i) {
      BOOST_CHECK(residuals[i] < 1e-6);
    }
  }
}
