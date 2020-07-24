// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "base/generalized_relative_pose"
#include "util/testing.h"

#include <array>

#include "base/pose.h"
#include "base/projection.h"
#include "base/similarity_transform.h"
#include "estimators/generalized_relative_pose.h"
#include "optim/loransac.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(Estimate) {
  SetPRNGSeed(0);

  const size_t kNumPoints = 100;

  std::vector<Eigen::Vector3d> points3D;
  for (size_t i = 0; i < kNumPoints; ++i) {
    points3D.emplace_back(RandomReal<double>(-10, 10),
                          RandomReal<double>(-10, 10),
                          RandomReal<double>(-10, 10));
  }

  for (double qx = 0; qx < 0.4; qx += 0.1) {
    for (double tx = 0; tx < 0.5; tx += 0.1) {
      const int kRefTform = 1;
      const int kNumTforms = 3;

      const std::array<SimilarityTransform3, kNumTforms> orig_tforms = {{
          SimilarityTransform3(1, Eigen::Vector4d(1, qx, 0, 0),
                               Eigen::Vector3d(tx, 0.1, 0)),
          SimilarityTransform3(1, Eigen::Vector4d(1, qx + 0.05, 0, 0),
                               Eigen::Vector3d(tx, 0.2, 0)),
          SimilarityTransform3(1, Eigen::Vector4d(1, qx + 0.1, 0, 0),
                               Eigen::Vector3d(tx, 0.3, 0)),
      }};

      std::array<Eigen::Matrix3x4d, kNumTforms> rel_tforms;
      for (size_t i = 0; i < kNumTforms; ++i) {
        Eigen::Vector4d rel_qvec;
        Eigen::Vector3d rel_tvec;
        ComputeRelativePose(orig_tforms[kRefTform].Rotation(),
                            orig_tforms[kRefTform].Translation(),
                            orig_tforms[i].Rotation(),
                            orig_tforms[i].Translation(), &rel_qvec, &rel_tvec);
        rel_tforms[i] = ComposeProjectionMatrix(rel_qvec, rel_tvec);
      }

      // Project points to cameras.
      std::vector<GR6PEstimator::X_t> points1;
      std::vector<GR6PEstimator::Y_t> points2;
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d point3D_camera1 =
            rel_tforms[i % kNumTforms] * points3D[i].homogeneous();
        Eigen::Vector3d point3D_camera2 = points3D[i];
        orig_tforms[(i + 1) % kNumTforms].TransformPoint(&point3D_camera2);

        if (point3D_camera1.z() < 0 || point3D_camera2.z() < 0) {
          continue;
        }

        points1.emplace_back();
        points1.back().rel_tform = rel_tforms[i % kNumTforms];
        points1.back().xy = point3D_camera1.hnormalized();

        points2.emplace_back();
        points2.back().rel_tform = rel_tforms[(i + 1) % kNumTforms];
        points2.back().xy = point3D_camera2.hnormalized();
      }

      RANSACOptions options;
      options.max_error = 1e-3;
      LORANSAC<GR6PEstimator, GR6PEstimator> ransac(options);
      const auto report = ransac.Estimate(points1, points2);

      BOOST_CHECK_EQUAL(report.success, true);

      const double matrix_diff =
          (orig_tforms[kRefTform].Matrix().topLeftCorner<3, 4>() - report.model)
              .norm();
      BOOST_CHECK_LE(matrix_diff, 1e-2);

      std::vector<double> residuals;
      GR6PEstimator::Residuals(points1, points2, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        BOOST_CHECK_LE(residuals[i], options.max_error);
      }
    }
  }
}
