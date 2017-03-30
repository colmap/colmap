// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "base/epnp"
#include "util/testing.h"

#include <Eigen/Core>

#include "base/pose.h"
#include "base/similarity_transform.h"
#include "estimators/epnp.h"
#include "estimators/essential_matrix.h"
#include "optim/ransac.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(Estimate) {
  SetPRNGSeed(0);

  std::vector<Eigen::Vector3d> points3D;
  points3D.emplace_back(1, 1, 1);
  points3D.emplace_back(0, 1, 1);
  points3D.emplace_back(3, 1.0, 4);
  points3D.emplace_back(3, 1.1, 4);
  points3D.emplace_back(3, 1.2, 4);
  points3D.emplace_back(3, 1.3, 4);
  points3D.emplace_back(3, 1.4, 4);
  points3D.emplace_back(2, 1, 7);

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  for (double qx = 0; qx < 1; qx += 0.2) {
    for (double tx = 0; tx < 1; tx += 0.1) {
      const SimilarityTransform3 orig_tform(1, Eigen::Vector4d(1, qx, 0, 0),
                                            Eigen::Vector3d(tx, 0, 0));

      // Project points to camera coordinate system.
      std::vector<Eigen::Vector2d> points2D;
      for (size_t i = 0; i < points3D.size(); ++i) {
        Eigen::Vector3d point3D_camera = points3D[i];
        orig_tform.TransformPoint(&point3D_camera);
        points2D.push_back(point3D_camera.hnormalized());
      }

      RANSACOptions options;
      options.max_error = 1e-5;
      RANSAC<EPnPEstimator> ransac(options);
      const auto report = ransac.Estimate(points2D, points3D);

      BOOST_CHECK_EQUAL(report.success, true);

      // Test if correct transformation has been determined.
      const double matrix_diff =
          (orig_tform.Matrix().topLeftCorner<3, 4>() - report.model).norm();
      BOOST_CHECK(matrix_diff < 1e-3);

      // Test residuals of exact points.
      std::vector<double> residuals;
      EPnPEstimator::Residuals(points2D, points3D, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        BOOST_CHECK(residuals[i] < 1e-3);
      }

      // Test residuals of faulty points.
      EPnPEstimator::Residuals(points2D, points3D_faulty, report.model,
                               &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        BOOST_CHECK(residuals[i] > 0.1);
      }
    }
  }
}
