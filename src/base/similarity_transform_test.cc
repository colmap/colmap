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

#define TEST_NAME "base/similarity_transform"
#include "util/testing.h"

#include <Eigen/Core>

#include "base/pose.h"
#include "base/similarity_transform.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestInitialization) {
  const Eigen::Vector4d qvec =
      NormalizeQuaternion(Eigen::Vector4d(0.1, 0.3, 0.2, 0.4));

  const SimilarityTransform3 tform(2, qvec, Eigen::Vector3d(100, 10, 0.5));

  BOOST_CHECK_CLOSE(tform.Scale(), 2, 1e-10);

  BOOST_CHECK_CLOSE(tform.Rotation()(0), qvec(0), 1e-10);
  BOOST_CHECK_CLOSE(tform.Rotation()(1), qvec(1), 1e-10);
  BOOST_CHECK_CLOSE(tform.Rotation()(2), qvec(2), 1e-10);
  BOOST_CHECK_CLOSE(tform.Rotation()(3), qvec(3), 1e-10);

  BOOST_CHECK_CLOSE(tform.Translation()[0], 100, 1e-10);
  BOOST_CHECK_CLOSE(tform.Translation()[1], 10, 1e-10);
  BOOST_CHECK_CLOSE(tform.Translation()[2], 0.5, 1e-10);
}

void TestEstimationWithNumCoords(const size_t num_coords) {
  const SimilarityTransform3 orig_tform(2, Eigen::Vector4d(0.1, 0.3, 0.2, 0.4),
                                        Eigen::Vector3d(100, 10, 0.5));

  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;

  for (size_t i = 0; i < num_coords; ++i) {
    src.emplace_back(i, i + 2, i * i);
    dst.push_back(src.back());
    orig_tform.TransformPoint(&dst.back());
  }

  SimilarityTransform3 est_tform;
  est_tform.Estimate(src, dst);

  BOOST_CHECK((orig_tform.Matrix() - est_tform.Matrix()).norm() < 1e-6);
}

BOOST_AUTO_TEST_CASE(TestEstimation) {
  TestEstimationWithNumCoords(3);
  TestEstimationWithNumCoords(100);
}
