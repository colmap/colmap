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

#define TEST_NAME "base/triangulation"
#include "util/testing.h"

#include <Eigen/Core>

#include "base/similarity_transform.h"
#include "base/triangulation.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestTriangulatePoint) {
  std::vector<Eigen::Vector3d> points3D(6);
  points3D[0] = Eigen::Vector3d(0, 0.1, 0.1);
  points3D[1] = Eigen::Vector3d(0, 1, 3);
  points3D[2] = Eigen::Vector3d(0, 1, 2);
  points3D[3] = Eigen::Vector3d(0.01, 0.2, 3);
  points3D[4] = Eigen::Vector3d(-1, 0.1, 1);
  points3D[5] = Eigen::Vector3d(0.1, 0.1, 0.2);

  Eigen::Matrix3x4d proj_matrix1 = Eigen::MatrixXd::Identity(3, 4);

  for (double qz = 0; qz < 1; qz += 0.2) {
    for (double tx = 0; tx < 10; tx += 2) {
      SimilarityTransform3 tform(1, Eigen::Vector4d(0.2, 0.3, 0.4, qz),
                                 Eigen::Vector3d(tx, 2, 3));

      Eigen::Matrix3x4d proj_matrix2 = tform.Matrix().topLeftCorner<3, 4>();

      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector4d point3D1(point3D(0), point3D(1), point3D(2), 1);
        Eigen::Vector3d point2D1 = proj_matrix1 * point3D1;
        Eigen::Vector3d point2D2 = proj_matrix2 * point3D1;
        point2D1 /= point2D1(2);
        point2D2 /= point2D2(2);

        const Eigen::Vector2d point2D1_N(point2D1(0), point2D1(1));
        const Eigen::Vector2d point2D2_N(point2D2(0), point2D2(1));

        const Eigen::Vector3d tri_point3D = TriangulatePoint(
            proj_matrix1, proj_matrix2, point2D1_N, point2D2_N);

        BOOST_CHECK((point3D - tri_point3D).norm() < 1e-10);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestCalculateTriangulationAngle) {
  const Eigen::Vector3d tvec1(0, 0, 0);
  const Eigen::Vector3d tvec2(0, 1, 0);

  BOOST_CHECK_CLOSE(
      CalculateTriangulationAngle(tvec1, tvec2, Eigen::Vector3d(0, 0, 100)),
      0.009999666687, 1e-8);
  BOOST_CHECK_CLOSE(
      CalculateTriangulationAngle(tvec1, tvec2, Eigen::Vector3d(0, 0, 50)),
      0.019997333973, 1e-8);
}
