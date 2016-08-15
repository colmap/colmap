// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "base/feature.h"

#include "util/math.h"

namespace colmap {

std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
    const FeatureKeypoints& keypoints) {
  std::vector<Eigen::Vector2d> points(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
  }
  return points;
}

Eigen::MatrixXf L2NormalizeFeatureDescriptors(
    const Eigen::MatrixXf& descriptors) {
  return descriptors.rowwise().normalized();
}

Eigen::MatrixXf L1RootNormalizeFeatureDescriptors(
    const Eigen::MatrixXf& descriptors) {
  Eigen::MatrixXf descriptors_normalized(descriptors.rows(),
                                         descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    const float norm = descriptors.row(r).lpNorm<1>();
    descriptors_normalized.row(r) = descriptors.row(r) / norm;
    descriptors_normalized.row(r) =
        descriptors_normalized.row(r).array().sqrt();
  }
  return descriptors_normalized;
}

FeatureDescriptors FeatureDescriptorsToUnsignedByte(
    const Eigen::MatrixXf& descriptors) {
  FeatureDescriptors descriptors_unsigned_byte(descriptors.rows(),
                                               descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      descriptors_unsigned_byte(r, c) =
          TruncateCast<float, uint8_t>(scaled_value);
    }
  }
  return descriptors_unsigned_byte;
}

}  // namespace colmap
