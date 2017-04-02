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

#ifndef COLMAP_SRC_BASE_SIMILARITY_TRANSFORM_H_
#define COLMAP_SRC_BASE_SIMILARITY_TRANSFORM_H_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// 3D similarity transformation with 7 degrees of freedom.
class SimilarityTransform3 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SimilarityTransform3();

  explicit SimilarityTransform3(const Eigen::Matrix3x4d& matrix);

  explicit SimilarityTransform3(
      const Eigen::Transform<double, 3, Eigen::Affine>& transform);

  SimilarityTransform3(const double scale, const Eigen::Vector4d& qvec,
                       const Eigen::Vector3d& tvec);

  void Estimate(const std::vector<Eigen::Vector3d>& src,
                const std::vector<Eigen::Vector3d>& dst);

  SimilarityTransform3 Inverse() const;

  void TransformPoint(Eigen::Vector3d* xyz) const;
  void TransformPose(Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) const;

  Eigen::Matrix4d Matrix() const;
  double Scale() const;
  Eigen::Vector4d Rotation() const;
  Eigen::Vector3d Translation() const;

 private:
  Eigen::Transform<double, 3, Eigen::Affine> transform_;
};

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::SimilarityTransform3)

#endif  // COLMAP_SRC_BASE_SIMILARITY_TRANSFORM_H_
