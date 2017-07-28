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

#ifndef COLMAP_SRC_ESTIMATORS_GENERALIZED_RELATIVE_POSE_H_
#define COLMAP_SRC_ESTIMATORS_GENERALIZED_RELATIVE_POSE_H_

#include <vector>

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/types.h"

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
  // the relative pose of the specific camera in the generalized camera and its
  // image observation.
  struct X_t {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // The relative transformation from the generalized camera to the camera
    // frame of the observation.
    Eigen::Matrix3x4d rel_tform;
    // The 2D image feature observation.
    Eigen::Vector2d xy;
  };

  // The normalized image feature points in the left camera.
  typedef X_t Y_t;
  // The relative transformation between the two generalized cameras.
  typedef Eigen::Matrix3x4d M_t;

  // The minimum number of samples needed to estimate a model. Note that in
  // theory the minimum required number of samples is 6 but Laurent Kneip showed
  // in his paper that using 8 samples is more stable.
  static const int kMinNumSamples = 8;

  // Estimate the most probable solution of the GR6P problem from a set of
  // six 2D-2D point correspondences.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                   const std::vector<Y_t>& points2);

  // Calculate the average squared reprojection error when triangulating and
  // reprojecting the correspondences between the two cameras.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& proj_matrix, std::vector<double>* residuals);
};

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::GR6PEstimator::X_t)

#endif  // COLMAP_SRC_ESTIMATORS_GENERALIZED_RELATIVE_POSE_H_
