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

#ifndef COLMAP_SRC_ESTIMATORS_GP3P_H_
#define COLMAP_SRC_ESTIMATORS_GP3P_H_

#include <vector>

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// Solver for the Generalized P3P problem (NP3P or GP3P), based on:
//
//      Lee, Gim Hee, et al. "Minimal solutions for pose estimation of a
//      multi-camera system." Robotics Research. Springer International
//      Publishing, 2016. 521-538.
//
// This class is based on an original implementation by Federico Camposeco.
class GP3PEstimator {
 public:
  // The generalized image observations, which is composed of the relative pose
  // of the specific camera in the generalized camera and its image observation.
  struct X_t {
    // The relative transformation from the generalized camera to the camera
    // frame of the observation.
    Eigen::Matrix<double, 3, 4, Eigen::DontAlign> rel_tform;
    // The 2D image feature observation.
    Eigen::Matrix<double, 2, 1, Eigen::DontAlign> xy;
  };

  // The observed 3D feature points in the world frame.
  typedef Eigen::Vector3d Y_t;
  // The transformation from the world to the generalized camera frame.
  typedef Eigen::Matrix3x4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate the most probable solution of the GP3P problem from a set of
  // three 2D-3D point correspondences.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points2D,
                                   const std::vector<Y_t>& points3D);

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix of the generalized camera.
  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& proj_matrix, std::vector<double>* residuals);
};

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_GP3P_H_
