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

#include "colmap/estimators/triangulation.h"

#include "colmap/estimators/essential_matrix.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/optim/combination_sampler.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/projection.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>

namespace colmap {

void TriangulationEstimator::SetMinTriAngle(const double min_tri_angle) {
  THROW_CHECK_GE(min_tri_angle, 0);
  min_tri_angle_ = min_tri_angle;
}

void TriangulationEstimator::SetResidualType(const ResidualType residual_type) {
  residual_type_ = residual_type;
}

void TriangulationEstimator::Estimate(const std::vector<X_t>& point_data,
                                      const std::vector<Y_t>& pose_data,
                                      std::vector<M_t>* models) const {
  THROW_CHECK_GE(point_data.size(), 2);
  THROW_CHECK_EQ(point_data.size(), pose_data.size());
  THROW_CHECK(models != nullptr);

  models->clear();

  if (point_data.size() == 2) {
    // Two-view triangulation.

    const M_t xyz = TriangulatePoint(pose_data[0].proj_matrix,
                                     pose_data[1].proj_matrix,
                                     point_data[0].point_normalized,
                                     point_data[1].point_normalized);

    if (HasPointPositiveDepth(pose_data[0].proj_matrix, xyz) &&
        HasPointPositiveDepth(pose_data[1].proj_matrix, xyz) &&
        CalculateTriangulationAngle(pose_data[0].proj_center,
                                    pose_data[1].proj_center,
                                    xyz) >= min_tri_angle_) {
      models->resize(1);
      (*models)[0] = xyz;
      return;
    }
  } else {
    // Multi-view triangulation.

    std::vector<Eigen::Matrix3x4d> proj_matrices;
    proj_matrices.reserve(point_data.size());
    std::vector<Eigen::Vector2d> points;
    points.reserve(point_data.size());
    for (size_t i = 0; i < point_data.size(); ++i) {
      proj_matrices.push_back(pose_data[i].proj_matrix);
      points.push_back(point_data[i].point_normalized);
    }

    const M_t xyz = TriangulateMultiViewPoint(proj_matrices, points);

    // Check for cheirality constraint.
    for (const auto& pose : pose_data) {
      if (!HasPointPositiveDepth(pose.proj_matrix, xyz)) {
        return;
      }
    }

    // Check for sufficient triangulation angle.
    for (size_t i = 0; i < pose_data.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        const double tri_angle = CalculateTriangulationAngle(
            pose_data[i].proj_center, pose_data[j].proj_center, xyz);
        if (tri_angle >= min_tri_angle_) {
          models->resize(1);
          (*models)[0] = xyz;
          return;
        }
      }
    }
  }
}

void TriangulationEstimator::Residuals(const std::vector<X_t>& point_data,
                                       const std::vector<Y_t>& pose_data,
                                       const M_t& xyz,
                                       std::vector<double>* residuals) const {
  THROW_CHECK_EQ(point_data.size(), pose_data.size());

  residuals->resize(point_data.size());

  for (size_t i = 0; i < point_data.size(); ++i) {
    if (residual_type_ == ResidualType::REPROJECTION_ERROR) {
      (*residuals)[i] =
          CalculateSquaredReprojectionError(point_data[i].point,
                                            xyz,
                                            pose_data[i].proj_matrix,
                                            *pose_data[i].camera);
    } else if (residual_type_ == ResidualType::ANGULAR_ERROR) {
      const double angular_error = CalculateNormalizedAngularError(
          point_data[i].point_normalized, xyz, pose_data[i].proj_matrix);
      (*residuals)[i] = angular_error * angular_error;
    }
  }
}

bool EstimateTriangulation(const EstimateTriangulationOptions& options,
                           const std::vector<Eigen::Vector2d>& points,
                           const std::vector<Rigid3d const*>& cams_from_world,
                           const std::vector<Camera const*>& cameras,
                           std::vector<char>* inlier_mask,
                           Eigen::Vector3d* xyz) {
  THROW_CHECK_NOTNULL(inlier_mask);
  THROW_CHECK_NOTNULL(xyz);
  THROW_CHECK_GE(points.size(), 2);
  THROW_CHECK_EQ(points.size(), cams_from_world.size());
  THROW_CHECK_EQ(points.size(), cameras.size());
  options.Check();

  std::vector<TriangulationEstimator::PointData> point_data;
  point_data.resize(points.size());
  std::vector<TriangulationEstimator::PoseData> pose_data;
  pose_data.resize(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    point_data[i].point = points[i];
    point_data[i].point_normalized = cameras[i]->CamFromImg(points[i]);
    pose_data[i].proj_matrix = cams_from_world[i]->ToMatrix();
    pose_data[i].proj_center = cams_from_world[i]->rotation.inverse() *
                               -cams_from_world[i]->translation;
    pose_data[i].camera = cameras[i];
  }

  // Robustly estimate track using LORANSAC.
  LORANSAC<TriangulationEstimator,
           TriangulationEstimator,
           InlierSupportMeasurer,
           CombinationSampler>
      ransac(options.ransac_options);
  ransac.estimator.SetMinTriAngle(options.min_tri_angle);
  ransac.estimator.SetResidualType(options.residual_type);
  ransac.local_estimator.SetMinTriAngle(options.min_tri_angle);
  ransac.local_estimator.SetResidualType(options.residual_type);
  const auto report = ransac.Estimate(point_data, pose_data);
  if (!report.success) {
    return false;
  }

  *inlier_mask = report.inlier_mask;
  *xyz = report.model;

  return report.success;
}

}  // namespace colmap
