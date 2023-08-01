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

#include "colmap/estimators/generalized_absolute_pose.h"

#include "colmap/estimators/generalized_absolute_pose_coeffs.h"
#include "colmap/math/polynomial.h"
#include "colmap/scene/projection.h"
#include "colmap/util/logging.h"

#include <array>

namespace colmap {
namespace {

// Check whether the rays are close to parallel.
bool CheckParallelRays(const Eigen::Vector3d& ray1,
                       const Eigen::Vector3d& ray2,
                       const Eigen::Vector3d& ray3) {
  const double kParallelThreshold = 1e-5;
  return ray1.cross(ray2).isApproxToConstant(0, kParallelThreshold) &&
         ray1.cross(ray3).isApproxToConstant(0, kParallelThreshold);
}

// Check whether the points are close to collinear.
bool CheckCollinearPoints(const Eigen::Vector3d& X1,
                          const Eigen::Vector3d& X2,
                          const Eigen::Vector3d& X3) {
  const double kMinNonCollinearity = 1e-5;
  const Eigen::Vector3d X12 = X2 - X1;
  const double non_collinearity_measure =
      X12.cross(X1 - X3).squaredNorm() / X12.squaredNorm();
  return non_collinearity_measure < kMinNonCollinearity;
}

Eigen::Vector6d ComposePlueckerLine(const Rigid3d& rig_from_cam,
                                    const Eigen::Vector3d& ray_in_cam) {
  const Eigen::Vector3d ray_in_rig =
      (rig_from_cam.rotation * ray_in_cam).normalized();
  Eigen::Vector6d pluecker;
  pluecker << ray_in_rig, rig_from_cam.translation.cross(ray_in_rig);
  return pluecker;
}

Eigen::Vector3d PointFromPlueckerLineAndDepth(const Eigen::Vector6d& pluecker,
                                              const double depth) {
  return pluecker.head<3>().cross(pluecker.tail<3>()) +
         depth * pluecker.head<3>();
}

// Compute the coefficients from the system of 3 equations, nonlinear in the
// depth of the points. Inputs are three Pluecker lines and the locations of
// their corresponding points in 3D. The system of equations comes from the
// distance constraints between 3D points:
//
//    || f_i - f_j ||^2 = || (q_i x q_i' + lambda_i * q_i) -
//                           (q_j x q_j' + lambda_j * q_j) ||^2
//
// where [q_i; q_i'] is the Pluecker coordinate of bearing i and f_i is the
// coordinate of the corresponding 3D point in the global coordinate system. A
// 3D point in the local camera coordinate system along this line is
// parameterized through the depth scalar lambda_i as:
//
//    B_fi = q_i x q_i' + lambda_i * q_i.
//
Eigen::Matrix<double, 3, 6> ComputePolynomialCoefficients(
    const std::vector<Eigen::Vector6d>& plueckers,
    const std::vector<Eigen::Vector3d>& points3D) {
  CHECK_EQ(plueckers.size(), 3);
  CHECK_EQ(points3D.size(), 3);

  Eigen::Matrix<double, 3, 6> K;
  const std::array<int, 3> is = {{0, 0, 1}};
  const std::array<int, 3> js = {{1, 2, 2}};

  for (int k = 0; k < 3; ++k) {
    const int i = is[k];
    const int j = js[k];
    const Eigen::Vector3d moment_difference =
        plueckers[i].head<3>().cross(plueckers[i].tail<3>()) -
        plueckers[j].head<3>().cross(plueckers[j].tail<3>());
    K(k, 0) = 1;
    K(k, 1) = -2 * plueckers[i].head<3>().dot(plueckers[j].head<3>());
    K(k, 2) = 2 * moment_difference.dot(plueckers[i].head<3>());
    K(k, 3) = 1;
    K(k, 4) = -2 * moment_difference.dot(plueckers[j].head<3>());
    K(k, 5) = moment_difference.squaredNorm() -
              (points3D[i] - points3D[j]).squaredNorm();
  }

  return K;
}

// Solve quadratics of the form: x^2 + bx + c = 0.
int SolveQuadratic(const double b, const double c, double* roots) {
  const double delta = b * b - 4 * c;
  // Do not allow complex solutions.
  if (delta >= 0) {
    const double sqrt_delta = std::sqrt(delta);
    roots[0] = -0.5 * (b + sqrt_delta);
    roots[1] = -0.5 * (b - sqrt_delta);
    return 2;
  } else {
    return 0;
  }
}

// Given lambda_j, return the values for lambda_i, where:
//     k1 lambda_i^2 + (k2 lambda_j + k3) lambda_i
//      + k4 lambda_j^2 + k5 lambda_j + k6          = 0.
void ComputeLambdaValues(const Eigen::Matrix<double, 3, 6>::ConstRowXpr& k,
                         const double lambda_j,
                         std::vector<double>* lambdas_i) {
  // Note that we solve x^2 + bx + c = 0, since k(0) is one.
  double roots[2];
  const int num_solutions =
      SolveQuadratic(k(1) * lambda_j + k(2),
                     lambda_j * (k(3) * lambda_j + k(4)) + k(5),
                     roots);
  for (int i = 0; i < num_solutions; ++i) {
    if (roots[i] > 0) {
      lambdas_i->push_back(roots[i]);
    }
  }
}

// Given the coefficients of the polynomial system return the depths of the
// points along the Pluecker lines. Use Sylvester resultant to get and 8th
// degree polynomial for lambda_3 and back-substite in the original equations.
std::vector<Eigen::Vector3d> ComputeDepthsSylvester(
    const Eigen::Matrix<double, 3, 6>& K) {
  const Eigen::Matrix<double, 9, 1> coeffs = ComputeDepthsSylvesterCoeffs(K);

  Eigen::VectorXd roots_real;
  Eigen::VectorXd roots_imag;
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
    return std::vector<Eigen::Vector3d>();
  }

  // Back-substitute every lambda_3 to the system of equations.
  std::vector<Eigen::Vector3d> depths;
  depths.reserve(roots_real.size());
  for (Eigen::VectorXd::Index i = 0; i < roots_real.size(); ++i) {
    const double kMaxRootImagRatio = 1e-3;
    if (std::abs(roots_imag(i)) > kMaxRootImagRatio * std::abs(roots_real(i))) {
      continue;
    }

    const double lambda_3 = roots_real(i);
    if (lambda_3 <= 0) {
      continue;
    }

    std::vector<double> lambdas_2;
    ComputeLambdaValues(K.row(2), lambda_3, &lambdas_2);

    // Now we have two depths, lambda_2 and lambda_3. From the two remaining
    // equations, we must get the same lambda_1, otherwise the solution is
    // invalid.
    for (const double lambda_2 : lambdas_2) {
      std::vector<double> lambdas_1_1;
      ComputeLambdaValues(K.row(0), lambda_2, &lambdas_1_1);
      std::vector<double> lambdas_1_2;
      ComputeLambdaValues(K.row(1), lambda_3, &lambdas_1_2);
      for (const double lambda_1_1 : lambdas_1_1) {
        for (const double lambda_1_2 : lambdas_1_2) {
          const double kMaxLambdaRatio = 1e-2;
          if (std::abs(lambda_1_1 - lambda_1_2) <
              kMaxLambdaRatio * std::max(lambda_1_1, lambda_1_2)) {
            const double lambda_1 = (lambda_1_1 + lambda_1_2) / 2;
            depths.emplace_back(lambda_1, lambda_2, lambda_3);
          }
        }
      }
    }
  }

  return depths;
}

}  // namespace

std::vector<GP3PEstimator::M_t> GP3PEstimator::Estimate(
    const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D) {
  CHECK_EQ(points2D.size(), 3);
  CHECK_EQ(points3D.size(), 3);

  if (CheckCollinearPoints(points3D[0], points3D[1], points3D[2])) {
    return {};
  }

  // Transform 2D points into compact Pluecker line representation.
  std::vector<Eigen::Vector6d> plueckers(3);
  for (size_t i = 0; i < 3; ++i) {
    plueckers[i] = ComposePlueckerLine(Inverse(points2D[i].cam_from_rig),
                                       points2D[i].ray_in_cam);
  }

  if (CheckParallelRays(plueckers[0].head<3>(),
                        plueckers[1].head<3>(),
                        plueckers[2].head<3>())) {
    return {};
  }

  // Compute the coefficients k1, k2, k3 using Eq. 4.
  const Eigen::Matrix<double, 3, 6> K =
      ComputePolynomialCoefficients(plueckers, points3D);

  // Compute the depths along the Pluecker lines of the observations.
  const std::vector<Eigen::Vector3d> depths = ComputeDepthsSylvester(K);
  if (depths.empty()) {
    return {};
  }

  // For all valid depth values, compute the transformation between points in
  // the camera and the world frame. This uses Umeyama's method rather than the
  // algorithm proposed in the paper, since Umeyama's method is numerically more
  // stable and this part is not a bottleneck.

  Eigen::Matrix3d points3D_in_world;
  for (size_t i = 0; i < 3; ++i) {
    points3D_in_world.col(i) = points3D[i];
  }

  std::vector<M_t> models(depths.size());
  for (size_t i = 0; i < depths.size(); ++i) {
    Eigen::Matrix3d points3D_in_rig;
    for (size_t j = 0; j < 3; ++j) {
      points3D_in_rig.col(j) =
          PointFromPlueckerLineAndDepth(plueckers[j], depths[i][j]);
    }

    const Eigen::Matrix4d rig_from_world =
        Eigen::umeyama(points3D_in_world, points3D_in_rig, false);
    models[i] =
        Rigid3d(Eigen::Quaterniond(rig_from_world.topLeftCorner<3, 3>()),
                rig_from_world.topRightCorner<3, 1>());
  }

  return models;
}

void GP3PEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& rig_from_world,
                              std::vector<double>* residuals) {
  CHECK_EQ(points2D.size(), points3D.size());
  residuals->resize(points2D.size(), 0);
  for (size_t i = 0; i < points2D.size(); ++i) {
    const Eigen::Vector3d point3D_in_cam =
        points2D[i].cam_from_rig * (rig_from_world * points3D[i]);
    // Check if 3D point is in front of camera.
    if (point3D_in_cam.z() > std::numeric_limits<double>::epsilon()) {
      if (residual_type == ResidualType::CosineDistance) {
        const double cosine_dist =
            1 - point3D_in_cam.normalized().dot(points2D[i].ray_in_cam);
        (*residuals)[i] = cosine_dist * cosine_dist;
      } else if (residual_type == ResidualType::ReprojectionError) {
        (*residuals)[i] = (point3D_in_cam.hnormalized() -
                           points2D[i].ray_in_cam.hnormalized())
                              .squaredNorm();
      } else {
        LOG(FATAL) << "Invalid residual type";
      }
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

}  // namespace colmap
