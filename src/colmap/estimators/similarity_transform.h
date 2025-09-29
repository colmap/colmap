// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/manifold.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/sim3.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>

namespace colmap {

// 3D point with associated covariance
struct PointWithCovariance3D {
  Eigen::Vector3d point;
  Eigen::Matrix3d covariance;

  PointWithCovariance3D() = default;
  PointWithCovariance3D(const Eigen::Vector3d& p, const Eigen::Matrix3d& cov)
      : point(p), covariance(cov) {}
  explicit PointWithCovariance3D(const Eigen::Vector3d& p)
      : point(p), covariance(Eigen::Matrix3d::Identity()) {}
};

// N-D similarity transform estimator from corresponding point pairs in the
// source and destination coordinate systems.
//
// This algorithm is based on the following paper:
//
//      S. Umeyama. Least-Squares Estimation of Transformation Parameters
//      Between Two Point Patterns. IEEE Transactions on Pattern Analysis and
//      Machine Intelligence, Volume 13 Issue 4, Page 376-380, 1991.
//      http://www.stanford.edu/class/cs273/refs/umeyama.pdf
//
// and uses the Eigen implementation.
template <int kDim, bool kEstimateScale = true>
class SimilarityTransformEstimator {
 public:
  typedef Eigen::Matrix<double, kDim, 1> X_t;
  typedef Eigen::Matrix<double, kDim, 1> Y_t;
  typedef Eigen::Matrix<double, kDim, kDim + 1> M_t;

  // The minimum number of samples needed to estimate a model. Note that
  // this only returns the true minimal sample in the two-dimensional case.
  // For higher dimensions, the system will alway be over-determined.
  static const int kMinNumSamples = kDim;

  // Estimate the similarity transform.
  //
  // @param src      Set of corresponding source points.
  // @param tgt      Set of corresponding destination points.
  //
  // @return         4x4 homogeneous transformation matrix.
  static void Estimate(const std::vector<X_t>& src,
                       const std::vector<Y_t>& tgt,
                       std::vector<M_t>* tgt_from_src);

  // Calculate the transformation error for each corresponding point pair.
  //
  // Residuals are defined as the squared transformation error when
  // transforming the source to the destination coordinates.
  //
  // @param src           Set of corresponding points in the source coordinate
  //                      system as a Nx3 matrix.
  // @param tgt           Set of corresponding points in the destination
  //                      coordinate system as a Nx3 matrix.
  // @param tgt_from_src  4x4 homogeneous transformation matrix.
  // @param residuals     Output vector of residuals for each point pair.
  static void Residuals(const std::vector<X_t>& src,
                        const std::vector<Y_t>& tgt,
                        const M_t& tgt_from_src,
                        std::vector<double>* residuals);
};

// Covariance-aware similarity transform estimator
// Uses Ceres and covariance whitening; embeds covariance with the sampled data
template <bool kEstimateScale = true>
class CovarianceSimilarityTransformEstimator {
 public:
  typedef PointWithCovariance3D X_t;
  typedef Eigen::Vector3d Y_t;
  typedef Eigen::Matrix3x4d M_t;
  static const int kMinNumSamples = 3;

  void Estimate(const std::vector<X_t>& src,
                const std::vector<Y_t>& tgt,
                std::vector<M_t>* tgt_from_src);

  void Residuals(const std::vector<X_t>& src,
                 const std::vector<Y_t>& tgt,
                 const M_t& tgt_from_src,
                 std::vector<double>* residuals);

 private:
  M_t EstimateWithCovariances(const std::vector<X_t>& src,
                              const std::vector<Y_t>& tgt) const;
};

bool EstimateRigid3d(const std::vector<Eigen::Vector3d>& src,
                     const std::vector<Eigen::Vector3d>& tgt,
                     Rigid3d& tgt_from_src);

typename RANSAC<SimilarityTransformEstimator<3, false>>::Report
EstimateRigid3dRobust(const std::vector<Eigen::Vector3d>& src,
                      const std::vector<Eigen::Vector3d>& tgt,
                      const RANSACOptions& options,
                      Rigid3d& tgt_from_src);

bool EstimateSim3d(const std::vector<Eigen::Vector3d>& src,
                   const std::vector<Eigen::Vector3d>& tgt,
                   Sim3d& tgt_from_src);

typename RANSAC<SimilarityTransformEstimator<3, true>>::Report
EstimateSim3dRobust(const std::vector<Eigen::Vector3d>& src,
                    const std::vector<Eigen::Vector3d>& tgt,
                    const RANSACOptions& options,
                    Sim3d& tgt_from_src);

typename RANSAC<CovarianceSimilarityTransformEstimator<true>>::Report
EstimateSim3dRobust(const std::vector<Eigen::Vector3d>& src,
                    const std::vector<Eigen::Vector3d>& tgt,
                    const std::vector<Eigen::Matrix3d>& covariances,
                    const RANSACOptions& options,
                    Sim3d& tgt_from_src);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int kDim, bool kEstimateScale>
void SimilarityTransformEstimator<kDim, kEstimateScale>::Estimate(
    const std::vector<X_t>& src,
    const std::vector<Y_t>& tgt,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(src.size(), tgt.size());
  THROW_CHECK_GE(src.size(), kMinNumSamples);
  THROW_CHECK(models != nullptr);

  models->clear();

  using MatrixType = Eigen::Matrix<double, kDim, Eigen::Dynamic>;
  const Eigen::Map<const MatrixType> src_mat(
      reinterpret_cast<const double*>(src.data()), kDim, src.size());
  const Eigen::Map<const MatrixType> tgt_mat(
      reinterpret_cast<const double*>(tgt.data()), kDim, tgt.size());

  if (Eigen::FullPivLU<MatrixType>(src_mat).rank() < kMinNumSamples ||
      Eigen::FullPivLU<MatrixType>(tgt_mat).rank() < kMinNumSamples) {
    return;
  }

  const M_t sol = Eigen::umeyama(src_mat, tgt_mat, kEstimateScale)
                      .template topLeftCorner<kDim, kDim + 1>();

  if (sol.hasNaN()) {
    return;
  }

  models->resize(1);
  (*models)[0] = sol;
}

template <int kDim, bool kEstimateScale>
void SimilarityTransformEstimator<kDim, kEstimateScale>::Residuals(
    const std::vector<X_t>& src,
    const std::vector<Y_t>& tgt,
    const M_t& tgt_from_src,
    std::vector<double>* residuals) {
  const size_t num_points = src.size();
  THROW_CHECK_EQ(num_points, tgt.size());
  residuals->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    (*residuals)[i] =
        (tgt[i] - tgt_from_src * src[i].homogeneous()).squaredNorm();
  }
}

template <bool kEstimateScale>
void CovarianceSimilarityTransformEstimator<kEstimateScale>::Estimate(
    const std::vector<X_t>& src,
    const std::vector<Y_t>& tgt,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(src.size(), tgt.size());
  THROW_CHECK_GE(src.size(), kMinNumSamples);
  THROW_CHECK(models != nullptr);

  models->clear();
  const M_t sol = EstimateWithCovariances(src, tgt);
  if (sol.hasNaN()) {
    return;
  }
  models->resize(1);
  (*models)[0] = sol;
}

template <bool kEstimateScale>
void CovarianceSimilarityTransformEstimator<kEstimateScale>::Residuals(
    const std::vector<X_t>& src,
    const std::vector<Y_t>& tgt,
    const M_t& tgt_from_src,
    std::vector<double>* residuals) {
  const size_t num_points = src.size();
  THROW_CHECK_EQ(num_points, tgt.size());
  residuals->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const Y_t transformed_src = tgt_from_src * src[i].point.homogeneous();
    const Y_t error = tgt[i] - transformed_src;
    Eigen::LLT<Eigen::Matrix3d> llt(src[i].covariance);
    THROW_CHECK(llt.info() == Eigen::Success)
        << "Covariance matrix is not positive definite:\n" << src[i].covariance;
    (*residuals)[i] = llt.matrixU().solve(error).squaredNorm();
  }
}

template <bool kEstimateScale>
typename CovarianceSimilarityTransformEstimator<kEstimateScale>::M_t
CovarianceSimilarityTransformEstimator<kEstimateScale>::EstimateWithCovariances(
    const std::vector<X_t>& src, const std::vector<Y_t>& tgt) const {
  const size_t num_points = src.size();
  THROW_CHECK_EQ(num_points, tgt.size());

  const Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  const double log_scale = 0.0;

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  double rotation_params[4] = {rotation.x(), rotation.y(), rotation.z(), rotation.w()};
  double translation_params[3] = {translation.x(), translation.y(), translation.z()};
  double scale_params[1] = {log_scale};

  for (size_t i = 0; i < num_points; ++i) {
    ceres::CostFunction* cost_function =
        CovarianceWeightedCostFunctor<SimilarityTransformCostFunctor>::Create(
            src[i].covariance, src[i].point, tgt[i]);
    problem.AddResidualBlock(cost_function,
                             nullptr,
                             rotation_params,
                             translation_params,
                             scale_params);
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, rotation_params);
    if constexpr (!kEstimateScale) {
      problem.SetParameterBlockConstant(scale_params);
    }
    // Constrain log-scale to a reasonable range for numerical stability
    problem.SetParameterLowerBound(scale_params, 0, -30.0);
    problem.SetParameterUpperBound(scale_params, 0, 30.0);
  }

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.logging_type = ceres::SILENT;
  solver_options.max_num_iterations = 50;
  solver_options.function_tolerance = 1e-12;
  solver_options.gradient_tolerance = 1e-12;
  solver_options.parameter_tolerance = 1e-12;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  const Eigen::Quaterniond final_rotation(rotation_params[3],
                                          rotation_params[0],
                                          rotation_params[1],
                                          rotation_params[2]);
  const Eigen::Vector3d final_translation(translation_params[0],
                                          translation_params[1],
                                          translation_params[2]);
  const double final_scale = std::exp(scale_params[0]);

  M_t final_transform;
  final_transform.template leftCols<3>() =
      final_scale * final_rotation.toRotationMatrix();
  final_transform.template rightCols<1>() = final_translation;
  return final_transform;
}

}  // namespace colmap
