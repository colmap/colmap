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

#include "colmap/estimators/similarity_transform.h"

namespace colmap {
namespace {

template <bool kEstimateScale>
inline bool EstimateRigidOrSim3d(const std::vector<Eigen::Vector3d>& src,
                                 const std::vector<Eigen::Vector3d>& tgt,
                                 Eigen::Matrix3x4d& tgt_from_src) {
  std::vector<Eigen::Matrix3x4d> models;
  SimilarityTransformEstimator<3, kEstimateScale>().Estimate(src, tgt, &models);
  if (models.empty()) {
    return false;
  }
  THROW_CHECK_EQ(models.size(), 1);
  tgt_from_src = models[0];
  return true;
}

template <bool kEstimateScale>
inline typename RANSAC<SimilarityTransformEstimator<3, kEstimateScale>>::Report
EstimateRigidOrSim3dRobust(const std::vector<Eigen::Vector3d>& src,
                           const std::vector<Eigen::Vector3d>& tgt,
                           const RANSACOptions& options,
                           Eigen::Matrix3x4d& tgt_from_src) {
  LORANSAC<SimilarityTransformEstimator<3, kEstimateScale>,
           SimilarityTransformEstimator<3, kEstimateScale>>
      ransac(options);
  auto report = ransac.Estimate(src, tgt);
  if (report.success) {
    tgt_from_src = report.model;
  }
  return report;
}

}  // namespace

bool EstimateRigid3d(const std::vector<Eigen::Vector3d>& src,
                     const std::vector<Eigen::Vector3d>& tgt,
                     Rigid3d& tgt_from_src) {
  Eigen::Matrix3x4d tgt_from_src_mat = Eigen::Matrix3x4d::Zero();
  if (!EstimateRigidOrSim3d<false>(src, tgt, tgt_from_src_mat)) {
    return false;
  }
  tgt_from_src = Rigid3d::FromMatrix(tgt_from_src_mat);
  return true;
}

typename RANSAC<SimilarityTransformEstimator<3, false>>::Report
EstimateRigid3dRobust(const std::vector<Eigen::Vector3d>& src,
                      const std::vector<Eigen::Vector3d>& tgt,
                      const RANSACOptions& options,
                      Rigid3d& tgt_from_src) {
  Eigen::Matrix3x4d tgt_from_src_mat = Eigen::Matrix3x4d::Zero();
  auto report =
      EstimateRigidOrSim3dRobust<false>(src, tgt, options, tgt_from_src_mat);
  tgt_from_src = Rigid3d::FromMatrix(tgt_from_src_mat);
  return report;
}

bool EstimateSim3d(const std::vector<Eigen::Vector3d>& src,
                   const std::vector<Eigen::Vector3d>& tgt,
                   Sim3d& tgt_from_src) {
  Eigen::Matrix3x4d tgt_from_src_mat = Eigen::Matrix3x4d::Zero();
  if (!EstimateRigidOrSim3d<true>(src, tgt, tgt_from_src_mat)) {
    return false;
  }
  tgt_from_src = Sim3d::FromMatrix(tgt_from_src_mat);
  return true;
}

typename RANSAC<SimilarityTransformEstimator<3, true>>::Report
EstimateSim3dRobust(const std::vector<Eigen::Vector3d>& src,
                    const std::vector<Eigen::Vector3d>& tgt,
                    const RANSACOptions& options,
                    Sim3d& tgt_from_src) {
  Eigen::Matrix3x4d tgt_from_src_mat = Eigen::Matrix3x4d::Zero();
  auto report =
      EstimateRigidOrSim3dRobust<true>(src, tgt, options, tgt_from_src_mat);
  tgt_from_src = Sim3d::FromMatrix(tgt_from_src_mat);
  return report;
}

bool WeightedUmeyama(const std::vector<Eigen::Vector3d>& src_points,
                     const std::vector<Eigen::Vector3d>& dst_points,
                     const std::vector<Eigen::Matrix3d>& covariances,
                     Sim3d* tgt_from_src) {
  CHECK_EQ(src_points.size(), dst_points.size());
  CHECK_EQ(src_points.size(), covariances.size());
  
  // Convert vector of points to Eigen matrices for the template version
  using MatrixType = Eigen::Matrix<double, 3, Eigen::Dynamic>;
  MatrixType src_mat(3, src_points.size());
  MatrixType dst_mat(3, dst_points.size());
  
  for (size_t i = 0; i < src_points.size(); ++i) {
    src_mat.col(i) = src_points[i];
    dst_mat.col(i) = dst_points[i];
  }
  
  // Call the template version to do the actual computation
  Eigen::Matrix<double, 3, 4> tgt_from_src_mat = 
      WeightedUmeyama(src_mat, dst_mat, covariances, true);
  
  // Check if result is valid
  if (tgt_from_src_mat.hasNaN()) {
    return false;
  }
  
  // Convert result to Sim3d
  *tgt_from_src = Sim3d::FromMatrix(tgt_from_src_mat);
  
  return true;
}

template <typename Derived1, typename Derived2>
Eigen::Matrix<typename Derived1::Scalar, Derived1::RowsAtCompileTime,
              Derived1::RowsAtCompileTime + 1>
WeightedUmeyama(const Eigen::MatrixBase<Derived1>& src,
               const Eigen::MatrixBase<Derived2>& dst,
               const std::vector<Eigen::Matrix3d>& covariances,
               bool with_scaling) {
  typedef typename Derived1::Scalar Scalar;
  typedef Eigen::Matrix<Scalar, Derived1::RowsAtCompileTime, 1> Vector;
  typedef Eigen::Matrix<Scalar, Derived1::RowsAtCompileTime,
                        Derived1::RowsAtCompileTime>
      Matrix;
  typedef Eigen::Matrix<Scalar, Derived1::RowsAtCompileTime,
                        Derived1::RowsAtCompileTime + 1>
      Result;

  const int rows = src.rows();
  const int cols = src.cols();

  THROW_CHECK_EQ(rows, dst.rows());
  THROW_CHECK_EQ(cols, dst.cols());
  THROW_CHECK_EQ(static_cast<size_t>(cols), covariances.size());

  // Prepare weight matrix for each axis based on the inverse of variance
  std::vector<Vector> weights(cols);
  
  // Create axis-specific weights from covariance diagonal
  for (int i = 0; i < cols; ++i) {
    // Extract axis variances from covariance
    Vector variances = covariances[i].diagonal();
    // Convert to weights (higher uncertainty = lower weight)
    // Apply a minimum threshold for numerical stability
    for (int j = 0; j < rows; j++) {
      weights[i](j) = 1.0 / std::max(variances(j), 1e-8);
    }
  }

  // Compute weighted centroids
  Vector src_mean = Vector::Zero(rows);
  Vector dst_mean = Vector::Zero(rows);
  Vector weight_sum = Vector::Zero(rows);
  
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < rows; j++) {
      src_mean(j) += weights[i](j) * src.col(i)(j);
      dst_mean(j) += weights[i](j) * dst.col(i)(j);
      weight_sum(j) += weights[i](j);
    }
  }
  
  // Normalize by weight sum
  for (int j = 0; j < rows; j++) {
    if (weight_sum(j) > 0) {
      src_mean(j) /= weight_sum(j);
      dst_mean(j) /= weight_sum(j);
    }
  }

  // Compute weighted covariance matrix
  Matrix covariance = Matrix::Zero(rows, rows);
  Scalar src_variance = 0.0;
  
  for (int i = 0; i < cols; ++i) {
    Vector src_diff = src.col(i) - src_mean;
    Vector dst_diff = dst.col(i) - dst_mean;
    
    // Weight each dimension according to its uncertainty
    Vector weighted_src_diff = src_diff;
    for (int j = 0; j < rows; j++) {
      weighted_src_diff(j) *= std::sqrt(weights[i](j));
      dst_diff(j) *= std::sqrt(weights[i](j));
    }
    
    // Build weighted cross-covariance
    covariance += dst_diff * weighted_src_diff.transpose();
    
    // For source variance
    src_variance += weighted_src_diff.squaredNorm();
  }
  
  // SVD
  Eigen::JacobiSVD<Matrix> svd(covariance, 
                            Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix rotation = svd.matrixU() * svd.matrixV().transpose();
  
  // Special case: reflection
  if (rotation.determinant() < 0) {
    Matrix S = Matrix::Identity(rows, rows);
    S(rows - 1, rows - 1) = -1;
    rotation = svd.matrixU() * S * svd.matrixV().transpose();
  }
  
  // Compute scaling factor
  Scalar scale = 1;
  if (with_scaling && src_variance > 0) {
    scale = svd.singularValues().sum() / src_variance;
  }
  
  // Compute translation
  Vector translation = dst_mean - scale * rotation * src_mean;
  
  // Compose similarity transformation matrix
  Result similarity;
  similarity.template block<Derived1::RowsAtCompileTime, 
                           Derived1::RowsAtCompileTime>(0, 0) = scale * rotation;
  similarity.col(rows) = translation;
  
  // Verify that the transformation actually improves the alignment
  double before_error = 0.0;
  double after_error = 0.0;
  
  for (int i = 0; i < cols; ++i) {
    Vector transformed = scale * rotation * src.col(i) + translation;
    
    // Calculate weighted MSE for each point
    for (int j = 0; j < rows; j++) {
      before_error += weights[i](j) * std::pow(dst.col(i)(j) - src.col(i)(j), 2);
      after_error += weights[i](j) * std::pow(dst.col(i)(j) - transformed(j), 2);
    }
  }
  
  // If the transform makes the error worse, return identity transform
  if (after_error > before_error) {
    LOG(WARNING) << "Weighted alignment increased error from " << before_error 
                 << " to " << after_error << ", falling back to identity transform";
    similarity.template block<Derived1::RowsAtCompileTime, 
                           Derived1::RowsAtCompileTime>(0, 0) = Matrix::Identity(rows, rows);
    similarity.col(rows) = Vector::Zero(rows);
  } else {
    LOG(INFO) << "Weighted alignment improved error from " << before_error 
              << " to " << after_error;
  }
  
  return similarity;
}

// Template instantiation for the 3D case we need
template Eigen::Matrix<double, 3, 4> WeightedUmeyama<
    Eigen::Matrix<double, 3, Eigen::Dynamic>,
    Eigen::Matrix<double, 3, Eigen::Dynamic>>(
    const Eigen::MatrixBase<Eigen::Matrix<double, 3, Eigen::Dynamic>>& src,
    const Eigen::MatrixBase<Eigen::Matrix<double, 3, Eigen::Dynamic>>& dst,
    const std::vector<Eigen::Matrix3d>& covariances,
    bool with_scaling);

bool EstimateWeightedSim3d(const std::vector<Eigen::Vector3d>& src,
                          const std::vector<Eigen::Vector3d>& tgt,
                          const std::vector<Eigen::Matrix3d>& covariances,
                          Sim3d& tgt_from_src) {
  CHECK_EQ(src.size(), tgt.size());
  CHECK_EQ(src.size(), covariances.size());
  
  if (src.size() < 3) {
    return false;
  }
  
  return WeightedUmeyama(src, tgt, covariances, &tgt_from_src);
}

void PrintAxisErrors(const std::vector<Eigen::Vector3d>& src,
                    const std::vector<Eigen::Vector3d>& tgt,
                    const std::vector<Eigen::Matrix3d>& covariances,
                    const Sim3d& tgt_from_src) {
  CHECK_EQ(src.size(), tgt.size());
  CHECK_EQ(src.size(), covariances.size());
  
  // Compute per-axis weights based on the inverse of variances
  std::vector<Eigen::Vector3d> weights(src.size());
  Eigen::Vector3d total_weights = Eigen::Vector3d::Zero();
  
  // Extract axis-specific weights from covariance diagonal
  for (size_t i = 0; i < src.size(); ++i) {
    // Get diagonal elements (variances for each axis)
    Eigen::Vector3d variances = covariances[i].diagonal();
    
    // Convert to weights (higher uncertainty = lower weight)
    for (int j = 0; j < 3; j++) {
      weights[i](j) = 1.0 / std::max(variances(j), 1e-8);
      total_weights(j) += weights[i](j);
    }
  }
  
  // Normalize weights per axis
  for (size_t i = 0; i < weights.size(); ++i) {
    for (int j = 0; j < 3; j++) {
      if (total_weights(j) > 0) {
        weights[i](j) /= total_weights(j);
      }
    }
  }
  
  Eigen::Vector3d weighted_mse = Eigen::Vector3d::Zero();
  double total_mse = 0.0;
  
  for (size_t i = 0; i < src.size(); ++i) {
    const Eigen::Vector3d transformed = tgt_from_src * src[i];
    const Eigen::Vector3d diff = tgt[i] - transformed;
    
    // Calculate per-axis weighted MSE
    for (int j = 0; j < 3; ++j) {
      weighted_mse(j) += weights[i](j) * diff(j) * diff(j);
    }
    
    // Calculate total weighted MSE (sum of per-axis MSEs)
    total_mse += diff(0) * diff(0) * weights[i](0) + 
                 diff(1) * diff(1) * weights[i](1) + 
                 diff(2) * diff(2) * weights[i](2);
  }
  
  LOG(INFO) << "Weighted MSE per axis - X: " << weighted_mse(0)
            << ", Y: " << weighted_mse(1)
            << ", Z: " << weighted_mse(2)
            << ", Total: " << total_mse;
}

}  // namespace colmap
