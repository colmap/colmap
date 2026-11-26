// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cmath>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gmock/gmock.h>

namespace colmap {

template <typename T>
bool EigenMatrixMatchAndExplainShape(const T& lhs,
                                     const T& rhs,
                                     testing::MatchResultListener* listener) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    *listener << " have different shape (" << lhs.rows() << ", " << lhs.cols()
              << ") vs. (" << rhs.rows() << ", " << rhs.cols() << ")";
    return false;
  }
  return true;
}

template <typename T>
class EigenMatrixEqMatcher : public testing::MatcherInterface<T> {
 public:
  explicit EigenMatrixEqMatcher(T rhs) : rhs_(std::forward<T>(rhs)) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    if (!EigenMatrixMatchAndExplainShape(lhs, rhs_, listener)) {
      return false;
    }
    return lhs == rhs_;
  }

 private:
  const T rhs_;
};

template <typename T>
testing::PolymorphicMatcher<EigenMatrixEqMatcher<T>> EigenMatrixEq(T rhs) {
  return testing::MakePolymorphicMatcher(
      EigenMatrixEqMatcher<T>(std::forward<T>(rhs)));
}

template <typename T>
class EigenMatrixNearMatcher : public testing::MatcherInterface<T> {
 public:
  EigenMatrixNearMatcher(T rhs, double tol)
      : rhs_(std::forward<T>(rhs)), tol_(tol) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    if (!EigenMatrixMatchAndExplainShape(lhs, rhs_, listener)) {
      return false;
    }
    if (rhs_.isZero()) {
      // isApprox() is not well-defined for zero matrices.
      return lhs.norm() <= tol_;
    } else {
      return lhs.isApprox(rhs_, tol_);
    }
  }

 private:
  const T rhs_;
  const double tol_;
};

template <typename T>
testing::PolymorphicMatcher<EigenMatrixNearMatcher<T>> EigenMatrixNear(
    T rhs, double tol = Eigen::NumTraits<double>::dummy_precision()) {
  return testing::MakePolymorphicMatcher(
      EigenMatrixNearMatcher<T>(std::forward<T>(rhs), tol));
}

class SamplesFollowCovMatcher {
 public:
  SamplesFollowCovMatcher(Eigen::MatrixXd cov,
                          int null_dim,
                          double num_std_errors)
      : cov_(std::move(cov)),
        null_dim_(null_dim),
        num_std_errors_(num_std_errors) {}

  void DescribeTo(std::ostream* os) const {
    *os << "follow covariance (dim=" << cov_.rows()
        << ", null_dim=" << null_dim_ << ")";
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "do not follow covariance (dim=" << cov_.rows()
        << ", null_dim=" << null_dim_ << ")";
  }

  template <typename SamplesT>
  bool MatchAndExplain(const SamplesT& samples,
                       testing::MatchResultListener* listener) const {
    if (samples.empty()) {
      *listener << "sample list is empty";
      return false;
    }
    const int dim = cov_.rows();
    if (dim == 0 || cov_.cols() != dim) {
      *listener << "covariance must be non-empty square, got (" << cov_.rows()
                << ", " << cov_.cols() << ")";
      return false;
    }
    if (null_dim_ < 0 || null_dim_ >= dim) {
      *listener << "invalid null_dim " << null_dim_ << " for dimension " << dim;
      return false;
    }
    if (!(num_std_errors_ > 0.0)) {
      *listener << "num_std_errors must be positive, got " << num_std_errors_;
      return false;
    }

    const int num_samples = samples.size();
    for (int i = 0; i < num_samples; ++i) {
      if (samples[i].size() != dim) {
        *listener << "sample " << i << " has size " << samples[i].size()
                  << ", expected " << dim;
        return false;
      }
    }

    const int dof = dim - null_dim_;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov_);
    if (eig.info() != Eigen::Success ||
        !(eig.eigenvalues().tail(dof).array() > 0.0).all()) {
      *listener << "covariance does not have " << dof
                << " positive eigenvalues";
      return false;
    }

    Eigen::VectorXd inv_eigenvalues = eig.eigenvalues().cwiseInverse();
    inv_eigenvalues.head(null_dim_).setZero();
    const Eigen::MatrixXd cov_pinv = eig.eigenvectors() *
                                     inv_eigenvalues.asDiagonal() *
                                     eig.eigenvectors().transpose();

    double mean_dist2 = 0.0;
    Eigen::VectorXd variances = Eigen::VectorXd::Zero(dim);
    for (const auto& sample : samples) {
      mean_dist2 += sample.dot(cov_pinv * sample) / num_samples;
      variances += sample.cwiseAbs2() / num_samples;
    }

    const double max_dist2_error =
        num_std_errors_ * std::sqrt(2.0 * dof / num_samples);
    if (!(std::abs(mean_dist2 - dof) <= max_dist2_error)) {
      *listener << "mean squared Mahalanobis distance " << mean_dist2
                << " differs from dof " << dof << " by more than "
                << max_dist2_error;
      return false;
    }

    const double max_variance_error =
        num_std_errors_ * std::sqrt(2.0 / num_samples);
    for (int i = 0; i < dim; ++i) {
      if (cov_(i, i) <= 0.0) {
        if (!(variances(i) == 0.0)) {
          *listener << "coordinate " << i << " has non-zero variance "
                    << variances(i) << " for non-positive predicted variance "
                    << cov_(i, i);
          return false;
        }
        continue;
      }
      const double ratio = variances(i) / cov_(i, i);
      if (!(std::abs(ratio - 1.0) <= max_variance_error)) {
        *listener << "coordinate " << i << " variance ratio " << ratio
                  << " differs from 1 by more than " << max_variance_error;
        return false;
      }
    }
    return true;
  }

 private:
  const Eigen::MatrixXd cov_;
  const int null_dim_;
  const double num_std_errors_;
};

// Checks that zero-mean samples follow the predicted covariance, which may
// have a null space of dimension `null_dim`: the mean squared Mahalanobis
// distance must match the chi-square degrees of freedom and each coordinate's
// sample variance must match the predicted variance within `num_std_errors`
// standard errors.
inline testing::PolymorphicMatcher<SamplesFollowCovMatcher> SamplesFollowCov(
    Eigen::MatrixXd cov, int null_dim = 0, double num_std_errors = 4.0) {
  return testing::MakePolymorphicMatcher(
      SamplesFollowCovMatcher(std::move(cov), null_dim, num_std_errors));
}

}  // namespace colmap
