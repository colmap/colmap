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
  SamplesFollowCovMatcher(Eigen::MatrixXd cov, int null_dim)
      : cov_(std::move(cov)), null_dim_(null_dim) {}

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
    const int dim = cov_.rows();
    const int num_samples = samples.size();
    if (num_samples == 0 || dim == 0 || cov_.cols() != dim || null_dim_ < 0 ||
        null_dim_ >= dim) {
      *listener << "invalid inputs: " << num_samples << " samples, covariance ("
                << cov_.rows() << ", " << cov_.cols() << "), null_dim "
                << null_dim_;
      return false;
    }

    const int dof = dim - null_dim_;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov_);
    if (eig.info() != Eigen::Success ||
        !(eig.eigenvalues().tail(dof).array() > 0.0).all()) {
      *listener << "covariance does not have " << dof
                << " positive eigenvalues";
      return false;
    }
    constexpr double kNullTol = 1e-6;
    if (null_dim_ > 0 &&
        !(eig.eigenvalues().head(null_dim_).cwiseAbs().maxCoeff() <=
          kNullTol * eig.eigenvalues()(null_dim_))) {
      *listener << "covariance null eigenvalues are not negligible";
      return false;
    }

    const Eigen::MatrixXd whitening =
        eig.eigenvalues().tail(dof).cwiseSqrt().cwiseInverse().asDiagonal() *
        eig.eigenvectors().rightCols(dof).transpose();
    const auto null_basis = eig.eigenvectors().leftCols(null_dim_);

    Eigen::MatrixXd whitened_cov = Eigen::MatrixXd::Zero(dof, dof);
    Eigen::VectorXd variances = Eigen::VectorXd::Zero(dim);
    double mean_null_norm2 = 0.0;
    double mean_sample_norm2 = 0.0;
    for (int i = 0; i < num_samples; ++i) {
      const auto& sample = samples[i];
      if (sample.size() != dim) {
        *listener << "sample " << i << " has size " << sample.size()
                  << ", expected " << dim;
        return false;
      }
      const Eigen::VectorXd z = whitening * sample;
      whitened_cov.noalias() += (z * z.transpose()) / num_samples;
      variances += sample.cwiseAbs2() / num_samples;
      if (null_dim_ > 0) {
        mean_null_norm2 +=
            (null_basis.transpose() * sample).squaredNorm() / num_samples;
        mean_sample_norm2 += sample.squaredNorm() / num_samples;
      }
    }

    if (null_dim_ > 0 && !(mean_null_norm2 <= kNullTol * mean_sample_norm2)) {
      *listener << "samples have non-negligible null-space component ("
                << mean_null_norm2 << " vs. " << mean_sample_norm2 << ")";
      return false;
    }

    constexpr double kNumStdErrors = 4.0;
    const double mean_dist2 = whitened_cov.trace();
    const double max_dist2_error =
        kNumStdErrors * std::sqrt(2.0 * dof / num_samples);
    if (!(std::abs(mean_dist2 - dof) <= max_dist2_error)) {
      *listener << "mean squared Mahalanobis distance " << mean_dist2
                << " differs from dof " << dof << " by more than "
                << max_dist2_error;
      return false;
    }

    const double max_variance_error =
        kNumStdErrors * std::sqrt(2.0 / num_samples);
    const double max_whitened_error =
        (whitened_cov - Eigen::MatrixXd::Identity(dof, dof))
            .cwiseAbs()
            .maxCoeff();
    if (!(max_whitened_error <= max_variance_error)) {
      *listener << "max whitened covariance error " << max_whitened_error
                << " exceeds " << max_variance_error;
      return false;
    }

    for (int i = 0; i < dim; ++i) {
      if (cov_(i, i) <= 0.0) {
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
};

// Checks that zero-mean samples follow the predicted covariance, which may
// have a null space of dimension `null_dim`: null-space components must be
// negligible, the mean squared Mahalanobis distance must match the chi-square
// degrees of freedom, the whitened sample covariance must be close to identity,
// and each coordinate's sample variance must match the predicted variance.
inline testing::PolymorphicMatcher<SamplesFollowCovMatcher> SamplesFollowCov(
    Eigen::MatrixXd cov, int null_dim = 0) {
  return testing::MakePolymorphicMatcher(
      SamplesFollowCovMatcher(std::move(cov), null_dim));
}

}  // namespace colmap
