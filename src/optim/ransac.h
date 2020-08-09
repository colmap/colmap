// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#ifndef COLMAP_SRC_OPTIM_RANSAC_H_
#define COLMAP_SRC_OPTIM_RANSAC_H_

#include <cfloat>
#include <random>
#include <stdexcept>
#include <vector>

#include "optim/random_sampler.h"
#include "optim/support_measurement.h"
#include "util/alignment.h"
#include "util/logging.h"

namespace colmap {

struct RANSACOptions {
  // Maximum error for a sample to be considered as an inlier. Note that
  // the residual of an estimator corresponds to a squared error.
  double max_error = 0.0;

  // A priori assumed minimum inlier ratio, which determines the maximum number
  // of iterations. Only applies if smaller than `max_num_trials`.
  double min_inlier_ratio = 0.1;

  // Abort the iteration if minimum probability that one sample is free from
  // outliers is reached.
  double confidence = 0.99;

  // The num_trials_multiplier to the dynamically computed maximum number of
  // iterations based on the specified confidence value.
  double dyn_num_trials_multiplier = 3.0;

  // Number of random trials to estimate model from random subset.
  size_t min_num_trials = 0;
  size_t max_num_trials = std::numeric_limits<size_t>::max();

  void Check() const {
    CHECK_GT(max_error, 0);
    CHECK_GE(min_inlier_ratio, 0);
    CHECK_LE(min_inlier_ratio, 1);
    CHECK_GE(confidence, 0);
    CHECK_LE(confidence, 1);
    CHECK_LE(min_num_trials, max_num_trials);
  }
};

template <typename Estimator, typename SupportMeasurer = InlierSupportMeasurer,
          typename Sampler = RandomSampler>
class RANSAC {
 public:
  struct Report {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Whether the estimation was successful.
    bool success = false;

    // The number of RANSAC trials / iterations.
    size_t num_trials = 0;

    // The support of the estimated model.
    typename SupportMeasurer::Support support;

    // Boolean mask which is true if a sample is an inlier.
    std::vector<char> inlier_mask;

    // The estimated model.
    typename Estimator::M_t model;
  };

  explicit RANSAC(const RANSACOptions& options);

  // Determine the maximum number of trials required to sample at least one
  // outlier-free random set of samples with the specified confidence,
  // given the inlier ratio.
  //
  // @param num_inliers				The number of inliers.
  // @param num_samples				The total number of samples.
  // @param confidence				Confidence that one sample is
  //								outlier-free.
  // @param num_trials_multiplier   Multiplication factor to the computed
  //							    number of trials.
  //
  // @return               The required number of iterations.
  static size_t ComputeNumTrials(const size_t num_inliers,
                                 const size_t num_samples,
                                 const double confidence,
                                 const double num_trials_multiplier);

  // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y);

  // Objects used in RANSAC procedure. Access useful to define custom behavior
  // through options or e.g. to compute residuals.
  Estimator estimator;
  Sampler sampler;
  SupportMeasurer support_measurer;

 protected:
  RANSACOptions options_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename SupportMeasurer, typename Sampler>
RANSAC<Estimator, SupportMeasurer, Sampler>::RANSAC(
    const RANSACOptions& options)
    : sampler(Sampler(Estimator::kMinNumSamples)), options_(options) {
  options.Check();

  // Determine max_num_trials based on assumed `min_inlier_ratio`.
  const size_t kNumSamples = 100000;
  const size_t dyn_max_num_trials = ComputeNumTrials(
      static_cast<size_t>(options_.min_inlier_ratio * kNumSamples), kNumSamples,
      options_.confidence, options_.dyn_num_trials_multiplier);
  options_.max_num_trials =
      std::min<size_t>(options_.max_num_trials, dyn_max_num_trials);
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
size_t RANSAC<Estimator, SupportMeasurer, Sampler>::ComputeNumTrials(
    const size_t num_inliers, const size_t num_samples, const double confidence,
    const double num_trials_multiplier) {
  const double inlier_ratio = num_inliers / static_cast<double>(num_samples);

  const double nom = 1 - confidence;
  if (nom <= 0) {
    return std::numeric_limits<size_t>::max();
  }

  const double denom = 1 - std::pow(inlier_ratio, Estimator::kMinNumSamples);
  if (denom <= 0) {
    return 1;
  }

  return static_cast<size_t>(
      std::ceil(std::log(nom) / std::log(denom) * num_trials_multiplier));
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report
RANSAC<Estimator, SupportMeasurer, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y) {
  CHECK_EQ(X.size(), Y.size());

  const size_t num_samples = X.size();

  Report report;
  report.success = false;
  report.num_trials = 0;

  if (num_samples < Estimator::kMinNumSamples) {
    return report;
  }

  typename SupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;

  bool abort = false;

  const double max_residual = options_.max_error * options_.max_error;

  std::vector<double> residuals(num_samples);

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
  std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

  sampler.Initialize(num_samples);

  size_t max_num_trials = options_.max_num_trials;
  max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
  size_t dyn_max_num_trials = max_num_trials;

  for (report.num_trials = 0; report.num_trials < max_num_trials;
       ++report.num_trials) {
    if (abort) {
      report.num_trials += 1;
      break;
    }

    sampler.SampleXY(X, Y, &X_rand, &Y_rand);

    // Estimate model for current subset.
    const std::vector<typename Estimator::M_t> sample_models =
        estimator.Estimate(X_rand, Y_rand);

    // Iterate through all estimated models.
    for (const auto& sample_model : sample_models) {
      estimator.Residuals(X, Y, sample_model, &residuals);
      CHECK_EQ(residuals.size(), num_samples);

      const auto support = support_measurer.Evaluate(residuals, max_residual);

      // Save as best subset if better than all previous subsets.
      if (support_measurer.Compare(support, best_support)) {
        best_support = support;
        best_model = sample_model;

        dyn_max_num_trials = ComputeNumTrials(
            best_support.num_inliers, num_samples, options_.confidence,
            options_.dyn_num_trials_multiplier);
      }

      if (report.num_trials >= dyn_max_num_trials &&
          report.num_trials >= options_.min_num_trials) {
        abort = true;
        break;
      }
    }
  }

  report.support = best_support;
  report.model = best_model;

  // No valid model was found.
  if (report.support.num_inliers < estimator.kMinNumSamples) {
    return report;
  }

  report.success = true;

  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  estimator.Residuals(X, Y, report.model, &residuals);
  CHECK_EQ(residuals.size(), num_samples);

  report.inlier_mask.resize(num_samples);
  for (size_t i = 0; i < residuals.size(); ++i) {
    report.inlier_mask[i] = residuals[i] <= max_residual;
  }

  return report;
}

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_RANSAC_H_
